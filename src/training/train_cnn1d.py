"""Training pipeline for the 1-D CNN classifier.

Trains on the FULL labelled training partition with BCEWithLogitsLoss, where
class imbalance is handled by pos_weight = N_legit / N_fraud (518.177) rather
than by oversampling or SMOTE. Early stopping and LR scheduling monitor validation PR-AUC.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import config, ExperimentConfig, CNN1DArchitectureConfig, CNN1DTrainingConfig
from src.models.cnn1d import CNN1DClassifier
from src.evaluation.evaluate_supervised import predict_dataloader
from src.utils import setup_logger, set_seed, save_json, ensure_directories

logger = setup_logger("CNN1DTrainer")


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Pick cuda, then Apple-Silicon mps, then cpu (unless a device is given)."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _max_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Best achievable F1 over all thresholds on the PR curve."""
    precisions, recalls, _ = precision_recall_curve(y_true, probs)
    denom = precisions + recalls
    f1 = np.where(denom > 0, 2 * precisions * recalls / np.where(denom > 0, denom, 1), 0.0)
    return float(f1.max()) if len(f1) > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one supervised epoch over the labelled training partition."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(features).view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return running_loss / max(total_samples, 1)


def train_cnn1d(
    train_loader: DataLoader,
    val_loader: DataLoader,
    exp_cfg: Any = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the 1-D CNN classifier with early stopping on validation PR-AUC.

    Parameters
    ----------
    train_loader : DataLoader
        Supervised training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    exp_cfg : ExperimentConfig or CNN1DTrainingConfig, optional
        Experiment configuration.

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        (best_model, training_history)
    """
    if exp_cfg is None:
        t_cfg = config.cnn1d_training
        m_cfg = config.cnn1d_model
    elif isinstance(exp_cfg, CNN1DTrainingConfig):
        t_cfg = exp_cfg
        m_cfg = config.cnn1d_model
    elif hasattr(exp_cfg, "cnn1d_training"):
        t_cfg = exp_cfg.cnn1d_training
        m_cfg = getattr(exp_cfg, "cnn1d_model", config.cnn1d_model)
    else:
        t_cfg = config.cnn1d_training
        m_cfg = config.cnn1d_model

    set_seed(t_cfg.random_seed)
    device = resolve_device(t_cfg.device)
    logger.info(f"Using device for training: {device}")

    model = CNN1DClassifier(m_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized CNN1DClassifier with {total_params:,} trainable parameters.")

    pos_weight = torch.tensor([t_cfg.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.learning_rate,
        weight_decay=t_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {
        "train_loss": [],
        "val_pr_auc": [],
        "val_f1": [],
        "learning_rate": [],
        "epoch_times": [],
        "best_epoch": 0,
        "best_val_pr_auc": 0.0,
        "pos_weight": t_cfg.pos_weight,
        "total_parameters": total_params,
    }

    ensure_directories(t_cfg.model_save_path.parent, t_cfg.history_save_path.parent)

    best_pr_auc = -1.0
    best_weights = None
    epochs_without_improvement = 0

    logger.info(f"Starting 1-D CNN training for up to {t_cfg.num_epochs} epochs (pos_weight={t_cfg.pos_weight:.3f}) ...")
    start_time = time.time()

    for epoch in range(1, t_cfg.num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_probs, y_val = predict_dataloader(model, val_loader, device)
        val_pr_auc = float(average_precision_score(y_val, val_probs))
        val_f1 = _max_f1(y_val, val_probs)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_pr_auc)

        history["train_loss"].append(train_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)
        history["epoch_times"].append(epoch_duration)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            epochs_without_improvement = 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            history["best_val_pr_auc"] = val_pr_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_pr_auc": val_pr_auc,
                    "val_f1": val_f1,
                    "config": {
                        "input_dim": m_cfg.input_dim,
                        "channels": list(m_cfg.channels),
                        "kernel_size": m_cfg.kernel_size,
                        "dropout": m_cfg.dropout,
                        "dense_dim": m_cfg.dense_dim,
                    },
                },
                t_cfg.model_save_path,
            )
            improvement_tag = " -> Best model saved"
        else:
            epochs_without_improvement += 1
            improvement_tag = ""

        logger.info(
            f"Epoch [{epoch:02d}/{t_cfg.num_epochs:02d}] "
            f"Train BCE: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.1e} | Time: {epoch_duration:.1f}s{improvement_tag}"
        )

        if epochs_without_improvement >= t_cfg.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Halting training.")
            break

    total_training_time = time.time() - start_time
    history["total_training_time"] = total_training_time
    logger.info(
        f"Training completed in {total_training_time:.1f}s. "
        f"Best epoch: {history['best_epoch']} with Val PR-AUC: {history['best_val_pr_auc']:.4f}"
    )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    save_json(history, t_cfg.history_save_path)
    logger.info(f"Training history saved to {t_cfg.history_save_path}")

    return model, history


def load_trained_cnn1d(checkpoint_path: Path, device: torch.device) -> CNN1DClassifier:
    """Load a trained 1-D CNN classifier from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved = checkpoint.get("config", {})
    m_cfg = CNN1DArchitectureConfig(**saved) if saved else CNN1DArchitectureConfig()
    model = CNN1DClassifier(m_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Successfully loaded 1-D CNN classifier from {checkpoint_path}")
    return model
