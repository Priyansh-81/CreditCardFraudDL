"""Supervised LSTM training with validation PR-AUC checkpointing."""
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import config, LSTMArchitectureConfig
from src.models.lstm import LSTMClassifier
from src.evaluation.evaluate_supervised import predict_dataloader
from src.utils import setup_logger, set_seed, save_json, ensure_directories
from src.training.train_mlp import resolve_device

logger = setup_logger("LSTMTrainer")


def _max_f1(y_true, probs):
    p, r, _ = precision_recall_curve(y_true, probs)
    denom = p + r
    scores = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(scores.max())


def train_lstm(train_loader, val_loader, exp_cfg=None):
    """Train on labelled sequence windows; early-stop on validation PR-AUC."""
    exp_cfg = exp_cfg or config
    t_cfg = exp_cfg.lstm_training
    m_cfg = exp_cfg.lstm_model
    set_seed(t_cfg.random_seed)
    device = resolve_device(t_cfg.device)
    model = LSTMClassifier(m_cfg).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([t_cfg.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg.learning_rate, weight_decay=t_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    history = {
        "train_loss": [], "val_pr_auc": [], "val_f1": [], "learning_rate": [],
        "epoch_times": [], "best_epoch": 0, "best_val_pr_auc": -1.0,
        "pos_weight": t_cfg.pos_weight,
    }
    ensure_directories(t_cfg.model_save_path.parent, t_cfg.history_save_path.parent)
    best_state = None
    stale_epochs = 0
    start = time.time()

    for epoch in range(1, t_cfg.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_count = 0.0, 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences).view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total_count += len(targets)

        train_loss = total_loss / max(total_count, 1)
        val_probs, y_val = predict_dataloader(model, val_loader, device)
        if len(np.unique(y_val)) < 2:
            raise ValueError("Validation split must contain both classes to compute PR-AUC/ROC-AUC.")
        val_pr_auc = float(average_precision_score(y_val, val_probs))
        val_f1 = _max_f1(y_val, val_probs)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)
        history["epoch_times"].append(duration)

        if val_pr_auc > history["best_val_pr_auc"]:
            stale_epochs = 0
            history["best_val_pr_auc"] = val_pr_auc
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch, "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pr_auc": val_pr_auc, "val_f1": val_f1,
                "config": vars(m_cfg),
            }, t_cfg.model_save_path)
            saved = " -> best checkpoint saved"
        else:
            stale_epochs += 1
            saved = ""

        logger.info(
            f"Epoch [{epoch:02d}/{t_cfg.num_epochs:02d}] "
            f"Train BCE: {train_loss:.5f} | Val PR-AUC: {val_pr_auc:.5f} | "
            f"Val F1: {val_f1:.5f} | LR: {current_lr:.2e}{saved}"
        )
        if stale_epochs >= t_cfg.patience:
            logger.info("Early stopping: validation PR-AUC did not improve.")
            break

    history["total_training_time"] = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    save_json(history, t_cfg.history_save_path)
    return model, history


def load_trained_lstm(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = LSTMArchitectureConfig(**checkpoint.get("config", {}))
    model = LSTMClassifier(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
