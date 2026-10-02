"""Training pipeline for Attention-based Tabular Autoencoder.

Trains the autoencoder strictly on legitimate transactions (Class == 0)
using Mean Squared Error (MSE) reconstruction loss. Tracks validation loss,
applies early stopping, and persists the best model checkpoint and history.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    config,
    ExperimentConfig,
    TrainingConfig,
    AutoencoderArchitectureConfig,
)
from src.dataset import TabularTransactionDataset, create_dataloaders
from src.models.attention_autoencoder import AttentionAutoencoder
from src.preprocessing import run_common_preprocessing
from src.utils import setup_logger, set_seed, save_json, ensure_directories

logger = setup_logger("AutoencoderTrainer")


class EarlyStopping:
    """Early stops training when validation reconstruction loss doesn't improve."""

    def __init__(self, patience: int = 6, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch of training on legitimate transactions only."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # If dataset returns (features, labels), grab features; otherwise batch is features
        features = batch[0] if isinstance(batch, (list, tuple)) else batch
        features = features.to(device)

        optimizer.zero_grad()
        reconstruction, _ = model(features)
        loss = criterion(reconstruction, features)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


def evaluate_reconstruction_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average reconstruction loss over validation transactions."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch[0] if isinstance(batch, (list, tuple)) else batch
            features = features.to(device)

            reconstruction, _ = model(features)
            loss = criterion(reconstruction, features)

            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

    val_loss = running_loss / max(total_samples, 1)
    return val_loss


def train_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    exp_cfg: ExperimentConfig = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the Attention Autoencoder.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader containing ONLY legitimate transactions from the training partition.
    val_loader : DataLoader
        DataLoader containing validation transactions.
    exp_cfg : ExperimentConfig, optional
        Experiment configuration object.

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        (best_model, training_history)
    """
    if exp_cfg is None:
        exp_cfg = config

    t_cfg = exp_cfg.training
    m_cfg = exp_cfg.model

    set_seed(t_cfg.random_seed)
    device = torch.device(t_cfg.device)
    logger.info(f"Using device for training: {device}")

    # Initialize model
    model = AttentionAutoencoder(m_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized AttentionAutoencoder with {total_params:,} trainable parameters.")

    # Reconstruction criterion: MSE
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.learning_rate,
        weight_decay=t_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    early_stopping = EarlyStopping(patience=t_cfg.patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_times": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "total_parameters": total_params,
    }

    ensure_directories(t_cfg.model_save_path.parent, t_cfg.history_save_path.parent)

    logger.info(f"Starting autoencoder training for up to {t_cfg.num_epochs} epochs ...")
    start_time = time.time()

    for epoch in range(1, t_cfg.num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_reconstruction_loss(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(epoch_duration)

        improved = early_stopping(val_loss, model)
        if improved:
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "input_dim": m_cfg.input_dim,
                        "d_model": m_cfg.d_model,
                        "nhead": m_cfg.nhead,
                        "num_layers": m_cfg.num_encoder_layers,
                        "bottleneck_dim": m_cfg.bottleneck_dim,
                    },
                },
                t_cfg.model_save_path,
            )
            improvement_tag = " -> Best model saved"
        else:
            improvement_tag = ""

        logger.info(
            f"Epoch [{epoch:02d}/{t_cfg.num_epochs:02d}] "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"Time: {epoch_duration:.1f}s{improvement_tag}"
        )

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Halting training.")
            break

    total_training_time = time.time() - start_time
    logger.info(
        f"Training completed in {total_training_time:.1f}s. "
        f"Best epoch: {history['best_epoch']} with Val MSE: {history['best_val_loss']:.6f}"
    )

    # Restore best weights to model
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)

    # Save training history
    save_json(history, t_cfg.history_save_path)
    logger.info(f"Training history saved to {t_cfg.history_save_path}")

    return model, history


def run_training_pipeline() -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience runner to execute preprocessing and train the Attention Autoencoder."""
    logger.info("Running training pipeline: Preparing preprocessed data ...")
    preprocessed = run_common_preprocessing()

    train_loader, val_loader, _ = create_dataloaders(
        X_train_legit=preprocessed["X_train_legit"],
        X_val=preprocessed["X_val"],
        y_val=preprocessed["y_val"],
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        batch_size=config.training.batch_size,
    )

    model, history = train_autoencoder(train_loader, val_loader)
    return model, history


if __name__ == "__main__":
    run_training_pipeline()
