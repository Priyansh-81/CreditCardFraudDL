"""Evaluation and threshold-selection module for Attention Autoencoder.

Computes sample-level reconstruction errors as anomaly scores, determines the
decision threshold strictly on the validation set, and evaluates the frozen
threshold once on the held-out test set.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from src.config import config, ExperimentConfig
from src.models.attention_autoencoder import AttentionAutoencoder
from src.utils import setup_logger, save_json, load_json, ensure_directories

logger = setup_logger("AutoencoderEvaluation")


def compute_reconstruction_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction MSE anomaly scores across a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained Attention Autoencoder model.
    dataloader : DataLoader
        DataLoader yielding (features, labels) or features.
    device : torch.device
        Computation device (cpu, cuda, mps).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (scores, labels):
        - scores: 1-D numpy array of reconstruction MSE errors
        - labels: 1-D numpy array of true binary labels (or empty if unlabelled)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                features, labels = batch[0], batch[1]
                all_labels.extend(labels.cpu().numpy().tolist())
            else:
                features = batch

            features = features.to(device)
            # Compute per-sample MSE across features
            recon_error = model.compute_reconstruction_error(features)
            all_scores.extend(recon_error.cpu().numpy().tolist())

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64) if all_labels else np.array([])
    return scores_arr, labels_arr


def find_optimal_threshold(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> Dict[str, Any]:
    """Determine the optimal anomaly detection threshold using the validation set ONLY.

    CRITICAL LEAKAGE PREVENTION:
    Threshold selection is performed strictly on validation data.
    The test set must never be used to pick or tune the decision threshold.

    Parameters
    ----------
    val_scores : np.ndarray
        Validation anomaly scores (reconstruction MSE).
    y_val : np.ndarray
        Validation ground-truth labels (0 = legit, 1 = fraud).
    metric : str
        Optimization objective on validation set (default: 'f1').

    Returns
    -------
    Dict[str, Any]
        Selected threshold and validation performance diagnostics.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)

    # Avoid zero-division in F1 calculation
    f1_scores = np.zeros_like(thresholds)
    denominator = precisions[:-1] + recalls[:-1]
    valid_idx = denominator > 0
    f1_scores[valid_idx] = (2 * precisions[:-1][valid_idx] * recalls[:-1][valid_idx]) / denominator[valid_idx]

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_val_f1 = float(f1_scores[best_idx])
    val_precision_at_thresh = float(precisions[best_idx])
    val_recall_at_thresh = float(recalls[best_idx])

    val_pr_auc = float(average_precision_score(y_val, val_scores))
    val_roc_auc = float(roc_auc_score(y_val, val_scores))

    logger.info(
        f"\n{'='*70}\n"
        f"VALIDATION THRESHOLD SELECTION (Criterion: Max {metric.upper()})\n"
        f"{'='*70}\n"
        f"Optimal Validation Threshold: {optimal_threshold:.6f}\n"
        f"Validation PR-AUC:           {val_pr_auc:.4f}\n"
        f"Validation ROC-AUC:          {val_roc_auc:.4f}\n"
        f"Validation F1 at Threshold:  {best_val_f1:.4f}\n"
        f"Validation Precision:        {val_precision_at_thresh:.4f}\n"
        f"Validation Recall:           {val_recall_at_thresh:.4f}\n"
        f"{'='*70}"
    )

    return {
        "optimal_threshold": optimal_threshold,
        "criterion": metric,
        "val_f1": best_val_f1,
        "val_precision": val_precision_at_thresh,
        "val_recall": val_recall_at_thresh,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate evaluation metrics on a given partition using a fixed threshold.

    Primary metric for extreme class imbalance is PR-AUC (Average Precision).
    """
    preds = (scores >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds).tolist()

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": cm,
    }


def evaluate_autoencoder_pipeline(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    exp_cfg: ExperimentConfig = None,
) -> Dict[str, Any]:
    """Execute validation threshold selection and final held-out test evaluation.

    The test set is evaluated ONCE with the threshold selected from the validation set.
    """
    if exp_cfg is None:
        exp_cfg = config

    device = torch.device(exp_cfg.training.device)
    logger.info(f"Computing anomaly scores on validation set ...")
    val_scores, y_val = compute_reconstruction_scores(model, val_loader, device)

    # 1. Select threshold strictly on validation partition
    threshold_results = find_optimal_threshold(val_scores, y_val, metric=exp_cfg.evaluation.threshold_metric)
    optimal_threshold = threshold_results["optimal_threshold"]

    # 2. Evaluate held-out test set ONCE with the frozen threshold
    logger.info(f"Evaluating held-out test set with frozen threshold {optimal_threshold:.6f} ...")
    test_scores, y_test = compute_reconstruction_scores(model, test_loader, device)
    test_metrics = compute_comprehensive_metrics(y_test, test_scores, threshold=optimal_threshold)

    # Log test evaluation results
    cm = test_metrics["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    logger.info(
        f"\n{'='*70}\n"
        f"HELD-OUT TEST SET EVALUATION RESULTS (Attention Autoencoder)\n"
        f"{'='*70}\n"
        f"Primary Metric (PR-AUC):   {test_metrics['pr_auc']:.4f}\n"
        f"ROC-AUC:                   {test_metrics['roc_auc']:.4f}\n"
        f"F1-Score:                  {test_metrics['f1']:.4f}\n"
        f"Precision:                 {test_metrics['precision']:.4f}\n"
        f"Recall:                    {test_metrics['recall']:.4f}\n"
        f"Accuracy:                  {test_metrics['accuracy']:.6f}\n"
        f"Decision Threshold:        {optimal_threshold:.6f}\n"
        f"Confusion Matrix:          TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}\n"
        f"{'='*70}\n"
    )

    # 3. Create machine-readable result files
    # Standard format for team comparison
    team_comparison_entry = {
        "model": "Attention Autoencoder",
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "threshold": optimal_threshold,
    }

    # Comprehensive detailed metrics
    detailed_metrics = {
        "model": "Attention Autoencoder",
        "architecture": {
            "num_encoder_layers": exp_cfg.model.num_encoder_layers,
            "nhead": exp_cfg.model.nhead,
            "bottleneck_dim": exp_cfg.model.bottleneck_dim,
            "d_model": exp_cfg.model.d_model,
        },
        "validation_selection": threshold_results,
        "test_metrics": test_metrics,
    }

    # Save to disk
    ensure_directories(exp_cfg.evaluation.metrics_save_path.parent)
    save_json(detailed_metrics, exp_cfg.evaluation.metrics_save_path)
    save_json(team_comparison_entry, exp_cfg.evaluation.comparison_summary_path)

    # Save test predictions and scores
    np.savez_compressed(
        exp_cfg.evaluation.predictions_save_path,
        y_test=y_test,
        test_scores=test_scores,
        test_predictions=(test_scores >= optimal_threshold).astype(int),
        threshold=optimal_threshold,
    )

    logger.info(f"Detailed metrics saved to: {exp_cfg.evaluation.metrics_save_path}")
    logger.info(f"Model comparison entry saved to: {exp_cfg.evaluation.comparison_summary_path}")
    logger.info(f"Test predictions and scores saved to: {exp_cfg.evaluation.predictions_save_path}")

    return {
        "comparison_entry": team_comparison_entry,
        "detailed_metrics": detailed_metrics,
        "test_scores": test_scores,
    }


def load_trained_autoencoder(
    checkpoint_path: Path,
    device: torch.device,
) -> AttentionAutoencoder:
    """Load trained Attention Autoencoder model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AttentionAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Successfully loaded Attention Autoencoder from {checkpoint_path}")
    return model
