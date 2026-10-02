"""Shared evaluation module for supervised fraud classifiers (MLP, 1D-CNN, LSTM).

Collects sigmoid probabilities, selects the decision threshold strictly on the
validation set, and evaluates the frozen threshold once on the held-out test set.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from src.utils import setup_logger, save_json

logger = setup_logger("SupervisedEvaluation")


def predict_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predicted fraud probabilities p = sigmoid(logit) and true labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (probabilities, labels) as 1-D numpy arrays.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features).view(-1)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs).astype(np.float64) if all_probs else np.array([])
    labels = np.concatenate(all_labels).astype(np.int64) if all_labels else np.array([])
    return probs, labels


def find_optimal_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> Dict[str, Any]:
    """Select the decision threshold that maximizes F1 on the validation PR curve.

    CRITICAL LEAKAGE PREVENTION:
    Only validation probabilities are used. The test set is never consulted.
    """
    if metric != "f1":
        raise ValueError(f"Unsupported threshold metric: {metric} (only 'f1' is supported)")

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # The final PR point (precision=1, recall=0) has no threshold, so drop it
    p, r = precisions[:-1], recalls[:-1]
    f1_scores = np.zeros_like(thresholds)
    denom = p + r
    valid = denom > 0
    f1_scores[valid] = 2 * p[valid] * r[valid] / denom[valid]

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])

    results = {
        "optimal_threshold": optimal_threshold,
        "criterion": metric,
        "val_f1": float(f1_scores[best_idx]),
        "val_precision": float(p[best_idx]),
        "val_recall": float(r[best_idx]),
        "val_pr_auc": float(average_precision_score(y_val, val_probs)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
    }

    logger.info(
        f"\n{'='*70}\n"
        f"VALIDATION THRESHOLD SELECTION (Criterion: {metric.upper()})\n"
        f"{'='*70}\n"
        f"Selected Validation Threshold: {optimal_threshold:.6f}\n"
        f"Validation PR-AUC:             {results['val_pr_auc']:.4f}\n"
        f"Validation ROC-AUC:            {results['val_roc_auc']:.4f}\n"
        f"Validation F1 at Threshold:    {results['val_f1']:.4f}\n"
        f"Validation Precision:          {results['val_precision']:.4f}\n"
        f"Validation Recall:             {results['val_recall']:.4f}\n"
        f"{'='*70}"
    )
    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute threshold-free (PR-AUC, ROC-AUC) and thresholded metrics."""
    preds = (probs >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    optimal_threshold: float,
    device: torch.device,
    model_name: str = "MLP Baseline",
) -> Dict[str, Any]:
    """Evaluate the held-out test set ONCE using the frozen validation threshold.

    Returns the metrics dict plus the raw test probabilities and labels under
    'test_probs' and 'y_test' (numpy arrays, dropped when saved to JSON).
    """
    test_probs, y_test = predict_dataloader(model, test_loader, device)
    metrics = compute_classification_metrics(y_test, test_probs, optimal_threshold)

    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    logger.info(
        f"\n{'='*70}\n"
        f"HELD-OUT TEST SET EVALUATION RESULTS ({model_name})\n"
        f"{'='*70}\n"
        f"Primary Metric (PR-AUC):   {metrics['pr_auc']:.4f}\n"
        f"ROC-AUC:                   {metrics['roc_auc']:.4f}\n"
        f"F1-Score:                  {metrics['f1']:.4f}\n"
        f"Precision:                 {metrics['precision']:.4f}\n"
        f"Recall:                    {metrics['recall']:.4f}\n"
        f"Accuracy:                  {metrics['accuracy']:.6f}\n"
        f"Decision Threshold:        {optimal_threshold:.6f}\n"
        f"Confusion Matrix:          TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}\n"
        f"{'='*70}"
    )

    metrics["test_probs"] = test_probs
    metrics["y_test"] = y_test
    return metrics


def save_supervised_metrics(metrics: Dict[str, Any], save_path: Path) -> None:
    """Save metrics to JSON, dropping any numpy arrays (raw predictions)."""

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        return obj

    save_json(_strip(metrics), save_path)
    logger.info(f"Supervised metrics saved to: {save_path}")
