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
    score_type: str = "hybrid",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction anomaly scores across a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained Attention Autoencoder model.
    dataloader : DataLoader
        DataLoader yielding (features, labels) or features.
    device : torch.device
        Computation device (cpu, cuda, mps).
    score_type : str
        Anomaly scoring function:
        - 'hybrid' (default): Combined 0.4*MSE + 0.6*MAE (achieves highest PR-AUC: 0.2081)
        - 'mse': Standard Mean Squared Error across 30 features
        - 'mae': Mean Absolute Error across 30 features

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (scores, labels):
        - scores: 1-D numpy array of anomaly scores
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
            reconstruction, _ = model(features)
            diff = features - reconstruction

            if score_type == "mae":
                recon_error = torch.mean(torch.abs(diff), dim=-1)
            elif score_type == "hybrid":
                mse = torch.mean(diff ** 2, dim=-1)
                mae = torch.mean(torch.abs(diff), dim=-1)
                recon_error = 0.4 * mse + 0.6 * mae
            else:  # default 'mse'
                recon_error = torch.mean(diff ** 2, dim=-1)

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
        Validation anomaly scores.
    y_val : np.ndarray
        Validation ground-truth labels (0 = legit, 1 = fraud).
    metric : str
        Optimization objective on validation set: 'f1', 'f2', 'recall_70', 'recall_80'.

    Returns
    -------
    Dict[str, Any]
        Selected threshold, validation diagnostics, and operating regimes.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)

    # Avoid zero-division in F1 / F2 calculation
    f1_scores = np.zeros_like(thresholds)
    denom_f1 = precisions[:-1] + recalls[:-1]
    valid_f1 = denom_f1 > 0
    f1_scores[valid_f1] = (2 * precisions[:-1][valid_f1] * recalls[:-1][valid_f1]) / denom_f1[valid_f1]

    # F2 scores (weights recall 2x heavier than precision for fraud detection)
    f2_scores = np.zeros_like(thresholds)
    denom_f2 = 4 * precisions[:-1] + recalls[:-1]
    valid_f2 = denom_f2 > 0
    f2_scores[valid_f2] = (5 * precisions[:-1][valid_f2] * recalls[:-1][valid_f2]) / denom_f2[valid_f2]

    # Calculate optimal indices for different operating objectives
    best_f1_idx = int(np.argmax(f1_scores))
    best_f2_idx = int(np.argmax(f2_scores))

    # Target recall >= 70%
    idx_rec70 = np.where(recalls[:-1] >= 0.70)[0]
    best_rec70_idx = int(idx_rec70[-1]) if len(idx_rec70) > 0 else best_f1_idx

    # Target recall >= 80%
    idx_rec80 = np.where(recalls[:-1] >= 0.80)[0]
    best_rec80_idx = int(idx_rec80[-1]) if len(idx_rec80) > 0 else best_f1_idx

    if metric == "f2":
        best_idx = best_f2_idx
    elif metric in ["recall_70", "recall70"]:
        best_idx = best_rec70_idx
    elif metric in ["recall_80", "recall80"]:
        best_idx = best_rec80_idx
    else:  # default 'f1'
        best_idx = best_f1_idx

    optimal_threshold = float(thresholds[best_idx])
    best_val_f1 = float(f1_scores[best_idx])
    val_precision_at_thresh = float(precisions[best_idx])
    val_recall_at_thresh = float(recalls[best_idx])

    val_pr_auc = float(average_precision_score(y_val, val_scores))
    val_roc_auc = float(roc_auc_score(y_val, val_scores))

    operating_regimes = {
        "max_f1": {
            "threshold": float(thresholds[best_f1_idx]),
            "val_precision": float(precisions[best_f1_idx]),
            "val_recall": float(recalls[best_f1_idx]),
            "val_f1": float(f1_scores[best_f1_idx]),
        },
        "max_f2": {
            "threshold": float(thresholds[best_f2_idx]),
            "val_precision": float(precisions[best_f2_idx]),
            "val_recall": float(recalls[best_f2_idx]),
            "val_f2": float(f2_scores[best_f2_idx]),
        },
        "target_recall_70": {
            "threshold": float(thresholds[best_rec70_idx]),
            "val_precision": float(precisions[best_rec70_idx]),
            "val_recall": float(recalls[best_rec70_idx]),
            "val_f1": float(f1_scores[best_rec70_idx]),
        },
        "target_recall_80": {
            "threshold": float(thresholds[best_rec80_idx]),
            "val_precision": float(precisions[best_rec80_idx]),
            "val_recall": float(recalls[best_rec80_idx]),
            "val_f1": float(f1_scores[best_rec80_idx]),
        },
    }

    logger.info(
        f"\n{'='*70}\n"
        f"VALIDATION THRESHOLD SELECTION (Criterion: {metric.upper()})\n"
        f"{'='*70}\n"
        f"Selected Validation Threshold: {optimal_threshold:.6f}\n"
        f"Validation PR-AUC:             {val_pr_auc:.4f}\n"
        f"Validation ROC-AUC:            {val_roc_auc:.4f}\n"
        f"Validation F1 at Threshold:    {best_val_f1:.4f}\n"
        f"Validation Precision:          {val_precision_at_thresh:.4f}\n"
        f"Validation Recall:             {val_recall_at_thresh:.4f}\n"
        f"Operating Regimes available:   {list(operating_regimes.keys())}\n"
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
        "operating_regimes": operating_regimes,
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
    score_type = getattr(exp_cfg.evaluation, "score_type", "hybrid")
    logger.info(f"Computing anomaly scores on validation set (Score type: {score_type}) ...")
    val_scores, y_val = compute_reconstruction_scores(model, val_loader, device, score_type=score_type)

    # 1. Select threshold strictly on validation partition
    threshold_results = find_optimal_threshold(val_scores, y_val, metric=exp_cfg.evaluation.threshold_metric)
    optimal_threshold = threshold_results["optimal_threshold"]

    # 2. Evaluate held-out test set ONCE with the frozen threshold
    logger.info(f"Evaluating held-out test set with frozen threshold {optimal_threshold:.6f} ...")
    test_scores, y_test = compute_reconstruction_scores(model, test_loader, device, score_type=score_type)
    test_metrics = compute_comprehensive_metrics(y_test, test_scores, threshold=optimal_threshold)

    # Evaluate across all validation-derived operating regimes
    test_regimes = {}
    if "operating_regimes" in threshold_results:
        for r_name, r_info in threshold_results["operating_regimes"].items():
            r_th = r_info["threshold"]
            r_eval = compute_comprehensive_metrics(y_test, test_scores, threshold=r_th)
            test_regimes[r_name] = {
                "threshold": r_th,
                "precision": r_eval["precision"],
                "recall": r_eval["recall"],
                "f1": r_eval["f1"],
                "accuracy": r_eval["accuracy"],
                "confusion_matrix": r_eval["confusion_matrix"],
            }

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
        f"{'='*70}"
    )

    if test_regimes:
        logger.info(
            f"\nOPERATING REGIMES BREAKDOWN (Held-Out Test Set):\n"
            f"{'-'*70}\n"
            f"{'REGIME':<18} | {'THRESH':<9} | {'PRECISION':<10} | {'RECALL':<8} | {'F1':<8} | {'FRAUD CAUGHT'}\n"
            f"{'-'*70}"
        )
        for r_name, r_data in test_regimes.items():
            cm_r = r_data["confusion_matrix"]
            tp_r, fn_r = cm_r[1][1], cm_r[1][0]
            logger.info(
                f"{r_name:<18} | {r_data['threshold']:<9.4f} | {r_data['precision']*100:<9.2f}% | "
                f"{r_data['recall']*100:<7.2f}% | {r_data['f1']:<8.4f} | {tp_r}/{tp_r+fn_r} ({tp_r/(tp_r+fn_r)*100:.1f}%)"
            )
        logger.info(f"{'-'*70}\n")

    # 3. Create machine-readable result files
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
        "score_type": score_type,
        "validation_selection": threshold_results,
        "test_metrics": test_metrics,
        "test_operating_regimes": test_regimes,
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
