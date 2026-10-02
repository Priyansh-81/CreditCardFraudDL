"""Master Evaluation Harness and Cross-Model Comparison.

Consolidates test-set performance metrics across all implemented model families:
1. Attention-based Autoencoder (Priyansh Nandan - Unsupervised/Semi-supervised)
2. Multilayer Perceptron Baseline (Pranav Kasliwal - Supervised Feed-Forward)
3. 1-D Convolutional Neural Network (Atharv Sharma - Supervised 1D-CNN)
4. Long Short-Term Memory Network (LSTM - if present)

Generates:
- Consolidated Markdown Table (outputs/metrics/comparison_table.md)
- Consolidated JSON Summary (outputs/metrics/consolidated_model_comparison.json)
- Comparative Precision-Recall Curves (outputs/metrics/comparative_pr_curves.png)
- Comparative ROC Curves (outputs/metrics/comparative_roc_curves.png)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from src.config import METRICS_DIR, OUTPUTS_DIR
from src.utils import setup_logger, save_json

logger = setup_logger("ModelComparison")

# Known model registry configuration
MODEL_REGISTRY = {
    "attention_autoencoder_metrics.json": {
        "family": "Attention Autoencoder",
        "default_name": "Attention Autoencoder",
        "prediction_file": "test_predictions.npz",
        "score_key": "test_scores",
        "color": "#1f77b4",  # Blue
        "line_style": "-",
    },
    "mlp_metrics.json": {
        "family": "Supervised Feed-Forward (MLP)",
        "default_name": "MLP Baseline",
        "prediction_file": "mlp_test_predictions.npz",
        "score_key": "test_probs",
        "color": "#ff7f0e",  # Orange
        "line_style": "--",
    },
    "cnn1d_metrics.json": {
        "family": "Supervised 1-D CNN",
        "default_name": "1-D CNN",
        "prediction_file": "cnn1d_test_predictions.npz",
        "score_key": "test_probs",
        "color": "#2ca02c",  # Green
        "line_style": "-.",
    },
    "lstm_metrics.json": {
        "family": "Supervised Recurrent (LSTM)",
        "default_name": "LSTM Classifier",
        "prediction_file": "lstm_test_predictions.npz",
        "score_key": "test_probs",
        "color": "#d62728",  # Red
        "line_style": ":",
    },
}


def load_metrics_files(metrics_dir: Path) -> List[Dict[str, Any]]:
    """Scan metrics directory and load all available model metric files."""
    models_data = []

    # First load known registry models in order
    for filename, meta in MODEL_REGISTRY.items():
        file_path = metrics_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                models_data.append({
                    "filename": filename,
                    "meta": meta,
                    "raw": content,
                })
                logger.info(f"Loaded metric file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to read {filename}: {e}")

    # Also check any other *_metrics.json not in registry
    for file_path in sorted(metrics_dir.glob("*_metrics.json")):
        if file_path.name not in MODEL_REGISTRY:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                models_data.append({
                    "filename": file_path.name,
                    "meta": {
                        "family": "Deep Learning Classifier",
                        "default_name": file_path.stem.replace("_metrics", "").upper(),
                        "prediction_file": file_path.stem.replace("_metrics", "_test_predictions.npz"),
                        "score_key": "test_probs",
                        "color": "#9467bd",
                        "line_style": "-",
                    },
                    "raw": content,
                })
                logger.info(f"Loaded additional metric file: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to read {file_path.name}: {e}")

    return models_data


def extract_standard_metrics(model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize extracted metrics into a standard schema."""
    raw = model_entry["raw"]
    meta = model_entry["meta"]

    # Locate test metrics dict
    if "test_metrics" in raw and isinstance(raw["test_metrics"], dict):
        test = raw["test_metrics"]
    else:
        test = raw

    # Model name
    model_name = raw.get("model", meta.get("default_name", "Unknown Model"))

    # Extract metrics with fallbacks
    pr_auc = float(test.get("pr_auc", 0.0))
    roc_auc = float(test.get("roc_auc", 0.0))
    f1 = float(test.get("f1", 0.0))
    precision = float(test.get("precision", 0.0))
    recall = float(test.get("recall", 0.0))
    threshold = float(test.get("threshold", raw.get("validation_selection", {}).get("optimal_threshold", 0.0)))
    accuracy = float(test.get("accuracy", 0.0))
    confusion = test.get("confusion_matrix", None)

    return {
        "family": meta["family"],
        "model_name": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "prediction_file": meta.get("prediction_file"),
        "score_key": meta.get("score_key"),
        "color": meta.get("color", "#333333"),
        "line_style": meta.get("line_style", "-"),
    }


def generate_markdown_table(models: List[Dict[str, Any]], output_path: Path) -> str:
    """Generate consolidated markdown comparison table."""
    # Sort models by PR-AUC descending
    sorted_models = sorted(models, key=lambda m: m["pr_auc"], reverse=True)

    header = (
        "# Consolidated Model Evaluation & Comparison Table\n\n"
        "> **Project**: Deep Learning for Credit Card Fraud Detection (ICT-4442)\n"
        "> **Evaluation Protocol**: Chronological Split (70% Train, 15% Val, 15% Test). "
        "Decision thresholds are chosen strictly to maximize $F_1$-score on the Validation partition "
        "and evaluated once on the unseen Test partition (42,722 transactions with 52 frauds, base rate 0.1217%).\n\n"
    )

    columns = [
        "Architecture Family",
        "Model Name",
        "PR-AUC (Primary)",
        "ROC-AUC",
        "F1-Score",
        "Precision",
        "Recall",
        "Optimal Threshold",
    ]

    table_lines = [
        "| " + " | ".join(columns) + " |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for m in sorted_models:
        row = (
            f"| {m['family']} "
            f"| **{m['model_name']}** "
            f"| **{m['pr_auc']:.4f}** "
            f"| {m['roc_auc']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| `{m['threshold']:.4f}` |"
        )
        table_lines.append(row)

    table_content = "\n".join(table_lines) + "\n\n"
    notes = (
        "### Key Findings & Architectural Observations\n"
        "- **Primary Performance Driver (PR-AUC)**: In highly skewed fraud detection (0.12% positive rate), "
        "Precision-Recall AUC is the primary evaluation metric because ROC-AUC is flattered by the 42,670 true negatives.\n"
        "- **Supervised Inductive Bias**: Supervised representations with positive loss reweighting (`pos_weight ≈ 518.177`) "
        "explicitly optimize fraud discriminability, while the Attention Autoencoder learns legitimate transaction manifold geometry.\n"
        "- **1-D CNN Regularization**: 1D convolutions across feature axes capture localized feature interactions with weight sharing, "
        "providing strong inductive bias against overfitting to small positive fraud sample sizes.\n"
    )

    full_markdown = header + table_content + notes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    logger.info(f"Consolidated markdown table written to: {output_path}")
    return full_markdown


def save_consolidated_json(models: List[Dict[str, Any]], output_path: Path) -> None:
    """Save structured comparison summary JSON."""
    sorted_models = sorted(models, key=lambda m: m["pr_auc"], reverse=True)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "primary_metric": "pr_auc",
        "dataset_split": {
            "protocol": "Chronological (70/15/15)",
            "test_sample_count": 42722,
            "test_fraud_count": 52,
            "test_fraud_percentage": 0.121717,
        },
        "ranking_by_pr_auc": [m["model_name"] for m in sorted_models],
        "models": sorted_models,
    }
    save_json(summary, output_path)
    logger.info(f"Consolidated comparison JSON saved to: {output_path}")


def generate_comparative_curves(
    models: List[Dict[str, Any]],
    metrics_dir: Path,
    pr_curve_path: Path,
    roc_curve_path: Path,
) -> Tuple[bool, bool]:
    """Generate comparative PR and ROC curves if prediction npz files exist."""
    curve_data = []

    for m in models:
        pred_file = m.get("prediction_file")
        if not pred_file:
            continue
        pred_path = metrics_dir / pred_file
        if not pred_path.exists():
            continue

        try:
            npz = np.load(pred_path)
            y_test = npz["y_test"]
            score_key = m.get("score_key", "test_probs")
            if score_key in npz:
                scores = npz[score_key]
            elif "test_probs" in npz:
                scores = npz["test_probs"]
            elif "test_scores" in npz:
                scores = npz["test_scores"]
            else:
                continue

            curve_data.append({
                "model_name": m["model_name"],
                "y_test": y_test,
                "scores": scores,
                "color": m.get("color", "#333333"),
                "line_style": m.get("line_style", "-"),
            })
            logger.info(f"Loaded prediction curve data for: {m['model_name']} ({pred_file})")
        except Exception as e:
            logger.warning(f"Failed to load curve data from {pred_path}: {e}")

    if not curve_data:
        logger.warning("No prediction .npz files found to generate comparative curves.")
        return False, False

    # 1. Comparative Precision-Recall Curves
    plt.figure(figsize=(9, 6.5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Base rate (no-skill) line
    sample_y = curve_data[0]["y_test"]
    base_rate = float(np.mean(sample_y))
    plt.plot(
        [0, 1],
        [base_rate, base_rate],
        linestyle=":",
        color="gray",
        linewidth=1.5,
        label=f"No-Skill Baseline (AP = {base_rate:.4f})",
    )

    for item in curve_data:
        prec, rec, _ = precision_recall_curve(item["y_test"], item["scores"])
        ap = average_precision_score(item["y_test"], item["scores"])
        plt.plot(
            rec,
            prec,
            label=f"{item['model_name']} (PR-AUC = {ap:.4f})",
            color=item["color"],
            linestyle=item["line_style"],
            linewidth=2.2,
        )

    plt.xlabel("Recall", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("Precision", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Comparative Precision-Recall Curves (Held-out Test Partition)", fontsize=14, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=10.5)
    plt.tight_layout()
    pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pr_curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparative PR curves saved to: {pr_curve_path}")

    # 2. Comparative ROC Curves
    plt.figure(figsize=(9, 6.5), dpi=300)
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        color="gray",
        linewidth=1.5,
        label="Random Chance (ROC-AUC = 0.5000)",
    )

    for item in curve_data:
        fpr, tpr, _ = roc_curve(item["y_test"], item["scores"])
        roc_auc = roc_auc_score(item["y_test"], item["scores"])
        plt.plot(
            fpr,
            tpr,
            label=f"{item['model_name']} (ROC-AUC = {roc_auc:.4f})",
            color=item["color"],
            linestyle=item["line_style"],
            linewidth=2.2,
        )

    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Comparative Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=10.5)
    plt.tight_layout()
    roc_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparative ROC curves saved to: {roc_curve_path}")

    return True, True


def compare_models(
    metrics_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete model comparison harness."""
    if metrics_dir is None:
        metrics_dir = METRICS_DIR
    if output_dir is None:
        output_dir = METRICS_DIR

    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running master model comparison in: {metrics_dir}")
    raw_entries = load_metrics_files(metrics_dir)

    if not raw_entries:
        logger.warning(f"No metric files found in {metrics_dir}. Cannot generate comparison table.")
        return {"models": [], "table_markdown": ""}

    standardized_models = [extract_standard_metrics(e) for e in raw_entries]

    # Generate Markdown Table
    table_path = output_dir / "comparison_table.md"
    markdown_content = generate_markdown_table(standardized_models, table_path)

    # Save JSON summary
    json_path = output_dir / "consolidated_model_comparison.json"
    save_consolidated_json(standardized_models, json_path)

    # Generate curves
    pr_curve_path = output_dir / "comparative_pr_curves.png"
    roc_curve_path = output_dir / "comparative_roc_curves.png"
    generate_comparative_curves(standardized_models, metrics_dir, pr_curve_path, roc_curve_path)

    # Print to console for immediate visibility
    print("\n" + "=" * 90)
    print("CONSOLIDATED MODEL COMPARISON TABLE")
    print("=" * 90)
    sorted_models = sorted(standardized_models, key=lambda m: m["pr_auc"], reverse=True)
    print(f"{'Architecture Family':<32} {'Model Name':<22} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 90)
    for m in sorted_models:
        print(
            f"{m['family']:<32} {m['model_name']:<22} "
            f"{m['pr_auc']:<10.4f} {m['roc_auc']:<10.4f} "
            f"{m['f1']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f}"
        )
    print("=" * 90 + "\n")

    return {
        "models": standardized_models,
        "table_markdown": markdown_content,
        "table_path": table_path,
        "json_path": json_path,
        "pr_curve_path": pr_curve_path,
        "roc_curve_path": roc_curve_path,
    }


if __name__ == "__main__":
    compare_models()
