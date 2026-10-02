"""Error analysis and result visualizations for supervised fraud classifiers.

Profiles test-set transactions by outcome (TP, TN, FP, FN) at the frozen
validation threshold to answer two questions from the project synopsis:
  - Do false negatives concentrate in micro-transactions?
  - Do false positives concentrate in high-dollar outliers?

Produces report figures (confusion matrix, PR curve with operating point,
per-feature residual comparison) and a JSON breakdown of the errors.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

from src.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_DIR
from src.utils import setup_logger, save_json, ensure_directories

logger = setup_logger("ErrorAnalysis")

OUTCOME_ORDER = ["TP", "TN", "FP", "FN"]
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


def assign_outcomes(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Label each sample as TP, TN, FP or FN."""
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds).astype(int)
    outcomes = np.empty(len(y_true), dtype=object)
    outcomes[(y_true == 1) & (preds == 1)] = "TP"
    outcomes[(y_true == 0) & (preds == 0)] = "TN"
    outcomes[(y_true == 0) & (preds == 1)] = "FP"
    outcomes[(y_true == 1) & (preds == 0)] = "FN"
    return outcomes


def _to_frame(X, scaler=None) -> pd.DataFrame:
    """Return test features as a DataFrame, inverse-scaled to raw units if a scaler is given."""
    values = X.values if hasattr(X, "values") else np.asarray(X)
    if scaler is not None:
        values = scaler.inverse_transform(values)
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def _standardized_mean_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Cohen's-d style difference (mean_a - mean_b) / pooled std, per feature."""
    if len(a) == 0 or len(b) == 0:
        return pd.Series(0.0, index=a.columns)
    pooled = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2).replace(0, np.nan)
    return ((a.mean() - b.mean()) / pooled).fillna(0.0)


def profile_errors(
    features: pd.DataFrame,
    outcomes: np.ndarray,
    probs: np.ndarray,
    micro_amount: float = 10.0,
    high_amount_quantile: float = 0.95,
) -> Dict[str, Any]:
    """Summarize Amount, probability and PCA-feature characteristics per outcome group."""
    amount = features["Amount"]
    legit_mask = np.isin(outcomes, ["TN", "FP"])
    high_amount_cutoff = float(amount[legit_mask].quantile(high_amount_quantile))

    groups = {}
    for name in OUTCOME_ORDER:
        mask = outcomes == name
        amt = amount[mask]
        groups[name] = {
            "count": int(mask.sum()),
            "amount_mean": float(amt.mean()) if mask.any() else None,
            "amount_median": float(amt.median()) if mask.any() else None,
            "amount_p90": float(amt.quantile(0.90)) if mask.any() else None,
            "amount_max": float(amt.max()) if mask.any() else None,
            "share_micro_transactions": float((amt <= micro_amount).mean()) if mask.any() else None,
            "share_high_amount": float((amt >= high_amount_cutoff).mean()) if mask.any() else None,
            "mean_fraud_probability": float(probs[mask].mean()) if mask.any() else None,
        }

    pca = features[PCA_COLUMNS]
    fn_vs_tp = _standardized_mean_diff(pca[outcomes == "FN"], pca[outcomes == "TP"])
    fp_vs_tn = _standardized_mean_diff(pca[outcomes == "FP"], pca[outcomes == "TN"])

    def _top(series: pd.Series, k: int = 5) -> Dict[str, float]:
        # An empty FP or FN group yields all-zero shifts; report nothing rather than fake ties
        if not series.abs().any():
            return {}
        top = series.abs().sort_values(ascending=False).head(k).index
        return {col: float(series[col]) for col in top}

    fn_micro = groups["FN"]["share_micro_transactions"]
    tp_micro = groups["TP"]["share_micro_transactions"]
    fp_high = groups["FP"]["share_high_amount"]
    tn_high = groups["TN"]["share_high_amount"]

    findings = {
        "fn_are_micro_transactions": (
            fn_micro is not None and tp_micro is not None and fn_micro > tp_micro
        ),
        "fp_are_high_amount_outliers": (
            fp_high is not None and tn_high is not None and fp_high > tn_high
        ),
        "fn_micro_share_vs_tp": [fn_micro, tp_micro],
        "fp_high_amount_share_vs_tn": [fp_high, tn_high],
    }

    return {
        "micro_amount_cutoff": micro_amount,
        "high_amount_cutoff": high_amount_cutoff,
        "high_amount_quantile": high_amount_quantile,
        "groups": groups,
        "findings": findings,
        "top_pca_shifts_fn_vs_tp": _top(fn_vs_tp),
        "top_pca_shifts_fp_vs_tn": _top(fp_vs_tn),
        "_fn_vs_tp_all": fn_vs_tp,
        "_fp_vs_tn_all": fp_vs_tn,
    }


def plot_confusion_matrix(y_true, preds, save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    # Log colour scale keeps the tiny fraud row visible next to ~42k legit samples
    im = ax.imshow(np.log10(cm + 1), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if np.log10(cm[i, j] + 1) > np.log10(cm.max() + 1) / 2 else "black",
                    fontsize=13)
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10(count + 1)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(y_true, probs, threshold: float, save_path: Path, title: str) -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    preds = probs >= threshold
    tp = int(((y_true == 1) & preds).sum())
    op_precision = tp / max(int(preds.sum()), 1)
    op_recall = tp / max(int((y_true == 1).sum()), 1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recalls, precisions, lw=2, label=f"PR curve (AP = {ap:.4f})")
    ax.axhline(y_true.mean(), color="grey", ls="--", lw=1, label=f"Prevalence = {y_true.mean():.4f}")
    ax.scatter([op_recall], [op_precision], color="crimson", zorder=5, s=60,
               label=f"Operating point (tau = {threshold:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_residuals(features: pd.DataFrame, outcomes: np.ndarray, profile: Dict[str, Any],
                           save_path: Path) -> None:
    """Per-feature standardized mean shift of errors vs correct predictions, plus Amount by outcome."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    x = np.arange(len(PCA_COLUMNS))

    axes[0].bar(x, profile["_fn_vs_tp_all"].values, color="#d62728")
    axes[0].set_title("False negatives vs true positives: standardized mean shift per PCA feature")
    axes[1].bar(x, profile["_fp_vs_tn_all"].values, color="#ff7f0e")
    axes[1].set_title("False positives vs true negatives: standardized mean shift per PCA feature")
    for ax, group in zip(axes[:2], ["FN", "FP"]):
        if not (outcomes == group).any():
            ax.text(0.5, 0.5, f"No {group}s at this threshold", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="grey")
        ax.set_xticks(x, PCA_COLUMNS, rotation=90)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Cohen's d")
        ax.grid(axis="y", alpha=0.3)

    data, labels = [], []
    for name in OUTCOME_ORDER:
        amt = features["Amount"][outcomes == name]
        if len(amt):
            data.append(np.log10(amt.clip(lower=0) + 1))
            labels.append(f"{name} (n={len(amt):,})")
    axes[2].boxplot(data, showfliers=False)
    axes[2].set_xticks(range(1, len(labels) + 1), labels)
    axes[2].axhline(np.log10(profile["micro_amount_cutoff"] + 1), color="grey", ls="--", lw=1,
                    label=f"Micro cutoff (${profile['micro_amount_cutoff']:.0f})")
    axes[2].axhline(np.log10(profile["high_amount_cutoff"] + 1), color="crimson", ls="--", lw=1,
                    label=f"High-amount cutoff (${profile['high_amount_cutoff']:.0f})")
    axes[2].set_ylabel("log10(Amount + 1)")
    axes[2].set_title("Transaction Amount by outcome")
    axes[2].legend(loc="upper right")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_error_analysis(
    X_test,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    threshold: float,
    scaler=None,
    model_name: str = "MLP Baseline",
    figures_dir: Path = FIGURES_DIR,
    summary_path: Path = METRICS_DIR / "error_analysis_summary.json",
    micro_amount: float = 10.0,
) -> Dict[str, Any]:
    """Profile FP/FN/TP/TN on the test set and write figures plus a JSON summary.

    Parameters
    ----------
    X_test : array-like
        Test features (scaled). Pass the fitted ``scaler`` so Amount is profiled in dollars.
    threshold : float
        Frozen decision threshold selected on the validation set.
    """
    y_test = np.asarray(y_test).astype(int)
    test_probs = np.asarray(test_probs, dtype=np.float64)
    preds = (test_probs >= threshold).astype(int)
    outcomes = assign_outcomes(y_test, preds)
    features = _to_frame(X_test, scaler)

    profile = profile_errors(features, outcomes, test_probs, micro_amount=micro_amount)

    ensure_directories(figures_dir)
    figure_paths = {
        "confusion_matrix": figures_dir / "confusion_matrix.png",
        "precision_recall_curve": figures_dir / "precision_recall_curve.png",
        "feature_residuals": figures_dir / "feature_residuals.png",
    }
    plot_confusion_matrix(y_test, preds, figure_paths["confusion_matrix"],
                          f"{model_name}: test confusion matrix")
    plot_precision_recall(y_test, test_probs, threshold, figure_paths["precision_recall_curve"],
                          f"{model_name}: test precision-recall")
    plot_feature_residuals(features, outcomes, profile, figure_paths["feature_residuals"])

    summary = {k: v for k, v in profile.items() if not k.startswith("_")}
    summary["model"] = model_name
    summary["threshold"] = float(threshold)
    summary["amount_units"] = "dollars" if scaler is not None else "scaled"
    summary["figures"] = {k: str(v) for k, v in figure_paths.items()}
    save_json(summary, summary_path)

    g = summary["groups"]
    f = summary["findings"]
    logger.info(
        f"\n{'='*70}\n"
        f"ERROR ANALYSIS ({model_name})\n"
        f"{'='*70}\n"
        f"Counts: TP={g['TP']['count']}, TN={g['TN']['count']:,}, FP={g['FP']['count']}, FN={g['FN']['count']}\n"
        f"Micro (<= ${micro_amount:.0f}) share  FN vs TP: {f['fn_micro_share_vs_tp']}\n"
        f"High-amount share         FP vs TN: {f['fp_high_amount_share_vs_tn']}\n"
        f"Top PCA shifts FN vs TP: {summary['top_pca_shifts_fn_vs_tp']}\n"
        f"Top PCA shifts FP vs TN: {summary['top_pca_shifts_fp_vs_tn']}\n"
        f"Figures saved to {figures_dir}; summary saved to {summary_path}\n"
        f"{'='*70}"
    )
    return summary
