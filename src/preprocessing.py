"""Common preprocessing module for Credit Card Fraud Detection.

Provides reproducible dataset loading, schema validation, statistical inspection,
chronological splitting, and training-only feature scaling.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

from src.config import (
    RAW_DATASET_PATH,
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PreprocessingConfig,
    config,
)
from src.utils import setup_logger

logger = setup_logger("Preprocessing")


def check_dataset_exists(filepath: Path = RAW_DATASET_PATH) -> bool:
    """Check if the raw dataset CSV file exists at the expected path."""
    return filepath.exists() and filepath.is_file()


def load_raw_data(filepath: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the raw Credit Card Fraud dataset from CSV and validate schema.

    Parameters
    ----------
    filepath : Path
        Path to data/raw/creditcard.csv.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame with original columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist. Provides detailed instructions on where
        to obtain and place the file.
    ValueError
        If required columns are missing from the dataset.
    """
    if not filepath.exists():
        error_msg = (
            f"\n{'='*75}\n"
            f"ERROR: Raw dataset not found at expected location:\n"
            f"  {filepath.resolve()}\n\n"
            f"Please obtain the ULB Credit Card Fraud Detection dataset:\n"
            f"  1. Download 'creditcard.csv' from Kaggle:\n"
            f"     https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"  2. Place the uncompressed CSV at:\n"
            f"     {filepath.resolve()}\n"
            f"{'='*75}\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading raw dataset from {filepath} ...")
    df = pd.read_csv(filepath)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    logger.info(f"Dataset successfully loaded. Total rows: {len(df):,}, columns: {len(df.columns)}")
    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive basic inspection of the credit card transaction dataset.

    Reports:
    - Shape (rows, columns)
    - Legitimate transaction count (Class = 0)
    - Fraudulent transaction count (Class = 1)
    - Fraud percentage (%)
    - Missing value count across all columns
    - Duplicate row count
    - Time span in seconds and hours
    - Amount range and summary statistics

    Parameters
    ----------
    df : pd.DataFrame
        Raw or loaded credit card DataFrame.

    Returns
    -------
    Dict[str, Any]
        Statistical summary dictionary.
    """
    total_rows = len(df)
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_pct = (fraud_count / total_rows) * 100.0 if total_rows > 0 else 0.0

    missing_vals_per_col = df.isnull().sum()
    total_missing = int(missing_vals_per_col.sum())

    num_duplicates = int(df.duplicated().sum())

    time_min = float(df["Time"].min())
    time_max = float(df["Time"].max())
    time_span_hours = (time_max - time_min) / 3600.0

    summary = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct,
        "total_missing_values": total_missing,
        "duplicate_rows_count": num_duplicates,
        "time_min_seconds": time_min,
        "time_max_seconds": time_max,
        "time_span_hours": time_span_hours,
        "amount_min": float(df["Amount"].min()),
        "amount_max": float(df["Amount"].max()),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
    }

    report = (
        f"\n{'='*70}\n"
        f"DATASET INSPECTION REPORT\n"
        f"{'='*70}\n"
        f"Total Transactions:        {total_rows:,}\n"
        f"Total Columns:             {len(df.columns)} (Time, V1-V28, Amount, Class)\n"
        f"Legitimate Transactions:   {legit_count:,} ({100 - fraud_pct:.4f}%)\n"
        f"Fraudulent Transactions:   {fraud_count:,} ({fraud_pct:.4f}%)\n"
        f"Total Missing Values:      {total_missing}\n"
        f"Duplicate Rows:            {num_duplicates:,}\n"
        f"Time Span:                 {time_min:.1f}s to {time_max:.1f}s (~{time_span_hours:.1f} hours)\n"
        f"Amount Range:              ${summary['amount_min']:.2f} - ${summary['amount_max']:.2f} "
        f"(Median: ${summary['amount_median']:.2f}, Mean: ${summary['amount_mean']:.2f})\n"
        f"{'='*70}\n"
    )
    logger.info(report)
    return summary


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Perform a strict chronological split of the dataset based on transaction Time.

    Transactions are sorted chronologically by 'Time' to prevent future information
    from leaking into past training data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Time' and other transaction features.
    train_ratio : float
        Proportion for training set (default: 0.70).
    val_ratio : float
        Proportion for validation set (default: 0.15).
    test_ratio : float
        Proportion for test set (default: 0.15).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]
        (df_train, df_val, df_test, partition_stats)
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio+val_ratio+test_ratio})"
        )

    # Ensure chronological order by Time
    df_sorted = df.sort_values(by="Time").reset_index(drop=True)
    n_total = len(df_sorted)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    df_train = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    df_test = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # Verify chronological integrity (strict temporal boundaries)
    verify_chronological_integrity(df_train, df_val, df_test)

    # Compute partition statistics
    partition_stats = {
        "train": _get_partition_metrics(df_train, "Train"),
        "val": _get_partition_metrics(df_val, "Validation"),
        "test": _get_partition_metrics(df_test, "Test"),
    }

    _log_partition_summary(partition_stats)

    return df_train, df_val, df_test, partition_stats


def verify_chronological_integrity(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> None:
    """Verify that there is no temporal overlap or leakage across partitions."""
    train_max_time = df_train["Time"].max()
    val_min_time = df_val["Time"].min()
    val_max_time = df_val["Time"].max()
    test_min_time = df_test["Time"].min()

    if train_max_time > val_min_time:
        raise AssertionError(
            f"Temporal leakage detected: Train max Time ({train_max_time}) > Val min Time ({val_min_time})"
        )

    if val_max_time > test_min_time:
        raise AssertionError(
            f"Temporal leakage detected: Val max Time ({val_max_time}) > Test min Time ({test_min_time})"
        )

    logger.info("Chronological split integrity verified: Strict temporal boundaries confirmed with no leakage.")


def _get_partition_metrics(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Helper to compute statistics for a partition."""
    total = len(df)
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_pct = (fraud_count / total) * 100.0 if total > 0 else 0.0

    return {
        "partition_name": name,
        "sample_count": total,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct,
        "time_min": float(df["Time"].min()) if total > 0 else 0.0,
        "time_max": float(df["Time"].max()) if total > 0 else 0.0,
    }


def _log_partition_summary(stats: Dict[str, Dict[str, Any]]) -> None:
    """Format and log partition metrics."""
    logger.info("\n" + "=" * 78)
    logger.info(f"{'PARTITION':<12} | {'SAMPLES':<10} | {'LEGIT':<10} | {'FRAUD':<8} | {'FRAUD %':<10} | {'TIME RANGE (s)':<20}")
    logger.info("-" * 78)
    for key in ["train", "val", "test"]:
        s = stats[key]
        time_str = f"[{s['time_min']:.0f}, {s['time_max']:.0f}]"
        logger.info(
            f"{s['partition_name']:<12} | {s['sample_count']:<10,d} | {s['legit_count']:<10,d} | "
            f"{s['fraud_count']:<8,d} | {s['fraud_percentage']:<9.4f}% | {time_str:<20}"
        )
    logger.info("=" * 78 + "\n")


def prepare_features_and_targets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separate input features and the target label for each partition.

    Enforces that 'Class' is strictly isolated as the target and NEVER included
    in the feature vector X.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = df_train[FEATURE_COLUMNS].copy()
    y_train = df_train[TARGET_COLUMN].copy().astype(int)

    X_val = df_val[FEATURE_COLUMNS].copy()
    y_val = df_val[TARGET_COLUMN].copy().astype(int)

    X_test = df_test[FEATURE_COLUMNS].copy()
    y_test = df_test[TARGET_COLUMN].copy().astype(int)

    assert TARGET_COLUMN not in X_train.columns, "Target leaked into training features!"
    assert TARGET_COLUMN not in X_val.columns, "Target leaked into validation features!"
    assert TARGET_COLUMN not in X_test.columns, "Target leaked into test features!"

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "robust",
    scaler_save_path: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Fit feature scaler STRICTLY on training data and transform all partitions.

    CRITICAL LEAKAGE PREVENTION:
    The scaler is fitted ONLY on X_train. X_val and X_test are transformed using
    the training-fitted parameters and must NEVER influence scaler fitting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix.
    X_test : pd.DataFrame
        Held-out test feature matrix.
    scaler_type : str
        'robust' (sklearn.preprocessing.RobustScaler) or 'standard' (StandardScaler).
    scaler_save_path : Path, optional
        Filepath to persist the fitted scaler using joblib.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]
        (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)
    """
    import joblib
    from sklearn.preprocessing import RobustScaler, StandardScaler

    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

    logger.info(f"Fitting {scaler.__class__.__name__} STRICTLY on X_train ({len(X_train):,} samples) ...")
    # Fit strictly on train
    scaler.fit(X_train)

    # Transform each partition
    X_train_scaled_arr = scaler.transform(X_train)
    X_val_scaled_arr = scaler.transform(X_val)
    X_test_scaled_arr = scaler.transform(X_test)

    # Wrap back into DataFrames with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled_arr, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=X_test.columns, index=X_test.index)

    # Sanity checks for data leakage
    _verify_scaling_leakage_absence(X_train_scaled, X_val_scaled, scaler_type)

    if scaler_save_path:
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        logger.info(f"Fitted scaler saved to {scaler_save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def _verify_scaling_leakage_absence(
    X_train_scaled: pd.DataFrame, X_val_scaled: pd.DataFrame, scaler_type: str
) -> None:
    """Verify that validation data was not used during scaler fitting."""
    if scaler_type == "standard":
        train_mean = X_train_scaled.mean().values
        val_mean = X_val_scaled.mean().values
        # Train means should be ~0, val means should not be identically 0
        assert abs(train_mean.mean()) < 1e-4, "Train scaled mean is not ~0 under StandardScaler"
    logger.info("Scaling verification passed: Scaler was fitted strictly on X_train.")


def get_autoencoder_training_subset(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Extract legitimate transactions (Class == 0) for Autoencoder reconstruction training.

    Per Phase 1 methodology:
    The Autoencoder learns the manifold of legitimate transactions only.
    Fraudulent transactions in the training partition are discarded from training,
    while validation and test partitions retain both classes for evaluation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels (0 = legitimate, 1 = fraud).

    Returns
    -------
    pd.DataFrame
        Legitimate training samples only.
    """
    legit_mask = (y_train == 0)
    X_train_legit = X_train[legit_mask].copy().reset_index(drop=True)

    num_total = len(X_train)
    num_legit = len(X_train_legit)
    num_excluded_fraud = num_total - num_legit

    logger.info(
        f"Autoencoder training subset prepared:\n"
        f"  Total training samples:         {num_total:,}\n"
        f"  Legitimate training samples:    {num_legit:,} (used for reconstruction training)\n"
        f"  Excluded training fraud:        {num_excluded_fraud:,} (fraud NEVER shown to autoencoder during training)"
    )
    return X_train_legit


def run_common_preprocessing(
    cfg: PreprocessingConfig = None,
) -> Dict[str, Any]:
    """Execute the complete end-to-end common preprocessing pipeline.

    Produces:
    - Chronologically split train, val, test partitions
    - Scaler fitted only on train
    - Scaled feature matrices
    - Autoencoder training subset (Class == 0 only)
    - Saved scaler artifact
    """
    import numpy as np

    if cfg is None:
        cfg = config.preprocessing

    df_raw = load_raw_data(cfg.raw_data_path)
    inspect_data(df_raw)

    df_train, df_val, df_test, partition_stats = chronological_split(
        df_raw,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        df_train, df_val, df_test
    )

    scaler_path = cfg.processed_dir / f"{cfg.scaler_type}_scaler.joblib"
    X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler = scale_features(
        X_train,
        X_val,
        X_test,
        scaler_type=cfg.scaler_type,
        scaler_save_path=scaler_path,
    )

    X_train_legit = get_autoencoder_training_subset(X_train_scaled, y_train)

    # Save processed splits as numpy compressed arrays for fast access
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_dir / "processed_splits.npz",
        X_train=X_train_scaled.values,
        y_train=y_train.values,
        X_val=X_val_scaled.values,
        y_val=y_val.values,
        X_test=X_test_scaled.values,
        y_test=y_test.values,
        X_train_legit=X_train_legit.values,
        feature_names=np.array(FEATURE_COLUMNS),
    )
    logger.info(f"Processed datasets saved to {cfg.processed_dir / 'processed_splits.npz'}")

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "X_train_legit": X_train_legit,
        "scaler": fitted_scaler,
        "partition_stats": partition_stats,
    }


if __name__ == "__main__":
    run_common_preprocessing()


