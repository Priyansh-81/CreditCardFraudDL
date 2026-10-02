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
