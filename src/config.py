"""Configuration module for Credit Card Fraud Detection experiments.

Centralizes all filepaths, random seeds, split ratios, scaling methods,
and model architecture parameters specified in the Phase 1 project synopsis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch


# Base directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Expected raw dataset path
RAW_DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"

# Column definitions
FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN: str = "Class"
REQUIRED_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]


@dataclass
class PreprocessingConfig:
    """Configuration for data loading, splitting, and scaling."""

    raw_data_path: Path = RAW_DATASET_PATH
    processed_dir: Path = PROCESSED_DATA_DIR
    random_seed: int = 42

    # Chronological split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Scaling configuration: "robust" (RobustScaler) or "standard" (StandardScaler)
    # RobustScaler is well suited for financial transactions with heavy outliers in Amount
    scaler_type: str = "robust"

    # Strictly fit scaler only on training data
    fit_on_train_only: bool = True

    # Expected dataset characteristics for validation
    expected_num_rows: int = 284807
    expected_num_features: int = 30
    expected_num_frauds: int = 492


@dataclass
class AutoencoderArchitectureConfig:
    """Architectural parameters for Attention-based Autoencoder.

    Must match the Phase 1 synopsis specifications:
    - 2-layer Transformer encoder
    - 4 attention heads
    - 8-unit bottleneck
    - Reconstruction error is the anomaly score
    """

    input_dim: int = 30  # Time, V1-V28, Amount
    d_model: int = 32  # Feature token embedding dimension
    nhead: int = 4  # Number of self-attention heads
    num_encoder_layers: int = 2  # Number of Transformer encoder layers
    dim_feedforward: int = 64  # Feed-forward layer dimension inside Transformer
    bottleneck_dim: int = 8  # 8-dimensional bottleneck
    dropout: float = 0.1  # Dropout rate
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for Autoencoder training."""

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 40
    patience: int = 6  # Early stopping patience
    loss_function: str = "mse"  # Mean Squared Error
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    num_workers: int = 0

    # Paths
    model_save_path: Path = MODELS_DIR / "best_attention_autoencoder.pt"
    history_save_path: Path = LOGS_DIR / "training_history.json"


@dataclass
class EvaluationConfig:
    """Configuration for threshold selection and test evaluation."""

    metrics_save_path: Path = METRICS_DIR / "attention_autoencoder_metrics.json"
    predictions_save_path: Path = METRICS_DIR / "test_predictions.npz"
    comparison_summary_path: Path = METRICS_DIR / "model_comparison_entry.json"
    threshold_metric: str = "f1"  # Criterion on validation set to pick threshold
    num_threshold_steps: int = 500


@dataclass
class ExperimentConfig:
    """Root configuration holding all sub-configs."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: AutoencoderArchitectureConfig = field(default_factory=AutoencoderArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# Default global experiment config
config = ExperimentConfig()
