"""Unit test suite for Credit Card Fraud Detection pipeline.

Covers all 10 required test criteria specified in the project requirements:
1. Dataset loading and missing-file guidance
2. Required-column validation
3. Chronological split ordering
4. No temporal overlap or index leakage between train/val/test
5. Scaler fitted ONLY on training data
6. Correct feature/target separation
7. Autoencoder forward pass
8. Output shape equals input shape
9. Reconstruction error calculation
10. Validation-based threshold selection and application
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    AutoencoderArchitectureConfig,
)
from src.preprocessing import (
    load_raw_data,
    inspect_data,
    chronological_split,
    prepare_features_and_targets,
    scale_features,
    get_autoencoder_training_subset,
)
from src.models.attention_autoencoder import AttentionAutoencoder, FeatureTokenizer
from src.evaluation.evaluate_autoencoder import (
    find_optimal_threshold,
    compute_comprehensive_metrics,
)


@pytest.fixture
def dummy_transaction_df():
    """Create a realistic synthetic transaction DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Ordered Time representing seconds over 2 days
    time_col = np.sort(np.random.uniform(0, 172800, n_samples))
    data = {"Time": time_col}

    # PCA features V1-V28
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_samples)

    # Amount with positive skew
    data["Amount"] = np.random.exponential(scale=50, size=n_samples)

    # Class: extreme imbalance (~1.5% fraud in synthetic test)
    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=15, replace=False)
    labels[fraud_indices] = 1
    data["Class"] = labels

    return pd.DataFrame(data)


# 1. Dataset Loading & Missing File Validation
def test_dataset_loading_and_missing_file_error():
    """Verify loading valid data and raising descriptive FileNotFoundError for missing paths."""
    missing_path = Path("/non/existent/path/creditcard.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_data(missing_path)
    assert "Raw dataset not found at expected location" in str(exc_info.value)
    assert "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" in str(exc_info.value)


# 2. Required-Column Validation
def test_required_columns_validation(dummy_transaction_df):
    """Verify validation passes when required columns exist and fails when one is missing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Valid CSV
        dummy_transaction_df.to_csv(tmp_path, index=False)
        loaded_df = load_raw_data(tmp_path)
        assert len(loaded_df) == len(dummy_transaction_df)
        for col in REQUIRED_COLUMNS:
            assert col in loaded_df.columns

        # Invalid CSV missing 'V14'
        invalid_df = dummy_transaction_df.drop(columns=["V14"])
        invalid_df.to_csv(tmp_path, index=False)
        with pytest.raises(ValueError) as exc_info:
            load_raw_data(tmp_path)
        assert "V14" in str(exc_info.value)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# 3. Chronological Split Ordering
def test_chronological_split(dummy_transaction_df):
    """Verify chronological split strictly preserves time order and specified ratios."""
    df_train, df_val, df_test, stats = chronological_split(
        dummy_transaction_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(df_train) == 700
    assert len(df_val) == 150
    assert len(df_test) == 150

    # Ensure each partition is internally sorted by Time
    assert df_train["Time"].is_monotonic_increasing
    assert df_val["Time"].is_monotonic_increasing
    assert df_test["Time"].is_monotonic_increasing


# 4. No Overlap Between Train/Validation/Test
def test_no_overlap_between_partitions(dummy_transaction_df):
    """Verify strict temporal boundaries with zero leakage or overlap."""
    df_train, df_val, df_test, _ = chronological_split(dummy_transaction_df)

    # Temporal boundaries
    assert df_train["Time"].max() <= df_val["Time"].min()
    assert df_val["Time"].max() <= df_test["Time"].min()

    # Total sample sum integrity
    assert len(df_train) + len(df_val) + len(df_test) == len(dummy_transaction_df)


# 5. Scaler Fitted ONLY on Training Data
def test_scaler_fitted_only_on_training_data(dummy_transaction_df):
    """Verify scaler is fitted strictly on X_train without influence from val or test."""
    from sklearn.preprocessing import RobustScaler

    df_train, df_val, df_test, _ = chronological_split(dummy_transaction_df)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        df_train, df_val, df_test
    )

    # Reference scaler fitted directly on X_train only
    ref_scaler = RobustScaler().fit(X_train)

    X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler = scale_features(
        X_train, X_val, X_test, scaler_type="robust"
    )

    # Verify fitted parameters match reference scaler fitted on train
    np.testing.assert_allclose(fitted_scaler.center_, ref_scaler.center_)
    np.testing.assert_allclose(fitted_scaler.scale_, ref_scaler.scale_)


# 6. Correct Feature/Target Separation
def test_correct_feature_target_separation(dummy_transaction_df):
    """Verify target column 'Class' is strictly separated and absent from feature matrix."""
    df_train, df_val, df_test, _ = chronological_split(dummy_transaction_df)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        df_train, df_val, df_test
    )

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_val.columns
    assert TARGET_COLUMN not in X_test.columns
    assert len(X_train.columns) == len(FEATURE_COLUMNS) == 30
    assert isinstance(y_train, pd.Series)

    # Verify autoencoder legitimate training subset excludes fraud
    X_train_legit = get_autoencoder_training_subset(X_train, y_train)
    assert len(X_train_legit) == (y_train == 0).sum()


# 7. Autoencoder Forward Pass
def test_autoencoder_forward_pass():
    """Verify model forward pass returns reconstructed tensor and latent representation."""
    cfg = AutoencoderArchitectureConfig(
        input_dim=30, d_model=16, nhead=4, num_encoder_layers=2, bottleneck_dim=8
    )
    model = AttentionAutoencoder(cfg)
    model.eval()

    dummy_input = torch.randn(8, 30)
    with torch.no_grad():
        recon, latent = model(dummy_input)

    assert isinstance(recon, torch.Tensor)
    assert isinstance(latent, torch.Tensor)


# 8. Output Shape Equals Input Shape
def test_output_shape_equals_input_shape():
    """Verify reconstructed output matches input shape (B, 30) and bottleneck matches (B, 8)."""
    cfg = AutoencoderArchitectureConfig(
        input_dim=30, d_model=32, nhead=4, num_encoder_layers=2, bottleneck_dim=8
    )
    model = AttentionAutoencoder(cfg)

    batch_sizes = [1, 16, 64]
    for b in batch_sizes:
        x = torch.randn(b, 30)
        recon, latent = model(x)
        assert recon.shape == (b, 30), f"Expected shape ({b}, 30), got {recon.shape}"
        assert latent.shape == (b, 8), f"Expected bottleneck shape ({b}, 8), got {latent.shape}"


# 9. Reconstruction Error Calculation
def test_reconstruction_error_calculation():
    """Verify per-sample reconstruction error computation."""
    cfg = AutoencoderArchitectureConfig(
        input_dim=30, d_model=16, nhead=4, num_encoder_layers=2, bottleneck_dim=8
    )
    model = AttentionAutoencoder(cfg)
    model.eval()

    x = torch.randn(10, 30)
    scores = model.compute_reconstruction_error(x)

    assert scores.shape == (10,)
    assert (scores >= 0).all(), "Reconstruction MSE error must be non-negative"


# 10. Threshold Application
def test_threshold_selection_and_application():
    """Verify validation threshold selection and application to test data."""
    np.random.seed(42)
    # Synthetic validation scores: legit ~ N(0.1, 0.05), fraud ~ N(1.5, 0.2)
    y_val = np.array([0] * 900 + [1] * 100)
    val_scores = np.concatenate([
        np.random.normal(0.1, 0.05, 900),
        np.random.normal(1.5, 0.2, 100),
    ])

    threshold_res = find_optimal_threshold(val_scores, y_val, metric="f1")
    threshold = threshold_res["optimal_threshold"]
    assert threshold > 0.1
    assert threshold < 1.5

    # Apply frozen threshold to test set
    y_test = np.array([0] * 450 + [1] * 50)
    test_scores = np.concatenate([
        np.random.normal(0.1, 0.05, 450),
        np.random.normal(1.5, 0.2, 50),
    ])

    metrics = compute_comprehensive_metrics(y_test, test_scores, threshold)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["f1"] > 0.8
    assert metrics["pr_auc"] > 0.8
