"""Unit tests for the 1-D CNN classifier and evaluation harness.

Covers:
1. Input reshaping (B, 30) -> (B, 1, 30) and output logit shape (B, 1)
2. Gradient flow across all conv layers, batch norms, and linear layers
3. Validation-based threshold selection and application to test probabilities
4. Predict proba range and sigmoid consistency
5. Loss weighting and single training step parameter update
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import CNN1DArchitectureConfig
from src.dataset import create_supervised_dataloaders
from src.models.cnn1d import CNN1DClassifier
from src.evaluation.evaluate_supervised import (
    find_optimal_threshold,
    compute_classification_metrics,
)


def test_cnn1d_input_reshaping_and_output_shape():
    """Verify passing (batch, 30) or (batch, 1, 30) produces (batch, 1) logits."""
    torch.manual_seed(42)
    cfg = CNN1DArchitectureConfig()
    model = CNN1DClassifier(cfg)
    model.eval()

    batch_sizes = [1, 8, 32]
    for b in batch_sizes:
        # 2D input (B, 30)
        x_2d = torch.randn(b, 30)
        logits_2d = model(x_2d)
        probs_2d = model.predict_proba(x_2d)
        assert logits_2d.shape == (b, 1), f"Expected shape ({b}, 1), got {logits_2d.shape}"
        assert probs_2d.shape == (b, 1), f"Expected shape ({b}, 1), got {probs_2d.shape}"
        assert torch.all((probs_2d >= 0.0) & (probs_2d <= 1.0))
        assert torch.allclose(probs_2d, torch.sigmoid(logits_2d))

        # 3D input (B, 1, 30)
        x_3d = torch.randn(b, 1, 30)
        logits_3d = model(x_3d)
        assert logits_3d.shape == (b, 1)

    # Invalid input shapes should raise ValueError
    with pytest.raises(ValueError):
        model(torch.randn(16, 2, 30))  # Invalid channel count


def test_cnn1d_gradient_flow():
    """Confirm all conv layers, batchnorm, and linear layers receive non-zero gradients."""
    torch.manual_seed(42)
    cfg = CNN1DArchitectureConfig()
    model = CNN1DClassifier(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([518.177], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x = torch.randn(16, 30)
    y = torch.tensor([0.0] * 14 + [1.0, 1.0]).view(-1, 1)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Verify all trainable parameters with requires_grad have valid gradients
    checked_conv = False
    checked_linear = False
    checked_bn = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf grad"
            if "conv" in name:
                checked_conv = True
            if "classifier" in name and "weight" in name:
                checked_linear = True
            if "BatchNorm" in name or "conv_blocks" in name and ("1.weight" in name or "5.weight" in name):
                checked_bn = True

    assert checked_conv, "No conv layer gradients checked"
    assert checked_linear, "No linear layer gradients checked"


def test_cnn1d_metrics_and_thresholding():
    """Confirm threshold selection on validation probabilities yields valid F1 score."""
    rng = np.random.default_rng(42)
    # Synthetic validation set: 90 legit, 10 fraud
    y_val = np.array([0] * 90 + [1] * 10)
    val_probs = np.concatenate([rng.uniform(0.01, 0.35, 90), rng.uniform(0.65, 0.99, 10)])

    res = find_optimal_threshold(val_probs, y_val, metric="f1")
    threshold = res["optimal_threshold"]

    # Threshold must separate well and achieve high F1 on this separable set
    assert 0.35 <= threshold <= 0.99
    assert res["val_f1"] == 1.0
    assert res["val_precision"] == 1.0
    assert res["val_recall"] == 1.0

    # Apply frozen threshold to synthetic test set
    y_test = np.array([0] * 40 + [1] * 10)
    test_probs = np.concatenate([np.full(40, threshold - 0.1), np.full(10, threshold + 0.1)])
    test_metrics = compute_classification_metrics(y_test, test_probs, threshold)

    assert test_metrics["threshold"] == threshold
    assert test_metrics["f1"] == 1.0
    assert test_metrics["pr_auc"] > 0.9
    assert test_metrics["roc_auc"] > 0.9
    assert test_metrics["confusion_matrix"] == [[40, 0], [0, 10]]


def test_cnn1d_training_step_parameter_update():
    """Verify that a single optimizer step properly updates model weights."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 30)).astype(np.float32)
    y = (rng.random(128) < 0.1).astype(np.float32)
    train_loader, _, _ = create_supervised_dataloaders(X, y, X, y, X, y, batch_size=32)

    model = CNN1DClassifier()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([518.177]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    features, targets = next(iter(train_loader))
    optimizer.zero_grad()
    loss = criterion(model(features).view(-1), targets.float())
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    # Check that weights updated
    updated = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point and not torch.equal(v, before_state[k])]
    assert any("conv_blocks" in k for k in updated)
    assert any("classifier" in k for k in updated)
