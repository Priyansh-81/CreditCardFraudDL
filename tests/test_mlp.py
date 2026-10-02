"""Unit tests for the supervised MLP baseline.

1. Forward pass output shape
2. pos_weight-weighted BCEWithLogitsLoss correctness
3. A single supervised training step updates the weights
4. Threshold is selected from validation probabilities only and applied to test
"""

import numpy as np
import torch
import torch.nn as nn

from src.config import MLPArchitectureConfig
from src.dataset import create_supervised_dataloaders
from src.models.mlp import MLPBaseline
from src.evaluation.evaluate_supervised import (
    find_optimal_threshold,
    compute_classification_metrics,
)
from src.evaluation.error_analysis import assign_outcomes


def test_mlp_forward_shape():
    torch.manual_seed(0)
    model = MLPBaseline(MLPArchitectureConfig())
    model.eval()
    x = torch.randn(16, 30)

    logits = model(x)
    probs = model.predict_proba(x)

    assert logits.shape == (16, 1)
    assert probs.shape == (16, 1)
    assert torch.all((probs >= 0) & (probs <= 1))
    # forward() must return raw logits, not probabilities
    assert torch.allclose(probs, torch.sigmoid(logits))

    # Architecture: three Linear->BatchNorm1d->ReLU->Dropout blocks, then Linear(32->1)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(30, 128), (128, 64), (64, 32), (32, 1)]
    assert sum(isinstance(m, nn.BatchNorm1d) for m in model.modules()) == 3
    assert all(m.p == 0.3 for m in model.modules() if isinstance(m, nn.Dropout))


def test_mlp_loss_computation():
    pos_weight = 518.177
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    p = torch.sigmoid(logits)
    manual = -(pos_weight * targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).mean()
    assert torch.allclose(criterion(logits, targets), manual, rtol=1e-5)

    # A missed fraud costs pos_weight times more than an equally confident false alarm
    miss_fraud = criterion(torch.tensor([-2.0]), torch.tensor([1.0]))
    false_alarm = criterion(torch.tensor([2.0]), torch.tensor([0.0]))
    assert torch.allclose(miss_fraud / false_alarm, torch.tensor(pos_weight), rtol=1e-4)


def test_mlp_supervised_training_step():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 30)).astype(np.float32)
    y = (rng.random(256) < 0.05).astype(np.float32)
    train_loader, _, _ = create_supervised_dataloaders(X, y, X, y, X, y, batch_size=64)

    model = MLPBaseline()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([518.177]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    features, targets = next(iter(train_loader))
    assert features.shape == (64, 30) and targets.shape == (64,)
    optimizer.zero_grad()
    loss = criterion(model(features).squeeze(), targets.float())
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    changed = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point and not torch.equal(v, before[k])]
    assert any("classifier.weight" in k for k in changed)
    assert any(k.startswith("feature_extractor.0.weight") for k in changed)


def test_mlp_threshold_selection():
    rng = np.random.default_rng(1)
    y_val = np.array([0] * 95 + [1] * 5)
    val_probs = np.concatenate([rng.uniform(0.0, 0.4, 95), rng.uniform(0.6, 0.9, 5)])

    result = find_optimal_threshold(val_probs, y_val)
    threshold = result["optimal_threshold"]

    # Perfectly separable validation set: threshold lies in the gap and F1 = 1
    assert 0.4 <= threshold <= 0.9
    assert result["val_f1"] == 1.0
    assert threshold in set(val_probs.tolist())

    # The threshold depends only on validation data: changing the test set cannot move it
    y_test = np.array([0, 0, 1, 1])
    test_probs = np.array([0.1, threshold + 0.01, threshold - 0.01, 0.95])
    assert find_optimal_threshold(val_probs, y_val)["optimal_threshold"] == threshold

    # Frozen threshold applied to test with >= rule
    metrics = compute_classification_metrics(y_test, test_probs, threshold)
    assert metrics["threshold"] == threshold
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert list(assign_outcomes(y_test, test_probs >= threshold)) == ["TN", "FP", "FN", "TP"]
