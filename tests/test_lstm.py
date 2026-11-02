"""Tests for causal LSTM sequence construction and classifier."""
import numpy as np
import torch

from src.config import LSTMArchitectureConfig
from src.dataset import SequenceTransactionDataset, create_sequence_dataloaders
from src.models.lstm import LSTMClassifier


def test_lstm_sequence_builder_shapes():
    x = np.arange(15 * 30, dtype=np.float32).reshape(15, 30)
    ds = SequenceTransactionDataset(x, np.zeros(15), window_size=10)
    assert len(ds) == 15
    assert ds[0][0].shape == (10, 30)
    assert ds[9][0].shape == (10, 30)
    assert ds[14][0].shape == (10, 30)


def test_temporal_leakage_in_sequence_dataset():
    # Feature zero acts as a monotonically increasing timestamp.
    train = np.zeros((12, 30), dtype=np.float32)
    val = np.zeros((5, 30), dtype=np.float32)
    test = np.zeros((4, 30), dtype=np.float32)
    train[:, 0] = np.arange(12)
    val[:, 0] = np.arange(12, 17)
    test[:, 0] = np.arange(17, 21)
    y_train = np.zeros(12)
    y_val = np.zeros(5)
    y_test = np.zeros(4)
    _, val_loader, test_loader = create_sequence_dataloaders(
        train, y_train, val, y_val, test, y_test, window_size=10, batch_size=32
    )
    val_windows, _ = next(iter(val_loader))
    test_windows, _ = next(iter(test_loader))
    for windows in (val_windows, test_windows):
        for window in windows:
            times = window[:, 0]
            assert torch.all(times[1:] >= times[:-1])
            assert torch.all(times <= times[-1])
    # First validation sample receives train history, never later validation rows.
    assert val_windows[0, -1, 0].item() == 12
    assert val_windows[0, -2, 0].item() == 11
    # First test sample receives only the immediately preceding train+val history.
    assert test_windows[0, -1, 0].item() == 17
    assert test_windows[0, -2, 0].item() == 16


def test_lstm_forward_pass_and_gradient():
    model = LSTMClassifier(LSTMArchitectureConfig())
    x = torch.randn(8, 10, 30)
    logits = model(x)
    assert logits.shape == (8,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.randint(0, 2, (8,)).float())
    loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None


def test_lstm_reproducibility():
    x = np.random.default_rng(42).normal(size=(20, 30)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    a = SequenceTransactionDataset(x, y, window_size=10)
    b = SequenceTransactionDataset(x, y, window_size=10)
    for i in range(len(a)):
        torch.testing.assert_close(a[i][0], b[i][0])
        assert a[i][1].item() == b[i][1].item()
