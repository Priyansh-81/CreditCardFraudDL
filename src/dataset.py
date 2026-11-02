"""PyTorch Dataset and DataLoader abstractions for tabular transaction data."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularTransactionDataset(Dataset):
    """PyTorch Dataset for Credit Card transactions.

    Supports both reconstruction training (unsupervised / legitimate only)
    and evaluation mode (features with true labels).

    Parameters
    ----------
    features : np.ndarray or pd.DataFrame
        Scaled 30-dimensional feature vectors (Time, V1-V28, Amount).
    labels : np.ndarray or pd.Series, optional
        Binary labels (0 = legit, 1 = fraud). Can be None during reconstruction.
    """

    def __init__(self, features, labels=None):
        if hasattr(features, "values"):
            features = features.values
        self.features = torch.tensor(features, dtype=torch.float32)

        if labels is not None:
            if hasattr(labels, "values"):
                labels = labels.values
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class SequenceTransactionDataset(Dataset):
    """Temporal sliding-window dataset for LSTM-based sequence classifiers.

    Each sample is a window of ``window_size`` consecutive transactions ending
    at (and including) the current transaction.  For the first few samples where
    fewer than ``window_size`` predecessors exist, the window is *causally
    padded* by repeating the earliest available transaction (or zeros when no
    prefix is supplied).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (N, D), e.g. (N, 30).
    labels : np.ndarray or None
        Binary labels of shape (N,). ``None`` for unsupervised / inference.
    window_size : int
        Number of time-steps in each window (default 10).
    prefix_features : np.ndarray or None
        Feature rows from *preceding* partitions (e.g. the last 9 training rows
        when building the validation dataset).  These are used exclusively for
        padding the initial windows of this partition so that every sample has a
        full-length context — they are **never** associated with labels and
        therefore cannot introduce label leakage.
    """

    def __init__(
        self,
        features,
        labels=None,
        window_size: int = 10,
        prefix_features=None,
    ):
        if hasattr(features, "values"):
            features = features.values
        self.features = np.asarray(features, dtype=np.float32)
        self.n_samples, self.n_features = self.features.shape
        self.window_size = window_size

        if labels is not None:
            if hasattr(labels, "values"):
                labels = labels.values
            self.labels = np.asarray(labels, dtype=np.float32)
        else:
            self.labels = None

        # Build the full feature timeline used for window extraction.
        # prefix_features provide causal history from earlier partitions.
        if prefix_features is not None:
            if hasattr(prefix_features, "values"):
                prefix_features = prefix_features.values
            prefix = np.asarray(prefix_features, dtype=np.float32)
            # Only keep the last (window_size - 1) prefix rows; that is the
            # maximum lookback any sample in this partition will ever need.
            max_prefix = self.window_size - 1
            if len(prefix) > max_prefix:
                prefix = prefix[-max_prefix:]
            self._timeline = np.concatenate([prefix, self.features], axis=0)
            self._offset = len(prefix)
        else:
            self._timeline = self.features
            self._offset = 0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # The current transaction sits at position (self._offset + idx) in the
        # full timeline.  We need ``window_size`` rows ending at that position.
        end = self._offset + idx + 1  # exclusive upper bound
        start = end - self.window_size

        if start >= 0:
            window = self._timeline[start:end]
        else:
            # Not enough history even after prefix — pad by repeating earliest
            # available row (or zeros if timeline is empty, which shouldn't
            # happen in practice).
            available = self._timeline[:end]
            pad_count = self.window_size - len(available)
            if len(available) > 0:
                pad = np.repeat(available[:1], pad_count, axis=0)
            else:
                pad = np.zeros((pad_count, self.n_features), dtype=np.float32)
            window = np.concatenate([pad, available], axis=0)

        window_tensor = torch.tensor(window, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return window_tensor, label_tensor
        return (window_tensor,)


def create_dataloaders(
    X_train_legit,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create standard PyTorch DataLoaders for training, validation, and testing.

    Training loader:
      - Uses ONLY legitimate training samples (Class == 0)
      - Shuffled across epochs for gradient descent stability
    Validation & Test loaders:
      - Retain both classes and true labels
      - Preserved sequential order (shuffle=False) for deterministic evaluation
    """
    train_dataset = TabularTransactionDataset(X_train_legit)
    val_dataset = TabularTransactionDataset(X_val, y_val)
    test_dataset = TabularTransactionDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_supervised_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create labelled DataLoaders for supervised models (MLP, 1D-CNN, LSTM).

    Training loader:
      - Uses the FULL training partition (both classes) with labels
      - Shuffled across epochs; no oversampling (imbalance handled via pos_weight)
    Validation & Test loaders:
      - Retain true labels in sequential order (shuffle=False)
    """
    train_dataset = TabularTransactionDataset(X_train, y_train)
    val_dataset = TabularTransactionDataset(X_val, y_val)
    test_dataset = TabularTransactionDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_sequence_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    window_size: int = 10,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create temporal-window DataLoaders for LSTM sequence classifiers.

    Training sequences are built from the training partition only.  Validation
    windows can look back into the last ``window_size - 1`` training rows for
    causal context.  Test windows can look back into the last
    ``window_size - 1`` rows of the combined train + validation timeline.

    This ensures:
    - No future transactions leak into any window.
    - No labels from other partitions leak (prefix rows carry features only).
    - Every sample has a full-length sequence (shape ``(window_size, D)``).
    """
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(X_val, "values"):
        X_val = X_val.values
    if hasattr(X_test, "values"):
        X_test = X_test.values

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # Training: no prefix (pad by repeating earliest train row)
    train_dataset = SequenceTransactionDataset(
        X_train, y_train, window_size=window_size, prefix_features=None
    )

    # Validation: prefix = last (window_size-1) training rows
    val_prefix = X_train[-(window_size - 1):] if len(X_train) >= (window_size - 1) else X_train
    val_dataset = SequenceTransactionDataset(
        X_val, y_val, window_size=window_size, prefix_features=val_prefix
    )

    # Test: prefix = last (window_size-1) rows of train+val combined
    combined = np.concatenate([X_train, X_val], axis=0)
    test_prefix = combined[-(window_size - 1):] if len(combined) >= (window_size - 1) else combined
    test_dataset = SequenceTransactionDataset(
        X_test, y_test, window_size=window_size, prefix_features=test_prefix
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
