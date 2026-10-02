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
