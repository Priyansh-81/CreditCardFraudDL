"""Multilayer Perceptron (MLP) baseline for supervised fraud classification.

Deep feed-forward network over the 30 scaled transaction features
(Time, V1-V28, Amount). Each hidden block is
Linear -> BatchNorm1d -> ReLU -> Dropout, and the output head emits a single
raw logit. Sigmoid is NOT applied in forward() because training uses
nn.BCEWithLogitsLoss, which fuses the sigmoid for numerical stability.
"""

import torch
import torch.nn as nn

from src.config import MLPArchitectureConfig


class MLPBaseline(nn.Module):
    """Feed-forward fraud classifier: 30 -> 128 -> 64 -> 32 -> 1 (logit)."""

    def __init__(self, config: MLPArchitectureConfig = None):
        super().__init__()
        if config is None:
            config = MLPArchitectureConfig()
        self.config = config

        layers = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch_size, 1)."""
        return self.classifier(self.feature_extractor(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probabilities sigmoid(logit) of shape (batch_size, 1)."""
        return torch.sigmoid(self.forward(x))
