"""1-D Convolutional Neural Network (1-D CNN) classifier for fraud detection.

Tabular features (Time, V1-V28, Amount) are treated as a 1-D spatial sequence (1, 30)
where local feature interactions and weight sharing regularize learning under extreme
class imbalance.
"""

from typing import Optional
import torch
import torch.nn as nn

from src.config import CNN1DArchitectureConfig


class CNN1DClassifier(nn.Module):
    """1-D Convolutional Neural Network fraud classifier."""

    def __init__(self, config: Optional[CNN1DArchitectureConfig] = None):
        super().__init__()
        if config is None:
            config = CNN1DArchitectureConfig()
        self.config = config

        channels = config.channels  # [32, 64, 128]
        kernel_size = config.kernel_size  # 3
        padding = kernel_size // 2  # 1

        conv_blocks = []
        in_c = 1
        for out_c in channels:
            conv_blocks.extend([
                nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_c = out_c

        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], config.dense_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tabular input of shape (B, 30) or sequence format (B, 1, 30).

        Returns
        -------
        torch.Tensor
            Raw scalar logits of shape (B, 1).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected input shape (B, 30) or (B, 1, 30), got {x.shape}")

        features = self.conv_blocks(x)  # (B, 128, 30)
        pooled = self.global_pool(features)  # (B, 128, 1)
        flattened = torch.flatten(pooled, 1)  # (B, 128)
        logits = self.classifier(flattened)  # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probabilities sigmoid(logit) of shape (B, 1)."""
        return torch.sigmoid(self.forward(x))
