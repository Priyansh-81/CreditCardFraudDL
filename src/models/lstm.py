"""LSTM classifier for causally ordered credit-card transaction windows."""
import torch
import torch.nn as nn
from src.config import LSTMArchitectureConfig


class LSTMClassifier(nn.Module):
    """Two-layer LSTM classifier returning one raw fraud logit per window."""
    def __init__(self, config: LSTMArchitectureConfig = None):
        super().__init__()
        self.config = config or LSTMArchitectureConfig()
        c = self.config
        self.lstm = nn.LSTM(
            input_size=c.input_dim, hidden_size=c.hidden_dim,
            num_layers=c.num_layers, batch_first=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(c.hidden_dim, c.dense_dim),
            nn.BatchNorm1d(c.dense_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.config.sequence_length or x.size(2) != self.config.input_dim:
            raise ValueError(
                f"Expected (B, {self.config.sequence_length}, {self.config.input_dim}), got {tuple(x.shape)}"
            )
        output, _ = self.lstm(x)
        representation = output[:, -1, :]
        if self.training and representation.size(0) == 1:
            # BatchNorm cannot estimate variance from a singleton final batch.
            linear = self.classifier[0](representation)
            bn = self.classifier[1]
            normalized = torch.nn.functional.batch_norm(
                linear, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                training=False, momentum=bn.momentum, eps=bn.eps,
            )
            hidden = self.classifier[2](normalized)
            hidden = self.classifier[3](hidden)
            logits = self.classifier[4](hidden)
        else:
            logits = self.classifier(representation)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
