"""Attention-based Tabular Autoencoder for Anomaly and Fraud Detection.

Strictly follows the Phase 1 synopsis specification:
- 2-layer Transformer encoder
- 4 attention heads
- 8-unit bottleneck
- Decoder reconstructs the original 30-dimensional feature vector
- Reconstruction error serves as the anomaly score
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import AutoencoderArchitectureConfig, config


class FeatureTokenizer(nn.Module):
    """Converts a continuous tabular feature vector into a sequence of feature tokens.

    Tabular transactions are not natural language sequences. To apply self-attention
    meaningfully across tabular features (similar to FT-Transformer, NeurIPS 2021),
    each of the D scalar features is projected into its own d_model-dimensional embedding space:
        Token_j = x_j * W_j + b_j + P_j
    where W_j in R^{1 x d_model}, b_j in R^{d_model}, and P_j is a learnable column/feature
    identity embedding.

    This produces an output of shape (Batch, num_features, d_model), enabling the
    multi-head self-attention mechanism to learn pairwise interactions across all features.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Dedicated weight and bias per feature
        # Weight shape: (num_features, d_model), Bias shape: (num_features, d_model)
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))

        # Learnable column identity (positional) embedding
        self.feature_embeddings = nn.Parameter(torch.empty(num_features, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.feature_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters

        ----------
        x : torch.Tensor
            Input batch of tabular features of shape (Batch, num_features).

        Returns
        -------
        torch.Tensor
            Embedded token sequence of shape (Batch, num_features, d_model).
        """
        # x is (B, D) -> unsqueeze to (B, D, 1)
        x_expanded = x.unsqueeze(-1)

        # Broadcast multiply with feature weights: (B, D, 1) * (D, d_model) -> (B, D, d_model)
        tokens = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Add column identity embedding
        tokens = tokens + self.feature_embeddings.unsqueeze(0)
        return tokens


class AttentionAutoencoder(nn.Module):
    """Attention-based Tabular Autoencoder.

    Architecture pipeline:
    1. Feature Tokenizer: Maps 30 input features to (B, 30, d_model)
    2. Transformer Encoder: 2 layers, 4 attention heads with self-attention across features
    3. Bottleneck: Compresses token sequence down to an 8-dimensional latent space
    4. Decoder: Non-linear MLP reconstructing the original 30 features
    """

    def __init__(self, cfg: AutoencoderArchitectureConfig = None):
        super().__init__()
        if cfg is None:
            cfg = config.model

        self.input_dim = cfg.input_dim
        self.d_model = cfg.d_model
        self.nhead = cfg.nhead
        self.num_layers = cfg.num_encoder_layers
        self.bottleneck_dim = cfg.bottleneck_dim
        self.dim_feedforward = cfg.dim_feedforward
        self.dropout_rate = cfg.dropout

        # 1. Feature Tokenizer Layer
        self.tokenizer = FeatureTokenizer(num_features=self.input_dim, d_model=self.d_model)

        # 2. Transformer Encoder (2 layers, 4 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # 3. Bottleneck Compression (Flatten -> Linear -> 8-dim latent bottleneck)
        flattened_dim = self.input_dim * self.d_model
        self.bottleneck = nn.Sequential(
            nn.Linear(flattened_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.bottleneck_dim),
        )

        # 4. Non-Linear Decoder: Maps 8-dim latent space back to 30 reconstructed features
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features into the 8-dimensional bottleneck.

        Parameters
        ----------
        x : torch.Tensor
            Batch of features, shape (Batch, 30).

        Returns
        -------
        torch.Tensor
            Bottleneck latent representation, shape (Batch, 8).
        """
        # (Batch, 30, d_model)
        tokens = self.tokenizer(x)
        # (Batch, 30, d_model)
        encoded_tokens = self.transformer_encoder(tokens)
        # Flatten across feature tokens: (Batch, 30 * d_model)
        flattened = encoded_tokens.reshape(encoded_tokens.size(0), -1)
        # (Batch, 8)
        latent = self.bottleneck(flattened)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent bottleneck vector back to reconstructed feature space.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (Batch, 8).

        Returns
        -------
        torch.Tensor
            Reconstructed feature vector, shape (Batch, 30).
        """
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Attention Autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (Batch, 30).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (reconstructed_x, latent_representation)
            - reconstructed_x shape: (Batch, 30)
            - latent shape: (Batch, 8)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample Mean Squared Error (MSE) reconstruction error.

        Used as the anomaly score:
            Score_i = (1 / D) * sum_{j=1}^D (x_{i,j} - x_hat_{i,j})^2

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (Batch, 30).

        Returns
        -------
        torch.Tensor
            1-D Tensor of anomaly scores, shape (Batch,).
        """
        reconstruction, _ = self.forward(x)
        # Mean squared error across feature dimension (dim=1)
        per_sample_mse = torch.mean((x - reconstruction) ** 2, dim=-1)
        return per_sample_mse
