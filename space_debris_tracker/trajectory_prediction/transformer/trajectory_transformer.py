"""
Trajectory Transformer
Multi-object interaction modeling using attention mechanism
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding"""
        return x + self.pe[: x.size(0)]


class TrajectoryTransformer(nn.Module):
    """
    Transformer for trajectory prediction with multi-object interactions
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize Trajectory Transformer

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Input projection (state -> d_model)
        self.input_projection = nn.Linear(6, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # Output projection (d_model -> state)
        self.output_projection = nn.Linear(d_model, 6)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: Source trajectory [batch, seq_len, 6]
            tgt: Target context (nearby objects) [batch, n_objects, d_model]

        Returns:
            Refined trajectory [batch, seq_len, 6]
        """
        # Project input
        src_embedded = self.input_projection(src)  # [batch, seq_len, d_model]

        # Add positional encoding
        src_embedded = src_embedded.transpose(0, 1)  # [seq_len, batch, d_model]
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded.transpose(0, 1)  # [batch, seq_len, d_model]

        # If no target, use source as target
        if tgt is None:
            tgt = src_embedded

        # Transformer
        output = self.transformer(src_embedded, tgt)

        # Project to output space
        trajectory = self.output_projection(output)

        return trajectory


if __name__ == "__main__":
    # Test transformer
    model = TrajectoryTransformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

    print(f"Trajectory Transformer:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 100
    src = torch.randn(batch_size, seq_len, 6)

    output = model(src)
    print(f"\nInput shape: {src.shape}")
    print(f"Output shape: {output.shape}")
