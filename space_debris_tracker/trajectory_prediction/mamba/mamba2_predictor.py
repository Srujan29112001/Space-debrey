"""
Mamba2 Predictor for Long-term Orbit Evolution
State Space Model for efficient long-sequence modeling
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class S6Block(nn.Module):
    """
    Simplified S6 (Selective State Space) block
    Based on Mamba architecture
    """

    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        # SSM parameters
        self.x_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # Input projection
        x_and_res = self.in_proj(x)
        x_ssm, res = x_and_res.chunk(2, dim=-1)

        # Convolution (along sequence dimension)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)

        # Activation
        x_conv = torch.nn.functional.silu(x_conv)

        # SSM (simplified)
        x_ssm = self._apply_ssm(x_conv)

        # Gated output
        output = x_ssm * torch.nn.functional.silu(res)

        # Output projection
        output = self.out_proj(output)

        # Residual connection
        return x + output

    def _apply_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective state space model"""
        # Simplified SSM implementation
        # In production, use full Mamba SSM with selective scan

        batch, seq_len, d_model = x.shape

        # Initialize state
        h = torch.zeros(batch, self.d_state, device=x.device)

        outputs = []

        for t in range(seq_len):
            # Discretization parameters
            dt = torch.sigmoid(self.dt_proj(x[:, t]))  # [batch, d_model]

            # State projection
            x_t = self.x_proj(x[:, t])  # [batch, d_state]

            # Update state (simplified)
            h = 0.9 * h + 0.1 * x_t

            # Output
            y_t = h @ torch.randn(self.d_state, d_model, device=x.device)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class Mamba2Predictor(nn.Module):
    """
    Mamba2 for long-term trajectory prediction
    Efficient state space model for very long sequences
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 24,
        seq_len: int = 10000,
    ):
        """
        Initialize Mamba2

        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            n_layers: Number of layers
            seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(6, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList(
            [S6Block(d_model=d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, 6)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input trajectory [batch, seq_len, 6]

        Returns:
            Predicted trajectory [batch, seq_len, 6]
        """
        # Project input
        x = self.input_proj(x)

        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        # Project to output
        trajectory = self.output_proj(x)

        return trajectory


if __name__ == "__main__":
    # Test Mamba2
    model = Mamba2Predictor(d_model=512, d_state=128, d_conv=4, n_layers=12, seq_len=10000)

    print(f"Mamba2 Predictor:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with long sequence
    batch_size = 2
    seq_len = 1000
    x = torch.randn(batch_size, seq_len, 6)

    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
