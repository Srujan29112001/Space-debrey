"""
Physics-Informed Neural Network (PINN)
Combines neural networks with orbital mechanics for trajectory prediction
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for orbit prediction
    Enforces orbital mechanics constraints during training
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: List[int] = [256, 512, 512, 256],
        output_dim: int = 6,
        physics_loss_weight: float = 0.1,
    ):
        """
        Initialize PINN

        Args:
            input_dim: Input dimension (position + velocity + time)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (position + velocity)
            physics_loss_weight: Weight for physics loss
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_loss_weight = physics_loss_weight

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Physics constants (Earth)
        self.mu = 398600.4418  # km^3/s^2 - Earth's gravitational parameter
        self.J2 = 1.08263e-3  # J2 coefficient
        self.Re = 6378.137  # Earth radius (km)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input [batch, input_dim] = [pos(3), vel(3), time(1)]

        Returns:
            Prediction [batch, output_dim] = [pos(3), vel(3)]
        """
        return self.network(x)

    def compute_physics_loss(
        self, prediction: torch.Tensor, state: torch.Tensor, dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute physics-based loss

        Args:
            prediction: Network prediction [batch, 6]
            state: Current state [batch, 6]
            dt: Time step

        Returns:
            Physics loss
        """
        # Extract positions and velocities
        pos_pred = prediction[:, :3]
        vel_pred = prediction[:, 3:6]

        pos_curr = state[:, :3]
        vel_curr = state[:, 3:6]

        # Compute expected acceleration from orbital mechanics
        acc_physics = self._compute_acceleration(pos_curr)

        # Predicted acceleration from velocity change
        acc_predicted = (vel_pred - vel_curr) / dt

        # Physics loss: difference between predicted and physics-based acceleration
        physics_loss = torch.mean((acc_predicted - acc_physics) ** 2)

        # Energy conservation loss
        energy_loss = self._compute_energy_loss(pos_curr, vel_curr, pos_pred, vel_pred)

        # Angular momentum conservation loss
        momentum_loss = self._compute_momentum_loss(
            pos_curr, vel_curr, pos_pred, vel_pred
        )

        # Total physics loss
        total_loss = physics_loss + 0.1 * energy_loss + 0.1 * momentum_loss

        return total_loss

    def _compute_acceleration(self, position: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational acceleration

        Args:
            position: Position vectors [batch, 3]

        Returns:
            Acceleration [batch, 3]
        """
        # Two-body gravity
        r = torch.norm(position, dim=1, keepdim=True)
        a_gravity = -self.mu * position / (r**3)

        # J2 perturbation
        x, y, z = position[:, 0:1], position[:, 1:2], position[:, 2:3]

        factor = 1.5 * self.J2 * self.mu * (self.Re**2) / (r**5)

        a_j2_x = factor * x * (5 * (z**2) / (r**2) - 1)
        a_j2_y = factor * y * (5 * (z**2) / (r**2) - 1)
        a_j2_z = factor * z * (5 * (z**2) / (r**2) - 3)

        a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=1)

        return a_gravity + a_j2

    def _compute_energy_loss(
        self,
        pos1: torch.Tensor,
        vel1: torch.Tensor,
        pos2: torch.Tensor,
        vel2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy conservation loss"""
        # Specific orbital energy: E = v^2/2 - mu/r
        r1 = torch.norm(pos1, dim=1)
        r2 = torch.norm(pos2, dim=1)

        v1_sq = torch.sum(vel1**2, dim=1)
        v2_sq = torch.sum(vel2**2, dim=1)

        E1 = v1_sq / 2 - self.mu / r1
        E2 = v2_sq / 2 - self.mu / r2

        # Energy should be conserved
        energy_loss = torch.mean((E2 - E1) ** 2)

        return energy_loss

    def _compute_momentum_loss(
        self,
        pos1: torch.Tensor,
        vel1: torch.Tensor,
        pos2: torch.Tensor,
        vel2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angular momentum conservation loss"""
        # Angular momentum: L = r x v
        L1 = torch.cross(pos1, vel1, dim=1)
        L2 = torch.cross(pos2, vel2, dim=1)

        # Angular momentum should be conserved
        momentum_loss = torch.mean(torch.sum((L2 - L1) ** 2, dim=1))

        return momentum_loss

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        Training step with physics loss

        Args:
            inputs: Input states [batch, input_dim]
            targets: Target states [batch, output_dim]
            optimizer: Optimizer

        Returns:
            (data_loss, physics_loss)
        """
        optimizer.zero_grad()

        # Forward pass
        predictions = self.forward(inputs)

        # Data loss (MSE)
        data_loss = nn.functional.mse_loss(predictions, targets)

        # Physics loss
        current_states = inputs[:, :6]  # Remove time dimension
        physics_loss = self.compute_physics_loss(predictions, current_states)

        # Total loss
        total_loss = data_loss + self.physics_loss_weight * physics_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return data_loss.item(), physics_loss.item()


if __name__ == "__main__":
    # Test PINN
    pinn = PhysicsInformedNN(
        input_dim=7, hidden_dims=[256, 512, 256], output_dim=6, physics_loss_weight=0.1
    )

    print(f"PINN architecture:")
    print(pinn)
    print(f"\nTotal parameters: {sum(p.numel() for p in pinn.parameters()):,}")

    # Test forward pass
    batch_size = 32
    inputs = torch.randn(batch_size, 7)
    outputs = pinn(inputs)

    print(f"\nInput shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    # Test physics loss
    current_states = inputs[:, :6]
    physics_loss = pinn.compute_physics_loss(outputs, current_states, dt=60.0)
    print(f"Physics loss: {physics_loss.item():.6f}")
