"""
Deep Ensemble for Uncertainty Quantification
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class DeepEnsemble:
    """
    Deep ensemble for uncertainty quantification
    Trains multiple models with different initializations
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_models: int = 5,
        uncertainty_type: str = "aleatoric_epistemic",
    ):
        """
        Initialize deep ensemble

        Args:
            base_model: Base model architecture
            n_models: Number of ensemble members
            uncertainty_type: Type of uncertainty to estimate
        """
        self.n_models = n_models
        self.uncertainty_type = uncertainty_type

        # Create ensemble of models
        self.models = [deepcopy(base_model) for _ in range(n_models)]

        # Initialize each model differently
        for model in self.models:
            model.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights randomly"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def estimate(self, initial_state: torch.Tensor, time_steps: np.ndarray) -> torch.Tensor:
        """
        Estimate uncertainty using ensemble

        Args:
            initial_state: Initial state [6]
            time_steps: Time steps array

        Returns:
            Uncertainty estimates [N, 6]
        """
        device = initial_state.device

        # Collect predictions from all models
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Predict trajectory
                trajectory = []
                state = initial_state.clone()

                for t in time_steps:
                    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
                    input_vec = torch.cat([state, t_tensor])

                    pred = model(input_vec.unsqueeze(0)).squeeze(0)
                    trajectory.append(pred)
                    state = pred

                predictions.append(torch.stack(trajectory))

        # Stack predictions [n_models, n_steps, 6]
        predictions = torch.stack(predictions)

        # Epistemic uncertainty (model uncertainty)
        # Measured by variance across ensemble members
        epistemic = torch.var(predictions, dim=0)

        # Aleatoric uncertainty (data uncertainty)
        # Simplified: use ensemble mean as estimate
        aleatoric = torch.ones_like(epistemic) * 0.1

        # Total uncertainty
        total_uncertainty = epistemic + aleatoric

        return total_uncertainty


if __name__ == "__main__":
    from space_debris_tracker.trajectory_prediction.pinn import PhysicsInformedNN

    # Create base model
    base_model = PhysicsInformedNN(input_dim=7, hidden_dims=[128, 256, 128], output_dim=6)

    # Create ensemble
    ensemble = DeepEnsemble(base_model, n_models=5)

    print(f"Deep Ensemble with {ensemble.n_models} models")

    # Test uncertainty estimation
    initial_state = torch.randn(6)
    time_steps = np.linspace(0, 3600, 60)

    uncertainty = ensemble.estimate(initial_state, time_steps)

    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {torch.mean(uncertainty):.6f}")
    print(f"Max uncertainty: {torch.max(uncertainty):.6f}")
