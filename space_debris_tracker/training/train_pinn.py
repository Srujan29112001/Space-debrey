"""
Physics-Informed Neural Network (PINN) Training
Combines data loss with physics constraints for orbital mechanics
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..trajectory_prediction.physics.orbital_mechanics import OrbitalMechanics
from ..trajectory_prediction.pinn.physics_informed_nn import PhysicsInformedNN
from .dataset import OrbitDataset, create_dataloaders


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting for multi-task learning
    Learns optimal weights for data and physics losses
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.num_tasks = num_tasks
        # Log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss

        Args:
            losses: Tensor of losses [num_tasks]

        Returns:
            Weighted loss
        """
        precision = torch.exp(-self.log_vars)
        weighted_loss = torch.sum(precision * losses + self.log_vars)
        return weighted_loss


class PINNTrainer:
    """
    PINN Trainer with physics-informed loss and adaptive weighting
    """

    def __init__(
        self,
        model: PhysicsInformedNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
        output_dir: str = "outputs/pinn_training",
    ):
        """
        Initialize PINN trainer

        Args:
            model: PINN model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            device: Device to train on
            output_dir: Output directory
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.epochs = config.get("epochs", 200)
        self.physics_loss_weight = config.get("physics_loss_weight", 0.1)
        self.adaptive_weights = config.get("adaptive_weights", True)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Adaptive loss weighting
        if self.adaptive_weights:
            self.loss_weighting = AdaptiveLossWeighting(num_tasks=2).to(device)
            self.weight_optimizer = optim.Adam(
                self.loss_weighting.parameters(), lr=0.001
            )
        else:
            self.loss_weighting = None

        # Orbital mechanics for validation
        self.orbital_mechanics = OrbitalMechanics()

        # Tensorboard
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        else:
            self.writer = None

        # Best metrics
        self.best_val_loss = float("inf")

        print(f"PINNTrainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Physics loss weight: {self.physics_loss_weight}")
        print(f"  Adaptive weighting: {self.adaptive_weights}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        lr = self.config.get("lr", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)

        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        metrics = {
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "total_loss": 0.0,
            "energy_conservation": 0.0,
            "momentum_conservation": 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch in pbar:
            input_seq = batch["input"].to(self.device)  # [B, T, 6]
            target_seq = batch["target"].to(self.device)  # [B, T', 6]

            batch_size, seq_len, _ = input_seq.shape

            # Forward pass - predict next states
            predictions = []
            current_state = input_seq[:, -1, :]  # Last state

            for t in range(target_seq.shape[1]):
                # Add time dimension
                time = torch.ones(batch_size, 1, device=self.device) * t
                model_input = torch.cat([current_state, time], dim=1)

                # Predict next state
                next_state = self.model(model_input)
                predictions.append(next_state)

                # Update current state
                current_state = next_state

            predictions = torch.stack(predictions, dim=1)  # [B, T', 6]

            # Data loss (MSE)
            data_loss = nn.functional.mse_loss(predictions, target_seq)

            # Physics loss
            physics_loss = torch.tensor(0.0, device=self.device)
            energy_violation = torch.tensor(0.0, device=self.device)
            momentum_violation = torch.tensor(0.0, device=self.device)

            for t in range(target_seq.shape[1] - 1):
                curr_pred = predictions[:, t, :]
                next_pred = predictions[:, t + 1, :]

                # Physics loss from model
                phys_loss = self.model.compute_physics_loss(
                    next_pred, curr_pred, dt=60.0  # 60 seconds
                )
                physics_loss += phys_loss

                # Energy conservation check
                energy_loss = self.model._compute_energy_loss(
                    curr_pred[:, :3],
                    curr_pred[:, 3:],
                    next_pred[:, :3],
                    next_pred[:, 3:],
                )
                energy_violation += energy_loss

                # Momentum conservation check
                momentum_loss = self.model._compute_momentum_loss(
                    curr_pred[:, :3],
                    curr_pred[:, 3:],
                    next_pred[:, :3],
                    next_pred[:, 3:],
                )
                momentum_violation += momentum_loss

            physics_loss /= target_seq.shape[1] - 1
            energy_violation /= target_seq.shape[1] - 1
            momentum_violation /= target_seq.shape[1] - 1

            # Total loss
            if self.adaptive_weights:
                # Adaptive weighting
                losses = torch.stack([data_loss, physics_loss])
                total_loss = self.loss_weighting(losses)

                # Update loss weights
                self.weight_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.weight_optimizer.step()
            else:
                # Fixed weighting
                total_loss = data_loss + self.physics_loss_weight * physics_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            metrics["data_loss"] += data_loss.item()
            metrics["physics_loss"] += physics_loss.item()
            metrics["total_loss"] += total_loss.item()
            metrics["energy_conservation"] += energy_violation.item()
            metrics["momentum_conservation"] += momentum_violation.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "data": f"{data_loss.item():.4f}",
                    "phys": f"{physics_loss.item():.4f}",
                    "total": f"{total_loss.item():.4f}",
                }
            )

        # Average metrics
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        metrics = {
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "total_loss": 0.0,
            "position_rmse": 0.0,
            "velocity_rmse": 0.0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_seq = batch["input"].to(self.device)
                target_seq = batch["target"].to(self.device)

                batch_size, seq_len, _ = input_seq.shape

                # Forward pass
                predictions = []
                current_state = input_seq[:, -1, :]

                for t in range(target_seq.shape[1]):
                    time = torch.ones(batch_size, 1, device=self.device) * t
                    model_input = torch.cat([current_state, time], dim=1)
                    next_state = self.model(model_input)
                    predictions.append(next_state)
                    current_state = next_state

                predictions = torch.stack(predictions, dim=1)

                # Data loss
                data_loss = nn.functional.mse_loss(predictions, target_seq)

                # Physics loss
                physics_loss = torch.tensor(0.0, device=self.device)
                for t in range(target_seq.shape[1] - 1):
                    curr_pred = predictions[:, t, :]
                    next_pred = predictions[:, t + 1, :]

                    phys_loss = self.model.compute_physics_loss(
                        next_pred, curr_pred, dt=60.0
                    )
                    physics_loss += phys_loss

                physics_loss /= target_seq.shape[1] - 1

                # Total loss
                total_loss = data_loss + self.physics_loss_weight * physics_loss

                # RMSE metrics
                position_rmse = torch.sqrt(
                    nn.functional.mse_loss(predictions[:, :, :3], target_seq[:, :, :3])
                )
                velocity_rmse = torch.sqrt(
                    nn.functional.mse_loss(predictions[:, :, 3:], target_seq[:, :, 3:])
                )

                # Update metrics
                metrics["data_loss"] += data_loss.item()
                metrics["physics_loss"] += physics_loss.item()
                metrics["total_loss"] += total_loss.item()
                metrics["position_rmse"] += position_rmse.item()
                metrics["velocity_rmse"] += velocity_rmse.item()

        # Average metrics
        n_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def train(self):
        """Run full training"""
        print(f"\nStarting PINN training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total_loss"])
                else:
                    self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
            print(f"    Data: {train_metrics['data_loss']:.6f}")
            print(f"    Physics: {train_metrics['physics_loss']:.6f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
            print(f"    Position RMSE: {val_metrics['position_rmse']:.6f} km")
            print(f"    Velocity RMSE: {val_metrics['velocity_rmse']:.6f} km/s")

            if self.adaptive_weights:
                weights = torch.exp(-self.loss_weighting.log_vars).detach().cpu()
                print(
                    f"  Adaptive weights: Data={weights[0]:.4f}, Physics={weights[1]:.4f}"
                )

            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

                if self.adaptive_weights:
                    weights = torch.exp(-self.loss_weighting.log_vars).detach()
                    self.writer.add_scalar("adaptive_weights/data", weights[0], epoch)
                    self.writer.add_scalar(
                        "adaptive_weights/physics", weights[1], epoch
                    )

            # Save checkpoint
            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model saved!")

            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\nTraining completed!")

        if self.writer is not None:
            self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.adaptive_weights:
            checkpoint["loss_weighting_state_dict"] = self.loss_weighting.state_dict()

        if is_best:
            path = self.output_dir / "best.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    # Example training configuration
    config = {
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "physics_loss_weight": 0.1,
        "adaptive_weights": True,
        "scheduler": "cosine",
    }

    print("PINN Training Script")
    print("=" * 50)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nInitializing PINN model...")
    model = PhysicsInformedNN(
        input_dim=7,
        hidden_dims=[256, 512, 512, 256],
        output_dim=6,
        physics_loss_weight=config["physics_loss_weight"],
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_type="orbit",
            data_path="data/orbits.h5",
            batch_size=config["batch_size"],
            num_workers=4,
            sequence_length=100,
            prediction_horizon=50,
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Dataset will create synthetic data...")

        # Create dataset with synthetic data
        dataset = OrbitDataset(
            data_path=Path("/tmp/dummy_orbits.h5"),
            sequence_length=100,
            prediction_horizon=50,
        )
        from torch.utils.data import random_split

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
        )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir="outputs/pinn_training",
    )

    # Start training
    print("\nStarting training...")
    trainer.train()
