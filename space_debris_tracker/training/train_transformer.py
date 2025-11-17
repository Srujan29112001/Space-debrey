"""
Trajectory Transformer Training
Sequence-to-sequence training with teacher forcing and multi-object modeling
"""

import random
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

from ..trajectory_prediction.transformer.trajectory_transformer import (
    TrajectoryTransformer,
)
from .dataset import OrbitDataset, create_dataloaders


class TransformerTrainer:
    """
    Transformer Trainer with teacher forcing and multi-step prediction
    """

    def __init__(
        self,
        model: TrajectoryTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
        output_dir: str = "outputs/transformer_training",
    ):
        """
        Initialize Transformer trainer

        Args:
            model: Trajectory Transformer model
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
        self.epochs = config.get("epochs", 100)
        self.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)
        self.teacher_forcing_decay = config.get("teacher_forcing_decay", 0.99)
        self.warmup_epochs = config.get("warmup_epochs", 10)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Loss function
        self.criterion = nn.MSELoss()

        # Tensorboard
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        else:
            self.writer = None

        # Best metrics
        self.best_val_loss = float("inf")

        # Current teacher forcing ratio
        self.current_tf_ratio = self.teacher_forcing_ratio

        print(f"TransformerTrainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Teacher forcing ratio: {self.teacher_forcing_ratio}")
        print(f"  Warmup epochs: {self.warmup_epochs}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with parameter groups"""
        lr = self.config.get("lr", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)

        # Separate embedding and transformer parameters
        embed_params = []
        transformer_params = []

        for name, param in self.model.named_parameters():
            if "projection" in name or "pos_encoder" in name:
                embed_params.append(param)
            else:
                transformer_params.append(param)

        return optim.AdamW(
            [
                {"params": embed_params, "lr": lr},
                {"params": transformer_params, "lr": lr},
            ],
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def _build_scheduler(self):
        """Build learning rate scheduler with warmup"""
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        total_steps = self.epochs * len(self.train_loader)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with teacher forcing

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        metrics = {"loss": 0.0, "position_mae": 0.0, "velocity_mae": 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_seq = batch["input"].to(self.device)  # [B, T_in, 6]
            target_seq = batch["target"].to(self.device)  # [B, T_out, 6]

            batch_size, input_len, _ = input_seq.shape
            _, target_len, _ = target_seq.shape

            # Teacher forcing decision
            use_teacher_forcing = random.random() < self.current_tf_ratio

            if use_teacher_forcing:
                # Teacher forcing: use ground truth as input
                # Concatenate input and target for decoder input
                decoder_input = torch.cat(
                    [
                        input_seq[:, -1:, :],  # Last input state
                        target_seq[:, :-1, :],  # All but last target
                    ],
                    dim=1,
                )

                # Forward pass
                predictions = self.model(input_seq, decoder_input)

                # Compute loss on predictions vs targets
                loss = self.criterion(predictions[:, 1:, :], target_seq)

            else:
                # Autoregressive: use predictions as input
                predictions = []
                decoder_state = input_seq[:, -1:, :]  # Start with last input

                for t in range(target_len):
                    # Predict next state
                    pred = self.model(input_seq, decoder_state)
                    next_state = pred[:, -1:, :]  # Take last prediction
                    predictions.append(next_state)

                    # Update decoder input
                    decoder_state = torch.cat([decoder_state, next_state], dim=1)

                predictions = torch.cat(predictions, dim=1)  # [B, T_out, 6]

                # Compute loss
                loss = self.criterion(predictions, target_seq)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.get("grad_clip", 1.0)
            )

            self.optimizer.step()
            self.scheduler.step()

            # Compute metrics
            with torch.no_grad():
                position_mae = torch.mean(torch.abs(predictions[:, :, :3] - target_seq[:, :, :3]))
                velocity_mae = torch.mean(torch.abs(predictions[:, :, 3:] - target_seq[:, :, 3:]))

            # Update metrics
            metrics["loss"] += loss.item()
            metrics["position_mae"] += position_mae.item()
            metrics["velocity_mae"] += velocity_mae.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "pos_mae": f"{position_mae.item():.6f}",
                    "tf_ratio": f"{self.current_tf_ratio:.3f}",
                }
            )

        # Average metrics
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches

        # Decay teacher forcing ratio
        self.current_tf_ratio *= self.teacher_forcing_decay
        self.current_tf_ratio = max(0.1, self.current_tf_ratio)

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model (always autoregressive)

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        metrics = {
            "loss": 0.0,
            "position_mae": 0.0,
            "velocity_mae": 0.0,
            "position_rmse": 0.0,
            "velocity_rmse": 0.0,
            "position_rmse_1step": 0.0,
            "position_rmse_final": 0.0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_seq = batch["input"].to(self.device)
                target_seq = batch["target"].to(self.device)

                batch_size, input_len, _ = input_seq.shape
                _, target_len, _ = target_seq.shape

                # Autoregressive prediction
                predictions = []
                decoder_state = input_seq[:, -1:, :]

                for t in range(target_len):
                    pred = self.model(input_seq, decoder_state)
                    next_state = pred[:, -1:, :]
                    predictions.append(next_state)
                    decoder_state = torch.cat([decoder_state, next_state], dim=1)

                predictions = torch.cat(predictions, dim=1)

                # Compute loss
                loss = self.criterion(predictions, target_seq)

                # Compute detailed metrics
                position_mae = torch.mean(torch.abs(predictions[:, :, :3] - target_seq[:, :, :3]))
                velocity_mae = torch.mean(torch.abs(predictions[:, :, 3:] - target_seq[:, :, 3:]))

                position_rmse = torch.sqrt(
                    torch.mean((predictions[:, :, :3] - target_seq[:, :, :3]) ** 2)
                )
                velocity_rmse = torch.sqrt(
                    torch.mean((predictions[:, :, 3:] - target_seq[:, :, 3:]) ** 2)
                )

                # First step error
                position_rmse_1step = torch.sqrt(
                    torch.mean((predictions[:, 0, :3] - target_seq[:, 0, :3]) ** 2)
                )

                # Final step error
                position_rmse_final = torch.sqrt(
                    torch.mean((predictions[:, -1, :3] - target_seq[:, -1, :3]) ** 2)
                )

                # Update metrics
                metrics["loss"] += loss.item()
                metrics["position_mae"] += position_mae.item()
                metrics["velocity_mae"] += velocity_mae.item()
                metrics["position_rmse"] += position_rmse.item()
                metrics["velocity_rmse"] += velocity_rmse.item()
                metrics["position_rmse_1step"] += position_rmse_1step.item()
                metrics["position_rmse_final"] += position_rmse_final.item()

        # Average metrics
        n_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def train(self):
        """Run full training"""
        print(f"\nStarting Transformer training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Log metrics
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"    Position RMSE: {val_metrics['position_rmse']:.6f}")
            print(f"    Velocity RMSE: {val_metrics['velocity_rmse']:.6f}")
            print(f"    1-step RMSE: {val_metrics['position_rmse_1step']:.6f}")
            print(f"    Final RMSE: {val_metrics['position_rmse_final']:.6f}")
            print(f"  Teacher forcing ratio: {self.current_tf_ratio:.3f}")
            print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                self.writer.add_scalar("teacher_forcing_ratio", self.current_tf_ratio, epoch)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Save checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
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
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "teacher_forcing_ratio": self.current_tf_ratio,
            "config": self.config,
        }

        if is_best:
            path = self.output_dir / "best.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_tf_ratio = checkpoint.get("teacher_forcing_ratio", self.teacher_forcing_ratio)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")


if __name__ == "__main__":
    # Example training configuration
    config = {
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "teacher_forcing_ratio": 0.5,
        "teacher_forcing_decay": 0.99,
        "warmup_epochs": 10,
        "grad_clip": 1.0,
    }

    print("Transformer Training Script")
    print("=" * 50)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nInitializing Trajectory Transformer...")
    model = TrajectoryTransformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
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
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir="outputs/transformer_training",
    )

    # Start training
    print("\nStarting training...")
    trainer.train()
