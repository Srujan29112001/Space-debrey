"""
Mamba2 Training for Long-Sequence Orbit Prediction
Efficient state space model training with memory-efficient batching
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
from datetime import datetime
import gc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .dataset import OrbitDataset, create_dataloaders
from ..trajectory_prediction.mamba.mamba2_predictor import Mamba2Predictor


class MemoryEfficientMambaTrainer:
    """
    Mamba2 Trainer with memory-efficient long-sequence handling
    """

    def __init__(
        self,
        model: Mamba2Predictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'outputs/mamba_training'
    ):
        """
        Initialize Mamba trainer

        Args:
            model: Mamba2 model
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
        self.epochs = config.get('epochs', 100)
        self.chunk_size = config.get('chunk_size', 1000)  # Process long sequences in chunks
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Loss function
        self.criterion = nn.MSELoss()

        # Tensorboard
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        else:
            self.writer = None

        # Best metrics
        self.best_val_loss = float('inf')

        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        print(f"MambaTrainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Gradient checkpointing: {self.gradient_checkpointing}")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for Mamba layers"""
        # Wrap forward pass of each layer with checkpoint
        for layer in self.model.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                layer.forward,
                use_reentrant=False
            )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        return optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None

    def _chunk_sequence(
        self,
        sequence: torch.Tensor,
        chunk_size: int
    ) -> List[torch.Tensor]:
        """
        Split long sequence into chunks for memory efficiency

        Args:
            sequence: [B, T, D]
            chunk_size: Chunk size

        Returns:
            List of chunks
        """
        seq_len = sequence.shape[1]
        chunks = []

        for i in range(0, seq_len, chunk_size):
            chunk = sequence[:, i:i + chunk_size, :]
            chunks.append(chunk)

        return chunks

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
            'loss': 0.0,
            'position_mae': 0.0,
            'velocity_mae': 0.0,
            'short_term_error': 0.0,  # 1-10 steps
            'medium_term_error': 0.0,  # 11-30 steps
            'long_term_error': 0.0   # 31+ steps
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_seq = batch['input'].to(self.device)  # [B, T_in, 6]
            target_seq = batch['target'].to(self.device)  # [B, T_out, 6]

            batch_size, input_len, _ = input_seq.shape
            _, target_len, _ = target_seq.shape

            # Autoregressive prediction
            # Concatenate input and target for full sequence
            full_seq = torch.cat([input_seq, target_seq], dim=1)

            # Process in chunks if sequence is too long
            if full_seq.shape[1] > self.chunk_size:
                # Chunked processing
                predictions = self._process_long_sequence(input_seq, target_len)
            else:
                # Direct processing
                # Use input to predict full sequence
                pred_full = self.model(full_seq)
                # Extract predictions for target portion
                predictions = pred_full[:, input_len:, :]

            # Compute loss
            loss = self.criterion(predictions, target_seq)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('grad_clip', 1.0)
            )

            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                position_mae = torch.mean(torch.abs(
                    predictions[:, :, :3] - target_seq[:, :, :3]
                ))
                velocity_mae = torch.mean(torch.abs(
                    predictions[:, :, 3:] - target_seq[:, :, 3:]
                ))

                # Error at different horizons
                if target_len >= 10:
                    short_error = torch.mean(torch.abs(
                        predictions[:, :10, :3] - target_seq[:, :10, :3]
                    ))
                else:
                    short_error = position_mae

                if target_len >= 30:
                    medium_error = torch.mean(torch.abs(
                        predictions[:, 10:30, :3] - target_seq[:, 10:30, :3]
                    ))
                    long_error = torch.mean(torch.abs(
                        predictions[:, 30:, :3] - target_seq[:, 30:, :3]
                    ))
                else:
                    medium_error = position_mae
                    long_error = position_mae

            # Update metrics
            metrics['loss'] += loss.item()
            metrics['position_mae'] += position_mae.item()
            metrics['velocity_mae'] += velocity_mae.item()
            metrics['short_term_error'] += short_error.item()
            metrics['medium_term_error'] += medium_error.item()
            metrics['long_term_error'] += long_error.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'pos_mae': f"{position_mae.item():.6f}",
                'mem': f"{torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
            })

            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Average metrics
        n_batches = len(self.train_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def _process_long_sequence(
        self,
        input_seq: torch.Tensor,
        target_len: int
    ) -> torch.Tensor:
        """
        Process long sequence in chunks

        Args:
            input_seq: Input sequence [B, T_in, 6]
            target_len: Target sequence length

        Returns:
            Predictions [B, target_len, 6]
        """
        predictions = []
        current_seq = input_seq

        for t in range(target_len):
            # Process current sequence
            if current_seq.shape[1] > self.chunk_size:
                # Only use last chunk_size states
                pred = self.model(current_seq[:, -self.chunk_size:, :])
                next_state = pred[:, -1:, :]
            else:
                pred = self.model(current_seq)
                next_state = pred[:, -1:, :]

            predictions.append(next_state)

            # Update sequence
            current_seq = torch.cat([current_seq, next_state], dim=1)

            # Keep only recent history to save memory
            if current_seq.shape[1] > self.chunk_size:
                current_seq = current_seq[:, -self.chunk_size:, :]

        return torch.cat(predictions, dim=1)

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
            'loss': 0.0,
            'position_mae': 0.0,
            'velocity_mae': 0.0,
            'position_rmse': 0.0,
            'velocity_rmse': 0.0,
            'short_term_rmse': 0.0,
            'medium_term_rmse': 0.0,
            'long_term_rmse': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)

                batch_size, input_len, _ = input_seq.shape
                _, target_len, _ = target_seq.shape

                # Predict
                if input_seq.shape[1] + target_len > self.chunk_size:
                    predictions = self._process_long_sequence(input_seq, target_len)
                else:
                    full_seq = torch.cat([input_seq, target_seq], dim=1)
                    pred_full = self.model(full_seq)
                    predictions = pred_full[:, input_len:, :]

                # Compute loss
                loss = self.criterion(predictions, target_seq)

                # Compute metrics
                position_mae = torch.mean(torch.abs(
                    predictions[:, :, :3] - target_seq[:, :, :3]
                ))
                velocity_mae = torch.mean(torch.abs(
                    predictions[:, :, 3:] - target_seq[:, :, 3:]
                ))

                position_rmse = torch.sqrt(torch.mean(
                    (predictions[:, :, :3] - target_seq[:, :, :3]) ** 2
                ))
                velocity_rmse = torch.sqrt(torch.mean(
                    (predictions[:, :, 3:] - target_seq[:, :, 3:]) ** 2
                ))

                # Horizon-specific RMSE
                if target_len >= 10:
                    short_rmse = torch.sqrt(torch.mean(
                        (predictions[:, :10, :3] - target_seq[:, :10, :3]) ** 2
                    ))
                else:
                    short_rmse = position_rmse

                if target_len >= 30:
                    medium_rmse = torch.sqrt(torch.mean(
                        (predictions[:, 10:30, :3] - target_seq[:, 10:30, :3]) ** 2
                    ))
                    long_rmse = torch.sqrt(torch.mean(
                        (predictions[:, 30:, :3] - target_seq[:, 30:, :3]) ** 2
                    ))
                else:
                    medium_rmse = position_rmse
                    long_rmse = position_rmse

                # Update metrics
                metrics['loss'] += loss.item()
                metrics['position_mae'] += position_mae.item()
                metrics['velocity_mae'] += velocity_mae.item()
                metrics['position_rmse'] += position_rmse.item()
                metrics['velocity_rmse'] += velocity_rmse.item()
                metrics['short_term_rmse'] += short_rmse.item()
                metrics['medium_term_rmse'] += medium_rmse.item()
                metrics['long_term_rmse'] += long_rmse.item()

        # Average metrics
        n_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def train(self):
        """Run full training"""
        print(f"\nStarting Mamba training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Metrics:")
            print(f"    Position RMSE: {val_metrics['position_rmse']:.6f} km")
            print(f"    Short-term (1-10): {val_metrics['short_term_rmse']:.6f} km")
            print(f"    Medium-term (11-30): {val_metrics['medium_term_rmse']:.6f} km")
            print(f"    Long-term (31+): {val_metrics['long_term_rmse']:.6f} km")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6e}")

            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar(
                    'lr',
                    self.optimizer.param_groups[0]['lr'],
                    epoch
                )

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
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
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if is_best:
            path = self.output_dir / 'best.pt'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    # Example training configuration
    config = {
        'epochs': 100,
        'batch_size': 16,  # Smaller batch for long sequences
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'chunk_size': 1000,
        'gradient_checkpointing': True,
        'grad_clip': 1.0,
        'scheduler': 'cosine'
    }

    print("Mamba2 Training Script")
    print("=" * 50)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
    print("\nInitializing Mamba2 model...")
    model = Mamba2Predictor(
        d_model=512,
        d_state=128,
        d_conv=4,
        n_layers=12,
        seq_len=5000  # Support very long sequences
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_type='orbit',
            data_path='data/orbits.h5',
            batch_size=config['batch_size'],
            num_workers=2,  # Fewer workers for large batches
            sequence_length=500,  # Longer input sequences
            prediction_horizon=500  # Longer predictions
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Dataset will create synthetic data...")

        # Create dataset with synthetic data
        dataset = OrbitDataset(
            data_path=Path('/tmp/dummy_orbits.h5'),
            sequence_length=500,
            prediction_horizon=500
        )
        from torch.utils.data import random_split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = MemoryEfficientMambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir='outputs/mamba_training'
    )

    # Start training
    print("\nStarting training...")
    trainer.train()
