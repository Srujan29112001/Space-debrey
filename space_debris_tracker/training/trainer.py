"""
Unified Training Framework
Generic trainer supporting all models with DDP, mixed precision, and W&B integration
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .checkpoint_manager import CheckpointManager
from .validation import ConjunctionEvaluator, DetectionEvaluator, TrajectoryEvaluator


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current score

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class UnifiedTrainer:
    """
    Unified trainer for all Space Debris Tracking models
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Distributed training (DDP)
    - Early stopping
    - W&B integration
    - Comprehensive logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        config: Dict,
        device: str = "cuda",
        output_dir: str = "outputs/training",
        use_ddp: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize unified trainer

        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            loss_fn: Loss function
            config: Training configuration
            device: Device to train on
            output_dir: Output directory
            use_ddp: Use distributed data parallel
            local_rank: Local rank for DDP
            world_size: World size for DDP
        """
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Distributed training
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = local_rank == 0

        # Model
        self.model = model.to(device)
        if use_ddp:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.loss_fn = loss_fn

        # Training parameters
        self.epochs = config.get("epochs", 100)
        self.gradient_accumulation_steps = config.get("gradient_accumulation", 1)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        early_stop_config = config.get("early_stopping", {})
        if early_stop_config.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get("patience", 10),
                min_delta=early_stop_config.get("min_delta", 0.0),
                mode=early_stop_config.get("mode", "min"),
            )
        else:
            self.early_stopping = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            max_checkpoints=config.get("max_checkpoints", 5),
            model_name=config.get("model_name", "model"),
        )

        # Logging
        if self.is_main_process:
            # Tensorboard
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
            else:
                self.writer = None

            # Weights & Biases
            self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE
            if self.use_wandb:
                wandb.init(
                    project=config.get("wandb_project", "space-debris-tracking"),
                    name=config.get(
                        "experiment_name",
                        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    ),
                    config=config,
                )
                wandb.watch(self.model, log="all", log_freq=100)
        else:
            self.writer = None
            self.use_wandb = False

        # Best metrics
        self.best_metric = (
            float("inf") if config.get("metric_mode", "min") == "min" else float("-inf")
        )
        self.metric_mode = config.get("metric_mode", "min")

        # Save config
        if self.is_main_process:
            self._save_config()

        if self.is_main_process:
            print(f"\nUnifiedTrainer initialized")
            print(f"  Model: {config.get('model_name', 'model')}")
            print(f"  Epochs: {self.epochs}")
            print(f"  Device: {device}")
            print(f"  Mixed precision: {self.use_amp}")
            print(f"  DDP: {use_ddp}")
            if use_ddp:
                print(f"  World size: {world_size}")
            print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"  W&B: {self.use_wandb}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config"""
        opt_config = self.config.get("optimizer", {})
        opt_type = opt_config.get("type", "adamw")
        lr = opt_config.get("lr", 1e-4)
        weight_decay = opt_config.get("weight_decay", 1e-5)

        if opt_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get("betas", (0.9, 0.999)),
            )
        elif opt_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=opt_config.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config.get("scheduler", {})
        sched_type = sched_config.get("type", "cosine")

        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=sched_config.get("eta_min", 1e-6),
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 30),
                gamma=sched_config.get("gamma", 0.1),
            )
        elif sched_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.metric_mode,
                factor=sched_config.get("factor", 0.5),
                patience=sched_config.get("patience", 10),
            )
        else:
            return None

    def _save_config(self):
        """Save training configuration"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        # Progress bar (only on main process)
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                loss = self.loss_fn(self.model, batch)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.get("grad_clip", 0) > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.get("grad_clip")
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1

            # Update progress bar
            if self.is_main_process:
                pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

        # Average loss
        avg_loss = total_loss / n_batches

        # Synchronize across processes
        if self.use_ddp:
            avg_loss = self._sync_metric(avg_loss)

        return {"loss": avg_loss}

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            if self.is_main_process:
                pbar = tqdm(self.val_loader, desc="Validation")
            else:
                pbar = self.val_loader

            for batch in pbar:
                batch = self._move_to_device(batch)

                with autocast(enabled=self.use_amp):
                    loss = self.loss_fn(self.model, batch)

                total_loss += loss.item()
                n_batches += 1

        # Average loss
        avg_loss = total_loss / n_batches

        # Synchronize across processes
        if self.use_ddp:
            avg_loss = self._sync_metric(avg_loss)

        return {"loss": avg_loss}

    def train(self):
        """Run full training loop"""
        if self.is_main_process:
            print(f"\nStarting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Set epoch for distributed sampler
            if self.use_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Log metrics
            if self.is_main_process:
                self._log_metrics(epoch, train_metrics, val_metrics)

                # Check if best model
                current_metric = val_metrics["loss"]
                is_best = self._is_best_metric(current_metric)

                # Save checkpoint
                if is_best or epoch % self.config.get("save_freq", 10) == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.module if self.use_ddp else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=is_best,
                    )

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(current_metric):
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break

        if self.is_main_process:
            print("\nTraining completed!")
            if self.writer is not None:
                self.writer.close()
            if self.use_wandb:
                wandb.finish()

    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch

    def _sync_metric(self, metric: float) -> float:
        """Synchronize metric across processes"""
        metric_tensor = torch.tensor(metric, device=self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item() / self.world_size

    def _is_best_metric(self, metric: float) -> bool:
        """Check if metric is best so far"""
        if self.metric_mode == "min":
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric

        return is_best

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console and trackers"""
        # Console
        print(f"\nEpoch {epoch}/{self.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6e}")

        # Tensorboard
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

        # Weights & Biases
        if self.use_wandb:
            log_dict = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value

            wandb.log(log_dict, step=epoch)


def setup_distributed():
    """Setup distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


if __name__ == "__main__":
    print("Unified Trainer - Example Usage")
    print("=" * 50)

    # Example configuration
    config = {
        "model_name": "example_model",
        "epochs": 10,
        "optimizer": {"type": "adamw", "lr": 1e-4, "weight_decay": 1e-5},
        "scheduler": {"type": "cosine", "eta_min": 1e-6},
        "use_amp": True,
        "gradient_accumulation": 4,
        "grad_clip": 1.0,
        "early_stopping": {"enabled": True, "patience": 5, "min_delta": 1e-4},
        "use_wandb": False,
        "save_freq": 5,
        "max_checkpoints": 3,
        "metric_mode": "min",
    }

    # Example model
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    # Example loss function
    def example_loss_fn(model, batch):
        x = batch["input"]
        y = batch["target"]
        pred = model(x)
        return nn.functional.mse_loss(pred, y)

    # Create dummy data
    from torch.utils.data import TensorDataset

    train_x = torch.randn(1000, 10)
    train_y = torch.randn(1000, 5)
    val_x = torch.randn(200, 10)
    val_y = torch.randn(200, 5)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    # Create dataloaders
    def collate_fn(batch):
        x, y = zip(*batch)
        return {"input": torch.stack(x), "target": torch.stack(y)}

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create model
    model = ExampleModel()

    # Setup distributed (if needed)
    rank, world_size, local_rank = setup_distributed()
    use_ddp = world_size > 1

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=example_loss_fn,
        config=config,
        device=device,
        output_dir="outputs/example_training",
        use_ddp=use_ddp,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Train
    trainer.train()

    print("\nExample completed!")
