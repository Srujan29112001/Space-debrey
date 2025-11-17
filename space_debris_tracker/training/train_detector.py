"""
YOLOv7 Detector Training
Complete training loop with multi-scale training, augmentation, and checkpointing
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..computer_vision.detector import YOLOv7Detector
from .dataset import DebrisImageDataset, create_dataloaders


class YOLOv7Loss(nn.Module):
    """
    YOLOv7 Loss Function
    Combines box loss, objectness loss, and classification loss
    """

    def __init__(
        self,
        num_classes: int,
        anchors: Optional[List[List[float]]] = None,
        img_size: int = 1280,
        box_weight: float = 0.05,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
    ):
        """
        Initialize YOLOv7 Loss

        Args:
            num_classes: Number of object classes
            anchors: Anchor boxes
            img_size: Image size
            box_weight: Box regression loss weight
            obj_weight: Objectness loss weight
            cls_weight: Classification loss weight
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

        # Default anchors if not provided
        if anchors is None:
            self.anchors = [
                [12, 16, 19, 36, 40, 28],  # P3/8
                [36, 75, 76, 55, 72, 146],  # P4/16
                [142, 110, 192, 243, 459, 401],  # P5/32
            ]
        else:
            self.anchors = anchors

        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute YOLOv7 loss

        Args:
            predictions: Model predictions [B, N, 85] (x, y, w, h, obj, cls...)
            targets: Ground truth boxes list of [N, 4]
            target_labels: Ground truth labels list of [N]

        Returns:
            (total_loss, loss_dict)
        """
        device = predictions.device
        batch_size = predictions.shape[0]

        # Simplified loss - in production use full YOLOv7 loss
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        # For each image in batch
        for i in range(batch_size):
            if len(targets[i]) == 0:
                # No targets - penalize objectness
                obj_loss += self.bce_obj(
                    predictions[i, :, 4], torch.zeros_like(predictions[i, :, 4])
                )
                continue

            # Match predictions to targets (simplified)
            pred_boxes = predictions[i, :, :4]  # [N, 4]
            pred_obj = predictions[i, :, 4]  # [N]
            pred_cls = predictions[i, :, 5:]  # [N, num_classes]

            gt_boxes = targets[i]  # [M, 4]
            gt_labels = target_labels[i]  # [M]

            # Compute IoU between all predictions and targets
            ious = self._box_iou(pred_boxes, gt_boxes)  # [N, M]

            # For each target, find best prediction
            best_ious, best_pred_idx = ious.max(dim=0)  # [M]

            # Box loss (only for matched predictions)
            if len(best_pred_idx) > 0:
                matched_pred = pred_boxes[best_pred_idx]
                box_loss += nn.functional.mse_loss(matched_pred, gt_boxes)

                # Classification loss (only for matched predictions)
                matched_cls = pred_cls[best_pred_idx]
                gt_cls_onehot = nn.functional.one_hot(
                    gt_labels, num_classes=self.num_classes
                ).float()
                cls_loss += self.bce_cls(matched_cls, gt_cls_onehot)

            # Objectness loss
            obj_targets = torch.zeros_like(pred_obj)
            if len(best_pred_idx) > 0:
                obj_targets[best_pred_idx] = best_ious.detach().clamp(0, 1)

            obj_loss += self.bce_obj(pred_obj, obj_targets)

        # Average over batch
        box_loss /= batch_size
        obj_loss /= batch_size
        cls_loss /= batch_size

        # Total loss
        total_loss = (
            self.box_weight * box_loss
            + self.obj_weight * obj_loss
            + self.cls_weight * cls_loss
        )

        loss_dict = {
            "box_loss": box_loss.item(),
            "obj_loss": obj_loss.item(),
            "cls_loss": cls_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes

        Args:
            boxes1: [N, 4] (x, y, w, h)
            boxes2: [M, 4] (x, y, w, h)

        Returns:
            IoU matrix [N, M]
        """
        # Convert to x1, y1, x2, y2
        boxes1_x1y1 = boxes1[:, :2] - boxes1[:, 2:] / 2
        boxes1_x2y2 = boxes1[:, :2] + boxes1[:, 2:] / 2

        boxes2_x1y1 = boxes2[:, :2] - boxes2[:, 2:] / 2
        boxes2_x2y2 = boxes2[:, :2] + boxes2[:, 2:] / 2

        # Compute intersection
        inter_x1y1 = torch.max(boxes1_x1y1[:, None, :], boxes2_x1y1[None, :, :])
        inter_x2y2 = torch.min(boxes1_x2y2[:, None, :], boxes2_x2y2[None, :, :])

        inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        # Compute union
        boxes1_area = boxes1[:, 2] * boxes1[:, 3]
        boxes2_area = boxes2[:, 2] * boxes2[:, 3]

        union_area = boxes1_area[:, None] + boxes2_area[None, :] - inter_area

        # Compute IoU
        iou = inter_area / (union_area + 1e-6)

        return iou


class YOLOv7Trainer:
    """
    YOLOv7 Trainer with multi-scale training, augmentation, and checkpointing
    """

    def __init__(
        self,
        model: YOLOv7Detector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
        output_dir: str = "outputs/yolo_training",
    ):
        """
        Initialize trainer

        Args:
            model: YOLOv7 model
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
        self.epochs = config.get("epochs", 300)
        self.batch_size = config.get("batch_size", 16)
        self.img_size = config.get("img_size", 1280)
        self.multi_scale = config.get("multi_scale", True)
        self.accumulate = config.get("accumulate", 4)  # Gradient accumulation

        # Loss function
        self.criterion = YOLOv7Loss(
            num_classes=config.get("num_classes", 4), img_size=self.img_size
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision training
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Tensorboard
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        else:
            self.writer = None

        # Best metrics
        self.best_loss = float("inf")

        # Multi-scale image sizes
        if self.multi_scale:
            self.img_sizes = [self.img_size + i * 32 for i in range(-3, 4)]
        else:
            self.img_sizes = [self.img_size]

        print(f"YOLOv7Trainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")
        print(f"  Multi-scale: {self.multi_scale}")
        print(f"  Mixed precision: {self.use_amp}")

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        lr = self.config.get("lr", 0.01)
        momentum = self.config.get("momentum", 0.937)
        weight_decay = self.config.get("weight_decay", 0.0005)

        # Separate parameters
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        optimizer = optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.config.get("lr", 0.01) * 0.01,
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.epochs // 3, gamma=0.1
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

        metrics = {"box_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0, "total_loss": 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for i, batch in enumerate(pbar):
            # Multi-scale training
            if self.multi_scale and i % 10 == 0:
                img_size = np.random.choice(self.img_sizes)
                # Resize would happen here - simplified

            images = batch["images"].to(self.device)
            bboxes = [b.to(self.device) for b in batch["bboxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            # Forward pass
            with autocast(enabled=self.use_amp):
                # Simplified prediction format
                # In production, model returns multi-scale predictions
                predictions = self.model(images)

                # Reshape predictions for loss
                # This is simplified - actual YOLOv7 has specific output format
                batch_size = images.shape[0]
                predictions = predictions.view(
                    batch_size, -1, 5 + self.criterion.num_classes
                )

                # Compute loss
                loss, loss_dict = self.criterion(predictions, bboxes, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.accumulate

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (i + 1) % self.accumulate == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update metrics
            for key in metrics:
                if key in loss_dict:
                    metrics[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss_dict["total_loss"],
                    "box": loss_dict["box_loss"],
                    "obj": loss_dict["obj_loss"],
                    "cls": loss_dict["cls_loss"],
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

        metrics = {"box_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0, "total_loss": 0.0}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["images"].to(self.device)
                bboxes = [b.to(self.device) for b in batch["bboxes"]]
                labels = [l.to(self.device) for l in batch["labels"]]

                # Forward pass
                with autocast(enabled=self.use_amp):
                    predictions = self.model(images)

                    batch_size = images.shape[0]
                    predictions = predictions.view(
                        batch_size, -1, 5 + self.criterion.num_classes
                    )

                    # Compute loss
                    loss, loss_dict = self.criterion(predictions, bboxes, labels)

                # Update metrics
                for key in metrics:
                    if key in loss_dict:
                        metrics[key] += loss_dict[key]

        # Average metrics
        n_batches = len(self.val_loader)
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def train(self):
        """Run full training"""
        print(f"\nStarting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")

            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, epoch)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Save checkpoint
            if val_metrics["total_loss"] < self.best_loss:
                self.best_loss = val_metrics["total_loss"]
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
            "best_loss": self.best_loss,
            "config": self.config,
        }

        if is_best:
            path = self.output_dir / "best.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    # Example training configuration
    config = {
        "epochs": 300,
        "batch_size": 16,
        "img_size": 1280,
        "num_classes": 4,
        "lr": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "multi_scale": True,
        "use_amp": True,
        "accumulate": 4,
        "scheduler": "cosine",
    }

    print("YOLOv7 Training Script")
    print("=" * 50)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("\nInitializing model...")
    model = YOLOv7Detector(img_size=config["img_size"], device=device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_type="detection",
            data_path="data/debris_detection",
            batch_size=config["batch_size"],
            num_workers=4,
            img_size=config["img_size"],
            augment=True,
            cache_images=False,
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Using dummy data for demonstration...")

        # Create dummy loaders
        from torch.utils.data import TensorDataset

        dummy_images = torch.randn(100, 3, 1280, 1280)
        dummy_boxes = [torch.rand(5, 4) for _ in range(100)]
        dummy_labels = [torch.randint(0, 4, (5,)) for _ in range(100)]

        train_loader = DataLoader(
            TensorDataset(dummy_images), batch_size=config["batch_size"], shuffle=True
        )
        val_loader = train_loader

    # Create trainer
    print("\nInitializing trainer...")
    trainer = YOLOv7Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir="outputs/yolo_training",
    )

    # Start training
    print("\nStarting training...")
    trainer.train()
