"""
Training Infrastructure for Space Debris Tracking Models
Includes dataset loaders, training loops, and validation
"""

# Checkpoint management
from .checkpoint_manager import CheckpointManager

# Dataset loaders
from .dataset import (
    ConjunctionDataset,
    DebrisImageDataset,
    OrbitDataset,
    collate_fn_detection,
    create_dataloaders,
)

# Trainers
from .train_detector import YOLOv7Loss, YOLOv7Trainer
from .train_mamba import MemoryEfficientMambaTrainer
from .train_pinn import AdaptiveLossWeighting, PINNTrainer
from .train_transformer import TransformerTrainer
from .trainer import EarlyStopping, UnifiedTrainer, setup_distributed

# Validation
from .validation import (
    ConjunctionEvaluator,
    ConjunctionMetrics,
    DetectionEvaluator,
    DetectionMetrics,
    TrajectoryEvaluator,
    TrajectoryMetrics,
    plot_calibration_curve,
)

__all__ = [
    # Datasets
    "DebrisImageDataset",
    "OrbitDataset",
    "ConjunctionDataset",
    "create_dataloaders",
    "collate_fn_detection",
    # Trainers
    "YOLOv7Trainer",
    "YOLOv7Loss",
    "PINNTrainer",
    "AdaptiveLossWeighting",
    "TransformerTrainer",
    "MemoryEfficientMambaTrainer",
    "UnifiedTrainer",
    "EarlyStopping",
    "setup_distributed",
    # Validation
    "DetectionEvaluator",
    "TrajectoryEvaluator",
    "ConjunctionEvaluator",
    "DetectionMetrics",
    "TrajectoryMetrics",
    "ConjunctionMetrics",
    "plot_calibration_curve",
    # Checkpoint management
    "CheckpointManager",
]
