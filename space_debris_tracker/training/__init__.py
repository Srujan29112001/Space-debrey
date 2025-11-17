"""
Training Infrastructure for Space Debris Tracking Models
Includes dataset loaders, training loops, and validation
"""

# Dataset loaders
from .dataset import (
    DebrisImageDataset,
    OrbitDataset,
    ConjunctionDataset,
    create_dataloaders,
    collate_fn_detection
)

# Trainers
from .train_detector import YOLOv7Trainer, YOLOv7Loss
from .train_pinn import PINNTrainer, AdaptiveLossWeighting
from .train_transformer import TransformerTrainer
from .train_mamba import MemoryEfficientMambaTrainer
from .trainer import UnifiedTrainer, EarlyStopping, setup_distributed

# Validation
from .validation import (
    DetectionEvaluator,
    TrajectoryEvaluator,
    ConjunctionEvaluator,
    DetectionMetrics,
    TrajectoryMetrics,
    ConjunctionMetrics,
    plot_calibration_curve
)

# Checkpoint management
from .checkpoint_manager import CheckpointManager

__all__ = [
    # Datasets
    'DebrisImageDataset',
    'OrbitDataset',
    'ConjunctionDataset',
    'create_dataloaders',
    'collate_fn_detection',

    # Trainers
    'YOLOv7Trainer',
    'YOLOv7Loss',
    'PINNTrainer',
    'AdaptiveLossWeighting',
    'TransformerTrainer',
    'MemoryEfficientMambaTrainer',
    'UnifiedTrainer',
    'EarlyStopping',
    'setup_distributed',

    # Validation
    'DetectionEvaluator',
    'TrajectoryEvaluator',
    'ConjunctionEvaluator',
    'DetectionMetrics',
    'TrajectoryMetrics',
    'ConjunctionMetrics',
    'plot_calibration_curve',

    # Checkpoint management
    'CheckpointManager',
]
