# Space Debris Tracking - Training Infrastructure

Complete production-ready training infrastructure for the Space Debris Tracking system.

## Overview

This module provides comprehensive training capabilities for all models in the system:

1. **YOLOv7 Detector** - Object detection in telescope images
2. **Physics-Informed Neural Networks (PINN)** - Orbital mechanics-constrained prediction
3. **Trajectory Transformer** - Multi-object interaction modeling
4. **Mamba2** - Long-sequence orbit prediction

## Module Structure

```
training/
├── dataset.py              # Dataset loaders for all model types
├── train_detector.py       # YOLOv7 training with multi-scale augmentation
├── train_pinn.py           # PINN training with physics constraints
├── train_transformer.py    # Transformer with teacher forcing
├── train_mamba.py          # Memory-efficient Mamba training
├── trainer.py              # Unified training framework (DDP, AMP, W&B)
├── validation.py           # Comprehensive evaluation metrics
├── checkpoint_manager.py   # Model versioning and export
└── README.md               # This file
```

## Dataset Loaders

### DebrisImageDataset

Load telescope images with YOLO-format annotations:

```python
from space_debris_tracker.training import DebrisImageDataset

dataset = DebrisImageDataset(
    image_dir='data/images',
    annotation_dir='data/annotations',
    img_size=1280,
    augment=True,
    cache_images=False
)
```

**Features:**
- YOLO format annotations (x, y, w, h normalized)
- Data augmentation: rotation, flip, brightness, noise
- Efficient caching
- Multi-scale support

### OrbitDataset

Load orbital trajectories for sequence prediction:

```python
from space_debris_tracker.training import OrbitDataset

dataset = OrbitDataset(
    data_path='data/orbits.h5',
    sequence_length=100,
    prediction_horizon=50,
    normalize=True
)
```

**Features:**
- HDF5 format support
- Automatic normalization
- Configurable sequence lengths
- Creates synthetic data if file not found

### ConjunctionDataset

Load conjunction events for risk prediction:

```python
from space_debris_tracker.training import ConjunctionDataset

dataset = ConjunctionDataset(
    data_path='data/conjunctions.json',
    lookback=100,
    time_to_tca=50
)
```

## Training Scripts

### 1. YOLOv7 Detector Training

```python
from space_debris_tracker.training import YOLOv7Trainer, create_dataloaders
from space_debris_tracker.computer_vision.detector import YOLOv7Detector

# Configuration
config = {
    'epochs': 300,
    'batch_size': 16,
    'img_size': 1280,
    'lr': 0.01,
    'multi_scale': True,
    'use_amp': True
}

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_type='detection',
    data_path='data/debris_detection',
    batch_size=config['batch_size']
)

# Initialize model and trainer
model = YOLOv7Detector(img_size=1280)
trainer = YOLOv7Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# Train
trainer.train()
```

**Features:**
- Multi-scale training
- Mosaic and MixUp augmentation
- Cosine learning rate scheduling
- Mixed precision (FP16)
- Tensorboard logging

### 2. PINN Training

```python
from space_debris_tracker.training import PINNTrainer
from space_debris_tracker.trajectory_prediction.pinn import PhysicsInformedNN

config = {
    'epochs': 200,
    'batch_size': 64,
    'lr': 1e-4,
    'physics_loss_weight': 0.1,
    'adaptive_weights': True
}

model = PhysicsInformedNN(
    input_dim=7,
    hidden_dims=[256, 512, 512, 256],
    output_dim=6
)

trainer = PINNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

trainer.train()
```

**Features:**
- Physics-informed loss (data + physics)
- Orbital mechanics constraints (J2, energy, momentum)
- Adaptive loss weighting
- Validation on conservation laws

### 3. Transformer Training

```python
from space_debris_tracker.training import TransformerTrainer
from space_debris_tracker.trajectory_prediction.transformer import TrajectoryTransformer

config = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'teacher_forcing_ratio': 0.5,
    'teacher_forcing_decay': 0.99,
    'warmup_epochs': 10
}

model = TrajectoryTransformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

trainer = TransformerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

trainer.train()
```

**Features:**
- Teacher forcing with decay
- Learning rate warmup
- Gradient clipping
- Multi-step prediction
- Horizon-specific metrics

### 4. Mamba Training

```python
from space_debris_tracker.training import MemoryEfficientMambaTrainer
from space_debris_tracker.trajectory_prediction.mamba import Mamba2Predictor

config = {
    'epochs': 100,
    'batch_size': 16,
    'lr': 1e-4,
    'chunk_size': 1000,
    'gradient_checkpointing': True
}

model = Mamba2Predictor(
    d_model=512,
    d_state=128,
    n_layers=12,
    seq_len=5000
)

trainer = MemoryEfficientMambaTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

trainer.train()
```

**Features:**
- Memory-efficient long-sequence training
- Gradient checkpointing
- Chunked processing
- Long-term prediction validation

## Unified Trainer

For maximum flexibility, use the unified trainer:

```python
from space_debris_tracker.training import UnifiedTrainer

config = {
    'model_name': 'my_model',
    'epochs': 100,
    'optimizer': {'type': 'adamw', 'lr': 1e-4},
    'scheduler': {'type': 'cosine'},
    'use_amp': True,
    'gradient_accumulation': 4,
    'early_stopping': {'enabled': True, 'patience': 10},
    'use_wandb': True,
    'wandb_project': 'space-debris'
}

def loss_fn(model, batch):
    x = batch['input']
    y = batch['target']
    pred = model(x)
    return F.mse_loss(pred, y)

trainer = UnifiedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    config=config
)

trainer.train()
```

**Features:**
- Mixed precision (FP16)
- Gradient accumulation
- Distributed training (DDP)
- Early stopping
- Weights & Biases integration
- Comprehensive logging

## Validation Metrics

### Detection Metrics

```python
from space_debris_tracker.training import DetectionEvaluator

evaluator = DetectionEvaluator(num_classes=4)

metrics = evaluator.evaluate(predictions, ground_truths)
print(f"mAP@0.5: {metrics.map_50:.3f}")
print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
```

**Metrics:**
- Precision, Recall, F1
- mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
- Per-class metrics

### Trajectory Metrics

```python
from space_debris_tracker.training import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()

metrics = evaluator.evaluate(predictions, targets)
print(f"Position RMSE: {metrics.position_rmse:.3f} km")
print(f"Radial error: {metrics.radial_error:.3f} km")
print(f"Along-track error: {metrics.along_track_error:.3f} km")
```

**Metrics:**
- Position/Velocity RMSE and MAE
- Radial, Along-Track, Cross-Track errors
- Horizon-specific errors
- Conservation law violations

### Conjunction Metrics

```python
from space_debris_tracker.training import ConjunctionEvaluator

evaluator = ConjunctionEvaluator()

metrics = evaluator.evaluate(predicted_probs, true_labels)
print(f"ROC-AUC: {metrics.roc_auc:.3f}")
print(f"Calibration error: {metrics.calibration_error:.3f}")
```

**Metrics:**
- ROC-AUC, PR-AUC
- Brier score
- Calibration error
- Detection rate at specific FAR

## Checkpoint Management

```python
from space_debris_tracker.training import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    max_checkpoints=5,
    model_name='debris_detector'
)

# Save checkpoint
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'loss': 0.5},
    is_best=True
)

# Load checkpoint
checkpoint = manager.load_checkpoint(
    checkpoint_path='checkpoints/best.pt',
    model=model,
    optimizer=optimizer
)

# Create version
manager.create_version(
    model=model,
    version_name='v1.0',
    description='Production release'
)

# Export to ONNX
manager.export_to_onnx(
    model=model,
    example_input=torch.randn(1, 3, 1280, 1280),
    export_name='detector'
)

# Export to TorchScript
manager.export_to_torchscript(
    model=model,
    example_input=torch.randn(1, 3, 1280, 1280),
    export_name='detector_ts'
)
```

## Distributed Training

```bash
# Single GPU
python -m space_debris_tracker.training.train_detector

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 \
    -m space_debris_tracker.training.train_detector \
    --config configs/detector.yaml
```

In code:

```python
from space_debris_tracker.training import setup_distributed

rank, world_size, local_rank = setup_distributed()

trainer = UnifiedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    config=config,
    use_ddp=True,
    local_rank=local_rank,
    world_size=world_size
)
```

## Best Practices

### 1. Data Preparation

- Use HDF5 for large datasets
- Normalize positions/velocities
- Cache images when memory allows
- Use appropriate augmentation

### 2. Training Configuration

- Start with small learning rate (1e-4)
- Use warmup for transformers
- Enable gradient clipping (1.0)
- Monitor validation metrics closely

### 3. Model Selection

- **Detection**: YOLOv7 (1280px, multi-scale)
- **Short-term prediction (<1 hour)**: PINN or Transformer
- **Long-term prediction (>1 hour)**: Mamba2
- **Multi-object interaction**: Transformer

### 4. Hyperparameter Tuning

- Batch size: Adjust based on GPU memory
- Learning rate: Use warmup and cosine decay
- Physics loss weight: Start at 0.1, adjust based on validation
- Teacher forcing: Start at 0.5, decay to 0.1

## Performance Tips

### Memory Optimization

```python
# 1. Gradient checkpointing
config['gradient_checkpointing'] = True

# 2. Mixed precision
config['use_amp'] = True

# 3. Gradient accumulation
config['gradient_accumulation'] = 4

# 4. Smaller batch size with accumulation
config['batch_size'] = 8  # effective: 8 * 4 = 32
```

### Speed Optimization

```python
# 1. More workers
num_workers = 4

# 2. Pin memory
pin_memory = True

# 3. Persistent workers
persistent_workers = True

# 4. Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Reduce sequence length
- Enable chunked processing (Mamba)

### Slow Training

- Increase num_workers
- Enable mixed precision
- Use distributed training
- Profile with PyTorch profiler

### Poor Convergence

- Check learning rate
- Verify data normalization
- Monitor gradient norms
- Try different optimizer
- Adjust loss weights

## Example Workflows

### Complete Training Pipeline

```python
# 1. Prepare data
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_type='orbit',
    data_path='data/orbits.h5',
    batch_size=32
)

# 2. Initialize model
from space_debris_tracker.trajectory_prediction.pinn import PhysicsInformedNN
model = PhysicsInformedNN()

# 3. Train
trainer = PINNTrainer(model, train_loader, val_loader, config)
trainer.train()

# 4. Evaluate
evaluator = TrajectoryEvaluator()
metrics = evaluator.evaluate(predictions, targets)

# 5. Save and export
manager = CheckpointManager()
manager.create_version(model, 'v1.0', 'Production model')
manager.export_to_onnx(model, example_input)
```

## Citation

If you use this training infrastructure, please cite:

```bibtex
@software{space_debris_training,
  title = {Space Debris Tracking Training Infrastructure},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/your/repo}
}
```

## License

[Your License Here]

## Support

For issues or questions:
- GitHub Issues: [link]
- Documentation: [link]
- Email: [email]
