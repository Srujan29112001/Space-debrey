"""
Checkpoint Manager for Model Versioning and Export
Handles saving, loading, versioning, and export to ONNX/TorchScript
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
from datetime import datetime
import shutil
import hashlib


class CheckpointManager:
    """
    Comprehensive checkpoint management system
    - Save/load checkpoints
    - Track best models
    - Model versioning
    - Resume training
    - Export to ONNX/TorchScript
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5,
        model_name: str = 'model'
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            model_name: Name of the model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.model_name = model_name

        # Subdirectories
        self.versions_dir = self.checkpoint_dir / 'versions'
        self.versions_dir.mkdir(exist_ok=True)

        self.exports_dir = self.checkpoint_dir / 'exports'
        self.exports_dir.mkdir(exist_ok=True)

        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self.metadata = self._load_metadata()

        print(f"CheckpointManager initialized at {self.checkpoint_dir}")

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'checkpoints': [],
            'best_checkpoint': None,
            'versions': [],
            'current_version': 0
        }

    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Training metrics
            extra_state: Additional state to save
            is_best: Whether this is the best model

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'timestamp': timestamp,
            'metrics': metrics or {},
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Compute model hash
        model_hash = self._compute_model_hash(model)
        checkpoint['model_hash'] = model_hash

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best.pt'
            self.metadata['best_checkpoint'] = {
                'path': str(checkpoint_path),
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': timestamp
            }
        else:
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_epoch_{epoch}_{timestamp}.pt'

        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'is_best': is_best,
            'model_hash': model_hash
        }
        self.metadata['checkpoints'].append(checkpoint_info)

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

        # Save metadata
        self._save_metadata()

        print(f"Checkpoint saved: {checkpoint_path}")
        if metrics:
            print(f"  Metrics: {metrics}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint (if None, load best)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            # Load best checkpoint
            if self.metadata['best_checkpoint'] is None:
                raise ValueError("No best checkpoint found")
            checkpoint_path = self.metadata['best_checkpoint']['path']

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  Model state loaded")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state loaded")

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Scheduler state loaded")

        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def create_version(
        self,
        model: nn.Module,
        version_name: str,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Create a versioned model snapshot

        Args:
            model: Model to version
            version_name: Version identifier
            description: Version description
            metadata: Additional metadata

        Returns:
            Path to versioned model
        """
        version_id = len(self.metadata['versions']) + 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        version_dir = self.versions_dir / f'v{version_id}_{version_name}_{timestamp}'
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'version_id': version_id,
            'version_name': version_name,
            'timestamp': timestamp,
            'description': description,
            'metadata': metadata or {}
        }, model_path)

        # Save version info
        version_info = {
            'version_id': version_id,
            'version_name': version_name,
            'path': str(version_dir),
            'timestamp': timestamp,
            'description': description,
            'metadata': metadata or {}
        }

        self.metadata['versions'].append(version_info)
        self.metadata['current_version'] = version_id
        self._save_metadata()

        print(f"Model version created: v{version_id} - {version_name}")
        print(f"  Path: {version_dir}")

        return version_dir

    def list_checkpoints(self) -> List[Dict]:
        """
        List all checkpoints

        Returns:
            List of checkpoint info dictionaries
        """
        return self.metadata['checkpoints']

    def list_versions(self) -> List[Dict]:
        """
        List all model versions

        Returns:
            List of version info dictionaries
        """
        return self.metadata['versions']

    def get_best_checkpoint(self) -> Optional[Dict]:
        """
        Get best checkpoint info

        Returns:
            Best checkpoint info
        """
        return self.metadata['best_checkpoint']

    def export_to_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        export_name: str = 'model',
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ) -> Path:
        """
        Export model to ONNX format

        Args:
            model: Model to export
            example_input: Example input tensor
            export_name: Export file name
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification

        Returns:
            Path to exported ONNX model
        """
        model.eval()

        export_path = self.exports_dir / f'{export_name}.onnx'

        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        print(f"Exporting model to ONNX: {export_path}")

        torch.onnx.export(
            model,
            example_input,
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

        print(f"  ONNX export successful")
        print(f"  Opset version: {opset_version}")

        # Verify export
        try:
            import onnx
            onnx_model = onnx.load(str(export_path))
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX model verified")
        except ImportError:
            print(f"  Warning: onnx not installed, skipping verification")
        except Exception as e:
            print(f"  Warning: ONNX verification failed: {e}")

        return export_path

    def export_to_torchscript(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        export_name: str = 'model',
        method: str = 'trace'
    ) -> Path:
        """
        Export model to TorchScript

        Args:
            model: Model to export
            example_input: Example input tensor
            export_name: Export file name
            method: 'trace' or 'script'

        Returns:
            Path to exported TorchScript model
        """
        model.eval()

        export_path = self.exports_dir / f'{export_name}.pt'

        print(f"Exporting model to TorchScript: {export_path}")
        print(f"  Method: {method}")

        if method == 'trace':
            traced_model = torch.jit.trace(model, example_input)
        elif method == 'script':
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")

        torch.jit.save(traced_model, export_path)

        print(f"  TorchScript export successful")

        # Test exported model
        try:
            loaded_model = torch.jit.load(str(export_path))
            with torch.no_grad():
                output = loaded_model(example_input)
            print(f"  TorchScript model verified")
        except Exception as e:
            print(f"  Warning: TorchScript verification failed: {e}")

        return export_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        # Keep best checkpoint and max_checkpoints most recent
        regular_checkpoints = [
            cp for cp in self.metadata['checkpoints']
            if not cp.get('is_best', False)
        ]

        if len(regular_checkpoints) > self.max_checkpoints:
            # Sort by timestamp
            regular_checkpoints.sort(key=lambda x: x['timestamp'])

            # Remove oldest
            for cp in regular_checkpoints[:-self.max_checkpoints]:
                cp_path = Path(cp['path'])
                if cp_path.exists():
                    cp_path.unlink()
                    print(f"Removed old checkpoint: {cp_path}")

            # Update metadata
            self.metadata['checkpoints'] = [
                cp for cp in self.metadata['checkpoints']
                if cp.get('is_best', False) or cp in regular_checkpoints[-self.max_checkpoints:]
            ]

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model state"""
        model_str = str(model.state_dict())
        return hashlib.md5(model_str.encode()).hexdigest()

    def resume_training(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Tuple[int, Dict]:
        """
        Resume training from checkpoint

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: Scheduler to load into
            device: Device to load to

        Returns:
            (start_epoch, checkpoint_dict)
        """
        checkpoint = self.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"\nResuming training from epoch {start_epoch}")

        return start_epoch, checkpoint


if __name__ == "__main__":
    print("Testing CheckpointManager...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir='test_checkpoints',
        max_checkpoints=3,
        model_name='test_model'
    )

    # Test saving checkpoints
    print("\n1. Saving checkpoints...")
    for epoch in range(5):
        metrics = {'loss': 1.0 / (epoch + 1), 'accuracy': epoch * 0.1}
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=(epoch == 3)
        )

    # Test listing checkpoints
    print("\n2. Listing checkpoints...")
    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"  Epoch {cp['epoch']}: {cp['metrics']}")

    # Test loading checkpoint
    print("\n3. Loading best checkpoint...")
    best_cp = manager.get_best_checkpoint()
    print(f"Best checkpoint: Epoch {best_cp['epoch']}")

    loaded_checkpoint = manager.load_checkpoint(model=model, optimizer=optimizer)
    print(f"Loaded epoch: {loaded_checkpoint['epoch']}")

    # Test creating version
    print("\n4. Creating model version...")
    version_dir = manager.create_version(
        model=model,
        version_name='stable',
        description='Stable release after 5 epochs'
    )

    # Test export to ONNX
    print("\n5. Exporting to ONNX...")
    try:
        example_input = torch.randn(1, 10)
        onnx_path = manager.export_to_onnx(
            model=model,
            example_input=example_input,
            export_name='test_model_onnx'
        )
    except Exception as e:
        print(f"ONNX export failed (expected if onnx not installed): {e}")

    # Test export to TorchScript
    print("\n6. Exporting to TorchScript...")
    try:
        example_input = torch.randn(1, 10)
        ts_path = manager.export_to_torchscript(
            model=model,
            example_input=example_input,
            export_name='test_model_torchscript',
            method='trace'
        )
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # Cleanup test directory
    import shutil
    print("\n7. Cleaning up...")
    if Path('test_checkpoints').exists():
        shutil.rmtree('test_checkpoints')
        print("Test directory removed")

    print("\nAll tests completed!")
