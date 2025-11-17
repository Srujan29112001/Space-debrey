"""
Dataset Loaders for Space Debris Tracking System
Supports telescope images, orbital trajectories, and conjunction events
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


@dataclass
class DebrisAnnotation:
    """Annotation for debris detection"""

    bbox: Tuple[float, float, float, float]  # x, y, w, h (normalized)
    class_id: int
    class_name: str
    confidence: float = 1.0


class DebrisImageDataset(Dataset):
    """
    Dataset for telescope images with debris annotations (YOLOv7 format)
    Supports data augmentation and efficient caching
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        img_size: int = 1280,
        augment: bool = True,
        cache_images: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize DebrisImageDataset

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing YOLO format annotations
            img_size: Target image size
            augment: Apply data augmentation
            cache_images: Cache images in memory
            class_names: List of class names
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_size = img_size
        self.augment = augment
        self.cache_images = cache_images

        # Class names
        self.class_names = class_names or [
            "debris",
            "satellite",
            "rocket_body",
            "fragment",
        ]
        self.num_classes = len(self.class_names)

        # Find all images
        self.image_files = sorted(
            list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        )

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        # Cache for images
        self.image_cache = {} if cache_images else None

        # Augmentation pipeline
        self.transform = self._build_transforms()

        print(f"DebrisImageDataset initialized with {len(self.image_files)} images")

    def _build_transforms(self) -> A.Compose:
        """Build augmentation pipeline"""
        if self.augment:
            return A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=5, p=1.0),
                            A.GaussianBlur(blur_limit=5, p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
        else:
            return A.Compose(
                [A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2()],
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item

        Returns:
            Dictionary with 'image', 'bboxes', 'labels'
        """
        # Load image
        img_path = self.image_files[idx]

        if self.image_cache is not None and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.image_cache is not None:
                self.image_cache[idx] = image.copy()

        # Resize to square
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

        # Pad to square
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Load annotations
        annotation_path = self.annotation_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = self._load_annotations(annotation_path)

        # Apply augmentation
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        except Exception as e:
            # Fallback: no augmentation
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            print(f"Augmentation failed for {img_path}: {e}")

        # Convert to tensors
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": class_labels,
            "image_path": str(img_path),
        }

    def _load_annotations(self, annotation_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Load YOLO format annotations

        Returns:
            (bboxes, class_labels)
        """
        bboxes = []
        class_labels = []

        if annotation_path.exists():
            with open(annotation_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)

        return bboxes, class_labels


class OrbitDataset(Dataset):
    """
    Dataset for orbital trajectories (PINN/Transformer/Mamba training)
    Supports efficient loading from HDF5 files
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 100,
        prediction_horizon: int = 50,
        stride: int = 1,
        normalize: bool = True,
    ):
        """
        Initialize OrbitDataset

        Args:
            data_path: Path to HDF5 file with orbit data
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            stride: Stride for sampling sequences
            normalize: Normalize positions and velocities
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.normalize = normalize

        # Load data
        self._load_data()

        print(f"OrbitDataset initialized with {len(self.trajectories)} trajectories")

    def _load_data(self):
        """Load orbit data from HDF5"""
        if not self.data_path.exists():
            # Create synthetic data for demonstration
            print(f"Data file not found, creating synthetic data...")
            self.trajectories = self._create_synthetic_data(n_trajectories=1000)
            self.timestamps = None
        else:
            with h5py.File(self.data_path, "r") as f:
                self.trajectories = f["trajectories"][:]  # [N, T, 6]
                if "timestamps" in f:
                    self.timestamps = f["timestamps"][:]
                else:
                    self.timestamps = None

        # Compute normalization statistics
        if self.normalize:
            self.pos_mean = np.mean(self.trajectories[:, :, :3], axis=(0, 1))
            self.pos_std = np.std(self.trajectories[:, :, :3], axis=(0, 1))
            self.vel_mean = np.mean(self.trajectories[:, :, 3:], axis=(0, 1))
            self.vel_std = np.std(self.trajectories[:, :, 3:], axis=(0, 1))

        # Compute valid indices for sequences
        self.indices = []
        for traj_idx in range(len(self.trajectories)):
            traj_len = self.trajectories.shape[1]
            for start_idx in range(
                0,
                traj_len - self.sequence_length - self.prediction_horizon,
                self.stride,
            ):
                self.indices.append((traj_idx, start_idx))

    def _create_synthetic_data(self, n_trajectories: int = 1000) -> np.ndarray:
        """Create synthetic orbital trajectories"""
        trajectories = []

        for _ in range(n_trajectories):
            # Random Keplerian elements
            a = np.random.uniform(6800, 42000)  # km
            e = np.random.uniform(0, 0.3)
            i = np.random.uniform(0, np.pi)

            # Simple 2-body propagation
            n_points = 500
            trajectory = np.zeros((n_points, 6))

            # Initial state
            r0 = a * (1 - e)
            v0 = np.sqrt(398600.4418 * (2 / r0 - 1 / a))

            trajectory[0, :3] = [r0, 0, 0]
            trajectory[0, 3:] = [0, v0 * np.cos(i), v0 * np.sin(i)]

            # Simple propagation
            dt = 60  # seconds
            for t in range(1, n_points):
                pos = trajectory[t - 1, :3]
                vel = trajectory[t - 1, 3:]

                r = np.linalg.norm(pos)
                acc = -398600.4418 * pos / (r**3)

                trajectory[t, 3:] = vel + acc * dt
                trajectory[t, :3] = pos + vel * dt

            trajectories.append(trajectory)

        return np.array(trajectories)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item

        Returns:
            Dictionary with 'input', 'target', 'timestamps'
        """
        traj_idx, start_idx = self.indices[idx]

        # Extract sequence
        input_seq = self.trajectories[traj_idx, start_idx : start_idx + self.sequence_length].copy()

        target_seq = self.trajectories[
            traj_idx,
            start_idx
            + self.sequence_length : start_idx
            + self.sequence_length
            + self.prediction_horizon,
        ].copy()

        # Normalize
        if self.normalize:
            input_seq[:, :3] = (input_seq[:, :3] - self.pos_mean) / (self.pos_std + 1e-8)
            input_seq[:, 3:] = (input_seq[:, 3:] - self.vel_mean) / (self.vel_std + 1e-8)
            target_seq[:, :3] = (target_seq[:, :3] - self.pos_mean) / (self.pos_std + 1e-8)
            target_seq[:, 3:] = (target_seq[:, 3:] - self.vel_mean) / (self.vel_std + 1e-8)

        # Get timestamps if available
        if self.timestamps is not None:
            input_times = self.timestamps[traj_idx, start_idx : start_idx + self.sequence_length]
            target_times = self.timestamps[
                traj_idx,
                start_idx
                + self.sequence_length : start_idx
                + self.sequence_length
                + self.prediction_horizon,
            ]
        else:
            input_times = np.arange(self.sequence_length) * 60.0
            target_times = np.arange(self.prediction_horizon) * 60.0

        return {
            "input": torch.from_numpy(input_seq).float(),
            "target": torch.from_numpy(target_seq).float(),
            "input_times": torch.from_numpy(input_times).float(),
            "target_times": torch.from_numpy(target_times).float(),
        }


class ConjunctionDataset(Dataset):
    """
    Dataset for conjunction events (risk prediction)
    """

    def __init__(self, data_path: Union[str, Path], lookback: int = 100, time_to_tca: int = 50):
        """
        Initialize ConjunctionDataset

        Args:
            data_path: Path to conjunction data (JSON or HDF5)
            lookback: Number of timesteps to look back
            time_to_tca: Time steps to Time of Closest Approach
        """
        self.data_path = Path(data_path)
        self.lookback = lookback
        self.time_to_tca = time_to_tca

        # Load data
        self._load_data()

        print(f"ConjunctionDataset initialized with {len(self.conjunctions)} events")

    def _load_data(self):
        """Load conjunction data"""
        if not self.data_path.exists():
            # Create synthetic data
            self.conjunctions = self._create_synthetic_conjunctions(n_events=500)
        else:
            if self.data_path.suffix == ".json":
                with open(self.data_path, "r") as f:
                    self.conjunctions = json.load(f)
            else:
                with h5py.File(self.data_path, "r") as f:
                    self.conjunctions = []
                    for i in range(len(f["primary_orbit"])):
                        self.conjunctions.append(
                            {
                                "primary_orbit": f["primary_orbit"][i],
                                "secondary_orbit": f["secondary_orbit"][i],
                                "miss_distance": f["miss_distance"][i],
                                "collision_probability": f["collision_probability"][i],
                            }
                        )

    def _create_synthetic_conjunctions(self, n_events: int = 500) -> List[Dict]:
        """Create synthetic conjunction events"""
        conjunctions = []

        for _ in range(n_events):
            # Create two nearby orbits
            a1 = np.random.uniform(7000, 8000)
            a2 = a1 + np.random.uniform(-100, 100)

            # Propagate both
            traj_len = self.lookback + self.time_to_tca
            primary = np.random.randn(traj_len, 6) * 100 + 7000
            secondary = primary + np.random.randn(traj_len, 6) * 50

            # Miss distance at TCA
            miss_distance = np.random.uniform(0.01, 10.0)  # km

            # Collision probability
            if miss_distance < 1.0:
                prob = np.random.uniform(1e-4, 1e-2)
            else:
                prob = np.random.uniform(1e-8, 1e-4)

            conjunctions.append(
                {
                    "primary_orbit": primary.tolist(),
                    "secondary_orbit": secondary.tolist(),
                    "miss_distance": float(miss_distance),
                    "collision_probability": float(prob),
                }
            )

        return conjunctions

    def __len__(self) -> int:
        return len(self.conjunctions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item

        Returns:
            Dictionary with conjunction data
        """
        event = self.conjunctions[idx]

        primary = np.array(event["primary_orbit"])[: self.lookback]
        secondary = np.array(event["secondary_orbit"])[: self.lookback]

        return {
            "primary": torch.from_numpy(primary).float(),
            "secondary": torch.from_numpy(secondary).float(),
            "miss_distance": torch.tensor(event["miss_distance"], dtype=torch.float32),
            "collision_probability": torch.tensor(
                event["collision_probability"], dtype=torch.float32
            ),
            "risk_label": torch.tensor(
                1 if event["collision_probability"] > 1e-4 else 0, dtype=torch.long
            ),
        }


def collate_fn_detection(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for detection dataset
    Handles variable number of bboxes per image
    """
    images = torch.stack([item["image"] for item in batch])

    # Bboxes and labels have variable length - return as list
    bboxes = [item["bboxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "bboxes": bboxes,
        "labels": labels,
        "image_paths": image_paths,
    }


def create_dataloaders(
    dataset_type: str,
    data_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders

    Args:
        dataset_type: 'detection', 'orbit', or 'conjunction'
        data_path: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers
        train_split: Training split ratio
        val_split: Validation split ratio
        **dataset_kwargs: Additional dataset arguments

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create dataset
    if dataset_type == "detection":
        dataset = DebrisImageDataset(
            image_dir=Path(data_path) / "images",
            annotation_dir=Path(data_path) / "annotations",
            **dataset_kwargs,
        )
        collate_fn = collate_fn_detection
    elif dataset_type == "orbit":
        dataset = OrbitDataset(data_path, **dataset_kwargs)
        collate_fn = None
    elif dataset_type == "conjunction":
        dataset = ConjunctionDataset(data_path, **dataset_kwargs)
        collate_fn = None
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing DebrisImageDataset...")

    # Create dummy data directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        ann_dir = tmpdir / "annotations"
        img_dir.mkdir()
        ann_dir.mkdir()

        # Create dummy image and annotation
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        with open(ann_dir / "test.txt", "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            f.write("1 0.3 0.7 0.05 0.05\n")

        # Test dataset
        try:
            dataset = DebrisImageDataset(
                image_dir=img_dir, annotation_dir=ann_dir, img_size=640, augment=True
            )

            item = dataset[0]
            print(f"Image shape: {item['image'].shape}")
            print(f"Bboxes: {item['bboxes']}")
            print(f"Labels: {item['labels']}")
            print("DebrisImageDataset test passed!")
        except Exception as e:
            print(f"DebrisImageDataset test failed: {e}")

    print("\nTesting OrbitDataset...")
    try:
        orbit_dataset = OrbitDataset(
            data_path=Path("/tmp/dummy_orbits.h5"),  # Will create synthetic
            sequence_length=100,
            prediction_horizon=50,
        )

        item = orbit_dataset[0]
        print(f"Input shape: {item['input'].shape}")
        print(f"Target shape: {item['target'].shape}")
        print("OrbitDataset test passed!")
    except Exception as e:
        print(f"OrbitDataset test failed: {e}")

    print("\nTesting ConjunctionDataset...")
    try:
        conj_dataset = ConjunctionDataset(
            data_path=Path("/tmp/dummy_conjunctions.json"),  # Will create synthetic
            lookback=100,
        )

        item = conj_dataset[0]
        print(f"Primary shape: {item['primary'].shape}")
        print(f"Secondary shape: {item['secondary'].shape}")
        print(f"Collision probability: {item['collision_probability']}")
        print("ConjunctionDataset test passed!")
    except Exception as e:
        print(f"ConjunctionDataset test failed: {e}")
