"""
Computer Vision Pipeline for Space Debris Detection
====================================================

Modules:
- detection: YOLOv7 and DINO v2 based debris detection
- tracking: DeepSORT multi-object tracking
- characterization: 3D Gaussian Splatting for shape/tumble analysis
- preprocessing: Space-specific image preprocessing
"""

from .detector import SpaceDebrisDetector
from .tracking.deepsort import DeepSORTTracker
from .preprocessing.space_image import SpaceImagePreprocessor

__all__ = [
    "SpaceDebrisDetector",
    "DeepSORTTracker",
    "SpaceImagePreprocessor",
]
