"""
Unit Tests for Space Debris Detector
Tests YOLOv7, DINO v2, and Gaussian Splatting components
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from space_debris_tracker.computer_vision.detector import (
    SpaceDebrisDetector,
    YOLOv7Detector,
    DINOv2Detector,
    Detection,
    Track
)
from tests.utils import generate_space_image


@pytest.mark.unit
class TestYOLOv7Detector:
    """Test YOLO detector"""

    def test_yolo_initialization(self, device):
        """Test YOLO detector initialization"""
        detector = YOLOv7Detector(device=device)

        assert detector is not None
        assert detector.device == device
        assert detector.conf_thres == 0.25
        assert detector.iou_thres == 0.45

    def test_yolo_custom_thresholds(self):
        """Test YOLO with custom thresholds"""
        detector = YOLOv7Detector(
            conf_thres=0.5,
            iou_thres=0.6,
            device='cpu'
        )

        assert detector.conf_thres == 0.5
        assert detector.iou_thres == 0.6

    def test_yolo_preprocessing(self, sample_telescope_image):
        """Test image preprocessing"""
        detector = YOLOv7Detector(device='cpu')

        img_tensor = detector._preprocess(sample_telescope_image)

        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.ndim == 4  # Batch dimension
        assert img_tensor.shape[1] == 3  # RGB channels
        assert img_tensor.shape[2] == detector.img_size
        assert img_tensor.shape[3] == detector.img_size

    def test_yolo_forward_pass(self):
        """Test YOLO forward pass"""
        detector = YOLOv7Detector(device='cpu')

        # Random input
        x = torch.randn(1, 3, 1280, 1280)
        output = detector(x)

        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_yolo_detect_returns_list(self, sample_telescope_image):
        """Test detection returns list"""
        detector = YOLOv7Detector(device='cpu')

        detections = detector.detect(sample_telescope_image)

        assert isinstance(detections, list)

    @pytest.mark.gpu
    def test_yolo_gpu_inference(self, sample_telescope_image):
        """Test YOLO GPU inference"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        detector = YOLOv7Detector(device='cuda')
        detections = detector.detect(sample_telescope_image)

        assert isinstance(detections, list)


@pytest.mark.unit
class TestDINOv2Detector:
    """Test DINO v2 detector"""

    def test_dino_initialization(self, device):
        """Test DINO initialization"""
        with patch('torch.hub.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            detector = DINOv2Detector(device=device)

            assert detector is not None
            assert detector.device == device

    def test_dino_feature_extraction(self, sample_telescope_image):
        """Test DINO feature extraction"""
        detector = DINOv2Detector(device='cpu')

        # Mock the model
        detector.model = Mock()
        detector.model.return_value = torch.randn(1, 768)

        features = detector.extract_features(sample_telescope_image)

        assert features is not None
        assert isinstance(features, torch.Tensor)

    def test_dino_no_model_fallback(self, sample_telescope_image):
        """Test DINO fallback when model unavailable"""
        detector = DINOv2Detector(device='cpu')
        detector.model = None

        features = detector.extract_features(sample_telescope_image)

        # Should return zero features
        assert features.shape == (768,)
        assert torch.all(features == 0)


@pytest.mark.unit
class TestSpaceDebrisDetector:
    """Test main debris detector system"""

    def test_detector_initialization(self, device):
        """Test detector initialization"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device=device)

            assert detector is not None
            assert detector.device == device
            assert detector.yolo is not None
            assert detector.dino is not None
            assert detector.tracker is not None
            assert detector.gaussian_splatter is not None

    def test_process_single_image(self, sample_telescope_image):
        """Test processing single telescope image"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            # Mock detections
            detector.yolo.detect = Mock(return_value=[])
            detector.dino.detect_novel_objects = Mock(return_value=[])

            results = detector.process_telescope_image(sample_telescope_image)

            assert isinstance(results, list)

    def test_process_with_previous_frames(self, image_sequence):
        """Test processing with frame history"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            # Mock detections
            detector.yolo.detect = Mock(return_value=[])
            detector.dino.detect_novel_objects = Mock(return_value=[])

            current_frame = image_sequence[-1]
            previous_frames = image_sequence[:-1]

            results = detector.process_telescope_image(
                current_frame,
                previous_frames=previous_frames
            )

            assert isinstance(results, list)

    def test_merge_detections(self):
        """Test merging detections from multiple sources"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            yolo_dets = [
                Detection((100, 100, 120, 120), 0.9, 0, 'debris'),
                Detection((200, 200, 220, 220), 0.85, 0, 'debris')
            ]

            dino_dets = [
                Detection((300, 300, 320, 320), 0.8, 1, 'unknown')
            ]

            merged = detector._merge_detections(yolo_dets, dino_dets)

            assert len(merged) == 3

    def test_detection_to_track_conversion(self):
        """Test converting detection to track"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            detection = Detection((100, 100, 120, 120), 0.9, 0, 'debris')
            track = detector._detection_to_track(detection, track_id=1)

            assert isinstance(track, Track)
            assert track.track_id == 1
            assert track.bbox == detection.bbox
            assert track.hits == 1
            assert track.age == 1

    def test_characterize_object(self, sample_telescope_image):
        """Test 3D object characterization"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            track = Track(
                track_id=1,
                bbox=(100, 100, 150, 150),
                velocity=(1.0, 2.0),
                hits=10,
                age=10,
                class_name='debris',
                intensities=[0.9, 0.85, 0.88]
            )

            char_obj = detector._characterize_object(
                track,
                sample_telescope_image,
                previous_frames=None
            )

            assert 'track_id' in char_obj
            assert 'shape' in char_obj
            assert 'tumble_rate' in char_obj
            assert 'size_estimate' in char_obj
            assert 'reflectivity' in char_obj

    def test_estimate_size(self):
        """Test object size estimation"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            bbox = (100, 100, 150, 180)
            shape_params = {}

            size = detector._estimate_size(shape_params, bbox)

            assert size > 0
            assert isinstance(size, float)

    def test_estimate_reflectivity(self):
        """Test reflectivity estimation"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            intensities = [0.9, 0.85, 0.88, 0.92]
            reflectivity = detector._estimate_reflectivity(intensities)

            assert 0.0 <= reflectivity <= 1.0
            assert reflectivity == np.mean(intensities)

    def test_empty_intensities_reflectivity(self):
        """Test reflectivity with empty intensities"""
        with patch('torch.hub.load'):
            detector = SpaceDebrisDetector(device='cpu')

            reflectivity = detector._estimate_reflectivity([])
            assert reflectivity == 0.0


@pytest.mark.unit
class TestDetectionDataclass:
    """Test Detection dataclass"""

    def test_detection_creation(self):
        """Test creating Detection"""
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            class_id=0,
            class_name='debris'
        )

        assert det.bbox == (100, 100, 200, 200)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == 'debris'
        assert det.features is None

    def test_detection_with_features(self):
        """Test Detection with features"""
        features = np.random.randn(128)
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            class_id=0,
            class_name='debris',
            features=features
        )

        assert det.features is not None
        assert det.features.shape == (128,)


@pytest.mark.unit
class TestTrackDataclass:
    """Test Track dataclass"""

    def test_track_creation(self):
        """Test creating Track"""
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            velocity=(1.5, 2.0),
            hits=5,
            age=10,
            class_name='debris',
            intensities=[0.9, 0.85]
        )

        assert track.track_id == 1
        assert track.bbox == (100, 100, 200, 200)
        assert track.velocity == (1.5, 2.0)
        assert track.hits == 5
        assert track.age == 10
        assert len(track.intensities) == 2
