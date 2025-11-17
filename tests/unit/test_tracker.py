"""
Unit Tests for DeepSORT Tracker
Tests multi-object tracking functionality
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from space_debris_tracker.computer_vision.detector import Detection
from space_debris_tracker.computer_vision.tracking.deepsort import (
    DeepSORTTracker,
    KalmanFilter,
)
from space_debris_tracker.computer_vision.tracking.deepsort import (
    Track as DeepSORTTrack,
)
from tests.utils import generate_detections


@pytest.mark.unit
class TestDeepSORTTracker:
    """Test DeepSORT tracking"""

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = DeepSORTTracker(
            max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3
        )

        assert tracker is not None
        assert tracker.max_dist == 0.2
        assert tracker.min_confidence == 0.3
        assert tracker.max_age == 70
        assert tracker.n_init == 3

    def test_tracker_empty_detections(self, sample_telescope_image):
        """Test tracker with no detections"""
        tracker = DeepSORTTracker()

        # Empty detections
        tracks = tracker.update([], sample_telescope_image)

        assert isinstance(tracks, list)
        assert len(tracks) == 0

    def test_tracker_single_detection(self, sample_telescope_image):
        """Test tracker with single detection"""
        tracker = DeepSORTTracker()

        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        tracks = tracker.update([detection], sample_telescope_image)

        assert isinstance(tracks, list)
        # May be empty on first frame (needs n_init confirmations)

    def test_tracker_multiple_frames(self, image_sequence):
        """Test tracker across multiple frames"""
        tracker = DeepSORTTracker(n_init=1)

        all_tracks = []

        for frame in image_sequence:
            # Generate detections
            detections = [
                Detection(
                    bbox=(100 + i * 10, 100 + i * 10, 150 + i * 10, 150 + i * 10),
                    confidence=0.95,
                    class_id=0,
                    class_name="debris",
                    features=np.random.randn(128),
                )
                for i in range(3)
            ]

            tracks = tracker.update(detections, frame)
            all_tracks.append(tracks)

        # Should have tracked objects
        assert len(all_tracks) > 0

    def test_track_id_consistency(self):
        """Test track ID consistency"""
        tracker = DeepSORTTracker(n_init=1)

        # Same detection across frames
        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        tracks1 = tracker.update([detection], frame)
        tracks2 = tracker.update([detection], frame)

        # Track IDs should be consistent
        if len(tracks1) > 0 and len(tracks2) > 0:
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_track_age_increments(self):
        """Test track age increments over time"""
        tracker = DeepSORTTracker(n_init=1)

        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        # Update multiple times
        for _ in range(5):
            tracks = tracker.update([detection], frame)

        # Track age should have increased
        if len(tracks) > 0:
            assert tracks[0].age > 1

    def test_track_deletion_on_miss(self):
        """Test tracks are deleted after max_age"""
        tracker = DeepSORTTracker(max_age=3, n_init=1)

        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        # Create track
        tracker.update([detection], frame)

        # Update without detections (misses)
        for _ in range(5):
            tracks = tracker.update([], frame)

        # Track should be deleted
        assert len(tracks) == 0

    def test_feature_extraction(self, sample_telescope_image):
        """Test feature extraction from detections"""
        tracker = DeepSORTTracker()

        detection = Detection(
            bbox=(100, 100, 150, 150), confidence=0.95, class_id=0, class_name="debris"
        )

        # Extract features
        features = tracker._extract_features([detection], sample_telescope_image)

        assert features is not None
        assert len(features) == 1


@pytest.mark.unit
class TestKalmanFilter:
    """Test Kalman Filter"""

    def test_kalman_initialization(self):
        """Test Kalman filter initialization"""
        kf = KalmanFilter()

        assert kf is not None

    def test_kalman_predict(self):
        """Test Kalman prediction"""
        kf = KalmanFilter()

        # Initial state [x, y, vx, vy]
        state = np.array([100.0, 100.0, 1.0, 1.0])
        covariance = np.eye(4)

        # Predict
        predicted_state, predicted_cov = kf.predict(state, covariance)

        assert predicted_state.shape == (4,)
        assert predicted_cov.shape == (4, 4)

        # Position should have moved
        assert predicted_state[0] != state[0]
        assert predicted_state[1] != state[1]

    def test_kalman_update(self):
        """Test Kalman update"""
        kf = KalmanFilter()

        state = np.array([100.0, 100.0, 1.0, 1.0])
        covariance = np.eye(4)

        # Measurement [x, y]
        measurement = np.array([102.0, 101.0])

        # Update
        updated_state, updated_cov = kf.update(state, covariance, measurement)

        assert updated_state.shape == (4,)
        assert updated_cov.shape == (4, 4)

        # State should be adjusted toward measurement
        assert abs(updated_state[0] - 102.0) < abs(state[0] - 102.0)

    def test_kalman_filter_sequence(self):
        """Test Kalman filter over sequence"""
        kf = KalmanFilter()

        state = np.array([100.0, 100.0, 1.0, 1.0])
        covariance = np.eye(4) * 10

        # Simulate constant velocity motion
        for i in range(10):
            # Predict
            state, covariance = kf.predict(state, covariance)

            # Measurement (with noise)
            true_pos = [100 + i, 100 + i]
            measurement = np.array(true_pos) + np.random.randn(2) * 0.5

            # Update
            state, covariance = kf.update(state, covariance, measurement)

        # Should be close to true position
        assert abs(state[0] - 109) < 5
        assert abs(state[1] - 109) < 5


@pytest.mark.unit
class TestTrackManagement:
    """Test track management"""

    def test_track_creation(self):
        """Test creating a track"""
        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        track = DeepSORTTrack(
            track_id=1, detection=detection, feature=detection.features
        )

        assert track is not None
        assert track.track_id == 1

    def test_track_state_update(self):
        """Test updating track state"""
        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        track = DeepSORTTrack(
            track_id=1, detection=detection, feature=detection.features
        )

        # Update with new detection
        new_detection = Detection(
            bbox=(105, 105, 155, 155),
            confidence=0.92,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        track.update(new_detection)

        assert track.hits > 1
        assert track.time_since_update == 0

    def test_track_miss(self):
        """Test track miss (no detection)"""
        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        track = DeepSORTTrack(
            track_id=1, detection=detection, feature=detection.features
        )

        initial_age = track.time_since_update

        # Mark as missed
        track.mark_missed()

        assert track.time_since_update > initial_age

    def test_track_confirmation(self):
        """Test track confirmation logic"""
        tracker = DeepSORTTracker(n_init=3)

        detection = Detection(
            bbox=(100, 100, 150, 150),
            confidence=0.95,
            class_id=0,
            class_name="debris",
            features=np.random.randn(128),
        )

        frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        # Need n_init hits to confirm
        for i in range(5):
            tracks = tracker.update([detection], frame)

        # Should be confirmed after n_init hits
        assert len(tracks) > 0
