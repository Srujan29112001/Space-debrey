"""
DeepSORT Multi-Object Tracking
Implements Kalman filtering and Hungarian algorithm for track assignment
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


@dataclass
class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes"""

    bbox: Tuple[int, int, int, int]
    track_id: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)

    kf: Optional[KalmanFilter] = field(default=None, init=False)
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize Kalman filter"""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # s (scale)
            [0, 0, 0, 1, 0, 0, 0],  # r (aspect ratio)
            [0, 0, 0, 0, 1, 0, 0],  # dx
            [0, 0, 0, 0, 0, 1, 0],  # dy
            [0, 0, 0, 0, 0, 0, 1],  # ds
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(self.bbox)

    def _bbox_to_z(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Convert bounding box to measurement space

        Args:
            bbox: (x1, y1, x2, y2)

        Returns:
            [x_center, y_center, scale, aspect_ratio]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h  # scale (area)
        r = w / float(h) if h > 0 else 1.0  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))

    def _z_to_bbox(self, z: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Convert measurement to bounding box

        Args:
            z: [x_center, y_center, scale, aspect_ratio]

        Returns:
            (x1, y1, x2, y2)
        """
        x, y, s, r = z.flatten()
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        x1 = int(x - w / 2.0)
        y1 = int(y - h / 2.0)
        x2 = int(x + w / 2.0)
        y2 = int(y + h / 2.0)
        return (x1, y1, x2, y2)

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next state

        Returns:
            Predicted bounding box
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = self._z_to_bbox(self.kf.x)
        return self.bbox

    def update(self, bbox: Tuple[int, int, int, int]):
        """
        Update with new detection

        Args:
            bbox: New detection bounding box
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))
        self.bbox = bbox

        # Update velocity
        if len(self.kf.x) >= 7:
            self.velocity = (float(self.kf.x[4]), float(self.kf.x[5]))

    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current state"""
        return self._z_to_bbox(self.kf.x)


class DeepSORTTracker:
    """
    DeepSORT tracker for multi-object tracking
    """

    def __init__(self,
                 max_dist: float = 0.2,
                 min_confidence: float = 0.3,
                 max_iou_distance: float = 0.7,
                 max_age: int = 70,
                 n_init: int = 3,
                 nn_budget: int = 100):
        """
        Initialize DeepSORT tracker

        Args:
            max_dist: Maximum distance for matching
            min_confidence: Minimum detection confidence
            max_iou_distance: Maximum IoU distance for matching
            max_age: Maximum frames to keep alive without detection
            n_init: Number of frames for track initialization
            nn_budget: Maximum size of feature gallery
        """
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget

        self.tracks: List[KalmanBoxTracker] = []
        self.next_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[KalmanBoxTracker]:
        """
        Update tracker with new detections

        Args:
            detections: List of Detection objects
            frame: Optional frame for feature extraction

        Returns:
            List of active tracks
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        # Filter detections by confidence
        if hasattr(detections[0] if detections else None, 'confidence'):
            detections = [d for d in detections if d.confidence >= self.min_confidence]

        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # Update matched tracks
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx].bbox)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.n_init]

    def _match(self, detections: List) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks

        Args:
            detections: List of detections

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Compute cost matrix (IoU distance)
        cost_matrix = np.zeros((len(detections), len(self.tracks)))

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                cost_matrix[d, t] = self._iou_distance(det.bbox, trk.bbox)

        # Hungarian algorithm for assignment
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by distance threshold
        matched = []
        unmatched_dets = []
        unmatched_trks = list(range(len(self.tracks)))

        for d, t in zip(det_indices, trk_indices):
            if cost_matrix[d, t] > self.max_iou_distance:
                unmatched_dets.append(d)
            else:
                matched.append((d, t))
                unmatched_trks.remove(t)

        # Add unmatched detections
        for d in range(len(detections)):
            if d not in det_indices:
                unmatched_dets.append(d)

        return matched, unmatched_dets, unmatched_trks

    def _iou_distance(self, bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute IoU distance between two bounding boxes

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            IoU distance (1 - IoU)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 1.0  # No intersection

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        # IoU
        iou = intersection / union if union > 0 else 0.0

        return 1.0 - iou

    def _initiate_track(self, detection):
        """
        Initialize a new track

        Args:
            detection: Detection object
        """
        track = KalmanBoxTracker(
            bbox=detection.bbox,
            track_id=self.next_id
        )
        self.tracks.append(track)
        self.next_id += 1


if __name__ == "__main__":
    # Example usage
    tracker = DeepSORTTracker()
    print("DeepSORT tracker initialized")
    print(f"Max age: {tracker.max_age}")
    print(f"Min confidence: {tracker.min_confidence}")
