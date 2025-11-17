"""
Test Utilities
Mock data generators, helpers, and test utilities
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ============================================================================
# TLE GENERATORS
# ============================================================================


def generate_tle(norad_id: int = 25544, name: str = "TEST SAT") -> Tuple[str, str, str]:
    """
    Generate a valid TLE

    Args:
        norad_id: NORAD catalog ID
        name: Satellite name

    Returns:
        Tuple of (name, line1, line2)
    """
    # Generate epoch
    epoch_year = 23
    epoch_day = 1.0

    # Line 1
    classification = "U"
    intl_designator = "98067A  "
    mean_motion_dot = " .00016717"
    mean_motion_ddot = " 00000-0"
    bstar = " 10270-3"
    ephemeris_type = "0"
    element_set = "900"

    line1_data = (
        f"1 {norad_id:5d}{classification} {intl_designator} "
        f"{epoch_year:02d}{epoch_day:012.8f}{mean_motion_dot} "
        f"{mean_motion_ddot} {bstar} {ephemeris_type} {element_set:>4s}"
    )

    # Calculate checksum for line 1
    checksum1 = calculate_tle_checksum(line1_data)
    line1 = f"{line1_data}{checksum1}"

    # Line 2
    inclination = "51.6416"
    raan = "247.4627"
    eccentricity = "0006703"
    arg_perigee = "130.5360"
    mean_anomaly = "325.0288"
    mean_motion = "15.72834465"
    rev_number = "00000"

    line2_data = (
        f"2 {norad_id:5d} {inclination:>8s} {raan:>8s} {eccentricity} "
        f"{arg_perigee:>8s} {mean_anomaly:>8s} {mean_motion:>11s}{rev_number:>5s}"
    )

    checksum2 = calculate_tle_checksum(line2_data)
    line2 = f"{line2_data}{checksum2}"

    return name, line1, line2


def calculate_tle_checksum(line: str) -> int:
    """Calculate TLE checksum"""
    total = 0
    for char in line:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def generate_tle_batch(count: int = 10) -> List[Tuple[str, str, str]]:
    """Generate batch of TLEs"""
    tles = []
    for i in range(count):
        norad_id = 25544 + i
        name = f"TEST SAT {i+1}"
        tles.append(generate_tle(norad_id, name))
    return tles


# ============================================================================
# IMAGE GENERATORS
# ============================================================================


def generate_space_image(
    width: int = 1024,
    height: int = 1024,
    n_stars: int = 100,
    n_debris: int = 5,
    add_noise: bool = True,
) -> np.ndarray:
    """
    Generate synthetic space telescope image

    Args:
        width: Image width
        height: Image height
        n_stars: Number of stars
        n_debris: Number of debris objects
        add_noise: Whether to add background noise

    Returns:
        Image array (H, W, 3)
    """
    # Create black image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add background noise
    if add_noise:
        noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add stars
    for _ in range(n_stars):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(180, 255)
        radius = np.random.randint(1, 3)
        cv2.circle(image, (x, y), radius, (brightness, brightness, brightness), -1)

    # Add debris (as small streaks or blobs)
    debris_positions = []
    for _ in range(n_debris):
        x = np.random.randint(10, width - 10)
        y = np.random.randint(10, height - 10)
        brightness = np.random.randint(150, 255)

        # Random shape: blob or streak
        if np.random.random() > 0.5:
            # Blob
            radius = np.random.randint(2, 5)
            cv2.circle(image, (x, y), radius, (brightness, brightness, brightness), -1)
            debris_positions.append((x - radius, y - radius, x + radius, y + radius))
        else:
            # Streak
            length = np.random.randint(5, 20)
            angle = np.random.random() * 2 * np.pi
            x2 = int(x + length * np.cos(angle))
            y2 = int(y + length * np.sin(angle))
            cv2.line(image, (x, y), (x2, y2), (brightness, brightness, brightness), 2)
            debris_positions.append((min(x, x2), min(y, y2), max(x, x2), max(y, y2)))

    return image, debris_positions


def generate_image_sequence(
    n_frames: int = 10, width: int = 1024, height: int = 1024, moving_objects: int = 3
) -> List[np.ndarray]:
    """
    Generate sequence of images with moving debris

    Args:
        n_frames: Number of frames
        width: Image width
        height: Image height
        moving_objects: Number of moving objects

    Returns:
        List of image arrays
    """
    frames = []

    # Initialize object positions and velocities
    objects = []
    for _ in range(moving_objects):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        vx = np.random.randint(-5, 5)
        vy = np.random.randint(-5, 5)
        brightness = np.random.randint(150, 255)
        objects.append({"x": x, "y": y, "vx": vx, "vy": vy, "brightness": brightness})

    # Generate frames
    for _ in range(n_frames):
        # Base image with stars
        image, _ = generate_space_image(width, height, n_stars=100, n_debris=0)

        # Add moving objects
        for obj in objects:
            cv2.circle(
                image,
                (int(obj["x"]), int(obj["y"])),
                3,
                (obj["brightness"], obj["brightness"], obj["brightness"]),
                -1,
            )

            # Update position
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # Bounce off edges
            if obj["x"] < 0 or obj["x"] >= width:
                obj["vx"] *= -1
            if obj["y"] < 0 or obj["y"] >= height:
                obj["vy"] *= -1

        frames.append(image)

    return frames


# ============================================================================
# STATE VECTOR GENERATORS
# ============================================================================


def generate_orbital_state(
    altitude: float = 400.0, inclination: float = 51.6, eccentricity: float = 0.0
) -> np.ndarray:
    """
    Generate orbital state vector

    Args:
        altitude: Altitude in km
        inclination: Inclination in degrees
        eccentricity: Orbital eccentricity

    Returns:
        State vector [x, y, z, vx, vy, vz]
    """
    Re = 6378.137  # Earth radius km
    r = Re + altitude

    # Circular orbit velocity
    mu = 398600.4418  # km^3/s^2
    v = np.sqrt(mu / r)

    # Convert inclination to radians
    inc_rad = np.radians(inclination)

    # Initial position (at ascending node)
    x = r
    y = 0.0
    z = 0.0

    # Initial velocity (perpendicular to position)
    vx = 0.0
    vy = v * np.cos(inc_rad)
    vz = v * np.sin(inc_rad)

    return np.array([x, y, z, vx, vy, vz])


def propagate_orbit_simple(state: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """
    Simple orbit propagation (two-body problem)

    Args:
        state: Initial state [x, y, z, vx, vy, vz]
        dt: Time step (seconds)
        n_steps: Number of steps

    Returns:
        Trajectory array (n_steps, 6)
    """
    mu = 398600.4418  # km^3/s^2
    trajectory = np.zeros((n_steps, 6))
    trajectory[0] = state

    for i in range(1, n_steps):
        # Current state
        r = trajectory[i - 1, :3]
        v = trajectory[i - 1, 3:6]

        # Acceleration (two-body)
        r_mag = np.linalg.norm(r)
        a = -mu * r / r_mag**3

        # Simple Euler integration
        v_new = v + a * dt
        r_new = r + v_new * dt

        trajectory[i] = np.concatenate([r_new, v_new])

    return trajectory


# ============================================================================
# DETECTION GENERATORS
# ============================================================================


def generate_detections(
    n_detections: int = 10, img_width: int = 1024, img_height: int = 1024
) -> List[Dict]:
    """
    Generate random detections

    Args:
        n_detections: Number of detections
        img_width: Image width
        img_height: Image height

    Returns:
        List of detection dictionaries
    """
    detections = []

    for i in range(n_detections):
        # Random bbox
        x1 = np.random.randint(0, img_width - 50)
        y1 = np.random.randint(0, img_height - 50)
        w = np.random.randint(10, 50)
        h = np.random.randint(10, 50)
        x2 = min(x1 + w, img_width)
        y2 = min(y1 + h, img_height)

        detection = {
            "bbox": (x1, y1, x2, y2),
            "confidence": np.random.uniform(0.5, 1.0),
            "class_id": np.random.randint(0, 3),
            "class_name": np.random.choice(["debris", "satellite", "unknown"]),
            "features": np.random.randn(128).astype(np.float32),
        }

        detections.append(detection)

    return detections


# ============================================================================
# KAFKA MESSAGE GENERATORS
# ============================================================================


def generate_kafka_message(message_type: str = "tle") -> Dict:
    """
    Generate Kafka message

    Args:
        message_type: Type of message ('tle', 'detection', 'alert')

    Returns:
        Message dictionary
    """
    if message_type == "tle":
        name, line1, line2 = generate_tle()
        return {
            "type": "tle",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"name": name, "line1": line1, "line2": line2},
        }

    elif message_type == "detection":
        return {
            "type": "detection",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "object_id": f"DEBRIS_{np.random.randint(10000, 99999)}",
                "position": np.random.randn(3).tolist(),
                "velocity": np.random.randn(3).tolist(),
                "confidence": np.random.uniform(0.5, 1.0),
            },
        }

    elif message_type == "alert":
        return {
            "type": "alert",
            "timestamp": datetime.utcnow().isoformat(),
            "severity": np.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "data": {
                "primary_object": 25544,
                "secondary_object": f"DEBRIS_{np.random.randint(10000, 99999)}",
                "collision_probability": np.random.uniform(0.0001, 0.01),
                "tca": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            },
        }

    return {}


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


def assert_valid_state_vector(state: np.ndarray, tolerance: float = 1e-6):
    """Assert state vector is valid"""
    assert state.shape == (6,), f"Invalid state shape: {state.shape}"
    assert not np.any(np.isnan(state)), "State contains NaN"
    assert not np.any(np.isinf(state)), "State contains Inf"

    # Check position magnitude (should be > Earth radius)
    r_mag = np.linalg.norm(state[:3])
    assert r_mag > 6378.0, f"Position magnitude too small: {r_mag} km"

    # Check velocity magnitude (should be reasonable for orbit)
    v_mag = np.linalg.norm(state[3:6])
    assert 0.5 < v_mag < 15.0, f"Velocity magnitude unreasonable: {v_mag} km/s"


def assert_valid_trajectory(trajectory: np.ndarray):
    """Assert trajectory is valid"""
    assert trajectory.ndim == 2, f"Invalid trajectory dimensions: {trajectory.ndim}"
    assert trajectory.shape[1] == 6, f"Invalid trajectory shape: {trajectory.shape}"

    # Check each state
    for state in trajectory:
        assert_valid_state_vector(state)


def assert_valid_detection(detection: Dict):
    """Assert detection is valid"""
    assert "bbox" in detection, "Detection missing bbox"
    assert "confidence" in detection, "Detection missing confidence"
    assert "class_id" in detection, "Detection missing class_id"

    # Check bbox format
    bbox = detection["bbox"]
    assert len(bbox) == 4, f"Invalid bbox length: {len(bbox)}"
    x1, y1, x2, y2 = bbox
    assert x2 > x1, "Invalid bbox: x2 <= x1"
    assert y2 > y1, "Invalid bbox: y2 <= y1"

    # Check confidence
    conf = detection["confidence"]
    assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"


def assert_arrays_close(
    arr1: np.ndarray, arr2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
):
    """Assert two arrays are close"""
    assert arr1.shape == arr2.shape, f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
    np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)


# ============================================================================
# TIMING UTILITIES
# ============================================================================


class Timer:
    """Simple timer context manager"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()

    def __str__(self):
        return f"{self.elapsed:.4f}s"
