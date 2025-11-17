"""
Pytest Configuration and Fixtures
Provides reusable test fixtures for the Space Debris Tracking System
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Generator
from pathlib import Path
import tempfile
import shutil

# Mock imports for components that might not be available
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# SESSION-LEVEL FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create temporary directory for test data"""
    temp_dir = Path(tempfile.mkdtemp(prefix="space_debris_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def device() -> str:
    """Get device for testing (CPU or CUDA if available)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# TLE FIXTURES
# ============================================================================

@pytest.fixture
def sample_tle_lines() -> tuple:
    """Sample TLE for ISS"""
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9005"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009"
    return name, line1, line2


@pytest.fixture
def sample_tle_file(test_data_dir, sample_tle_lines) -> Path:
    """Create sample TLE file"""
    name, line1, line2 = sample_tle_lines

    tle_file = test_data_dir / "sample_tle.txt"
    with open(tle_file, 'w') as f:
        f.write(f"{name}\n{line1}\n{line2}\n")

        # Add a few more satellites
        f.write("NOAA 15\n")
        f.write("1 25338U 98030A   23001.00000000  .00000050  00000-0  44814-4 0  9991\n")
        f.write("2 25338  98.7139 332.8019 0010571  72.5356 287.7187 14.26017399000000\n")

        f.write("HUBBLE SPACE TELESCOPE\n")
        f.write("1 20580U 90037B   23001.00000000  .00001363  00000-0  73063-4 0  9999\n")
        f.write("2 20580  28.4714 261.5419 0002713 330.7671  29.3086 15.09726543000003\n")

    return tle_file


@pytest.fixture
def orbital_elements() -> Dict:
    """Sample orbital elements"""
    return {
        'norad_id': 25544,
        'name': 'ISS (ZARYA)',
        'epoch': datetime(2023, 1, 1),
        'mean_motion': 15.72834465,
        'eccentricity': 0.0006703,
        'inclination': 51.6416,
        'raan': 247.4627,
        'arg_perigee': 130.5360,
        'mean_anomaly': 325.0288,
        'bstar': 0.00010270,
        'classification': 'U',
        'element_set_number': 900,
        'revolution_number': 0,
        'semi_major_axis': 6778.0,
        'period': 91.5,
        'apogee': 418.0,
        'perigee': 408.0
    }


# ============================================================================
# IMAGE FIXTURES
# ============================================================================

@pytest.fixture
def sample_telescope_image() -> np.ndarray:
    """Generate sample telescope image with stars and debris"""
    # Create 1024x1024 image
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Add background noise
    noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
    image = image + noise

    # Add stars (bright points)
    n_stars = 100
    for _ in range(n_stars):
        x = np.random.randint(0, 1024)
        y = np.random.randint(0, 1024)
        brightness = np.random.randint(200, 255)
        image[y:y+2, x:x+2] = brightness

    # Add debris objects (small streaks)
    n_debris = 5
    for _ in range(n_debris):
        x = np.random.randint(0, 1000)
        y = np.random.randint(0, 1000)
        length = np.random.randint(5, 20)
        brightness = np.random.randint(150, 255)
        image[y:y+2, x:x+length] = brightness

    return image


@pytest.fixture
def image_sequence(sample_telescope_image) -> List[np.ndarray]:
    """Generate sequence of telescope images"""
    images = []
    base_image = sample_telescope_image.copy()

    for i in range(10):
        # Slightly modify each frame
        frame = base_image.copy()
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(frame)

    return images


# ============================================================================
# STATE VECTOR FIXTURES
# ============================================================================

@pytest.fixture
def sample_state_vector() -> np.ndarray:
    """Sample state vector [x, y, z, vx, vy, vz]"""
    # ISS-like orbit
    return np.array([
        6778.0, 0.0, 0.0,  # Position (km)
        0.0, 7.66, 0.0     # Velocity (km/s)
    ])


@pytest.fixture
def sample_trajectory(sample_state_vector) -> Dict:
    """Sample trajectory data"""
    n_steps = 100
    time = np.linspace(0, 5400, n_steps)  # 90 minutes

    positions = []
    velocities = []

    # Simple circular orbit
    r = 6778.0
    v = 7.66

    for t in time:
        theta = (v / r) * t
        positions.append([
            r * np.cos(theta),
            r * np.sin(theta),
            0.0
        ])
        velocities.append([
            -v * np.sin(theta),
            v * np.cos(theta),
            0.0
        ])

    return {
        'time': time,
        'position': np.array(positions),
        'velocity': np.array(velocities)
    }


# ============================================================================
# DETECTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_detections() -> List[Dict]:
    """Sample debris detections"""
    return [
        {
            'bbox': (100, 100, 120, 120),
            'confidence': 0.95,
            'class_id': 0,
            'class_name': 'debris',
            'features': np.random.randn(128)
        },
        {
            'bbox': (300, 400, 330, 430),
            'confidence': 0.87,
            'class_id': 0,
            'class_name': 'debris',
            'features': np.random.randn(128)
        },
        {
            'bbox': (500, 600, 525, 625),
            'confidence': 0.92,
            'class_id': 1,
            'class_name': 'satellite',
            'features': np.random.randn(128)
        }
    ]


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO detector"""
    mock = MagicMock()
    mock.detect.return_value = []
    mock.conf_thres = 0.25
    mock.iou_thres = 0.45
    return mock


@pytest.fixture
def mock_pinn_model():
    """Mock Physics-Informed Neural Network"""
    mock = MagicMock()
    mock.forward.return_value = torch.randn(1, 6)
    return mock


@pytest.fixture
def mock_transformer_model():
    """Mock Trajectory Transformer"""
    mock = MagicMock()
    mock.forward.return_value = torch.randn(1, 100, 6)
    return mock


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock query results
    mock_result = MagicMock()
    mock_record = MagicMock()
    mock_record.__getitem__.return_value = {'norad_id': 25544, 'name': 'ISS'}
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    return mock_driver


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    mock = MagicMock()
    mock.poll.return_value = None
    mock.subscribe.return_value = None
    return mock


# ============================================================================
# API FIXTURES
# ============================================================================

@pytest.fixture
def api_test_client():
    """FastAPI test client"""
    from fastapi.testclient import TestClient
    from space_debris_tracker.api.server import app

    return TestClient(app)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config() -> Dict:
    """Test configuration"""
    return {
        'database': {
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'test_password'
        },
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'topic': 'space_debris_test'
        },
        'models': {
            'yolo_weights': None,
            'device': 'cpu'
        },
        'prediction': {
            'time_horizon': 86400,
            'dt': 60.0
        }
    }


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def cleanup_files():
    """Cleanup created files after test"""
    files_to_cleanup = []

    def register_file(filepath):
        files_to_cleanup.append(Path(filepath))

    yield register_file

    # Cleanup
    for filepath in files_to_cleanup:
        if filepath.exists():
            if filepath.is_file():
                filepath.unlink()
            else:
                shutil.rmtree(filepath)


# ============================================================================
# SKIP CONDITIONS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests requiring network"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ============================================================================
# BENCHMARKING FIXTURES
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for benchmarking tests"""
    return {
        'rounds': 10,
        'warmup_rounds': 2,
        'min_time': 0.01,
        'max_time': 10.0
    }
