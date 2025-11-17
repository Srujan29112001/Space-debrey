"""
Unit Tests for Data Validator
Tests data validation and sanitization logic
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from space_debris_tracker.data_ingestion.data_validator import DataValidator


@pytest.mark.unit
class TestDataValidator:
    """Test data validation functionality"""

    def test_validator_initialization(self):
        """Test DataValidator initialization"""
        validator = DataValidator()
        assert validator is not None

    def test_validate_state_vector_valid(self):
        """Test validation of valid state vector"""
        validator = DataValidator()

        # Valid ISS-like state
        state = np.array([6778.0, 0.0, 0.0, 0.0, 7.66, 0.0])

        result = validator.validate_state_vector(state)
        assert result is True

    def test_validate_state_vector_invalid_shape(self):
        """Test rejection of invalid state vector shape"""
        validator = DataValidator()

        # Wrong shape
        state = np.array([6778.0, 0.0, 0.0])

        result = validator.validate_state_vector(state)
        assert result is False

    def test_validate_state_vector_suborbital(self):
        """Test rejection of suborbital position"""
        validator = DataValidator()

        # Position below Earth surface
        state = np.array([6000.0, 0.0, 0.0, 0.0, 7.66, 0.0])

        result = validator.validate_state_vector(state)
        assert result is False

    def test_validate_state_vector_excessive_velocity(self):
        """Test rejection of excessive velocity"""
        validator = DataValidator()

        # Escape velocity exceeded
        state = np.array([6778.0, 0.0, 0.0, 0.0, 20.0, 0.0])

        result = validator.validate_state_vector(state)
        assert result is False

    def test_validate_state_vector_nan(self):
        """Test rejection of NaN values"""
        validator = DataValidator()

        state = np.array([6778.0, np.nan, 0.0, 0.0, 7.66, 0.0])

        result = validator.validate_state_vector(state)
        assert result is False

    def test_validate_state_vector_inf(self):
        """Test rejection of infinite values"""
        validator = DataValidator()

        state = np.array([6778.0, 0.0, np.inf, 0.0, 7.66, 0.0])

        result = validator.validate_state_vector(state)
        assert result is False

    def test_validate_orbital_elements(self):
        """Test validation of orbital elements"""
        validator = DataValidator()

        elements = {
            'norad_id': 25544,
            'eccentricity': 0.001,
            'inclination': 51.6,
            'mean_motion': 15.5,
            'semi_major_axis': 6778.0
        }

        result = validator.validate_orbital_elements(elements)
        assert result is True

    def test_validate_orbital_elements_invalid_eccentricity(self):
        """Test rejection of invalid eccentricity"""
        validator = DataValidator()

        # Eccentricity > 1 (hyperbolic)
        elements = {
            'norad_id': 25544,
            'eccentricity': 1.5,
            'inclination': 51.6,
            'mean_motion': 15.5
        }

        result = validator.validate_orbital_elements(elements)
        assert result is False

    def test_validate_orbital_elements_invalid_inclination(self):
        """Test rejection of invalid inclination"""
        validator = DataValidator()

        # Inclination > 180
        elements = {
            'norad_id': 25544,
            'eccentricity': 0.001,
            'inclination': 200.0,
            'mean_motion': 15.5
        }

        result = validator.validate_orbital_elements(elements)
        assert result is False

    def test_sanitize_string_input(self):
        """Test string sanitization"""
        validator = DataValidator()

        # Test XSS prevention
        dirty = "<script>alert('xss')</script>Test"
        clean = validator.sanitize_string(dirty)

        assert '<script>' not in clean
        assert 'Test' in clean

    def test_sanitize_numeric_input(self):
        """Test numeric sanitization"""
        validator = DataValidator()

        # Convert string to float
        result = validator.sanitize_numeric("123.45", float)
        assert result == 123.45

        # Handle invalid input
        result = validator.sanitize_numeric("invalid", float, default=0.0)
        assert result == 0.0

    def test_validate_timestamp(self):
        """Test timestamp validation"""
        validator = DataValidator()

        # Valid timestamp
        now = datetime.utcnow()
        result = validator.validate_timestamp(now)
        assert result is True

        # Future timestamp (within tolerance)
        future = now + timedelta(minutes=5)
        result = validator.validate_timestamp(future)
        assert result is True

        # Far future timestamp (invalid)
        far_future = now + timedelta(days=365)
        result = validator.validate_timestamp(far_future)
        assert result is False

        # Old timestamp (invalid)
        old = now - timedelta(days=365*10)
        result = validator.validate_timestamp(old)
        assert result is False

    def test_validate_detection_bbox(self):
        """Test bounding box validation"""
        validator = DataValidator()

        # Valid bbox
        bbox = (100, 100, 200, 200)
        result = validator.validate_bbox(bbox, img_width=1024, img_height=1024)
        assert result is True

        # Invalid bbox (x2 < x1)
        bbox = (200, 100, 100, 200)
        result = validator.validate_bbox(bbox, img_width=1024, img_height=1024)
        assert result is False

        # Out of bounds
        bbox = (100, 100, 2000, 200)
        result = validator.validate_bbox(bbox, img_width=1024, img_height=1024)
        assert result is False

    def test_validate_confidence_score(self):
        """Test confidence score validation"""
        validator = DataValidator()

        # Valid confidence
        assert validator.validate_confidence(0.5) is True
        assert validator.validate_confidence(0.0) is True
        assert validator.validate_confidence(1.0) is True

        # Invalid confidence
        assert validator.validate_confidence(-0.1) is False
        assert validator.validate_confidence(1.5) is False
        assert validator.validate_confidence(np.nan) is False

    @pytest.mark.parametrize("norad_id,expected", [
        (25544, True),   # Valid
        (99999, True),   # Valid
        (0, False),      # Invalid
        (-1, False),     # Invalid
        (100000, False)  # Invalid (too large)
    ])
    def test_validate_norad_id(self, norad_id, expected):
        """Test NORAD ID validation"""
        validator = DataValidator()
        result = validator.validate_norad_id(norad_id)
        assert result == expected

    def test_validate_trajectory(self):
        """Test trajectory validation"""
        validator = DataValidator()

        # Valid trajectory
        trajectory = np.random.randn(100, 6) * 1000 + 7000
        trajectory[:, 3:] = np.random.randn(100, 3) * 2 + 7.5  # Velocities

        result = validator.validate_trajectory(trajectory)
        assert result is True

        # Invalid shape
        trajectory = np.random.randn(100, 3)
        result = validator.validate_trajectory(trajectory)
        assert result is False

    def test_batch_validation(self):
        """Test batch validation of multiple items"""
        validator = DataValidator()

        items = [
            {'norad_id': 25544, 'state': np.array([6778.0, 0, 0, 0, 7.66, 0])},
            {'norad_id': 25338, 'state': np.array([7178.0, 0, 0, 0, 7.5, 0])},
            {'norad_id': 0, 'state': np.array([6778.0, 0, 0, 0, 7.66, 0])},  # Invalid NORAD
        ]

        valid_items = validator.batch_validate(items)
        assert len(valid_items) == 2

    def test_data_range_validation(self):
        """Test data range validation"""
        validator = DataValidator()

        assert validator.validate_range(50.0, 0.0, 100.0) is True
        assert validator.validate_range(-10.0, 0.0, 100.0) is False
        assert validator.validate_range(150.0, 0.0, 100.0) is False
