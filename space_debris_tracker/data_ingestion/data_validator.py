"""
Data Validation and Normalization
Validates and normalizes incoming space debris data
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_data: Optional[Dict] = None


class DataValidator:
    """Validates and normalizes space debris tracking data"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data validator

        Args:
            config: Validation configuration
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """Default validation configuration"""
        return {
            # Orbital element ranges
            "semi_major_axis_min": 6400,  # km (LEO minimum)
            "semi_major_axis_max": 42164,  # km (GEO)
            "eccentricity_min": 0.0,
            "eccentricity_max": 0.99,  # Highly elliptical orbits
            "inclination_min": 0.0,
            "inclination_max": 180.0,  # degrees
            "raan_min": 0.0,
            "raan_max": 360.0,  # degrees
            "arg_perigee_min": 0.0,
            "arg_perigee_max": 360.0,  # degrees
            "mean_anomaly_min": 0.0,
            "mean_anomaly_max": 360.0,  # degrees
            # State vector ranges
            "position_min": 6400,  # km
            "position_max": 50000,  # km
            "velocity_min": 0.5,  # km/s
            "velocity_max": 15.0,  # km/s (escape velocity ~11.2)
            # Image validation
            "image_min_size": (100, 100),
            "image_max_size": (10000, 10000),
            "image_bit_depth": [8, 12, 14, 16],
            # Radar validation
            "range_min": 100,  # km
            "range_max": 10000,  # km
            "range_rate_min": -15,  # km/s
            "range_rate_max": 15,  # km/s
            "rcs_min": 0.001,  # m^2
            "rcs_max": 1000,  # m^2
            "snr_min": 3,  # dB
            "snr_max": 100,  # dB
            # Time validation
            "max_age_hours": 24,  # Maximum data age
        }

    def validate_orbital_elements(self, elements: Dict) -> ValidationResult:
        """
        Validate Keplerian orbital elements

        Args:
            elements: Dictionary with orbital elements

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        required_fields = [
            "semi_major_axis",
            "eccentricity",
            "inclination",
            "raan",
            "arg_perigee",
            "mean_anomaly",
        ]

        for field in required_fields:
            if field not in elements:
                errors.append(f"Missing required field: {field}")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate ranges
        if not (
            self.config["semi_major_axis_min"]
            <= elements["semi_major_axis"]
            <= self.config["semi_major_axis_max"]
        ):
            errors.append(f"Semi-major axis out of range: {elements['semi_major_axis']:.1f} km")

        if not (
            self.config["eccentricity_min"]
            <= elements["eccentricity"]
            <= self.config["eccentricity_max"]
        ):
            errors.append(f"Eccentricity out of range: {elements['eccentricity']:.4f}")

        if not (
            self.config["inclination_min"]
            <= elements["inclination"]
            <= self.config["inclination_max"]
        ):
            errors.append(f"Inclination out of range: {elements['inclination']:.2f}°")

        # Normalize angles to [0, 360)
        normalized = elements.copy()
        for angle_field in ["raan", "arg_perigee", "mean_anomaly"]:
            normalized[angle_field] = elements[angle_field] % 360.0

        # Check for physically impossible orbits
        # Perigee must be above Earth surface
        Re = 6378.137  # Earth radius km
        a = elements["semi_major_axis"]
        e = elements["eccentricity"]
        perigee = a * (1 - e) - Re

        if perigee < -100:  # Allow some margin for decay
            errors.append(f"Perigee below Earth surface: {perigee:.1f} km")
        elif perigee < 100:
            warnings.append(f"Very low perigee (rapid decay expected): {perigee:.1f} km")

        # Check epoch if provided
        if "epoch" in elements:
            try:
                epoch = (
                    datetime.fromisoformat(elements["epoch"])
                    if isinstance(elements["epoch"], str)
                    else elements["epoch"]
                )

                age = datetime.utcnow() - epoch
                if age.total_seconds() / 3600 > self.config["max_age_hours"]:
                    warnings.append(
                        f"Old orbital elements: {age.total_seconds()/3600:.1f} hours old"
                    )
            except Exception as e:
                errors.append(f"Invalid epoch format: {e}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_data=normalized if is_valid else None,
        )

    def validate_state_vector(self, state: Dict) -> ValidationResult:
        """
        Validate Cartesian state vector

        Args:
            state: Dictionary with position and velocity

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        if "position" not in state or "velocity" not in state:
            errors.append("Missing position or velocity")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        position = np.array(state["position"])
        velocity = np.array(state["velocity"])

        # Validate dimensions
        if position.shape != (3,):
            errors.append(f"Position must be 3D vector, got shape {position.shape}")
        if velocity.shape != (3,):
            errors.append(f"Velocity must be 3D vector, got shape {velocity.shape}")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate ranges
        pos_magnitude = np.linalg.norm(position)
        vel_magnitude = np.linalg.norm(velocity)

        if not (self.config["position_min"] <= pos_magnitude <= self.config["position_max"]):
            errors.append(f"Position magnitude out of range: {pos_magnitude:.1f} km")

        if not (self.config["velocity_min"] <= vel_magnitude <= self.config["velocity_max"]):
            errors.append(f"Velocity magnitude out of range: {vel_magnitude:.3f} km/s")

        # Check if orbit is bound (negative total energy)
        mu = 398600.4418  # Earth gravitational parameter km^3/s^2
        kinetic_energy = 0.5 * vel_magnitude**2
        potential_energy = -mu / pos_magnitude
        total_energy = kinetic_energy + potential_energy

        if total_energy > 0:
            warnings.append("Hyperbolic orbit detected (escape trajectory)")

        # Check for atmospheric re-entry
        Re = 6378.137  # Earth radius km
        altitude = pos_magnitude - Re

        if altitude < 100:
            errors.append(f"Object below atmosphere boundary: {altitude:.1f} km")
        elif altitude < 200:
            warnings.append(f"Object in upper atmosphere (rapid decay): {altitude:.1f} km")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_data=state if is_valid else None,
        )

    def validate_telescope_image(self, image_data: Dict) -> ValidationResult:
        """
        Validate telescope image data

        Args:
            image_data: Dictionary with image and metadata

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        required_fields = ["image_array", "timestamp", "ra", "dec"]
        for field in required_fields:
            if field not in image_data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        image = image_data["image_array"]

        # Validate image dimensions
        if image.ndim not in [2, 3]:
            errors.append(f"Image must be 2D or 3D, got {image.ndim}D")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        height, width = image.shape[:2]

        min_h, min_w = self.config["image_min_size"]
        max_h, max_w = self.config["image_max_size"]

        if not (min_h <= height <= max_h):
            errors.append(f"Image height out of range: {height}")
        if not (min_w <= width <= max_w):
            errors.append(f"Image width out of range: {width}")

        # Validate bit depth
        if image.dtype == np.uint8:
            bit_depth = 8
        elif image.dtype == np.uint16:
            bit_depth = 16
        else:
            warnings.append(f"Unusual image dtype: {image.dtype}")
            bit_depth = 0

        if bit_depth not in self.config["image_bit_depth"]:
            warnings.append(f"Non-standard bit depth: {bit_depth}")

        # Validate celestial coordinates
        ra = image_data["ra"]
        dec = image_data["dec"]

        if not (0 <= ra < 360):
            errors.append(f"Right Ascension out of range [0, 360): {ra}°")
        if not (-90 <= dec <= 90):
            errors.append(f"Declination out of range [-90, 90]: {dec}°")

        # Validate timestamp
        try:
            timestamp = (
                datetime.fromisoformat(image_data["timestamp"])
                if isinstance(image_data["timestamp"], str)
                else image_data["timestamp"]
            )

            age = datetime.utcnow() - timestamp
            if age.total_seconds() > 3600 * self.config["max_age_hours"]:
                warnings.append(f"Old image: {age.total_seconds()/3600:.1f} hours old")
        except Exception as e:
            errors.append(f"Invalid timestamp: {e}")

        # Check for saturated pixels
        if bit_depth == 16:
            max_value = 65535
            saturation_threshold = 0.9 * max_value
            saturated_pixels = np.sum(image >= saturation_threshold)
            saturation_percent = 100 * saturated_pixels / image.size

            if saturation_percent > 5:
                warnings.append(f"High saturation: {saturation_percent:.1f}% of pixels")
        elif bit_depth == 8:
            max_value = 255
            saturation_threshold = 0.9 * max_value
            saturated_pixels = np.sum(image >= saturation_threshold)
            saturation_percent = 100 * saturated_pixels / image.size

            if saturation_percent > 5:
                warnings.append(f"High saturation: {saturation_percent:.1f}% of pixels")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_data=image_data if is_valid else None,
        )

    def validate_radar_return(self, radar_data: Dict) -> ValidationResult:
        """
        Validate radar return data

        Args:
            radar_data: Dictionary with radar measurements

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check required fields
        required_fields = ["timestamp", "range", "range_rate", "azimuth", "elevation"]
        for field in required_fields:
            if field not in radar_data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate range
        range_val = radar_data["range"]
        if not (self.config["range_min"] <= range_val <= self.config["range_max"]):
            errors.append(f"Range out of bounds: {range_val:.1f} km")

        # Validate range rate
        range_rate = radar_data["range_rate"]
        if not (self.config["range_rate_min"] <= range_rate <= self.config["range_rate_max"]):
            errors.append(f"Range rate out of bounds: {range_rate:.3f} km/s")

        # Validate angles
        azimuth = radar_data["azimuth"]
        elevation = radar_data["elevation"]

        if not (0 <= azimuth < 360):
            radar_data["azimuth"] = azimuth % 360
            warnings.append("Normalized azimuth to [0, 360)")

        if not (0 <= elevation <= 90):
            errors.append(f"Elevation out of range [0, 90]: {elevation}°")

        # Validate optional fields
        if "rcs" in radar_data:
            rcs = radar_data["rcs"]
            if not (self.config["rcs_min"] <= rcs <= self.config["rcs_max"]):
                warnings.append(f"RCS out of typical range: {rcs:.3f} m^2")

        if "snr" in radar_data:
            snr = radar_data["snr"]
            if not (self.config["snr_min"] <= snr <= self.config["snr_max"]):
                warnings.append(f"SNR out of typical range: {snr:.1f} dB")
            elif snr < 10:
                warnings.append(f"Low SNR (unreliable detection): {snr:.1f} dB")

        # Validate timestamp
        try:
            timestamp = (
                datetime.fromisoformat(radar_data["timestamp"])
                if isinstance(radar_data["timestamp"], str)
                else radar_data["timestamp"]
            )

            age = datetime.utcnow() - timestamp
            if age.total_seconds() > 3600 * self.config["max_age_hours"]:
                warnings.append(f"Old radar data: {age.total_seconds()/3600:.1f} hours old")
        except Exception as e:
            errors.append(f"Invalid timestamp: {e}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_data=radar_data if is_valid else None,
        )

    def validate_batch(self, data_list: List[Dict], data_type: str) -> Tuple[List[Dict], List[str]]:
        """
        Validate batch of data

        Args:
            data_list: List of data dictionaries
            data_type: Type of data ('orbital_elements', 'state_vector',
                      'telescope_image', 'radar_return')

        Returns:
            Tuple of (valid_data, error_messages)
        """
        validators = {
            "orbital_elements": self.validate_orbital_elements,
            "state_vector": self.validate_state_vector,
            "telescope_image": self.validate_telescope_image,
            "radar_return": self.validate_radar_return,
        }

        if data_type not in validators:
            raise ValueError(f"Unknown data type: {data_type}")

        validator = validators[data_type]
        valid_data = []
        all_errors = []

        for i, data in enumerate(data_list):
            result = validator(data)

            if result.is_valid:
                valid_data.append(result.normalized_data)
            else:
                all_errors.extend([f"Item {i}: {error}" for error in result.errors])

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Item {i}: {warning}")

        logger.info(
            f"Validated {len(data_list)} items: "
            f"{len(valid_data)} valid, {len(data_list) - len(valid_data)} invalid"
        )

        return valid_data, all_errors


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    validator = DataValidator()

    # Test orbital elements
    elements = {
        "semi_major_axis": 6900,
        "eccentricity": 0.001,
        "inclination": 51.6,
        "raan": 120.5,
        "arg_perigee": 45.2,
        "mean_anomaly": 180.0,
        "epoch": datetime.utcnow().isoformat(),
    }

    result = validator.validate_orbital_elements(elements)
    print(f"Orbital elements valid: {result.is_valid}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")

    # Test state vector
    state = {"position": [7000, 0, 0], "velocity": [0, 7.5, 0]}

    result = validator.validate_state_vector(state)
    print(f"State vector valid: {result.is_valid}")
