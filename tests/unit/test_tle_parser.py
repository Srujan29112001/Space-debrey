"""
Unit Tests for TLE Parser
Tests TLE parsing, validation, and SGP4 propagation
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from space_debris_tracker.data_ingestion.tle_parser import OrbitalElements, TLEParser
from tests.utils import assert_valid_state_vector, generate_tle


@pytest.mark.unit
class TestTLEParser:
    """Test TLE parsing functionality"""

    def test_parser_initialization(self):
        """Test TLE parser initialization"""
        parser = TLEParser()
        assert parser is not None
        assert parser.base_url == "https://www.space-track.org"

    def test_parse_valid_tle(self, sample_tle_lines):
        """Test parsing valid TLE"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)

        assert elements is not None
        assert elements.norad_id == 25544
        assert elements.name == name
        assert 0.0 < elements.eccentricity < 1.0
        assert 0.0 <= elements.inclination <= 180.0
        assert elements.mean_motion > 0

    def test_parse_tle_validates_checksum(self):
        """Test TLE checksum validation"""
        parser = TLEParser()

        # Invalid checksum
        line1 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9009"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009"

        with pytest.raises(ValueError, match="Invalid TLE format"):
            parser.parse_tle(line1, line2)

    def test_parse_tle_validates_format(self):
        """Test TLE format validation"""
        parser = TLEParser()

        # Too short
        line1 = "1 25544U 98067A"
        line2 = "2 25544"

        with pytest.raises(ValueError, match="Invalid TLE format"):
            parser.parse_tle(line1, line2)

    def test_parse_tle_file(self, sample_tle_file):
        """Test parsing TLE file"""
        parser = TLEParser()
        elements_list = parser.parse_tle_file(str(sample_tle_file))

        assert len(elements_list) == 3
        assert elements_list[0].name == "ISS (ZARYA)"
        assert elements_list[1].name == "NOAA 15"
        assert elements_list[2].name == "HUBBLE SPACE TELESCOPE"

    def test_orbital_elements_dataclass(self, orbital_elements):
        """Test OrbitalElements dataclass"""
        elements = OrbitalElements(**orbital_elements)

        assert elements.norad_id == 25544
        assert elements.name == "ISS (ZARYA)"
        assert elements.eccentricity < 1.0

    def test_calculate_derived_elements(self, sample_tle_lines):
        """Test calculation of derived orbital elements"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)

        # Check derived elements calculated
        assert elements.semi_major_axis is not None
        assert elements.period is not None
        assert elements.apogee is not None
        assert elements.perigee is not None

        # Sanity checks
        assert elements.semi_major_axis > 6378.0  # Greater than Earth radius
        assert 80.0 < elements.period < 200.0  # Reasonable LEO period
        assert elements.apogee > elements.perigee  # Basic sanity

    def test_sgp4_propagation(self, sample_tle_lines):
        """Test SGP4 orbit propagation"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)

        # Propagate 1 hour into future
        target_time = elements.epoch + timedelta(hours=1)
        position, velocity = parser.propagate_sgp4(elements, target_time)

        # Validate results
        assert position.shape == (3,)
        assert velocity.shape == (3,)

        # Check magnitudes
        r_mag = np.linalg.norm(position)
        v_mag = np.linalg.norm(velocity)

        assert 6500.0 < r_mag < 7000.0  # LEO range
        assert 7.0 < v_mag < 8.0  # LEO velocity range

    def test_sgp4_multiple_epochs(self, sample_tle_lines):
        """Test SGP4 propagation at multiple epochs"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)

        # Propagate at multiple times
        times = [elements.epoch + timedelta(minutes=i * 10) for i in range(10)]
        positions = []
        velocities = []

        for t in times:
            pos, vel = parser.propagate_sgp4(elements, t)
            positions.append(pos)
            velocities.append(vel)

        positions = np.array(positions)
        velocities = np.array(velocities)

        # Check trajectory is continuous
        assert positions.shape == (10, 3)
        assert velocities.shape == (10, 3)

        # Check adjacent positions are close
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(positions[i + 1] - positions[i])
            assert dist < 500.0  # Should move less than 500 km in 10 min

    def test_epoch_year_handling(self):
        """Test correct handling of 2-digit year in TLE"""
        parser = TLEParser()

        # Year 23 -> 2023
        line1_2023 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9005"
        line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009"

        elements = parser.parse_tle(line1_2023, line2)
        assert elements.epoch.year == 2023

        # Year 99 -> 1999
        line1_1999 = "1 25544U 98067A   99001.00000000  .00016717  00000-0  10270-3 0  9005"
        elements = parser.parse_tle(line1_1999, line2)
        assert elements.epoch.year == 1999

    def test_to_dict_conversion(self, sample_tle_lines):
        """Test conversion of OrbitalElements to dictionary"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)
        data_dict = elements.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict["norad_id"] == 25544
        assert data_dict["name"] == name
        assert "epoch" in data_dict
        assert "semi_major_axis" in data_dict

    @pytest.mark.parametrize(
        "norad_id,expected_name",
        [(25544, "ISS (ZARYA)"), (25338, "NOAA 15"), (20580, "HUBBLE SPACE TELESCOPE")],
    )
    def test_parse_multiple_satellites(self, sample_tle_file, norad_id, expected_name):
        """Test parsing file with multiple satellites"""
        parser = TLEParser()
        elements_list = parser.parse_tle_file(str(sample_tle_file))

        # Find satellite by NORAD ID
        satellite = next((e for e in elements_list if e.norad_id == norad_id), None)

        assert satellite is not None
        assert satellite.name == expected_name

    def test_tle_checksum_calculation(self):
        """Test TLE checksum calculation"""
        parser = TLEParser()

        line = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  900"
        checksum = parser._validate_tle(
            line + "5",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009",
        )

        assert checksum  # Should be valid

    def test_invalid_tle_line_numbers(self):
        """Test validation of line numbers"""
        parser = TLEParser()

        # Wrong line numbers
        line1 = "2 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9005"
        line2 = "1 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009"

        with pytest.raises(ValueError):
            parser.parse_tle(line1, line2)

    def test_generated_tle(self):
        """Test parsing generated TLE"""
        parser = TLEParser()
        name, line1, line2 = generate_tle(99999, "GENERATED SAT")

        elements = parser.parse_tle(line1, line2, name)

        assert elements.norad_id == 99999
        assert elements.name == "GENERATED SAT"

    @pytest.mark.slow
    def test_long_propagation(self, sample_tle_lines):
        """Test propagation over extended period"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines

        elements = parser.parse_tle(line1, line2, name)

        # Propagate 7 days
        target_time = elements.epoch + timedelta(days=7)
        position, velocity = parser.propagate_sgp4(elements, target_time)

        # Should still be in orbit
        r_mag = np.linalg.norm(position)
        assert 6500.0 < r_mag < 7000.0


@pytest.mark.unit
class TestOrbitalElements:
    """Test OrbitalElements class"""

    def test_orbital_elements_creation(self):
        """Test creating OrbitalElements"""
        elements = OrbitalElements(
            norad_id=25544,
            name="TEST",
            epoch=datetime.now(),
            mean_motion=15.5,
            eccentricity=0.001,
            inclination=51.6,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            bstar=0.0001,
            classification="U",
            element_set_number=1,
            revolution_number=0,
        )

        assert elements.norad_id == 25544
        assert elements.name == "TEST"

    def test_derived_elements_calculation(self):
        """Test calculation of derived elements"""
        elements = OrbitalElements(
            norad_id=25544,
            name="TEST",
            epoch=datetime.now(),
            mean_motion=15.72834465,
            eccentricity=0.0006703,
            inclination=51.6,
            raan=0.0,
            arg_perigee=0.0,
            mean_anomaly=0.0,
            bstar=0.0001,
            classification="U",
            element_set_number=1,
            revolution_number=0,
        )

        elements.calculate_derived_elements()

        assert elements.semi_major_axis is not None
        assert elements.period is not None
        assert elements.apogee is not None
        assert elements.perigee is not None

        # Check values are reasonable for ISS
        assert 6700 < elements.semi_major_axis < 6900
        assert 88 < elements.period < 95
        assert 380 < elements.perigee < 420
        assert 400 < elements.apogee < 440
