"""
TLE (Two-Line Element) Parser
Parses orbital element data from NORAD and Space-Track.org
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from sgp4 import exporter
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)


@dataclass
class OrbitalElements:
    """Keplerian orbital elements"""

    norad_id: int
    name: str
    epoch: datetime
    mean_motion: float  # revolutions per day
    eccentricity: float
    inclination: float  # degrees
    raan: float  # Right Ascension of Ascending Node (degrees)
    arg_perigee: float  # Argument of perigee (degrees)
    mean_anomaly: float  # degrees
    bstar: float  # drag term
    classification: str  # U=unclassified, C=classified, S=secret
    element_set_number: int
    revolution_number: int

    # Derived quantities
    semi_major_axis: Optional[float] = None  # km
    period: Optional[float] = None  # minutes
    apogee: Optional[float] = None  # km
    perigee: Optional[float] = None  # km

    def calculate_derived_elements(self):
        """Calculate semi-major axis, period, apogee, perigee"""
        mu = 398600.4418  # Earth gravitational parameter km^3/s^2

        # Mean motion in radians per minute
        n = self.mean_motion * 2 * np.pi / 1440.0  # convert from rev/day to rad/min

        # Semi-major axis from mean motion
        # n^2 * a^3 = mu
        self.semi_major_axis = (mu / (n * 60) ** 2) ** (1 / 3)  # convert to rad/sec

        # Period in minutes
        self.period = 1440.0 / self.mean_motion

        # Apogee and perigee
        Re = 6378.137  # Earth radius km
        self.apogee = self.semi_major_axis * (1 + self.eccentricity) - Re
        self.perigee = self.semi_major_axis * (1 - self.eccentricity) - Re

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "norad_id": self.norad_id,
            "name": self.name,
            "epoch": self.epoch.isoformat(),
            "mean_motion": self.mean_motion,
            "eccentricity": self.eccentricity,
            "inclination": self.inclination,
            "raan": self.raan,
            "arg_perigee": self.arg_perigee,
            "mean_anomaly": self.mean_anomaly,
            "bstar": self.bstar,
            "classification": self.classification,
            "element_set_number": self.element_set_number,
            "revolution_number": self.revolution_number,
            "semi_major_axis": self.semi_major_axis,
            "period": self.period,
            "apogee": self.apogee,
            "perigee": self.perigee,
        }


class TLEParser:
    """Parse and process Two-Line Element sets"""

    def __init__(
        self,
        space_track_username: Optional[str] = None,
        space_track_password: Optional[str] = None,
    ):
        """
        Initialize TLE Parser

        Args:
            space_track_username: Space-Track.org username
            space_track_password: Space-Track.org password
        """
        self.space_track_username = space_track_username
        self.space_track_password = space_track_password
        self.session = requests.Session()
        self.base_url = "https://www.space-track.org"

    def parse_tle(self, line1: str, line2: str, name: str = "") -> OrbitalElements:
        """
        Parse a TLE from two lines

        Args:
            line1: First line of TLE
            line2: Second line of TLE
            name: Satellite name (optional)

        Returns:
            OrbitalElements object
        """
        # Validate TLE format
        if not self._validate_tle(line1, line2):
            raise ValueError("Invalid TLE format")

        # Parse line 1
        norad_id = int(line1[2:7])
        classification = line1[7]
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        bstar = float(line1[53:61])
        element_set_num = int(line1[64:68])

        # Handle year (00-57 = 2000-2057, 58-99 = 1958-1999)
        if epoch_year < 57:
            epoch_year += 2000
        else:
            epoch_year += 1900

        # Convert epoch day to datetime
        epoch = datetime(epoch_year, 1, 1) + datetime.timedelta(days=epoch_day - 1)

        # Parse line 2
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33])
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        revolution_number = int(line2[63:68])

        # Create orbital elements
        elements = OrbitalElements(
            norad_id=norad_id,
            name=name.strip(),
            epoch=epoch,
            mean_motion=mean_motion,
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            arg_perigee=arg_perigee,
            mean_anomaly=mean_anomaly,
            bstar=bstar,
            classification=classification,
            element_set_number=element_set_num,
            revolution_number=revolution_number,
        )

        # Calculate derived elements
        elements.calculate_derived_elements()

        return elements

    def parse_tle_file(self, filepath: str) -> List[OrbitalElements]:
        """
        Parse TLE file with multiple satellites

        Args:
            filepath: Path to TLE file

        Returns:
            List of OrbitalElements
        """
        elements_list = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Process in groups of 3 (name, line1, line2)
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break

            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            try:
                elements = self.parse_tle(line1, line2, name)
                elements_list.append(elements)
            except ValueError as e:
                logger.warning(f"Failed to parse TLE for {name}: {e}")
                continue

        logger.info(f"Parsed {len(elements_list)} TLE sets from {filepath}")
        return elements_list

    def propagate_sgp4(
        self, elements: OrbitalElements, target_time: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit using SGP4

        Args:
            elements: Orbital elements
            target_time: Time to propagate to

        Returns:
            position (km), velocity (km/s) in TEME frame
        """
        # Create satellite object
        # Convert orbital elements to SGP4 format
        sat = Satrec()
        sat.sgp4init(
            whichconst=exporter.WGS72,  # gravity model
            opsmode="i",  # improved mode
            satnum=str(elements.norad_id),
            epoch=(elements.epoch - datetime(1949, 12, 31)).total_seconds() / 86400.0,
            bstar=elements.bstar,
            ndot=0.0,  # Not in standard TLE
            nddot=0.0,  # Not in standard TLE
            ecco=elements.eccentricity,
            argpo=np.radians(elements.arg_perigee),
            inclo=np.radians(elements.inclination),
            mo=np.radians(elements.mean_anomaly),
            no_kozai=elements.mean_motion * 2 * np.pi / 1440.0,  # rad/min
            nodeo=np.radians(elements.raan),
        )

        # Propagate to target time
        jd, fr = jday(
            target_time.year,
            target_time.month,
            target_time.day,
            target_time.hour,
            target_time.minute,
            target_time.second,
        )

        error_code, position, velocity = sat.sgp4(jd, fr)

        if error_code != 0:
            raise RuntimeError(f"SGP4 propagation error: {error_code}")

        return np.array(position), np.array(velocity)

    def fetch_latest_tle(self, norad_id: int) -> Optional[OrbitalElements]:
        """
        Fetch latest TLE from Space-Track.org

        Args:
            norad_id: NORAD catalog ID

        Returns:
            OrbitalElements or None if not found
        """
        if not self.space_track_username or not self.space_track_password:
            logger.warning("Space-Track credentials not provided")
            return None

        # Login
        login_url = f"{self.base_url}/ajaxauth/login"
        login_data = {
            "identity": self.space_track_username,
            "password": self.space_track_password,
        }

        try:
            response = self.session.post(login_url, data=login_data)
            response.raise_for_status()

            # Query TLE
            query_url = (
                f"{self.base_url}/basicspacedata/query/class/tle_latest/"
                f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/3le"
            )

            response = self.session.get(query_url)
            response.raise_for_status()

            # Parse response
            lines = response.text.strip().split("\n")
            if len(lines) >= 3:
                return self.parse_tle(lines[1], lines[2], lines[0])
            else:
                logger.error(f"Invalid response for NORAD ID {norad_id}")
                return None

        except requests.RequestException as e:
            logger.error(f"Failed to fetch TLE: {e}")
            return None

    def fetch_catalog(
        self, classification: str = "U", output_file: Optional[str] = None
    ) -> List[OrbitalElements]:
        """
        Fetch entire satellite catalog

        Args:
            classification: U=unclassified, C=classified
            output_file: Optional file to save TLEs

        Returns:
            List of OrbitalElements
        """
        if not self.space_track_username or not self.space_track_password:
            logger.warning("Space-Track credentials not provided")
            return []

        # Login
        login_url = f"{self.base_url}/ajaxauth/login"
        login_data = {
            "identity": self.space_track_username,
            "password": self.space_track_password,
        }

        try:
            response = self.session.post(login_url, data=login_data)
            response.raise_for_status()

            # Query catalog
            query_url = (
                f"{self.base_url}/basicspacedata/query/class/tle_latest/"
                f"ORDINAL/1/CLASSIFICATION/{classification}/"
                f"orderby/NORAD_CAT_ID/format/3le"
            )

            response = self.session.get(query_url)
            response.raise_for_status()

            # Save to file if requested
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    f.write(response.text)
                logger.info(f"Saved catalog to {output_file}")

            # Parse TLEs
            lines = response.text.strip().split("\n")
            elements_list = []

            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break

                try:
                    elements = self.parse_tle(lines[i + 1], lines[i + 2], lines[i])
                    elements_list.append(elements)
                except ValueError:
                    continue

            logger.info(f"Fetched {len(elements_list)} TLEs from catalog")
            return elements_list

        except requests.RequestException as e:
            logger.error(f"Failed to fetch catalog: {e}")
            return []

    @staticmethod
    def _validate_tle(line1: str, line2: str) -> bool:
        """Validate TLE checksum and format"""
        if len(line1) != 69 or len(line2) != 69:
            return False

        if line1[0] != "1" or line2[0] != "2":
            return False

        # Validate checksums
        def checksum(line):
            total = 0
            for char in line[:-1]:
                if char.isdigit():
                    total += int(char)
                elif char == "-":
                    total += 1
            return total % 10

        if checksum(line1) != int(line1[68]):
            return False
        if checksum(line2) != int(line2[68]):
            return False

        return True


# Example usage
if __name__ == "__main__":
    # Parse TLE
    parser = TLEParser()

    # Example TLE for ISS
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9005"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72834465 00009"

    elements = parser.parse_tle(line1, line2, name)
    print(f"Parsed: {elements.name}")
    print(f"  Altitude: {elements.perigee:.1f} x {elements.apogee:.1f} km")
    print(f"  Period: {elements.period:.2f} min")
    print(f"  Inclination: {elements.inclination:.2f}°")

    # Propagate orbit
    target_time = datetime.utcnow()
    pos, vel = parser.propagate_sgp4(elements, target_time)
    print(f"  Position at {target_time}: {pos}")
    print(f"  Velocity: {vel}")
