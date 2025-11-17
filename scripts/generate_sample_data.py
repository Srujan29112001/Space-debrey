#!/usr/bin/env python3
"""
Generate Sample Data for Space Debris Tracking System

This script generates synthetic data for testing and development purposes:
- Telescope images with simulated debris
- TLE (Two-Line Element) data
- Conjunction events
- Radar observations

Usage:
    python scripts/generate_sample_data.py --type images --count 100
    python scripts/generate_sample_data.py --type tle --count 500
    python scripts/generate_sample_data.py --all
"""

import argparse
import datetime
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Try to import optional dependencies
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Image generation will be skipped.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available. HDF5 generation will be skipped.")


class SampleDataGenerator:
    """Generate synthetic data for testing"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_telescope_images(self, count: int = 100):
        """Generate synthetic telescope images with debris"""
        if not HAS_PIL:
            print("Skipping image generation (PIL not available)")
            return

        print(f"Generating {count} synthetic telescope images...")

        image_dir = self.output_dir / "telescope_images" / "raw"
        annotation_dir = self.output_dir / "telescope_images" / "annotated"
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            # Create blank image (space background)
            img = Image.new('RGB', (1280, 1280), color=(0, 0, 5))
            draw = ImageDraw.Draw(img)

            # Add stars
            num_stars = random.randint(50, 200)
            annotations = []

            for _ in range(num_stars):
                x = random.randint(0, 1279)
                y = random.randint(0, 1279)
                brightness = random.randint(200, 255)
                size = random.randint(1, 3)
                draw.ellipse([x-size, y-size, x+size, y+size],
                           fill=(brightness, brightness, brightness))

            # Add debris objects
            num_debris = random.randint(1, 5)

            for _ in range(num_debris):
                # Random position
                x = random.randint(100, 1180)
                y = random.randint(100, 1180)

                # Random size (1-10cm, 10-100cm, >100cm)
                debris_class = random.randint(0, 2)
                if debris_class == 0:  # 1-10cm
                    width = random.randint(5, 15)
                    height = random.randint(5, 15)
                elif debris_class == 1:  # 10-100cm
                    width = random.randint(15, 40)
                    height = random.randint(15, 40)
                else:  # >100cm
                    width = random.randint(40, 80)
                    height = random.randint(40, 80)

                # Draw debris (brighter than stars)
                brightness = random.randint(220, 255)
                draw.ellipse([x-width//2, y-height//2, x+width//2, y+height//2],
                           fill=(brightness, brightness-20, brightness-40))

                # Add motion blur (simulate movement)
                for j in range(3):
                    offset_x = random.randint(-2, 2)
                    offset_y = random.randint(-2, 2)
                    draw.ellipse([x-width//2+offset_x, y-height//2+offset_y,
                                x+width//2+offset_x, y+height//2+offset_y],
                               fill=(brightness//2, brightness//2-20, brightness//2-40))

                # YOLO format annotation: class x_center y_center width height
                x_center = x / 1280.0
                y_center = y / 1280.0
                norm_width = width / 1280.0
                norm_height = height / 1280.0

                annotations.append(f"{debris_class} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Save image
            image_path = image_dir / f"telescope_image_{i:05d}.png"
            img.save(image_path)

            # Save annotation
            annotation_path = annotation_dir / f"telescope_image_{i:05d}.txt"
            with open(annotation_path, 'w') as f:
                f.write('\n'.join(annotations))

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} images")

        print(f"✓ Generated {count} telescope images in {image_dir}")

    def generate_tle_data(self, count: int = 500):
        """Generate synthetic TLE (Two-Line Element) data"""
        print(f"Generating {count} synthetic TLE entries...")

        tle_dir = self.output_dir / "tle_data" / "current"
        tle_dir.mkdir(parents=True, exist_ok=True)

        tle_data = []

        for i in range(count):
            # Generate object name
            obj_types = ["DEB", "PAYLOAD", "R/B", "UNKNOWN"]
            obj_type = random.choice(obj_types)
            name = f"{obj_type}-{random.randint(1000, 99999)}"

            # Generate NORAD ID
            norad_id = 25544 + i

            # Generate orbital elements
            epoch_year = 24  # 2024
            epoch_day = random.uniform(1, 365)
            inclination = random.uniform(0, 180)  # degrees
            raan = random.uniform(0, 360)  # degrees
            eccentricity = random.uniform(0.0001, 0.1)  # Low Earth orbit
            arg_perigee = random.uniform(0, 360)
            mean_anomaly = random.uniform(0, 360)
            mean_motion = random.uniform(12, 16)  # revs/day for LEO
            rev_number = random.randint(1000, 99999)

            # Format TLE (simplified checksum calculation)
            line1 = f"1 {norad_id:5d}U 98067A   {epoch_year:02d}{epoch_day:012.8f}  .00012345  00000-0  12345-3 0  9999"
            line2 = f"2 {norad_id:5d} {inclination:8.4f} {raan:8.4f} {int(eccentricity*10000000):07d} {arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_number:5d}9"

            # Calculate checksums
            line1 = line1[:-1] + str(self._tle_checksum(line1[:-1]))
            line2 = line2[:-1] + str(self._tle_checksum(line2[:-1]))

            tle_data.append({
                'name': name,
                'line1': line1,
                'line2': line2
            })

        # Save TLE data
        tle_path = tle_dir / "sample_tle.txt"
        with open(tle_path, 'w') as f:
            for tle in tle_data:
                f.write(f"{tle['name']}\n")
                f.write(f"{tle['line1']}\n")
                f.write(f"{tle['line2']}\n")

        print(f"✓ Generated {count} TLE entries in {tle_path}")

    def _tle_checksum(self, line: str) -> int:
        """Calculate TLE checksum"""
        checksum = 0
        for char in line[:-1]:  # Exclude checksum position
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return checksum % 10

    def generate_conjunction_events(self, count: int = 50):
        """Generate synthetic conjunction events"""
        print(f"Generating {count} synthetic conjunction events...")

        conj_dir = self.output_dir / "training" / "conjunctions"
        conj_dir.mkdir(parents=True, exist_ok=True)

        conjunctions = []

        for i in range(count):
            # Time of closest approach (random future date)
            days_ahead = random.uniform(1, 30)
            tca = datetime.datetime.utcnow() + datetime.timedelta(days=days_ahead)

            # Miss distance (km)
            miss_distance = random.uniform(0.1, 10.0)

            # Collision probability (depends on miss distance)
            if miss_distance < 1.0:
                probability = random.uniform(1e-4, 1e-2)
                risk_level = "HIGH"
            elif miss_distance < 5.0:
                probability = random.uniform(1e-5, 1e-4)
                risk_level = "MEDIUM"
            else:
                probability = random.uniform(1e-7, 1e-5)
                risk_level = "LOW"

            conjunction = {
                "conjunction_id": f"CONJ-2024-{i:05d}",
                "primary_object": {
                    "norad_id": 25544 + random.randint(0, 500),
                    "name": f"SAT-{random.randint(1000, 9999)}",
                    "type": random.choice(["PAYLOAD", "R/B"])
                },
                "secondary_object": {
                    "norad_id": 25544 + random.randint(501, 1000),
                    "name": f"DEB-{random.randint(1000, 9999)}",
                    "type": "DEBRIS"
                },
                "time_of_closest_approach": tca.isoformat() + "Z",
                "miss_distance_km": round(miss_distance, 3),
                "collision_probability": probability,
                "relative_velocity_km_s": round(random.uniform(10, 15), 2),
                "separation": {
                    "radial_km": round(random.uniform(0, miss_distance), 3),
                    "in_track_km": round(random.uniform(0, miss_distance), 3),
                    "cross_track_km": round(random.uniform(0, miss_distance), 3)
                },
                "risk_level": risk_level
            }

            conjunctions.append(conjunction)

        # Save conjunctions
        conj_path = conj_dir / "sample_conjunctions.json"
        with open(conj_path, 'w') as f:
            json.dump(conjunctions, f, indent=2)

        print(f"✓ Generated {count} conjunction events in {conj_path}")

    def generate_orbit_data(self, count: int = 100):
        """Generate synthetic orbit trajectories (HDF5 format)"""
        if not HAS_H5PY:
            print("Skipping orbit data generation (h5py not available)")
            return

        print(f"Generating {count} synthetic orbit trajectories...")

        orbit_dir = self.output_dir / "training" / "orbits"
        orbit_dir.mkdir(parents=True, exist_ok=True)

        orbit_path = orbit_dir / "sample_orbits.h5"

        with h5py.File(orbit_path, 'w') as f:
            for i in range(count):
                # Generate orbital trajectory
                num_points = 1440  # 24 hours at 1-minute intervals

                # Orbital parameters
                semi_major_axis = random.uniform(6571, 8000)  # km (LEO)
                eccentricity = random.uniform(0.0001, 0.1)
                inclination = np.radians(random.uniform(0, 180))

                # Generate trajectory points
                time_points = np.arange(0, num_points * 60, 60)  # seconds
                positions = []
                velocities = []

                for t in time_points:
                    # Simplified Keplerian orbit (circular approximation)
                    theta = 2 * np.pi * t / (86400)  # One day orbit
                    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))

                    # Position in orbital plane
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    z = 0

                    # Rotate by inclination
                    x_rot = x
                    y_rot = y * np.cos(inclination)
                    z_rot = y * np.sin(inclination)

                    positions.append([x_rot, y_rot, z_rot])

                    # Velocity (perpendicular to position)
                    v_mag = np.sqrt(398600.4418 / r)  # GM_Earth / r
                    vx = -v_mag * np.sin(theta)
                    vy = v_mag * np.cos(theta)
                    vz = 0

                    vx_rot = vx
                    vy_rot = vy * np.cos(inclination)
                    vz_rot = vy * np.sin(inclination)

                    velocities.append([vx_rot, vy_rot, vz_rot])

                # Create HDF5 group for this orbit
                grp = f.create_group(f"orbit_{i:05d}")
                grp.attrs['object_id'] = 25544 + i
                grp.attrs['epoch'] = datetime.datetime.utcnow().isoformat()
                grp.create_dataset('time', data=time_points)
                grp.create_dataset('position', data=np.array(positions))
                grp.create_dataset('velocity', data=np.array(velocities))

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{count} orbits")

        print(f"✓ Generated {count} orbit trajectories in {orbit_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for Space Debris Tracking System"
    )
    parser.add_argument(
        '--type',
        choices=['images', 'tle', 'conjunctions', 'orbits'],
        help='Type of data to generate'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output',
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all types of sample data'
    )

    args = parser.parse_args()

    generator = SampleDataGenerator(output_dir=args.output)

    if args.all:
        print("Generating all sample data types...\n")
        generator.generate_telescope_images(count=100)
        print()
        generator.generate_tle_data(count=500)
        print()
        generator.generate_conjunction_events(count=50)
        print()
        generator.generate_orbit_data(count=100)
        print("\n✓ All sample data generated successfully!")

    elif args.type == 'images':
        generator.generate_telescope_images(count=args.count)

    elif args.type == 'tle':
        generator.generate_tle_data(count=args.count)

    elif args.type == 'conjunctions':
        generator.generate_conjunction_events(count=args.count)

    elif args.type == 'orbits':
        generator.generate_orbit_data(count=args.count)

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/generate_sample_data.py --all")
        print("  python scripts/generate_sample_data.py --type images --count 100")
        print("  python scripts/generate_sample_data.py --type tle --count 500")


if __name__ == "__main__":
    main()
