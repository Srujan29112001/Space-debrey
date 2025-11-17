# Data Directory Structure

This directory contains all data used by the Space Debris Tracking System.

## Directory Structure

```
data/
├── telescope_images/    # Optical telescope imagery
│   ├── raw/             # Unprocessed telescope images
│   ├── preprocessed/    # Star-removed, enhanced images
│   └── annotated/       # YOLO format annotations
│
├── radar_data/          # Radar observations
│   ├── raw/             # Raw radar returns
│   └── processed/       # Calibrated radar data
│
├── tle_data/            # Two-Line Element sets
│   ├── current/         # Latest TLE data
│   ├── historical/      # Historical TLEs for training
│   └── validated/       # Quality-checked TLEs
│
├── training/            # Training datasets
│   ├── debris_images/   # Labeled debris images (YOLO)
│   ├── orbits/          # Historical orbit data (HDF5)
│   └── conjunctions/    # Past conjunction events (JSON)
│
├── validation/          # Validation datasets
│   ├── debris_images/
│   ├── orbits/
│   └── conjunctions/
│
└── test/                # Test datasets
    ├── debris_images/
    ├── orbits/
    └── conjunctions/
```

## Data Formats

### Telescope Images
- **Format**: PNG, FITS
- **Resolution**: 1280x1280 pixels
- **Bit Depth**: 16-bit grayscale
- **Annotations**: YOLO format (txt files)

Example annotation format:
```
# class x_center y_center width height
0 0.512 0.384 0.023 0.019
```

### Radar Data
- **Format**: HDF5
- **Fields**: range, azimuth, elevation, doppler, rcs
- **Sample Rate**: 1 kHz
- **Range Resolution**: 10 meters

Example HDF5 structure:
```python
{
    'metadata': {
        'timestamp': '2024-01-15T14:30:00Z',
        'radar_id': 'RADAR-001',
        'frequency_ghz': 10.5
    },
    'measurements': {
        'range': [...],      # km
        'azimuth': [...],    # degrees
        'elevation': [...],  # degrees
        'doppler': [...],    # Hz
        'rcs': [...]         # dBsm
    }
}
```

### TLE Data
- **Format**: Text (TLE format)
- **Source**: Space-Track.org, CelesTrak
- **Update Frequency**: Daily

Example TLE:
```
ISS (ZARYA)
1 25544U 98067A   24015.50000000  .00012345  00000-0  12345-3 0  9993
2 25544  51.6400 247.4627 0002340 123.4567 234.5678 15.54012345123456
```

### Orbits (HDF5)
- **Format**: HDF5
- **Fields**: position, velocity, time, covariance
- **Reference Frame**: J2000 ECI

Example:
```python
{
    'object_id': 25544,
    'epoch': '2024-01-15T00:00:00Z',
    'state_vectors': {
        'position': [[x1, y1, z1], [x2, y2, z2], ...],  # km
        'velocity': [[vx1, vy1, vz1], [vx2, vy2, vz2], ...],  # km/s
        'time': [0, 60, 120, ...]  # seconds from epoch
    },
    'covariance': [...],  # 6x6 covariance matrix
    'metadata': {...}
}
```

### Conjunctions (JSON)
- **Format**: JSON
- **Fields**: objects, tca, miss_distance, probability

Example:
```json
{
    "conjunction_id": "CONJ-2024-001",
    "primary_object": {
        "norad_id": 25544,
        "name": "ISS (ZARYA)",
        "type": "PAYLOAD"
    },
    "secondary_object": {
        "norad_id": 12345,
        "name": "DEBRIS-12345",
        "type": "DEBRIS"
    },
    "time_of_closest_approach": "2024-01-20T15:30:45Z",
    "miss_distance_km": 0.523,
    "collision_probability": 2.3e-5,
    "relative_velocity_km_s": 14.2,
    "separation": {
        "radial_km": 0.15,
        "in_track_km": 0.32,
        "cross_track_km": 0.38
    },
    "covariance_combined": [...],
    "risk_level": "MEDIUM"
}
```

## Data Acquisition

### Telescope Images
1. **Ground-based Telescopes**: 0.5m+ aperture with wide field
2. **Space-based Sensors**: SpaceX Starlink tracking cameras
3. **Synthetic Data**: Blender-generated imagery for training

### Radar Data
1. **Space Surveillance Network (SSN)**: US Space Force radars
2. **Commercial Radars**: LeoLabs, NorthStar
3. **Simulated Data**: STK-generated radar returns

### TLE Data
1. **Space-Track.org**: https://www.space-track.org (requires account)
2. **CelesTrak**: https://celestrak.org (public access)
3. **Update Script**: `scripts/update_tle_data.py`

## Data Pipeline

```
┌─────────────────┐
│ Data Ingestion  │
│  - Kafka        │
│  - REST API     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Validation      │
│  - Format check │
│  - Range check  │
│  - Checksum     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Preprocessing   │
│  - Normalization│
│  - Augmentation │
│  - Star removal │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Storage         │
│  - Local disk   │
│  - S3 bucket    │
│  - Database     │
└─────────────────┘
```

## Sample Data Generation

To generate sample data for testing:

```bash
# Generate sample telescope images
python scripts/generate_sample_data.py --type images --count 100

# Generate sample TLE data
python scripts/generate_sample_data.py --type tle --count 500

# Generate sample conjunction events
python scripts/generate_sample_data.py --type conjunctions --count 50
```

## Data Privacy & Security

- **Classification**: Most debris tracking data is public
- **Restricted Data**: Military satellite orbits (NORAD catalog)
- **ITAR Compliance**: Radar performance specifications
- **Encryption**: At-rest encryption for sensitive orbits

## Gitignore Note

This directory is gitignored to prevent large data files from being committed.
Only structure and documentation are version-controlled.

## References

- [Space-Track.org](https://www.space-track.org)
- [CelesTrak](https://celestrak.org)
- [YOLO Annotation Format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [TLE Format Specification](https://en.wikipedia.org/wiki/Two-line_element_set)
