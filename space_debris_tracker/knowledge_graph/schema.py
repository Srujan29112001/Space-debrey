"""
Graph Schema Definition
Defines the structure of the space knowledge graph
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ObjectStatus(Enum):
    """Space object status"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DECAYED = "DECAYED"
    UNKNOWN = "UNKNOWN"


class SizeCategory(Enum):
    """Debris size category"""
    LARGE = "LARGE"  # > 10 cm
    MEDIUM = "MEDIUM"  # 1-10 cm
    SMALL = "SMALL"  # < 1 cm
    UNKNOWN = "UNKNOWN"


@dataclass
class GraphSchema:
    """
    Space Knowledge Graph Schema

    Node Types:
    - Satellite: Active satellites
    - Debris: Space debris objects
    - Orbit: Orbital elements
    - Conjunction: Close approach events
    - Operator: Satellite operators
    - Maneuver: Avoidance maneuvers

    Relationship Types:
    - HAS_ORBIT: Satellite -> Orbit
    - ORIGINATED_FROM: Debris -> Satellite
    - INVOLVES: Conjunction -> Satellite/Debris
    - AT_RISK: Satellite -> Debris
    - OPERATED_BY: Satellite -> Operator
    - EXECUTED: Satellite -> Maneuver
    - AVOIDED: Maneuver -> Conjunction
    """

    # Node schemas
    SATELLITE_SCHEMA = {
        'norad_id': int,  # Primary key
        'name': str,
        'international_designator': str,
        'operator': str,
        'launch_date': str,  # ISO format
        'mass': float,  # kg
        'status': str,  # ObjectStatus
        'area_to_mass': float,  # m^2/kg
        'updated': str  # ISO timestamp
    }

    DEBRIS_SCHEMA = {
        'object_id': str,  # Primary key
        'size_category': str,  # SizeCategory
        'origin': str,  # Origin satellite/event
        'detection_date': str,  # ISO format
        'rcs': float,  # Radar cross section (m^2)
        'estimated_mass': float,  # kg
        'updated': str
    }

    ORBIT_SCHEMA = {
        'semi_major_axis': float,  # km
        'eccentricity': float,
        'inclination': float,  # degrees
        'raan': float,  # Right ascension of ascending node (deg)
        'arg_perigee': float,  # Argument of perigee (deg)
        'mean_anomaly': float,  # degrees
        'epoch': str,  # ISO timestamp
        'period': float,  # minutes
        'apogee': float,  # km
        'perigee': float  # km
    }

    CONJUNCTION_SCHEMA = {
        'primary_object': int,  # NORAD ID or object ID
        'secondary_object': str,
        'tca': str,  # Time of closest approach (ISO)
        'miss_distance': float,  # km
        'probability': float,  # Collision probability
        'relative_velocity': float,  # km/s
        'radial_separation': float,  # km
        'in_track_separation': float,  # km
        'cross_track_separation': float,  # km
        'screening_volume_radius': float,  # km
        'updated': str
    }

    OPERATOR_SCHEMA = {
        'name': str,  # Primary key
        'country': str,
        'organization_type': str,  # Government, Commercial, etc.
        'contact': str,
        'satellite_count': int
    }

    MANEUVER_SCHEMA = {
        'maneuver_id': str,  # Primary key
        'satellite_id': int,  # NORAD ID
        'execution_time': str,  # ISO timestamp
        'delta_v': float,  # m/s
        'delta_v_vector': List[float],  # [x, y, z] m/s
        'fuel_cost': float,  # kg
        'reason': str,  # Collision avoidance, orbit maintenance, etc.
        'success': bool
    }

    # Relationship schemas
    HAS_ORBIT_REL = {
        'valid_from': str,  # ISO timestamp
        'valid_to': str,
        'source': str  # TLE, GPS, etc.
    }

    AT_RISK_REL = {
        'level': str,  # RiskLevel
        'updated': str,
        'factors': List[str]  # Contributing risk factors
    }

    AVOIDED_REL = {
        'planned_miss_distance': float,  # km before maneuver
        'actual_miss_distance': float,  # km after maneuver
        'improvement': float  # km
    }


# Cypher query templates
QUERIES = {
    'find_high_risk_conjunctions': """
        MATCH (c:Conjunction)
        WHERE c.probability > $threshold
          AND c.tca > datetime()
          AND c.tca < datetime() + duration({days: $days_ahead})
        MATCH (c)-[:INVOLVES]->(obj)
        RETURN c, collect(obj) as objects
        ORDER BY c.probability DESC
        LIMIT $limit
    """,

    'find_satellites_in_danger': """
        MATCH (s:Satellite)-[r:AT_RISK]->(d:Debris)
        WHERE r.level IN ['HIGH', 'CRITICAL']
        OPTIONAL MATCH (s)-[:HAS_ORBIT]->(o:Orbit)
        RETURN s, o, count(d) as debris_count
        ORDER BY debris_count DESC
    """,

    'find_conjunction_clusters': """
        MATCH (c:Conjunction)
        WHERE c.tca > datetime()
          AND c.tca < datetime() + duration({hours: 24})
        MATCH (c)-[:INVOLVES]->(s:Satellite)
        MATCH (c)-[:INVOLVES]->(d:Debris)
        RETURN s.name as satellite,
               count(c) as conjunction_count,
               max(c.probability) as max_risk
        ORDER BY conjunction_count DESC
        LIMIT 10
    """,

    'analyze_debris_population': """
        MATCH (d:Debris)-[:ORIGINATED_FROM]->(s:Satellite)
        RETURN s.name as origin,
               count(d) as debris_count,
               d.size_category as size
        ORDER BY debris_count DESC
    """,

    'track_maneuver_effectiveness': """
        MATCH (m:Maneuver)-[a:AVOIDED]->(c:Conjunction)
        WHERE m.execution_time > datetime() - duration({days: 30})
        RETURN avg(a.improvement) as avg_improvement,
               count(m) as total_maneuvers,
               sum(m.fuel_cost) as total_fuel
    """
}


if __name__ == "__main__":
    schema = GraphSchema()

    print("Space Knowledge Graph Schema")
    print("=" * 50)
    print("\nNode Types:")
    for attr in dir(schema):
        if attr.endswith('_SCHEMA') and not attr.startswith('_'):
            print(f"  {attr.replace('_SCHEMA', '')}")

    print("\nSample Queries:")
    for name, query in QUERIES.items():
        print(f"  - {name}")
