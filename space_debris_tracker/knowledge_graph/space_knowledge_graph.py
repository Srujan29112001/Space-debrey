"""
Space Knowledge Graph
Neo4j-based graph for satellites, debris, orbits, and conjunctions
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not available. Install with: pip install neo4j")


class SpaceKnowledgeGraph:
    """
    Neo4j-based knowledge graph for space situational awareness
    """

    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "space_catalog"):
        """
        Initialize Space Knowledge Graph

        Args:
            uri: Neo4j URI
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.database = database

        # Connect to Neo4j
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self._create_schema()
                print(f"Connected to Neo4j at {uri}")
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}")
                self.driver = None
        else:
            self.driver = None

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def _create_schema(self):
        """Create graph schema with constraints and indices"""
        if not self.driver:
            return

        queries = [
            # Constraints
            """
            CREATE CONSTRAINT satellite_id IF NOT EXISTS
            FOR (s:Satellite) REQUIRE s.norad_id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT debris_id IF NOT EXISTS
            FOR (d:Debris) REQUIRE d.object_id IS UNIQUE
            """,

            # Indices
            """
            CREATE INDEX satellite_name IF NOT EXISTS
            FOR (s:Satellite) ON (s.name)
            """,

            """
            CREATE INDEX orbit_altitude IF NOT EXISTS
            FOR (o:Orbit) ON (o.semi_major_axis)
            """,

            """
            CREATE INDEX conjunction_tca IF NOT EXISTS
            FOR (c:Conjunction) ON (c.tca)
            """,
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    # Constraint/index might already exist
                    pass

    def add_satellite(self, satellite_data: Dict) -> Dict:
        """
        Add satellite to graph

        Args:
            satellite_data: Satellite information

        Returns:
            Created satellite node
        """
        if not self.driver:
            return {}

        query = """
        MERGE (s:Satellite {norad_id: $norad_id})
        SET s.name = $name,
            s.international_designator = $intl_des,
            s.operator = $operator,
            s.launch_date = datetime($launch_date),
            s.mass = $mass,
            s.status = $status,
            s.updated = datetime()
        RETURN s
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                norad_id=satellite_data['norad_id'],
                name=satellite_data.get('name', ''),
                intl_des=satellite_data.get('international_designator', ''),
                operator=satellite_data.get('operator', ''),
                launch_date=satellite_data.get('launch_date', datetime.now().isoformat()),
                mass=satellite_data.get('mass', 0.0),
                status=satellite_data.get('status', 'ACTIVE')
            )

            record = result.single()
            return dict(record['s']) if record else {}

    def add_orbit(self, norad_id: int, orbit_data: Dict) -> Dict:
        """
        Add orbit to satellite

        Args:
            norad_id: Satellite NORAD ID
            orbit_data: Orbital elements

        Returns:
            Created orbit relationship
        """
        if not self.driver:
            return {}

        query = """
        MATCH (s:Satellite {norad_id: $norad_id})
        CREATE (o:Orbit {
            semi_major_axis: $a,
            eccentricity: $e,
            inclination: $i,
            raan: $omega,
            arg_perigee: $w,
            mean_anomaly: $M,
            epoch: datetime($epoch)
        })
        CREATE (s)-[r:HAS_ORBIT]->(o)
        RETURN o, r
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                norad_id=norad_id,
                a=orbit_data['semi_major_axis'],
                e=orbit_data['eccentricity'],
                i=orbit_data['inclination'],
                omega=orbit_data['raan'],
                w=orbit_data['arg_perigee'],
                M=orbit_data['mean_anomaly'],
                epoch=orbit_data.get('epoch', datetime.now().isoformat())
            )

            record = result.single()
            return dict(record['o']) if record else {}

    def add_debris(self, debris_data: Dict) -> Dict:
        """
        Add debris object to graph

        Args:
            debris_data: Debris information

        Returns:
            Created debris node
        """
        if not self.driver:
            return {}

        query = """
        MERGE (d:Debris {object_id: $object_id})
        SET d.size_category = $size,
            d.origin = $origin,
            d.detection_date = datetime($date),
            d.rcs = $rcs,
            d.updated = datetime()
        RETURN d
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                object_id=debris_data['object_id'],
                size=debris_data.get('size_category', 'UNKNOWN'),
                origin=debris_data.get('origin', ''),
                date=debris_data.get('detection_date', datetime.now().isoformat()),
                rcs=debris_data.get('rcs', 0.0)
            )

            record = result.single()
            return dict(record['d']) if record else {}

    def update_conjunction_assessment(self,
                                     obj1_id: int,
                                     obj2_id: str,
                                     prediction: Dict) -> Dict:
        """
        Update conjunction assessment

        Args:
            obj1_id: Primary object NORAD ID
            obj2_id: Secondary object ID
            prediction: Conjunction prediction data

        Returns:
            Updated conjunction
        """
        if not self.driver:
            return {}

        query = """
        MATCH (s:Satellite {norad_id: $obj1})
        MATCH (d:Debris {object_id: $obj2})

        MERGE (c:Conjunction {
            primary_object: $obj1,
            secondary_object: $obj2,
            tca: datetime($tca)
        })
        SET c.miss_distance = $distance,
            c.probability = $probability,
            c.relative_velocity = $rel_vel,
            c.updated = datetime(),
            c.radial_separation = $radial,
            c.in_track_separation = $in_track,
            c.cross_track_separation = $cross_track

        MERGE (c)-[:INVOLVES]->(s)
        MERGE (c)-[:INVOLVES]->(d)

        MERGE (s)-[r:AT_RISK]->(d)
        SET r.level = CASE
            WHEN $probability > 0.001 THEN 'HIGH'
            WHEN $probability > 0.0001 THEN 'MEDIUM'
            ELSE 'LOW'
        END,
        r.updated = datetime()

        RETURN c, r
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                obj1=obj1_id,
                obj2=obj2_id,
                tca=prediction['tca'],
                distance=prediction['miss_distance'],
                probability=prediction['collision_probability'],
                rel_vel=prediction.get('relative_velocity', 0.0),
                radial=prediction.get('radial_separation', 0.0),
                in_track=prediction.get('in_track_separation', 0.0),
                cross_track=prediction.get('cross_track_separation', 0.0)
            )

            record = result.single()
            return dict(record['c']) if record else {}

    def get_historical_conjunctions(self,
                                   satellite_id: int,
                                   days_back: int = 30) -> List[Dict]:
        """
        Get historical conjunctions for satellite

        Args:
            satellite_id: NORAD ID
            days_back: Days to look back

        Returns:
            List of historical conjunctions
        """
        if not self.driver:
            return []

        query = """
        MATCH (s:Satellite {norad_id: $sat_id})
        MATCH (c:Conjunction)-[:INVOLVES]->(s)
        WHERE c.tca > datetime() - duration({days: $days})
        OPTIONAL MATCH (c)-[:INVOLVES]->(other)
        WHERE other <> s
        RETURN c,
               other,
               c.miss_distance as miss_distance,
               c.probability as probability
        ORDER BY c.tca DESC
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                sat_id=satellite_id,
                days=days_back
            )

            conjunctions = []
            for record in result:
                conjunctions.append({
                    'conjunction': dict(record['c']),
                    'other_object': dict(record['other']) if record['other'] else None,
                    'miss_distance': record['miss_distance'],
                    'probability': record['probability']
                })

            return conjunctions

    def get_high_risk_satellites(self, threshold: float = 0.0001) -> List[Dict]:
        """
        Get satellites with high collision risk

        Args:
            threshold: Probability threshold

        Returns:
            List of high-risk satellites
        """
        if not self.driver:
            return []

        query = """
        MATCH (s:Satellite)-[r:AT_RISK]->(d:Debris)
        WHERE r.level IN ['HIGH', 'MEDIUM']
        MATCH (c:Conjunction)-[:INVOLVES]->(s)
        WHERE c.probability > $threshold
        RETURN s,
               count(c) as conjunction_count,
               max(c.probability) as max_probability,
               min(c.miss_distance) as min_distance
        ORDER BY max_probability DESC
        LIMIT 100
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, threshold=threshold)

            satellites = []
            for record in result:
                satellites.append({
                    'satellite': dict(record['s']),
                    'conjunction_count': record['conjunction_count'],
                    'max_probability': record['max_probability'],
                    'min_distance': record['min_distance']
                })

            return satellites

    def find_debris_by_origin(self, origin_satellite: int) -> List[Dict]:
        """
        Find debris originating from satellite

        Args:
            origin_satellite: Origin satellite NORAD ID

        Returns:
            List of debris objects
        """
        if not self.driver:
            return []

        query = """
        MATCH (s:Satellite {norad_id: $sat_id})
        MATCH (d:Debris)-[:ORIGINATED_FROM]->(s)
        RETURN d
        ORDER BY d.detection_date DESC
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, sat_id=origin_satellite)

            debris_list = []
            for record in result:
                debris_list.append(dict(record['d']))

            return debris_list

    def get_satellites_in_orbit_range(self,
                                     altitude_min: float,
                                     altitude_max: float) -> List[Dict]:
        """
        Get satellites in altitude range

        Args:
            altitude_min: Minimum altitude (km)
            altitude_max: Maximum altitude (km)

        Returns:
            List of satellites
        """
        if not self.driver:
            return []

        query = """
        MATCH (s:Satellite)-[:HAS_ORBIT]->(o:Orbit)
        WHERE o.semi_major_axis >= $a_min AND o.semi_major_axis <= $a_max
        RETURN s, o
        ORDER BY o.semi_major_axis
        """

        # Convert altitude to semi-major axis (approximate)
        Re = 6378.137  # Earth radius km
        a_min = Re + altitude_min
        a_max = Re + altitude_max

        with self.driver.session(database=self.database) as session:
            result = session.run(query, a_min=a_min, a_max=a_max)

            satellites = []
            for record in result:
                satellites.append({
                    'satellite': dict(record['s']),
                    'orbit': dict(record['o'])
                })

            return satellites


if __name__ == "__main__":
    # Example usage
    kg = SpaceKnowledgeGraph()

    # Add satellite
    satellite = {
        'norad_id': 25544,
        'name': 'ISS (ZARYA)',
        'operator': 'ISS',
        'launch_date': '1998-11-20',
        'mass': 419700,
        'status': 'ACTIVE'
    }

    result = kg.add_satellite(satellite)
    print(f"Added satellite: {result.get('name', 'N/A')}")

    # Add orbit
    orbit = {
        'semi_major_axis': 6778.0,
        'eccentricity': 0.0001,
        'inclination': 51.6,
        'raan': 0.0,
        'arg_perigee': 0.0,
        'mean_anomaly': 0.0,
        'epoch': datetime.now().isoformat()
    }

    orbit_result = kg.add_orbit(25544, orbit)
    print(f"Added orbit with SMA: {orbit['semi_major_axis']} km")

    # Close connection
    kg.close()
