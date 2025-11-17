"""
Unit Tests for Space Knowledge Graph
Tests Neo4j graph operations
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from space_debris_tracker.knowledge_graph.space_knowledge_graph import (
    SpaceKnowledgeGraph,
)


@pytest.mark.unit
@pytest.mark.neo4j
class TestSpaceKnowledgeGraph:
    """Test knowledge graph operations"""

    def test_kg_initialization(self, mock_neo4j_driver):
        """Test knowledge graph initialization"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password="test")

            assert kg is not None
            assert kg.uri == "bolt://localhost:7687"

    def test_add_satellite(self, mock_neo4j_driver):
        """Test adding satellite to graph"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            satellite_data = {
                "norad_id": 25544,
                "name": "ISS (ZARYA)",
                "operator": "ISS",
                "launch_date": "1998-11-20",
                "mass": 419700,
                "status": "ACTIVE",
            }

            result = kg.add_satellite(satellite_data)

            assert isinstance(result, dict)

    def test_add_orbit(self, mock_neo4j_driver):
        """Test adding orbit to satellite"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            orbit_data = {
                "semi_major_axis": 6778.0,
                "eccentricity": 0.0001,
                "inclination": 51.6,
                "raan": 0.0,
                "arg_perigee": 0.0,
                "mean_anomaly": 0.0,
                "epoch": datetime.now().isoformat(),
            }

            result = kg.add_orbit(25544, orbit_data)

            assert isinstance(result, dict)

    def test_add_debris(self, mock_neo4j_driver):
        """Test adding debris to graph"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            debris_data = {
                "object_id": "DEBRIS_12345",
                "size_category": "MEDIUM",
                "origin": "COLLISION",
                "detection_date": datetime.now().isoformat(),
                "rcs": 0.5,
            }

            result = kg.add_debris(debris_data)

            assert isinstance(result, dict)

    def test_update_conjunction_assessment(self, mock_neo4j_driver):
        """Test updating conjunction assessment"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            prediction = {
                "tca": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                "miss_distance": 2.5,
                "collision_probability": 0.00015,
                "relative_velocity": 10.2,
            }

            result = kg.update_conjunction_assessment(25544, "DEBRIS_12345", prediction)

            assert isinstance(result, dict)

    def test_get_historical_conjunctions(self, mock_neo4j_driver):
        """Test retrieving historical conjunctions"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            conjunctions = kg.get_historical_conjunctions(25544, days_back=30)

            assert isinstance(conjunctions, list)

    def test_get_high_risk_satellites(self, mock_neo4j_driver):
        """Test retrieving high-risk satellites"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            satellites = kg.get_high_risk_satellites(threshold=0.0001)

            assert isinstance(satellites, list)

    def test_find_debris_by_origin(self, mock_neo4j_driver):
        """Test finding debris by origin satellite"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            debris_list = kg.find_debris_by_origin(25544)

            assert isinstance(debris_list, list)

    def test_get_satellites_in_orbit_range(self, mock_neo4j_driver):
        """Test getting satellites in altitude range"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            satellites = kg.get_satellites_in_orbit_range(altitude_min=350.0, altitude_max=450.0)

            assert isinstance(satellites, list)

    def test_close_connection(self, mock_neo4j_driver):
        """Test closing Neo4j connection"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()
            kg.close()

            # Should call driver.close()
            if kg.driver:
                kg.driver.close.assert_called_once()

    def test_no_driver_fallback(self):
        """Test fallback when Neo4j not available"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.NEO4J_AVAILABLE",
            False,
        ):
            kg = SpaceKnowledgeGraph()

            assert kg.driver is None

            # Operations should return empty results
            result = kg.add_satellite({"norad_id": 25544, "name": "TEST"})
            assert result == {}

    def test_schema_creation(self, mock_neo4j_driver):
        """Test schema creation"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            # _create_schema should have been called during init
            # Verify session was created
            assert mock_neo4j_driver.session.called

    @pytest.mark.parametrize(
        "altitude_min,altitude_max",
        [(350, 450), (10000, 12000), (35000, 36000)],  # LEO  # MEO  # GEO
    )
    def test_orbit_range_queries(self, mock_neo4j_driver, altitude_min, altitude_max):
        """Test orbit range queries for different regimes"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            satellites = kg.get_satellites_in_orbit_range(altitude_min, altitude_max)

            assert isinstance(satellites, list)


@pytest.mark.unit
class TestKnowledgeGraphQueries:
    """Test graph query construction"""

    def test_satellite_query_format(self, mock_neo4j_driver):
        """Test satellite query format"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            satellite_data = {
                "norad_id": 25544,
                "name": "ISS",
                "operator": "ISS",
                "launch_date": "1998-11-20",
                "mass": 419700,
                "status": "ACTIVE",
            }

            # Add satellite
            kg.add_satellite(satellite_data)

            # Verify query was run
            assert mock_neo4j_driver.session.called

    def test_conjunction_query_format(self, mock_neo4j_driver):
        """Test conjunction query format"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            kg = SpaceKnowledgeGraph()

            prediction = {
                "tca": datetime.utcnow().isoformat(),
                "miss_distance": 2.5,
                "collision_probability": 0.00015,
                "relative_velocity": 10.2,
                "radial_separation": 1.0,
                "in_track_separation": 1.5,
                "cross_track_separation": 1.2,
            }

            kg.update_conjunction_assessment(25544, "DEBRIS_12345", prediction)

            assert mock_neo4j_driver.session.called
