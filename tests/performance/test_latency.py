"""
Performance Tests - Latency
Measure system response latency
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from space_debris_tracker.computer_vision.detector import SpaceDebrisDetector
from space_debris_tracker.trajectory_prediction.orbit_predictor import (
    OrbitPredictionEngine,
)
from tests.utils import Timer, generate_orbital_state, generate_space_image


@pytest.mark.performance
class TestDetectionLatency:
    """Test detection latency"""

    def test_single_image_latency(self, sample_telescope_image):
        """Measure single image detection latency"""
        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")
            detector.yolo.detect = lambda x: []

            latencies = []

            for _ in range(10):
                with Timer() as timer:
                    detector.process_telescope_image(sample_telescope_image)

                latencies.append(timer.elapsed)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            print(f"\nDetection latency:")
            print(f"  Average: {avg_latency*1000:.2f}ms")
            print(f"  P95: {p95_latency*1000:.2f}ms")
            print(f"  P99: {p99_latency*1000:.2f}ms")

            # Should process within reasonable time
            assert avg_latency < 5.0  # 5 seconds
            assert p99_latency < 10.0  # 10 seconds

    def test_preprocessing_latency(self, sample_telescope_image):
        """Measure preprocessing latency"""
        from space_debris_tracker.computer_vision.preprocessing.space_image import (
            SpaceImagePreprocessor,
        )

        preprocessor = SpaceImagePreprocessor()

        with Timer() as timer:
            preprocessor.preprocess(sample_telescope_image)

        print(f"\nPreprocessing latency: {timer.elapsed*1000:.2f}ms")

        # Preprocessing should be fast
        assert timer.elapsed < 1.0

    def test_detection_breakdown(self, sample_telescope_image):
        """Measure latency breakdown"""
        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")

            # Mock individual components
            yolo_times = []
            dino_times = []
            tracking_times = []

            def mock_yolo(img):
                with Timer() as t:
                    time.sleep(0.01)  # Simulate processing
                yolo_times.append(t.elapsed)
                return []

            def mock_dino(img):
                with Timer() as t:
                    time.sleep(0.005)
                dino_times.append(t.elapsed)
                return []

            detector.yolo.detect = mock_yolo
            detector.dino.detect_novel_objects = mock_dino

            detector.process_telescope_image(sample_telescope_image)

            print(f"\nLatency breakdown:")
            if yolo_times:
                print(f"  YOLO: {np.mean(yolo_times)*1000:.2f}ms")
            if dino_times:
                print(f"  DINO: {np.mean(dino_times)*1000:.2f}ms")


@pytest.mark.performance
class TestPredictionLatency:
    """Test trajectory prediction latency"""

    def test_short_term_prediction_latency(self, sample_state_vector):
        """Measure short-term prediction latency"""
        predictor = OrbitPredictionEngine(device="cpu")

        latencies = []

        for _ in range(10):
            with Timer() as timer:
                predictor.predict_trajectory(
                    initial_state=sample_state_vector,
                    time_horizon=3600,
                    dt=60.0,
                    include_uncertainty=False,
                )

            latencies.append(timer.elapsed)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\nShort-term prediction latency:")
        print(f"  Average: {avg_latency*1000:.2f}ms")
        print(f"  P95: {p95_latency*1000:.2f}ms")

        # Should complete quickly
        assert avg_latency < 2.0

    def test_collision_assessment_latency(self):
        """Measure collision assessment latency"""
        predictor = OrbitPredictionEngine(device="cpu")

        state1 = generate_orbital_state(altitude=400.0)
        state2 = generate_orbital_state(altitude=405.0)

        predictor.add_nearby_object(state2)

        with Timer() as timer:
            predictor.predict_trajectory(
                initial_state=state1, time_horizon=3600, dt=60.0
            )

        print(f"\nCollision assessment latency: {timer.elapsed*1000:.2f}ms")

        assert timer.elapsed < 5.0

    def test_propagation_step_latency(self, sample_state_vector):
        """Measure single propagation step latency"""
        from space_debris_tracker.trajectory_prediction.physics.orbital_mechanics import (
            OrbitalMechanics,
        )

        mechanics = OrbitalMechanics()

        position = sample_state_vector[:3]
        velocity = sample_state_vector[3:6]

        latencies = []

        for _ in range(100):
            with Timer() as timer:
                mechanics.propagate(
                    position, velocity, dt=60.0, include_j2=True, include_drag=True
                )

            latencies.append(timer.elapsed)

        avg_latency = np.mean(latencies)

        print(f"\nPropagation step latency: {avg_latency*1000:.4f}ms")

        # Each step should be very fast
        assert avg_latency < 0.01  # 10ms


@pytest.mark.performance
class TestAPILatency:
    """Test API response latency"""

    def test_satellite_query_latency(self, api_test_client):
        """Measure satellite query latency"""
        latencies = []

        for _ in range(10):
            start = time.time()
            response = api_test_client.get("/api/satellites/25544")
            latency = time.time() - start

            assert response.status_code == 200
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\nAPI query latency:")
        print(f"  Average: {avg_latency*1000:.2f}ms")
        print(f"  P95: {p95_latency*1000:.2f}ms")

        # API should respond quickly
        assert avg_latency < 1.0  # 1 second
        assert p95_latency < 2.0

    def test_trajectory_request_latency(self, api_test_client):
        """Measure trajectory request latency"""
        with Timer() as timer:
            response = api_test_client.get(
                "/api/satellites/25544/trajectory?time_horizon=3600"
            )

        assert response.status_code == 200

        print(f"\nTrajectory request latency: {timer.elapsed*1000:.2f}ms")

        # Should complete within reasonable time
        assert timer.elapsed < 5.0

    def test_conjunction_assessment_latency(self, api_test_client):
        """Measure conjunction assessment latency"""
        with Timer() as timer:
            response = api_test_client.post(
                "/api/conjunctions/assess",
                json={
                    "primary_id": 25544,
                    "secondary_id": "DEBRIS_12345",
                    "time_horizon": 604800,
                },
            )

        assert response.status_code == 200

        print(f"\nConjunction assessment latency: {timer.elapsed*1000:.2f}ms")

        assert timer.elapsed < 10.0


@pytest.mark.performance
class TestDatabaseLatency:
    """Test database query latency"""

    def test_neo4j_query_latency(self, mock_neo4j_driver):
        """Measure Neo4j query latency"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            from space_debris_tracker.knowledge_graph.space_knowledge_graph import (
                SpaceKnowledgeGraph,
            )

            kg = SpaceKnowledgeGraph()

            with Timer() as timer:
                kg.get_high_risk_satellites(threshold=0.0001)

            print(f"\nNeo4j query latency: {timer.elapsed*1000:.2f}ms")

            # Database queries should be fast
            assert timer.elapsed < 1.0

    def test_satellite_lookup_latency(self, mock_neo4j_driver):
        """Measure satellite lookup latency"""
        with patch(
            "space_debris_tracker.knowledge_graph.space_knowledge_graph.GraphDatabase"
        ) as mock_gdb:
            mock_gdb.driver.return_value = mock_neo4j_driver

            from space_debris_tracker.knowledge_graph.space_knowledge_graph import (
                SpaceKnowledgeGraph,
            )

            kg = SpaceKnowledgeGraph()

            latencies = []

            for i in range(10):
                with Timer() as timer:
                    kg.add_satellite(
                        {"norad_id": 25544 + i, "name": f"SAT_{i}", "operator": "TEST"}
                    )

                latencies.append(timer.elapsed)

            avg_latency = np.mean(latencies)

            print(f"\nSatellite lookup latency: {avg_latency*1000:.2f}ms")

            assert avg_latency < 0.5


@pytest.mark.performance
class TestLatencyUnderLoad:
    """Test latency under load"""

    def test_detection_latency_under_load(self):
        """Measure detection latency under concurrent load"""
        import concurrent.futures

        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")
            detector.yolo.detect = lambda x: []

            images = [generate_space_image()[0] for _ in range(20)]

            def process_image(img):
                start = time.time()
                detector.process_telescope_image(img)
                return time.time() - start

            # Sequential processing
            sequential_latencies = [process_image(img) for img in images[:5]]

            # Concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                concurrent_latencies = list(executor.map(process_image, images))

            seq_avg = np.mean(sequential_latencies)
            conc_avg = np.mean(concurrent_latencies)

            print(f"\nLatency under load:")
            print(f"  Sequential: {seq_avg*1000:.2f}ms")
            print(f"  Concurrent: {conc_avg*1000:.2f}ms")

            # Concurrent latency should not degrade significantly
            assert conc_avg < seq_avg * 3  # Allow 3x degradation
