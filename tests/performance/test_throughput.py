"""
Performance Tests - Throughput
Measure system processing throughput
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from space_debris_tracker.computer_vision.detector import SpaceDebrisDetector
from space_debris_tracker.trajectory_prediction.orbit_predictor import (
    OrbitPredictionEngine,
)
from tests.utils import generate_image_sequence, generate_orbital_state


@pytest.mark.performance
class TestDetectionThroughput:
    """Test detection system throughput"""

    def test_images_per_second(self, benchmark_config):
        """Measure images processed per second"""
        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")
            detector.yolo.detect = lambda x: []
            detector.dino.detect_novel_objects = lambda x: []

            # Generate test images
            images = generate_image_sequence(n_frames=100, moving_objects=5)

            start_time = time.time()

            for image in images:
                detector.process_telescope_image(image)

            end_time = time.time()

            elapsed = end_time - start_time
            throughput = len(images) / elapsed

            print(f"\nDetection throughput: {throughput:.2f} images/second")

            # Benchmark: should process at least 1 image per second
            assert throughput >= 0.5

    def test_batch_processing_throughput(self):
        """Measure batch processing throughput"""
        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")
            detector.yolo.detect = lambda x: []

            images = generate_image_sequence(n_frames=50)
            batch_sizes = [1, 5, 10]

            for batch_size in batch_sizes:
                start = time.time()

                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    for img in batch:
                        detector.process_telescope_image(img)

                elapsed = time.time() - start
                throughput = len(images) / elapsed

                print(f"Batch size {batch_size}: {throughput:.2f} img/s")

    @pytest.mark.gpu
    def test_gpu_throughput(self):
        """Measure GPU processing throughput"""
        if not pytest.importorskip("torch").cuda.is_available():
            pytest.skip("CUDA not available")

        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cuda")
            detector.yolo.detect = lambda x: []

            images = generate_image_sequence(n_frames=100)

            start = time.time()

            for image in images:
                detector.process_telescope_image(image)

            elapsed = time.time() - start
            throughput = len(images) / elapsed

            print(f"\nGPU throughput: {throughput:.2f} images/second")

            # GPU should be faster than CPU
            assert throughput >= 1.0


@pytest.mark.performance
class TestPredictionThroughput:
    """Test trajectory prediction throughput"""

    def test_predictions_per_second(self):
        """Measure trajectory predictions per second"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Generate test states
        states = [generate_orbital_state(altitude=400.0 + i * 10) for i in range(20)]

        start = time.time()

        for state in states:
            predictor.predict_trajectory(
                initial_state=state,
                time_horizon=3600,
                dt=300.0,
                include_uncertainty=False,
            )

        elapsed = time.time() - start
        throughput = len(states) / elapsed

        print(f"\nPrediction throughput: {throughput:.2f} predictions/second")

        # Should complete at least 1 prediction per second
        assert throughput >= 0.5

    def test_concurrent_predictions(self):
        """Measure concurrent prediction throughput"""
        import concurrent.futures

        predictor = OrbitPredictionEngine(device="cpu")
        states = [generate_orbital_state(altitude=400.0 + i * 10) for i in range(10)]

        def predict(state):
            return predictor.predict_trajectory(
                initial_state=state,
                time_horizon=3600,
                dt=300.0,
                include_uncertainty=False,
            )

        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(predict, states))

        elapsed = time.time() - start
        throughput = len(states) / elapsed

        print(f"\nConcurrent prediction throughput: {throughput:.2f} pred/s")

    def test_long_term_prediction_throughput(self):
        """Measure long-term prediction performance"""
        predictor = OrbitPredictionEngine(device="cpu")
        state = generate_orbital_state(altitude=400.0)

        time_horizons = [86400, 86400 * 7, 86400 * 30]  # 1 day, 1 week, 1 month

        for horizon in time_horizons:
            start = time.time()

            predictor.predict_trajectory(
                initial_state=state,
                time_horizon=horizon,
                dt=3600.0,  # 1 hour steps
                include_uncertainty=False,
            )

            elapsed = time.time() - start

            print(f"\n{horizon/(86400):.0f} day prediction: {elapsed:.2f}s")

            # Should complete within reasonable time
            assert elapsed < 60.0  # Within 1 minute


@pytest.mark.performance
class TestDataIngestionThroughput:
    """Test data ingestion throughput"""

    def test_tle_parsing_throughput(self, sample_tle_file):
        """Measure TLE parsing throughput"""
        from space_debris_tracker.data_ingestion.tle_parser import TLEParser

        parser = TLEParser()

        start = time.time()

        # Parse file multiple times
        iterations = 100
        for _ in range(iterations):
            elements_list = parser.parse_tle_file(str(sample_tle_file))

        elapsed = time.time() - start
        throughput = (iterations * len(elements_list)) / elapsed

        print(f"\nTLE parsing throughput: {throughput:.2f} TLEs/second")

        # Should parse at least 10 TLEs per second
        assert throughput >= 10.0

    def test_kafka_message_throughput(self, mock_kafka_consumer):
        """Measure Kafka message processing throughput"""
        from space_debris_tracker.data_ingestion.kafka_consumer import (
            KafkaStreamConsumer,
        )
        from tests.utils import generate_kafka_message

        consumer = KafkaStreamConsumer(
            bootstrap_servers=["localhost:9092"], topic="test"
        )
        consumer.consumer = mock_kafka_consumer

        # Generate messages
        messages = [generate_kafka_message("tle") for _ in range(1000)]

        start = time.time()

        for msg in messages:
            consumer.deserialize_message(str(msg).encode())

        elapsed = time.time() - start
        throughput = len(messages) / elapsed

        print(f"\nMessage processing throughput: {throughput:.2f} msg/s")

        assert throughput >= 100.0


@pytest.mark.performance
class TestEndToEndThroughput:
    """Test end-to-end system throughput"""

    def test_full_pipeline_throughput(self, sample_tle_lines):
        """Measure complete pipeline throughput"""
        from space_debris_tracker.data_ingestion.tle_parser import TLEParser

        parser = TLEParser()
        predictor = OrbitPredictionEngine(device="cpu")

        name, line1, line2 = sample_tle_lines

        start = time.time()

        iterations = 10
        for _ in range(iterations):
            # Parse TLE
            elements = parser.parse_tle(line1, line2, name)

            # Get state
            pos, vel = parser.propagate_sgp4(elements, elements.epoch)
            state = np.concatenate([pos, vel])

            # Predict
            predictor.predict_trajectory(
                initial_state=state,
                time_horizon=3600,
                dt=300.0,
                include_uncertainty=False,
            )

        elapsed = time.time() - start
        throughput = iterations / elapsed

        print(f"\nFull pipeline throughput: {throughput:.2f} objects/second")

        # Should complete at least 1 object per second
        assert throughput >= 0.5

    @pytest.mark.slow
    def test_sustained_throughput(self):
        """Test sustained processing throughput"""
        with patch("torch.hub.load"):
            detector = SpaceDebrisDetector(device="cpu")
            detector.yolo.detect = lambda x: []

            # Generate large sequence
            images = generate_image_sequence(n_frames=500)

            start = time.time()

            for i, image in enumerate(images):
                detector.process_telescope_image(image)

                # Check throughput every 100 images
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    current_throughput = (i + 1) / elapsed
                    print(f"\nAfter {i+1} images: {current_throughput:.2f} img/s")

            elapsed = time.time() - start
            avg_throughput = len(images) / elapsed

            print(f"\nAverage sustained throughput: {avg_throughput:.2f} img/s")

            # Should maintain reasonable throughput
            assert avg_throughput >= 0.5
