"""
Kafka Consumers for Space Debris Data
Handles telescope images, radar returns, and streaming data
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import msgpack
import base64
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class TelescopeImage:
    """Telescope image data structure"""
    timestamp: datetime
    telescope_id: str
    ra: float  # Right Ascension (degrees)
    dec: float  # Declination (degrees)
    exposure_time: float  # seconds
    filter_type: str  # B, V, R, I bands
    image_data: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class RadarReturn:
    """Radar return data structure"""
    timestamp: datetime
    radar_id: str
    azimuth: float  # degrees
    elevation: float  # degrees
    range: float  # km
    range_rate: float  # km/s
    doppler_shift: float  # Hz
    rcs: float  # Radar Cross Section (m^2)
    snr: float  # Signal-to-Noise Ratio (dB)
    metadata: Dict[str, Any]


class TelescopeImageConsumer:
    """
    Kafka consumer for telescope images
    Consumes high-resolution images from optical telescopes
    """

    def __init__(self,
                 bootstrap_servers: List[str],
                 topic: str = 'telescope-images',
                 group_id: str = 'debris-detector',
                 max_workers: int = 4):
        """
        Initialize telescope image consumer

        Args:
            bootstrap_servers: List of Kafka brokers
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            max_workers: Number of worker threads for processing
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'bytes_received': 0,
            'processing_time_total': 0.0,
        }

        # Create consumer
        self.consumer = None
        self._running = False
        self._callbacks: List[Callable[[TelescopeImage], None]] = []

    def register_callback(self, callback: Callable[[TelescopeImage], None]):
        """Register callback for processed images"""
        self._callbacks.append(callback)

    def start(self):
        """Start consuming messages"""
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=self._deserialize_message,
            max_poll_records=10,  # Process in small batches
            fetch_max_bytes=52428800,  # 50 MB
            max_partition_fetch_bytes=10485760,  # 10 MB per partition
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        self._running = True
        logger.info(f"Started consuming from topic: {self.topic}")

    def stop(self):
        """Stop consuming messages"""
        self._running = False
        if self.consumer:
            self.consumer.close()
        self.executor.shutdown(wait=True)
        logger.info("Stopped telescope image consumer")

    def consume(self, timeout_ms: int = 1000):
        """
        Consume messages from Kafka

        Args:
            timeout_ms: Poll timeout in milliseconds
        """
        if not self.consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        while self._running:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=timeout_ms)

                for topic_partition, records in messages.items():
                    for record in records:
                        self.stats['messages_received'] += 1
                        self.stats['bytes_received'] += len(record.value) if isinstance(record.value, bytes) else 0

                        # Process in thread pool
                        future = self.executor.submit(self._process_message, record.value)
                        future.add_done_callback(self._process_callback)

            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
                self.stats['messages_failed'] += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.stats['messages_failed'] += 1

    def _process_message(self, message: Dict) -> TelescopeImage:
        """Process raw message into TelescopeImage"""
        start_time = time.time()

        try:
            # Extract metadata
            timestamp = datetime.fromisoformat(message['timestamp'])
            telescope_id = message['telescope_id']
            ra = message['ra']
            dec = message['dec']
            exposure_time = message['exposure_time']
            filter_type = message.get('filter', 'R')

            # Decode image data
            if 'image_base64' in message:
                # Base64 encoded image
                image_bytes = base64.b64decode(message['image_base64'])
                image_array = np.frombuffer(image_bytes, dtype=np.uint16)
                height = message['height']
                width = message['width']
                image_data = image_array.reshape((height, width))
            elif 'image_url' in message:
                # Download from URL
                import urllib.request
                response = urllib.request.urlopen(message['image_url'])
                image_bytes = response.read()
                image_data = cv2.imdecode(np.frombuffer(image_bytes, np.uint8),
                                         cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError("No image data found in message")

            # Create TelescopeImage object
            telescope_image = TelescopeImage(
                timestamp=timestamp,
                telescope_id=telescope_id,
                ra=ra,
                dec=dec,
                exposure_time=exposure_time,
                filter_type=filter_type,
                image_data=image_data,
                metadata=message.get('metadata', {})
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats['processing_time_total'] += processing_time
            self.stats['messages_processed'] += 1

            logger.debug(f"Processed telescope image from {telescope_id} "
                        f"in {processing_time:.3f}s")

            return telescope_image

        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            self.stats['messages_failed'] += 1
            raise

    def _process_callback(self, future):
        """Callback when message processing is complete"""
        try:
            telescope_image = future.result()

            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(telescope_image)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error in process callback: {e}")

    @staticmethod
    def _deserialize_message(raw_message: bytes) -> Dict:
        """Deserialize Kafka message"""
        try:
            # Try MessagePack first (more efficient for binary data)
            return msgpack.unpackb(raw_message, raw=False)
        except:
            # Fall back to JSON
            return json.loads(raw_message.decode('utf-8'))

    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        avg_processing_time = (self.stats['processing_time_total'] /
                              self.stats['messages_processed']
                              if self.stats['messages_processed'] > 0 else 0)

        return {
            **self.stats,
            'avg_processing_time': avg_processing_time,
            'throughput_mbps': (self.stats['bytes_received'] / 1024 / 1024 /
                               max(self.stats['processing_time_total'], 1))
        }


class RadarDataConsumer:
    """
    Kafka consumer for radar returns
    Processes phased array radar detections
    """

    def __init__(self,
                 bootstrap_servers: List[str],
                 topic: str = 'radar-returns',
                 group_id: str = 'debris-tracker',
                 max_workers: int = 4):
        """
        Initialize radar data consumer

        Args:
            bootstrap_servers: List of Kafka brokers
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            max_workers: Number of worker threads
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'detections': 0,
        }

        self.consumer = None
        self._running = False
        self._callbacks: List[Callable[[RadarReturn], None]] = []

    def register_callback(self, callback: Callable[[RadarReturn], None]):
        """Register callback for processed radar returns"""
        self._callbacks.append(callback)

    def start(self):
        """Start consuming messages"""
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        self._running = True
        logger.info(f"Started consuming from topic: {self.topic}")

    def stop(self):
        """Stop consuming messages"""
        self._running = False
        if self.consumer:
            self.consumer.close()
        self.executor.shutdown(wait=True)
        logger.info("Stopped radar data consumer")

    def consume(self, timeout_ms: int = 1000):
        """
        Consume messages from Kafka

        Args:
            timeout_ms: Poll timeout in milliseconds
        """
        if not self.consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        while self._running:
            try:
                messages = self.consumer.poll(timeout_ms=timeout_ms)

                for topic_partition, records in messages.items():
                    for record in records:
                        self.stats['messages_received'] += 1

                        # Process in thread pool
                        future = self.executor.submit(self._process_message, record.value)
                        future.add_done_callback(self._process_callback)

            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
                self.stats['messages_failed'] += 1
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.stats['messages_failed'] += 1

    def _process_message(self, message: Dict) -> RadarReturn:
        """Process raw message into RadarReturn"""
        try:
            radar_return = RadarReturn(
                timestamp=datetime.fromisoformat(message['timestamp']),
                radar_id=message['radar_id'],
                azimuth=message['azimuth'],
                elevation=message['elevation'],
                range=message['range'],
                range_rate=message['range_rate'],
                doppler_shift=message.get('doppler_shift', 0.0),
                rcs=message.get('rcs', 0.0),
                snr=message.get('snr', 0.0),
                metadata=message.get('metadata', {})
            )

            self.stats['messages_processed'] += 1
            self.stats['detections'] += 1

            return radar_return

        except Exception as e:
            logger.error(f"Failed to process radar message: {e}")
            self.stats['messages_failed'] += 1
            raise

    def _process_callback(self, future):
        """Callback when message processing is complete"""
        try:
            radar_return = future.result()

            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(radar_return)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error in process callback: {e}")

    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        return self.stats.copy()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Telescope image consumer
    telescope_consumer = TelescopeImageConsumer(
        bootstrap_servers=['localhost:9092'],
        topic='telescope-images'
    )

    def process_telescope_image(image: TelescopeImage):
        print(f"Received image from {image.telescope_id} at {image.timestamp}")
        print(f"  Shape: {image.image_data.shape}")
        print(f"  RA/Dec: {image.ra:.4f}, {image.dec:.4f}")

    telescope_consumer.register_callback(process_telescope_image)
    telescope_consumer.start()

    # Radar data consumer
    radar_consumer = RadarDataConsumer(
        bootstrap_servers=['localhost:9092'],
        topic='radar-returns'
    )

    def process_radar_return(radar: RadarReturn):
        print(f"Received radar return from {radar.radar_id} at {radar.timestamp}")
        print(f"  Range: {radar.range:.1f} km")
        print(f"  RCS: {radar.rcs:.2f} m^2")

    radar_consumer.register_callback(process_radar_return)
    radar_consumer.start()

    # Consume for a while
    try:
        while True:
            telescope_consumer.consume()
            radar_consumer.consume()
            time.sleep(0.1)
    except KeyboardInterrupt:
        telescope_consumer.stop()
        radar_consumer.stop()
        print("Consumer statistics:")
        print(f"  Telescope: {telescope_consumer.get_stats()}")
        print(f"  Radar: {radar_consumer.get_stats()}")
