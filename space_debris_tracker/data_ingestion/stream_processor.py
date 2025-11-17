"""
Stream Processing Pipeline
Real-time processing of space debris data streams
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque
import numpy as np
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for stream processing"""
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    avg_latency_ms: float = 0.0
    throughput_eps: float = 0.0  # events per second
    last_update: datetime = field(default_factory=datetime.utcnow)


class StreamProcessor:
    """
    Real-time stream processing for space debris data
    Handles validation, enrichment, and routing of data streams
    """

    def __init__(self,
                 max_queue_size: int = 10000,
                 batch_size: int = 100,
                 batch_timeout_ms: int = 1000):
        """
        Initialize stream processor

        Args:
            max_queue_size: Maximum queue size before backpressure
            batch_size: Number of events to batch together
            batch_timeout_ms: Timeout for batch accumulation
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms

        # Processing queues
        self.input_queue: asyncio.Queue = None
        self.output_queues: Dict[str, asyncio.Queue] = {}

        # Processing stages
        self.processors: List[Callable] = []
        self.enrichers: List[Callable] = []
        self.filters: List[Callable] = []

        # Metrics
        self.metrics = ProcessingMetrics()
        self._running = False

        # Windowing for aggregations
        self.windows: Dict[str, deque] = {}
        self.window_sizes: Dict[str, int] = {}

        logger.info("Initialized StreamProcessor")

    async def start(self):
        """Start stream processor"""
        self.input_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._running = True
        logger.info("Started stream processor")

    async def stop(self):
        """Stop stream processor"""
        self._running = False
        logger.info("Stopped stream processor")

    def register_processor(self, processor: Callable[[Dict], Dict]):
        """
        Register processing function

        Args:
            processor: Function that processes event dict
        """
        self.processors.append(processor)
        logger.info(f"Registered processor: {processor.__name__}")

    def register_enricher(self, enricher: Callable[[Dict], Dict]):
        """
        Register enrichment function

        Args:
            enricher: Function that enriches event with additional data
        """
        self.enrichers.append(enricher)
        logger.info(f"Registered enricher: {enricher.__name__}")

    def register_filter(self, filter_func: Callable[[Dict], bool]):
        """
        Register filter function

        Args:
            filter_func: Function that returns True if event should be kept
        """
        self.filters.append(filter_func)
        logger.info(f"Registered filter: {filter_func.__name__}")

    def register_output(self, name: str, queue_size: int = 1000):
        """
        Register output queue

        Args:
            name: Output name (e.g., 'detection', 'tracking', 'collision')
            queue_size: Queue size
        """
        self.output_queues[name] = asyncio.Queue(maxsize=queue_size)
        logger.info(f"Registered output queue: {name}")

    def register_window(self, name: str, size: int):
        """
        Register sliding window for aggregations

        Args:
            name: Window name
            size: Window size (number of events)
        """
        self.windows[name] = deque(maxlen=size)
        self.window_sizes[name] = size
        logger.info(f"Registered window '{name}' with size {size}")

    async def submit(self, event: Dict):
        """
        Submit event to processing pipeline

        Args:
            event: Event dictionary
        """
        if self.input_queue.full():
            logger.warning("Input queue full - applying backpressure")

        await self.input_queue.put(event)
        self.metrics.total_events += 1

    async def process_stream(self):
        """Main processing loop"""
        batch = []
        last_batch_time = time.time()

        while self._running:
            try:
                # Collect batch
                timeout = (self.batch_timeout_ms -
                          (time.time() - last_batch_time) * 1000) / 1000

                if timeout > 0:
                    try:
                        event = await asyncio.wait_for(
                            self.input_queue.get(),
                            timeout=timeout
                        )
                        batch.append(event)
                    except asyncio.TimeoutError:
                        pass

                # Process batch when full or timeout
                if (len(batch) >= self.batch_size or
                    (time.time() - last_batch_time) * 1000 >= self.batch_timeout_ms):

                    if batch:
                        await self._process_batch(batch)
                        batch = []
                        last_batch_time = time.time()

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.metrics.failed_events += len(batch)
                batch = []

    async def _process_batch(self, batch: List[Dict]):
        """
        Process batch of events

        Args:
            batch: List of event dictionaries
        """
        start_time = time.time()

        processed_batch = []

        for event in batch:
            try:
                # Apply filters
                if not self._apply_filters(event):
                    continue

                # Apply processors
                processed_event = event
                for processor in self.processors:
                    processed_event = processor(processed_event)
                    if processed_event is None:
                        break

                if processed_event is None:
                    continue

                # Apply enrichers
                for enricher in self.enrichers:
                    processed_event = enricher(processed_event)

                # Add processing timestamp
                processed_event['_processed_at'] = datetime.utcnow().isoformat()

                processed_batch.append(processed_event)
                self.metrics.processed_events += 1

            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self.metrics.failed_events += 1

        # Route to output queues
        await self._route_events(processed_batch)

        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.avg_latency_ms = (
            0.9 * self.metrics.avg_latency_ms +
            0.1 * (processing_time * 1000 / len(batch))
        )
        self.metrics.throughput_eps = len(batch) / processing_time
        self.metrics.last_update = datetime.utcnow()

    def _apply_filters(self, event: Dict) -> bool:
        """Apply all filters to event"""
        for filter_func in self.filters:
            try:
                if not filter_func(event):
                    return False
            except Exception as e:
                logger.error(f"Error in filter {filter_func.__name__}: {e}")
                return False
        return True

    async def _route_events(self, events: List[Dict]):
        """Route processed events to output queues"""
        for event in events:
            # Determine routing based on event type
            event_type = event.get('type', 'default')

            # Send to all matching output queues
            for output_name, output_queue in self.output_queues.items():
                if self._should_route(event, output_name):
                    try:
                        await output_queue.put(event)
                    except asyncio.QueueFull:
                        logger.warning(f"Output queue {output_name} full")

            # Update windows
            for window_name, window in self.windows.items():
                if self._should_add_to_window(event, window_name):
                    window.append(event)

    def _should_route(self, event: Dict, output_name: str) -> bool:
        """Determine if event should be routed to output"""
        # Default routing logic
        event_type = event.get('type', '')

        routing_rules = {
            'detection': lambda e: 'detection' in e.get('type', ''),
            'tracking': lambda e: 'track' in e.get('type', ''),
            'collision': lambda e: e.get('collision_risk', 0) > 0.0001,
            'high_risk': lambda e: e.get('risk_level', '') == 'HIGH',
            'alerts': lambda e: e.get('alert', False),
        }

        if output_name in routing_rules:
            return routing_rules[output_name](event)

        return True  # Default: route to all

    def _should_add_to_window(self, event: Dict, window_name: str) -> bool:
        """Determine if event should be added to window"""
        # Default: add all events
        return True

    async def get_from_output(self, output_name: str,
                             timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get event from output queue

        Args:
            output_name: Output queue name
            timeout: Timeout in seconds

        Returns:
            Event dictionary or None
        """
        if output_name not in self.output_queues:
            raise ValueError(f"Unknown output: {output_name}")

        try:
            if timeout is not None:
                event = await asyncio.wait_for(
                    self.output_queues[output_name].get(),
                    timeout=timeout
                )
            else:
                event = await self.output_queues[output_name].get()

            return event

        except asyncio.TimeoutError:
            return None

    def get_window_data(self, window_name: str) -> List[Dict]:
        """
        Get data from window

        Args:
            window_name: Window name

        Returns:
            List of events in window
        """
        if window_name not in self.windows:
            raise ValueError(f"Unknown window: {window_name}")

        return list(self.windows[window_name])

    def compute_window_aggregation(self, window_name: str,
                                   field: str,
                                   aggregation: str = 'mean') -> float:
        """
        Compute aggregation over window

        Args:
            window_name: Window name
            field: Field to aggregate
            aggregation: Aggregation type ('mean', 'sum', 'max', 'min', 'count')

        Returns:
            Aggregated value
        """
        data = self.get_window_data(window_name)

        if not data:
            return 0.0

        values = [event.get(field, 0) for event in data if field in event]

        if not values:
            return 0.0

        if aggregation == 'mean':
            return np.mean(values)
        elif aggregation == 'sum':
            return np.sum(values)
        elif aggregation == 'max':
            return np.max(values)
        elif aggregation == 'min':
            return np.min(values)
        elif aggregation == 'count':
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_metrics(self) -> Dict:
        """Get processing metrics"""
        return {
            'total_events': self.metrics.total_events,
            'processed_events': self.metrics.processed_events,
            'failed_events': self.metrics.failed_events,
            'success_rate': (self.metrics.processed_events /
                           max(self.metrics.total_events, 1)),
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'throughput_eps': self.metrics.throughput_eps,
            'last_update': self.metrics.last_update.isoformat(),
            'queue_size': self.input_queue.qsize() if self.input_queue else 0,
            'output_queue_sizes': {
                name: queue.qsize()
                for name, queue in self.output_queues.items()
            }
        }


# Example processors
def deduplicate_processor(event: Dict) -> Optional[Dict]:
    """Remove duplicate events based on ID"""
    # Simple deduplication - in production, use Redis or similar
    if '_seen_ids' not in deduplicate_processor.__dict__:
        deduplicate_processor._seen_ids = set()

    event_id = event.get('id')
    if event_id in deduplicate_processor._seen_ids:
        return None

    deduplicate_processor._seen_ids.add(event_id)
    return event


def timestamp_enricher(event: Dict) -> Dict:
    """Add processing timestamp"""
    event['_ingested_at'] = datetime.utcnow().isoformat()
    return event


def collision_risk_filter(event: Dict) -> bool:
    """Filter for high collision risk events"""
    return event.get('collision_probability', 0) > 1e-5


# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)

    # Create processor
    processor = StreamProcessor(
        max_queue_size=10000,
        batch_size=50,
        batch_timeout_ms=100
    )

    # Register components
    processor.register_processor(deduplicate_processor)
    processor.register_enricher(timestamp_enricher)
    processor.register_filter(lambda e: e.get('valid', True))

    # Register outputs
    processor.register_output('detection')
    processor.register_output('high_risk')

    # Register window for statistics
    processor.register_window('recent_detections', size=1000)

    # Start processor
    await processor.start()

    # Start processing loop
    process_task = asyncio.create_task(processor.process_stream())

    # Simulate events
    for i in range(100):
        event = {
            'id': f'event_{i}',
            'type': 'detection',
            'timestamp': datetime.utcnow().isoformat(),
            'collision_probability': np.random.rand() * 0.001,
            'valid': np.random.rand() > 0.1
        }
        await processor.submit(event)

    # Wait a bit for processing
    await asyncio.sleep(2)

    # Get metrics
    metrics = processor.get_metrics()
    print(f"Processing metrics: {metrics}")

    # Stop
    await processor.stop()
    process_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
