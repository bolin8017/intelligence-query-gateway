"""BatchingService for dynamic request batching with adaptive parameters.

This service implements dynamic batching to improve inference throughput
by aggregating multiple concurrent requests into batches before sending
them to the model.

Adaptive Batching:
- Low load: shorter wait times (5ms) to minimize latency
- High load: longer wait times (up to 15ms) and larger batches to maximize throughput
"""

import asyncio
import time
from dataclasses import dataclass

from src.core.logging import get_logger
from src.core.metrics import BATCH_QUEUE_SIZE, BATCH_WAIT_TIME
from src.services.classifier import ClassifierService, ClassifyResult

logger = get_logger(__name__)

# Adaptive batching thresholds
_LOW_LOAD_QUEUE_THRESHOLD = 4
_HIGH_LOAD_QUEUE_THRESHOLD = 16


@dataclass
class BatchRequest:
    """Represents a single request in the batch queue.

    Attributes:
        text: Query text to classify.
        future: Future to set with the classification result.
        enqueue_time: Timestamp when request was added to queue.
    """

    text: str
    future: asyncio.Future[ClassifyResult]
    enqueue_time: float


class BatchingService:
    """Dynamic batching service for request aggregation.

    This service uses asyncio.Queue to collect incoming classification
    requests and processes them in batches. Batches are triggered by
    either reaching the max_batch_size or exceeding max_wait_time.

    The batch processor runs as a background task that continuously
    monitors the queue and processes batches when triggered.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        max_batch_size: int = 32,
        max_wait_ms: int = 10,
    ) -> None:
        """Initialize BatchingService with adaptive batching.

        Args:
            classifier: ClassifierService instance for inference.
            max_batch_size: Base maximum batch size (adaptive range: 8 to 2x this value).
            max_wait_ms: Base maximum wait time in ms (adaptive range: 5ms to 1.5x this value).
        """
        self._classifier = classifier
        # Base values for adaptive adjustment
        self._base_batch_size = max_batch_size
        self._base_wait_ms = max_wait_ms
        # Current adaptive values
        self._max_batch_size = max_batch_size
        self._max_wait_sec = max_wait_ms / 1000.0

        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._batch_task: asyncio.Task | None = None
        self._shutdown = False

        logger.info(
            "BatchingService initialized with adaptive batching",
            base_batch_size=max_batch_size,
            base_wait_ms=max_wait_ms,
        )

    async def start(self) -> None:
        """Start the background batch processor task.

        This should be called during application startup (in lifespan).
        """
        if self._batch_task is not None:
            logger.warning("BatchingService already started")
            return

        self._shutdown = False
        self._batch_task = asyncio.create_task(self._batch_processor())
        logger.info("BatchingService started")

    async def stop(self) -> None:
        """Stop the background batch processor gracefully.

        Waits for the queue to be empty before stopping.
        This should be called during application shutdown (in lifespan).
        """
        if self._batch_task is None:
            logger.warning("BatchingService not running")
            return

        logger.info("Stopping BatchingService gracefully")
        self._shutdown = True

        # Wait for queue to be processed
        await self._queue.join()

        # Cancel the batch task
        self._batch_task.cancel()
        try:
            await self._batch_task
        except asyncio.CancelledError:
            pass

        logger.info("BatchingService stopped")

    async def classify(self, text: str) -> ClassifyResult:
        """Submit a classification request and wait for the result.

        This method adds the request to the batch queue and returns
        a Future that will be resolved when the batch is processed.

        Args:
            text: Query text to classify.

        Returns:
            ClassifyResult from batched inference.

        Raises:
            RuntimeError: If batching service is not started.
        """
        if self._batch_task is None:
            raise RuntimeError("BatchingService not started")

        # Create a future for this request's result
        loop = asyncio.get_event_loop()
        future: asyncio.Future[ClassifyResult] = loop.create_future()

        # Create batch request with enqueue timestamp
        request = BatchRequest(
            text=text,
            future=future,
            enqueue_time=time.perf_counter(),
        )

        # Add to queue
        await self._queue.put(request)

        # Update queue size metric
        BATCH_QUEUE_SIZE.set(self._queue.qsize())

        # Wait for result
        result = await future
        return result

    async def _batch_processor(self) -> None:
        """Background task that processes batches.

        This runs continuously, collecting requests from the queue and
        processing them in batches when either:
        1. Batch size reaches max_batch_size
        2. Wait time exceeds max_wait_sec
        """
        logger.info("Batch processor started")

        while not self._shutdown:
            try:
                batch = await self._collect_batch()

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                logger.info("Batch processor cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in batch processor",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Continue processing despite errors
                await asyncio.sleep(0.01)

        logger.info("Batch processor stopped")

    def _adapt_parameters(self) -> None:
        """Adapt batch size and wait time based on current queue depth.

        Strategy:
        - Low load (queue < 4): Minimize latency with shorter waits
        - Medium load: Use base parameters
        - High load (queue > 16): Maximize throughput with larger batches
        """
        queue_size = self._queue.qsize()

        if queue_size < _LOW_LOAD_QUEUE_THRESHOLD:
            # Low load: prioritize latency
            self._max_batch_size = max(8, self._base_batch_size // 2)
            self._max_wait_sec = 0.005  # 5ms
        elif queue_size > _HIGH_LOAD_QUEUE_THRESHOLD:
            # High load: prioritize throughput
            self._max_batch_size = min(64, self._base_batch_size * 2)
            self._max_wait_sec = self._base_wait_ms * 1.5 / 1000.0  # 15ms at default
        else:
            # Medium load: use base parameters
            self._max_batch_size = self._base_batch_size
            self._max_wait_sec = self._base_wait_ms / 1000.0

    async def _collect_batch(self) -> list[BatchRequest]:
        """Collect requests from queue until batch is full or timeout.

        Uses adaptive parameters based on current load.

        Returns:
            List of BatchRequest objects to process.
        """
        # Adapt parameters before collecting
        self._adapt_parameters()

        batch: list[BatchRequest] = []
        deadline = time.perf_counter() + self._max_wait_sec

        while len(batch) < self._max_batch_size:
            # Calculate remaining wait time
            remaining_time = deadline - time.perf_counter()

            if remaining_time <= 0:
                # Timeout reached
                break

            try:
                # Wait for next request with timeout
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining_time,
                )
                batch.append(request)

            except TimeoutError:
                # Timeout waiting for next request
                break

        return batch

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a batch of requests through the classifier.

        Args:
            batch: List of BatchRequest objects to process.
        """
        if not batch:
            return

        try:
            # Extract texts
            texts = [req.text for req in batch]

            # Perform batch inference
            results = self._classifier.classify_batch(texts)

            # Distribute results to futures and record wait times
            current_time = time.perf_counter()
            for request, result in zip(batch, results, strict=False):
                # Calculate and record wait time
                wait_time = current_time - request.enqueue_time
                BATCH_WAIT_TIME.observe(wait_time)

                # Set result on future
                if not request.future.done():
                    request.future.set_result(result)

                # Mark task as done in queue
                self._queue.task_done()

            # Update queue size metric
            BATCH_QUEUE_SIZE.set(self._queue.qsize())

            logger.debug(
                "Batch processed",
                batch_size=len(batch),
                queue_size=self._queue.qsize(),
            )

        except Exception as e:
            # If batch processing fails, set exception on all futures
            logger.error(
                "Batch processing failed",
                error=str(e),
                error_type=type(e).__name__,
                batch_size=len(batch),
            )
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                self._queue.task_done()

            # Update queue size metric
            BATCH_QUEUE_SIZE.set(self._queue.qsize())

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if batch processor is running."""
        return self._batch_task is not None and not self._batch_task.done()
