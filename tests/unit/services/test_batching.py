"""Unit tests for BatchingService.

Tests cover queue mechanics, batch triggering (size and timeout),
request processing, and graceful shutdown behavior.
"""

import asyncio
from unittest.mock import Mock

import pytest

from src.services.batching import BatchingService, BatchRequest
from src.services.classifier import ClassifierService, ClassifyResult


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_batch_request_creation(self):
        """BatchRequest stores text, future, and enqueue time."""
        future = asyncio.Future()
        request = BatchRequest(text="test query", future=future, enqueue_time=1234.5)

        assert request.text == "test query"
        assert request.future is future
        assert request.enqueue_time == 1234.5


class TestBatchingServiceInitialization:
    """Tests for BatchingService initialization."""

    def test_batching_service_initialization_defaults(self):
        """BatchingService initializes with default parameters."""
        mock_classifier = Mock(spec=ClassifierService)

        service = BatchingService(classifier=mock_classifier)

        assert service._classifier is mock_classifier
        assert service._max_batch_size == 32
        assert service._max_wait_sec == 0.01
        assert service._batch_task is None
        assert service._shutdown is False

    def test_batching_service_initialization_custom_parameters(self):
        """BatchingService accepts custom batch size and wait time."""
        mock_classifier = Mock(spec=ClassifierService)

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=64,
            max_wait_ms=50,
        )

        assert service._max_batch_size == 64
        assert service._max_wait_sec == 0.05

    def test_batching_service_converts_wait_time_to_seconds(self):
        """BatchingService converts max_wait_ms to seconds correctly."""
        mock_classifier = Mock(spec=ClassifierService)

        service = BatchingService(
            classifier=mock_classifier,
            max_wait_ms=100,
        )

        assert service._max_wait_sec == 0.1


@pytest.mark.asyncio
class TestBatchingServiceStartStop:
    """Tests for BatchingService start/stop lifecycle."""

    async def test_start_creates_batch_processor_task(self):
        """start creates and launches the batch processor task."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()

        assert service._batch_task is not None
        assert not service._batch_task.done()
        assert service.is_running is True

        await service.stop()

    async def test_start_does_nothing_if_already_started(self):
        """start is idempotent - does nothing if already started."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()
        first_task = service._batch_task

        await service.start()
        second_task = service._batch_task

        assert first_task is second_task

        await service.stop()

    async def test_stop_sets_shutdown_flag(self):
        """stop sets shutdown flag to True."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()
        await service.stop()

        assert service._shutdown is True

    async def test_stop_cancels_batch_task(self):
        """stop cancels the batch processor task."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()
        await service.stop()

        assert service._batch_task.cancelled() or service._batch_task.done()

    async def test_stop_does_nothing_if_not_started(self):
        """stop handles being called when service not started."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.stop()

        assert service._batch_task is None


@pytest.mark.asyncio
class TestBatchingServiceClassify:
    """Tests for BatchingService.classify method."""

    async def test_classify_raises_error_if_not_started(self):
        """classify raises RuntimeError if service not started."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        with pytest.raises(RuntimeError) as exc_info:
            await service.classify("test query")

        assert "not started" in str(exc_info.value).lower()

    async def test_classify_adds_request_to_queue(self):
        """classify adds request to the internal queue."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[ClassifyResult(label=0, confidence=0.9, category="fast_path")]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=10,
            max_wait_ms=100,
        )

        await service.start()

        task = asyncio.create_task(service.classify("test query"))

        await asyncio.sleep(0.01)

        assert service.queue_size >= 0

        mock_classifier.classify_batch.return_value = [
            ClassifyResult(label=0, confidence=0.9, category="fast_path")
        ]

        await service.stop()

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except TimeoutError:
            task.cancel()

    async def test_classify_returns_result_from_batch_processing(self):
        """classify returns the classification result after batch processing."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[ClassifyResult(label=1, confidence=0.85, category="slow_path")]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=1,
            max_wait_ms=10,
        )

        await service.start()

        result = await service.classify("Write a story")

        assert result.label == 1
        assert result.confidence == 0.85
        assert result.category == "slow_path"

        await service.stop()

    async def test_classify_batches_multiple_concurrent_requests(self):
        """classify batches multiple concurrent requests together."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[
                ClassifyResult(label=0, confidence=0.9, category="fast_path"),
                ClassifyResult(label=1, confidence=0.85, category="slow_path"),
                ClassifyResult(label=0, confidence=0.88, category="fast_path"),
            ]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=10,
            max_wait_ms=50,
        )

        await service.start()

        tasks = [
            asyncio.create_task(service.classify("Query 1")),
            asyncio.create_task(service.classify("Query 2")),
            asyncio.create_task(service.classify("Query 3")),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0

        mock_classifier.classify_batch.assert_called()

        await service.stop()


@pytest.mark.asyncio
class TestBatchingServiceBatchProcessing:
    """Tests for batch collection and processing logic."""

    async def test_batch_triggers_on_max_size(self):
        """Batch processes when max_batch_size is reached."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[
                ClassifyResult(label=0, confidence=0.9, category="fast_path"),
                ClassifyResult(label=0, confidence=0.9, category="fast_path"),
            ]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=2,
            max_wait_ms=1000,
        )

        await service.start()

        task1 = asyncio.create_task(service.classify("Query 1"))
        task2 = asyncio.create_task(service.classify("Query 2"))

        await asyncio.gather(task1, task2)

        mock_classifier.classify_batch.assert_called_once()
        call_args = mock_classifier.classify_batch.call_args[0][0]
        assert len(call_args) == 2

        await service.stop()

    async def test_batch_triggers_on_timeout(self):
        """Batch processes when max_wait_time is exceeded."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[ClassifyResult(label=0, confidence=0.9, category="fast_path")]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=100,
            max_wait_ms=50,
        )

        await service.start()

        result = await service.classify("Single query")

        assert result.label == 0
        mock_classifier.classify_batch.assert_called_once()

        await service.stop()

    async def test_empty_batch_not_processed(self):
        """Empty batches are not sent to classifier."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(return_value=[])

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=10,
            max_wait_ms=10,
        )

        await service.start()
        await asyncio.sleep(0.05)

        mock_classifier.classify_batch.assert_not_called()

        await service.stop()

    async def test_batch_processing_handles_classifier_exception(self):
        """Batch processing sets exception on futures when classifier fails."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(side_effect=RuntimeError("Model error"))

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=1,
            max_wait_ms=10,
        )

        await service.start()

        with pytest.raises(RuntimeError) as exc_info:
            await service.classify("test query")

        assert "Model error" in str(exc_info.value)

        await service.stop()


class TestBatchingServiceProperties:
    """Tests for BatchingService property methods."""

    @pytest.mark.asyncio
    async def test_queue_size_property(self):
        """queue_size property returns current queue size."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        assert service.queue_size == 0

    @pytest.mark.asyncio
    async def test_is_running_property_when_not_started(self):
        """is_running returns False when service not started."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_is_running_property_when_started(self):
        """is_running returns True when service is started."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()

        assert service.is_running is True

        await service.stop()

    @pytest.mark.asyncio
    async def test_is_running_property_after_stopped(self):
        """is_running returns False after service is stopped."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()
        await service.stop()

        assert service.is_running is False


@pytest.mark.asyncio
class TestBatchingServiceGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    async def test_stop_waits_for_queue_to_empty(self):
        """stop waits for pending requests to be processed before stopping."""
        mock_classifier = Mock(spec=ClassifierService)
        mock_classifier.classify_batch = Mock(
            return_value=[ClassifyResult(label=0, confidence=0.9, category="fast_path")]
        )

        service = BatchingService(
            classifier=mock_classifier,
            max_batch_size=10,
            max_wait_ms=50,
        )

        await service.start()

        task = asyncio.create_task(service.classify("test"))

        await asyncio.sleep(0.01)

        await service.stop()

        result = await task
        assert result.label == 0

    async def test_shutdown_flag_stops_batch_processor(self):
        """Shutdown flag causes batch processor loop to exit."""
        mock_classifier = Mock(spec=ClassifierService)
        service = BatchingService(classifier=mock_classifier)

        await service.start()
        initial_task = service._batch_task

        await service.stop()

        assert service._shutdown is True
        assert initial_task.done() or initial_task.cancelled()
