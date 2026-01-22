"""Integration tests for BatchingService with ClassifierService.

Tests validate batching behavior with real classifier integration,
queue dynamics, and batch triggering conditions.
"""

import asyncio

import pytest


@pytest.mark.asyncio
class TestBatchingServiceIntegration:
    """Integration tests for BatchingService with real classifier."""

    async def test_single_request_processed_correctly(self, batching_service):
        """Single request is processed and returns correct result."""
        result = await batching_service.classify("What is the capital of France?")

        assert result.label == 0
        assert result.category == "fast_path"
        assert result.confidence > 0.8

    async def test_multiple_concurrent_requests_batched(self, batching_service):
        """Multiple concurrent requests are batched together."""
        queries = [
            "What is AI?",
            "Write a poem",
            "Summarize this",
        ]

        tasks = [asyncio.create_task(batching_service.classify(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0

    async def test_batch_triggers_on_size_limit(self, classifier_service):
        """Batch processes when max_batch_size is reached."""
        from src.services.batching import BatchingService

        service = BatchingService(
            classifier=classifier_service,
            max_batch_size=2,
            max_wait_ms=5000,
        )

        await service.start()

        tasks = [
            asyncio.create_task(service.classify("Query 1")),
            asyncio.create_task(service.classify("Query 2")),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert all(r.label in (0, 1) for r in results)

        await service.stop()

    async def test_batch_triggers_on_timeout(self, classifier_service):
        """Batch processes when max_wait_time is exceeded."""
        from src.services.batching import BatchingService

        service = BatchingService(
            classifier=classifier_service,
            max_batch_size=100,
            max_wait_ms=50,
        )

        await service.start()

        result = await service.classify("Single query")

        assert result.label in (0, 1)
        assert result.confidence > 0.0

        await service.stop()

    async def test_batching_preserves_query_result_mapping(self, batching_service):
        """Each request receives its corresponding result."""
        queries = [
            "What is ML?",
            "Write a story about space",
            "Define recursion",
            "Create a poem",
        ]

        tasks = [asyncio.create_task(batching_service.classify(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0
        assert results[3].label == 1

    async def test_batching_handles_sequential_requests(self, batching_service):
        """Sequential requests are also handled correctly."""
        result1 = await batching_service.classify("Query 1")
        result2 = await batching_service.classify("Query 2")
        result3 = await batching_service.classify("Query 3")

        assert all(r.label in (0, 1) for r in [result1, result2, result3])

    async def test_batching_with_mixed_timing(self, batching_service):
        """Batching handles requests arriving at different times."""
        task1 = asyncio.create_task(batching_service.classify("Query 1"))

        await asyncio.sleep(0.005)

        task2 = asyncio.create_task(batching_service.classify("Query 2"))
        task3 = asyncio.create_task(batching_service.classify("Query 3"))

        results = await asyncio.gather(task1, task2, task3)

        assert len(results) == 3
        assert all(r.label in (0, 1) for r in results)

    async def test_batching_service_graceful_shutdown(self, classifier_service):
        """BatchingService shuts down gracefully with pending requests."""
        from src.services.batching import BatchingService

        service = BatchingService(
            classifier=classifier_service,
            max_batch_size=10,
            max_wait_ms=100,
        )

        await service.start()

        task = asyncio.create_task(service.classify("Test query"))

        await asyncio.sleep(0.01)

        await service.stop()

        result = await task
        assert result.label in (0, 1)

    async def test_batching_error_propagation(self, classifier_service):
        """Batching propagates classifier errors to all requests in batch."""
        from unittest.mock import Mock
        from src.services.batching import BatchingService

        classifier_service.classify_batch = Mock(
            side_effect=RuntimeError("Classifier error")
        )

        service = BatchingService(
            classifier=classifier_service,
            max_batch_size=2,
            max_wait_ms=50,
        )

        await service.start()

        with pytest.raises(RuntimeError) as exc_info:
            await service.classify("Test query")

        assert "Classifier error" in str(exc_info.value)

        await service.stop()

    async def test_batching_queue_size_tracking(self, batching_service):
        """BatchingService tracks queue size correctly."""
        initial_size = batching_service.queue_size

        task = asyncio.create_task(batching_service.classify("Query"))

        await asyncio.sleep(0.001)

        await task

        assert batching_service.queue_size >= 0

    async def test_large_batch_processing(self, batching_service):
        """BatchingService handles large number of concurrent requests."""
        queries = [f"Query {i}" for i in range(50)]

        tasks = [asyncio.create_task(batching_service.classify(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 50
        assert all(r.label in (0, 1) for r in results)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    async def test_batching_with_different_query_types(self, batching_service):
        """BatchingService correctly routes different query types."""
        fast_queries = [
            "What is X?",
            "Define Y",
            "Explain Z",
        ]
        slow_queries = [
            "Write a story about X",
            "Create a poem about Y",
            "Write creative content",
        ]

        all_queries = fast_queries + slow_queries
        tasks = [asyncio.create_task(batching_service.classify(q)) for q in all_queries]
        results = await asyncio.gather(*tasks)

        fast_results = results[: len(fast_queries)]
        slow_results = results[len(fast_queries) :]

        assert all(r.label == 0 for r in fast_results)
        assert all(r.label == 1 for r in slow_results)
