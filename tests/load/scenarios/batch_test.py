"""Batch processing performance test scenario.

This scenario focuses on testing dynamic batching efficiency with
unique queries that won't hit cache.

Usage:
    locust -f scenarios/batch_test.py --host=http://localhost:8000 \\
        --users=100 --spawn-rate=20 --run-time=120s --headless \\
        --html=reports/batch_test_report.html --csv=reports/batch_test
"""

from locust import HttpUser, between, task

from tests.load.utils.data_generator import generate_unique_query


class BatchUser(HttpUser):
    """User that makes unique queries to force batch processing.

    Expected results:
    - Average batch size > 8 (under high load)
    - Speedup from batching: 7-11x (Phase 2 results)
    - P99 latency < 100ms
    """

    wait_time = between(0.01, 0.05)  # Very fast to build batches

    @task
    def unique_query(self):
        """Make a unique query that will be processed in batch."""
        query = generate_unique_query()
        self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            name="classify (batch-only)",
        )
