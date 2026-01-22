"""Cache performance test scenario.

This scenario focuses on testing L1 cache performance with high
cache hit rates.

Usage:
    locust -f scenarios/cache_test.py --host=http://localhost:8000 \\
        --users=50 --spawn-rate=10 --run-time=60s --headless \\
        --html=reports/cache_test_report.html --csv=reports/cache_test
"""

import random

from locust import HttpUser, between, task

from tests.load.utils.data_generator import get_common_queries


class CacheUser(HttpUser):
    """User that only makes common queries to maximize cache hits.

    Expected results:
    - Cache hit rate > 90%
    - P99 latency < 5ms (Phase 2 achieved < 1ms)
    """

    wait_time = between(0.05, 0.2)  # Fast requests to stress cache

    def on_start(self):
        """Load common queries for cache testing."""
        self.common_queries = get_common_queries()

    @task
    def cached_query(self):
        """Make a query that should hit L1 cache."""
        query = random.choice(self.common_queries)
        self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            name="classify (cache-only)",
        )
