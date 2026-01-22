"""Locust load testing scenarios for Query Gateway.

This file defines the main load testing user behaviors for the
Intelligence Query Gateway service. It includes:

- Common query testing (cache hits)
- Unique query testing (batch processing)
- Health check monitoring

Usage:
    # Headless mode
    locust -f locustfile.py --host=http://localhost:8000 \\
        --users=50 --spawn-rate=5 --run-time=60s --headless

    # Web UI mode
    locust -f locustfile.py --host=http://localhost:8000
"""

import random

from locust import HttpUser, between, events, task
from utils.data_generator import (
    generate_common_query,
    generate_unique_query,
    get_common_queries,
)


class QueryGatewayUser(HttpUser):
    """Simulates a user making queries to the Query Gateway.

    This user performs three main tasks with different weights:
    - Common queries (weight 3): Tests cache hit scenarios
    - Unique queries (weight 7): Tests batch processing
    - Health checks (weight 1): Monitors system health

    The wait time between requests simulates realistic user behavior.
    """

    # Wait 100-500ms between requests (realistic user behavior)
    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Initialize user session.

        Loads common queries for cache hit testing and prepares
        the user for making requests.
        """
        self.common_queries = get_common_queries()
        print(f"User started with {len(self.common_queries)} common queries loaded")

    @task(3)
    def classify_common_query(self):
        """Test cache hit scenario with common queries.

        Weight: 3 (30% of requests)
        Expected: < 5ms latency (cache hit)
        """
        query = generate_common_query()

        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="POST /v1/query-classify (cached)",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure (API returns "label")
                    if "label" in data:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code {response.status_code}")

    @task(7)
    def classify_unique_query(self):
        """Test batch processing with unique queries.

        Weight: 7 (70% of requests)
        Expected: Benefits from batching under load
        """
        query = generate_unique_query()

        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="POST /v1/query-classify (unique)",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "label" in data:
                        response.success()
                    else:
                        response.failure("Invalid response structure")
                except ValueError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code {response.status_code}")

    @task(1)
    def health_check(self):
        """Monitor service health.

        Weight: 1 (10% of requests)
        Expected: Very fast response
        """
        with self.client.get(
            "/health/live",
            catch_response=True,
            name="GET /health/live",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Event handler called when the load test starts.

    Args:
        environment: Locust environment object.
        **kwargs: Additional keyword arguments.
    """
    print("=" * 60)
    print("Query Gateway Load Test Started")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if environment.runner else 'N/A'}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Event handler called when the load test stops.

    Args:
        environment: Locust environment object.
        **kwargs: Additional keyword arguments.
    """
    print("=" * 60)
    print("Query Gateway Load Test Completed")
    print("=" * 60)

    # Try to collect final metrics
    try:
        from utils.metrics_collector import get_metrics_summary

        summary = get_metrics_summary(f"{environment.host}/metrics")
        if summary:
            print("\nFinal Metrics Summary:")
            print(f"  Total Requests: {summary['total_requests']}")
            print(f"  Cache Hit Rate: {summary['cache_hit_rate']:.2f}%")
            print(f"  Average Batch Size: {summary['average_batch_size']:.2f}")
            print(f"  P50 Latency: {summary['p50_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {summary['p95_latency_ms']:.2f}ms")
            print(f"  P99 Latency: {summary['p99_latency_ms']:.2f}ms")
    except Exception as e:
        print(f"\nFailed to collect final metrics: {e}")

    print("=" * 60)


# Additional user classes for specific scenarios

class CacheTestUser(HttpUser):
    """User that only makes common queries to test cache performance.

    Use this class to test maximum cache hit rate and L1 cache efficiency.
    """

    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.common_queries = get_common_queries()

    @task
    def cache_hit_query(self):
        """Only make queries that should hit cache."""
        query = random.choice(self.common_queries)

        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="POST /v1/query-classify (cache-test)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")


class BatchTestUser(HttpUser):
    """User that only makes unique queries to test batch processing.

    Use this class to test maximum batch efficiency without cache interference.
    """

    wait_time = between(0.01, 0.05)  # Fast requests to build batches

    @task
    def batch_query(self):
        """Only make unique queries to force batching."""
        query = generate_unique_query()

        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="POST /v1/query-classify (batch-test)",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")
