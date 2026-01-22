"""Baseline performance test scenario.

This scenario establishes performance baselines with minimal load.
Used to measure single-user and low-concurrency performance.

Usage:
    locust -f scenarios/baseline.py --host=http://localhost:8000 \\
        --users=10 --spawn-rate=2 --run-time=60s --headless \\
        --html=reports/baseline_report.html --csv=reports/baseline
"""

from locust import HttpUser, between, task

from tests.load.utils.data_generator import generate_common_query, generate_unique_query


class BaselineUser(HttpUser):
    """Baseline user for low-load performance testing.

    Simulates minimal concurrent users to establish performance baselines.
    """

    wait_time = between(0.5, 1.0)  # Slower pace for baseline

    @task(3)
    def classify_common_query(self):
        """Common query - cache hit scenario."""
        query = generate_common_query()
        self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            name="classify (cached)",
        )

    @task(7)
    def classify_unique_query(self):
        """Unique query - batch processing scenario."""
        query = generate_unique_query()
        self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            name="classify (unique)",
        )
