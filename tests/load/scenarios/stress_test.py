"""Stress test scenario.

This scenario pushes the system to find performance limits and
potential bottlenecks.

Usage:
    locust -f scenarios/stress_test.py --host=http://localhost:8000 \\
        --users=200 --spawn-rate=20 --run-time=120s --headless \\
        --html=reports/stress_test_report.html --csv=reports/stress_test
"""

from locust import HttpUser, task, between
from tests.load.utils.data_generator import generate_common_query, generate_unique_query


class StressUser(HttpUser):
    """User for stress testing with aggressive request patterns.

    This scenario tests system limits:
    - Error rate should stay < 0.1%
    - Service should not crash or hang
    - Metrics should remain accurate
    """

    wait_time = between(0.01, 0.1)  # Very aggressive

    @task(2)
    def stress_common_query(self):
        """Stress test with common queries."""
        query = generate_common_query()
        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="classify (stress-cached)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}")

    @task(8)
    def stress_unique_query(self):
        """Stress test with unique queries."""
        query = generate_unique_query()
        with self.client.post(
            "/v1/query-classify",
            json={"text": query},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="classify (stress-unique)",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}")
