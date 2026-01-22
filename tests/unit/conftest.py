"""Fixtures specific to unit tests.

Provides mocked dependencies and test data for isolated unit testing.
"""

import pytest


@pytest.fixture
def mock_time_counter():
    """Create a controllable time counter for deterministic testing.

    Returns:
        A callable that increments time on each call, starting from 1000.0.
    """
    time_value = [1000.0]

    def counter():
        result = time_value[0]
        time_value[0] += 0.1  # Increment by 100ms
        return result

    return counter
