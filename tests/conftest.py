"""Shared pytest fixtures for all test layers.

This module provides common test utilities and fixtures that are used
across both unit and integration tests.
"""

import pytest


@pytest.fixture
def sample_query_texts() -> list[str]:
    """Sample query texts for testing classification.

    Returns:
        List of representative query strings covering different lengths
        and complexity levels.
    """
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a creative story about a dragon",
        "Summarize the following text",
        "How do I reset my password?",
    ]


@pytest.fixture
def sample_query_single() -> str:
    """Single sample query for basic tests.

    Returns:
        A simple factual question string.
    """
    return "What is the capital of France?"
