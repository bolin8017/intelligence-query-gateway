"""Fixtures for integration tests.

Provides mocked models, real service instances, and test clients
for integration testing with controlled dependencies.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.models.semantic_router import ClassificationResult, SemanticRouter
from src.services.batching import BatchingService
from src.services.cache import CacheService
from src.services.classifier import ClassifierService


@pytest.fixture
def mock_semantic_router():
    """Create a mock SemanticRouter for testing.

    Returns:
        Mock SemanticRouter that simulates model behavior without
        requiring actual model files or torch inference.
    """
    mock_router = Mock(spec=SemanticRouter)
    mock_router.is_loaded = True

    def mock_predict(texts: list[str]) -> list[ClassificationResult]:
        """Mock predict that returns deterministic results based on text content."""
        results = []
        for text in texts:
            text_lower = text.lower()

            if any(
                keyword in text_lower
                for keyword in ["write", "creative", "story", "poem"]
            ):
                label = 1
                confidence = 0.85
                probabilities = [0.15, 0.85]
            else:
                label = 0
                confidence = 0.92
                probabilities = [0.92, 0.08]

            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )
        return results

    mock_router.predict = Mock(side_effect=mock_predict)
    mock_router.predict_single = Mock(
        side_effect=lambda text: mock_predict([text])[0]
    )

    return mock_router


@pytest.fixture
def classifier_service(mock_semantic_router):
    """Create a real ClassifierService with mocked model.

    Args:
        mock_semantic_router: Mocked SemanticRouter fixture.

    Returns:
        ClassifierService instance using the mocked model.
    """
    return ClassifierService(model=mock_semantic_router)


@pytest.fixture
def cache_service():
    """Create a real CacheService for integration testing.

    Returns:
        CacheService instance with test-appropriate settings.
    """
    return CacheService(max_size=100, ttl_seconds=300, level="L1-Test")


@pytest.fixture
async def batching_service(classifier_service):
    """Create and start a real BatchingService for integration testing.

    Args:
        classifier_service: ClassifierService fixture.

    Yields:
        Started BatchingService instance, automatically stopped after test.
    """
    service = BatchingService(
        classifier=classifier_service,
        max_batch_size=32,
        max_wait_ms=10,
    )

    await service.start()

    yield service

    await service.stop()


@pytest.fixture
def app_with_mocked_model(mock_semantic_router):
    """Create FastAPI app instance with mocked model for API testing.

    Args:
        mock_semantic_router: Mocked SemanticRouter fixture.

    Returns:
        FastAPI application instance configured for testing.
    """
    from src.api.dependencies import (
        get_cache_service,
        get_classifier_service,
    )
    from src.main import app

    classifier = ClassifierService(model=mock_semantic_router)
    cache = CacheService(max_size=100, ttl_seconds=300)

    async def override_get_classifier():
        return classifier

    async def override_get_cache():
        return cache

    app.dependency_overrides[get_classifier_service] = override_get_classifier
    app.dependency_overrides[get_cache_service] = override_get_cache

    return app


@pytest.fixture
def test_client(app_with_mocked_model):
    """Create TestClient for API endpoint testing.

    Args:
        app_with_mocked_model: FastAPI app fixture with mocked dependencies.

    Returns:
        TestClient instance for making HTTP requests.
    """
    return TestClient(app_with_mocked_model)
