"""Integration tests for ClassifierService with SemanticRouter.

Tests validate the interaction between ClassifierService and the
SemanticRouter model with realistic (mocked) model behavior.
"""

import pytest

from src.core.exceptions import ModelNotReadyError


class TestClassifierServiceIntegration:
    """Integration tests for ClassifierService."""

    def test_classify_single_query_fast_path(self, classifier_service):
        """Classifier correctly identifies fast path queries."""
        result = classifier_service.classify("What is the capital of France?")

        assert result.label == 0
        assert result.category == "fast_path"
        assert result.confidence > 0.8

    def test_classify_single_query_slow_path(self, classifier_service):
        """Classifier correctly identifies slow path queries."""
        result = classifier_service.classify("Write a creative story about dragons")

        assert result.label == 1
        assert result.category == "slow_path"
        assert result.confidence > 0.8

    def test_classify_batch_mixed_queries(self, classifier_service):
        """Classifier handles mixed fast/slow path queries in batch."""
        queries = [
            "What is machine learning?",
            "Write a poem about the ocean",
            "Summarize this article",
            "Create a story",
        ]

        results = classifier_service.classify_batch(queries)

        assert len(results) == 4
        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0
        assert results[3].label == 1

    def test_classify_batch_maintains_order(self, classifier_service):
        """Classifier preserves query order in batch results."""
        queries = [f"Query {i}" for i in range(10)]

        results = classifier_service.classify_batch(queries)

        assert len(results) == len(queries)
        for result in results:
            assert result.label in (0, 1)
            assert result.confidence > 0.0

    def test_classify_empty_batch(self, classifier_service):
        """Classifier handles empty batch gracefully."""
        results = classifier_service.classify_batch([])

        assert results == []

    def test_classify_with_unloaded_model(self, mock_semantic_router):
        """Classifier raises error when model not loaded."""
        from src.services.classifier import ClassifierService

        mock_semantic_router.is_loaded = False
        service = ClassifierService(model=mock_semantic_router)

        with pytest.raises(ModelNotReadyError):
            service.classify("test query")

    def test_classify_batch_calls_model_predict_once(
        self, classifier_service, mock_semantic_router
    ):
        """Classifier calls model.predict once for batch."""
        queries = ["Query 1", "Query 2", "Query 3"]

        classifier_service.classify_batch(queries)

        mock_semantic_router.predict.assert_called_once()
        call_args = mock_semantic_router.predict.call_args[0][0]
        assert call_args == queries

    def test_classify_result_structure(self, classifier_service):
        """Classifier returns complete result structure."""
        result = classifier_service.classify("Test query")

        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "category")
        assert isinstance(result.label, int)
        assert isinstance(result.confidence, float)
        assert isinstance(result.category, str)

    def test_classifier_service_ready_state(self, classifier_service):
        """Classifier correctly reports ready state."""
        assert classifier_service.is_ready is True

    def test_classify_with_various_text_lengths(self, classifier_service):
        """Classifier handles queries of different lengths."""
        short = "AI?"
        medium = "What is artificial intelligence?"
        long = "Explain artificial intelligence " * 20

        results = classifier_service.classify_batch([short, medium, long])

        assert len(results) == 3
        assert all(r.label in (0, 1) for r in results)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_classify_with_special_characters(self, classifier_service):
        """Classifier handles special characters in queries."""
        queries = [
            "What's the cost? $100",
            "Email: user@example.com",
            "Math: 2 + 2 = 4",
        ]

        results = classifier_service.classify_batch(queries)

        assert len(results) == 3
        assert all(r.label in (0, 1) for r in results)
