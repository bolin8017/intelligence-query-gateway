"""Unit tests for ClassifierService.

Tests cover classification logic, batch processing, metrics emission,
and error handling with mocked SemanticRouter.
"""

from unittest.mock import Mock

import pytest

from src.core.exceptions import ModelNotReadyError
from src.models.semantic_router import ClassificationResult
from src.services.classifier import ClassifierService, ClassifyResult


class TestClassifyResult:
    """Tests for ClassifyResult dataclass."""

    def test_classify_result_creation(self):
        """ClassifyResult stores label, confidence, and category."""
        result = ClassifyResult(label=0, confidence=0.95, category="fast_path")

        assert result.label == 0
        assert result.confidence == 0.95
        assert result.category == "fast_path"

    def test_classify_result_from_model_result_fast_path(self):
        """from_model_result maps label 0 to fast_path category."""
        model_result = ClassificationResult(
            label=0,
            confidence=0.92,
            probabilities=[0.92, 0.08],
        )

        result = ClassifyResult.from_model_result(model_result)

        assert result.label == 0
        assert result.confidence == 0.92
        assert result.category == "fast_path"

    def test_classify_result_from_model_result_slow_path(self):
        """from_model_result maps label 1 to slow_path category."""
        model_result = ClassificationResult(
            label=1,
            confidence=0.88,
            probabilities=[0.12, 0.88],
        )

        result = ClassifyResult.from_model_result(model_result)

        assert result.label == 1
        assert result.confidence == 0.88
        assert result.category == "slow_path"

    def test_classify_result_from_model_result_unknown_label(self):
        """from_model_result maps unknown labels to 'unknown' category."""
        model_result = ClassificationResult(
            label=99,
            confidence=0.5,
            probabilities=[0.5, 0.5],
        )

        result = ClassifyResult.from_model_result(model_result)

        assert result.label == 99
        assert result.category == "unknown"


class TestClassifierServiceInitialization:
    """Tests for ClassifierService initialization."""

    def test_classifier_service_initialization(self):
        """ClassifierService initializes with SemanticRouter model."""
        mock_model = Mock()
        mock_model.is_loaded = True

        service = ClassifierService(model=mock_model)

        assert service._model is mock_model

    def test_classifier_service_is_ready_when_model_loaded(self):
        """is_ready returns True when model is loaded."""
        mock_model = Mock()
        mock_model.is_loaded = True

        service = ClassifierService(model=mock_model)

        assert service.is_ready is True

    def test_classifier_service_is_ready_when_model_not_loaded(self):
        """is_ready returns False when model not loaded."""
        mock_model = Mock()
        mock_model.is_loaded = False

        service = ClassifierService(model=mock_model)

        assert service.is_ready is False


class TestClassifierServiceClassify:
    """Tests for ClassifierService.classify method."""

    def test_classify_single_query_calls_classify_batch(self):
        """classify delegates to classify_batch with single-item list."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.95, probabilities=[0.95, 0.05])
        ]

        service = ClassifierService(model=mock_model)
        result = service.classify("What is AI?")

        assert result.label == 0
        assert result.confidence == 0.95
        assert result.category == "fast_path"
        mock_model.predict.assert_called_once_with(["What is AI?"])

    def test_classify_raises_error_when_model_not_loaded(self):
        """classify raises ModelNotReadyError when model not loaded."""
        mock_model = Mock()
        mock_model.is_loaded = False

        service = ClassifierService(model=mock_model)

        with pytest.raises(ModelNotReadyError) as exc_info:
            service.classify("test query")

        assert "not loaded" in str(exc_info.value.message).lower()

    def test_classify_returns_correct_result_structure(self):
        """classify returns ClassifyResult with correct structure."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=1, confidence=0.87, probabilities=[0.13, 0.87])
        ]

        service = ClassifierService(model=mock_model)
        result = service.classify("Write a story")

        assert isinstance(result, ClassifyResult)
        assert result.label == 1
        assert result.confidence == 0.87
        assert result.category == "slow_path"


class TestClassifierServiceClassifyBatch:
    """Tests for ClassifierService.classify_batch method."""

    def test_classify_batch_returns_empty_list_for_empty_input(self):
        """classify_batch returns empty list when given empty texts."""
        mock_model = Mock()
        mock_model.is_loaded = True

        service = ClassifierService(model=mock_model)
        results = service.classify_batch([])

        assert results == []
        mock_model.predict.assert_not_called()

    def test_classify_batch_processes_single_item(self):
        """classify_batch handles single-item batch correctly."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.92, probabilities=[0.92, 0.08])
        ]

        service = ClassifierService(model=mock_model)
        results = service.classify_batch(["What is ML?"])

        assert len(results) == 1
        assert results[0].label == 0
        assert results[0].confidence == 0.92

    def test_classify_batch_processes_multiple_items(self):
        """classify_batch handles multi-item batch correctly."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.91, probabilities=[0.91, 0.09]),
            ClassificationResult(label=1, confidence=0.85, probabilities=[0.15, 0.85]),
            ClassificationResult(label=0, confidence=0.88, probabilities=[0.88, 0.12]),
        ]

        service = ClassifierService(model=mock_model)
        texts = ["Query 1", "Query 2", "Query 3"]
        results = service.classify_batch(texts)

        assert len(results) == 3
        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0

    def test_classify_batch_raises_error_when_model_not_loaded(self):
        """classify_batch raises ModelNotReadyError when model not loaded."""
        mock_model = Mock()
        mock_model.is_loaded = False

        service = ClassifierService(model=mock_model)

        with pytest.raises(ModelNotReadyError):
            service.classify_batch(["test"])

    def test_classify_batch_preserves_result_order(self):
        """classify_batch preserves order of results matching input order."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.9, probabilities=[0.9, 0.1]),
            ClassificationResult(label=1, confidence=0.8, probabilities=[0.2, 0.8]),
            ClassificationResult(label=0, confidence=0.7, probabilities=[0.7, 0.3]),
        ]

        service = ClassifierService(model=mock_model)
        texts = ["Fast query", "Slow query", "Another fast"]
        results = service.classify_batch(texts)

        assert results[0].label == 0
        assert results[1].label == 1
        assert results[2].label == 0

    def test_classify_batch_converts_all_model_results_to_classify_results(self):
        """classify_batch converts all model results to ClassifyResult."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.95, probabilities=[0.95, 0.05]),
            ClassificationResult(label=1, confidence=0.88, probabilities=[0.12, 0.88]),
        ]

        service = ClassifierService(model=mock_model)
        results = service.classify_batch(["Query 1", "Query 2"])

        assert all(isinstance(r, ClassifyResult) for r in results)
        assert results[0].category == "fast_path"
        assert results[1].category == "slow_path"

    def test_classify_batch_calls_model_predict_once(self):
        """classify_batch calls model.predict exactly once with all texts."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.9, probabilities=[0.9, 0.1]),
            ClassificationResult(label=0, confidence=0.8, probabilities=[0.8, 0.2]),
        ]

        service = ClassifierService(model=mock_model)
        texts = ["Text 1", "Text 2"]
        service.classify_batch(texts)

        mock_model.predict.assert_called_once_with(texts)

    def test_classify_batch_handles_various_confidence_values(self):
        """classify_batch correctly handles different confidence values."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(
                label=0, confidence=0.99, probabilities=[0.99, 0.01]
            ),
            ClassificationResult(label=1, confidence=0.5, probabilities=[0.5, 0.5]),
            ClassificationResult(
                label=0, confidence=0.75, probabilities=[0.75, 0.25]
            ),
        ]

        service = ClassifierService(model=mock_model)
        results = service.classify_batch(["High conf", "Low conf", "Med conf"])

        assert results[0].confidence == 0.99
        assert results[1].confidence == 0.5
        assert results[2].confidence == 0.75

    def test_classify_batch_model_prediction_error_propagates(self):
        """classify_batch propagates exceptions from model.predict."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.side_effect = RuntimeError("Model inference failed")

        service = ClassifierService(model=mock_model)

        with pytest.raises(RuntimeError) as exc_info:
            service.classify_batch(["test"])

        assert "Model inference failed" in str(exc_info.value)


class TestClassifierServiceIntegrationBehavior:
    """Integration-style tests validating complete classifier behaviors."""

    def test_classify_and_classify_batch_consistency(self):
        """classify produces same result as classify_batch for single item."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.93, probabilities=[0.93, 0.07])
        ]

        service = ClassifierService(model=mock_model)

        batch_result = service.classify_batch(["Test query"])[0]

        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.93, probabilities=[0.93, 0.07])
        ]

        single_result = service.classify("Test query")

        assert batch_result.label == single_result.label
        assert batch_result.confidence == single_result.confidence
        assert batch_result.category == single_result.category

    def test_service_handles_model_ready_state_changes(self):
        """Service correctly responds to model ready state changes."""
        mock_model = Mock()
        mock_model.is_loaded = False

        service = ClassifierService(model=mock_model)
        assert service.is_ready is False

        with pytest.raises(ModelNotReadyError):
            service.classify("test")

        mock_model.is_loaded = True
        mock_model.predict.return_value = [
            ClassificationResult(label=0, confidence=0.9, probabilities=[0.9, 0.1])
        ]

        assert service.is_ready is True
        result = service.classify("test")
        assert result.label == 0
