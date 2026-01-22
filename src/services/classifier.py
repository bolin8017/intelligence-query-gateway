"""ClassifierService for query classification.

This service provides the high-level interface for classifying queries,
orchestrating the model inference and result formatting.
"""

from dataclasses import dataclass

from src.core.exceptions import ModelNotReadyError
from src.core.logging import get_logger
from src.core.metrics import (
    CLASSIFICATION_COUNT,
    CONFIDENCE_SCORE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_LATENCY,
)
from src.models.semantic_router import ClassificationResult, SemanticRouter

import time

logger = get_logger(__name__)


@dataclass
class ClassifyResult:
    """Result of query classification with metadata.

    Attributes:
        label: Classification label (0=Fast Path, 1=Slow Path).
        confidence: Confidence score (0.0 to 1.0).
        category: Human-readable category name.
    """

    label: int
    confidence: float
    category: str

    @classmethod
    def from_model_result(cls, result: ClassificationResult) -> "ClassifyResult":
        """Create ClassifyResult from model output.

        Args:
            result: Raw model classification result.

        Returns:
            ClassifyResult with category name mapped.
        """
        category_map = {
            0: "fast_path",
            1: "slow_path",
        }
        return cls(
            label=result.label,
            confidence=result.confidence,
            category=category_map.get(result.label, "unknown"),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/Redis storage.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "label": self.label,
            "confidence": self.confidence,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassifyResult":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with label, confidence, category keys.

        Returns:
            ClassifyResult instance.
        """
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            category=data["category"],
        )


class ClassifierService:
    """Service for classifying queries using the semantic router.

    This service wraps the SemanticRouter model and provides:
    - Batch classification support
    - Metrics collection
    - Error handling

    In Phase 2, this will be integrated with BatchingService and CacheService.
    """

    def __init__(self, model: SemanticRouter) -> None:
        """Initialize ClassifierService.

        Args:
            model: SemanticRouter instance for inference.
        """
        self._model = model
        logger.info("ClassifierService initialized")

    @property
    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests."""
        return self._model.is_loaded

    def classify(self, text: str) -> ClassifyResult:
        """Classify a single query.

        Args:
            text: Query text to classify.

        Returns:
            ClassifyResult with label, confidence, and category.

        Raises:
            ModelNotReadyError: If model is not loaded.
        """
        results = self.classify_batch([text])
        return results[0]

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        """Classify a batch of queries.

        Args:
            texts: List of query texts to classify.

        Returns:
            List of ClassifyResult objects.

        Raises:
            ModelNotReadyError: If model is not loaded.
        """
        if not self._model.is_loaded:
            raise ModelNotReadyError("Model is not loaded")

        if not texts:
            return []

        # Record batch size
        INFERENCE_BATCH_SIZE.observe(len(texts))

        # Perform inference with timing
        start_time = time.perf_counter()
        model_results = self._model.predict(texts)
        latency = time.perf_counter() - start_time

        # Record metrics
        INFERENCE_LATENCY.observe(latency)

        # Convert to service results and record per-result metrics
        results = []
        for model_result in model_results:
            result = ClassifyResult.from_model_result(model_result)
            results.append(result)

            # Record classification metrics
            CLASSIFICATION_COUNT.labels(label=str(result.label)).inc()
            CONFIDENCE_SCORE.labels(label=str(result.label)).observe(result.confidence)

        logger.debug(
            "Batch classification completed",
            batch_size=len(texts),
            latency_ms=latency * 1000,
        )

        return results
