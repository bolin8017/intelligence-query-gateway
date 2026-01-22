"""SemanticRouter model wrapper for query classification.

This module provides a high-level interface to the fine-tuned DistilBERT
model for classifying queries into Fast Path (0) or Slow Path (1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.logging import get_logger
from src.utils.model_downloader import ensure_model_exists

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Result of a single classification.

    Attributes:
        label: Classification label (0=Fast Path, 1=Slow Path).
        confidence: Confidence score (0.0 to 1.0).
        probabilities: Raw probability distribution over all classes.
    """

    label: int
    confidence: float
    probabilities: list[float]


class SemanticRouter:
    """Semantic router model for query classification.

    Wraps a fine-tuned DistilBERT model to classify queries into
    Fast Path (simple tasks) or Slow Path (complex tasks).

    The model expects text input and outputs:
    - Label 0: Fast Path (classification, summarization)
    - Label 1: Slow Path (creative_writing, open_qa)
    """

    # Label descriptions for interpretability
    LABEL_NAMES = {
        0: "fast_path",
        1: "slow_path",
    }

    def __init__(
        self,
        model_path: str | Path,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        max_length: int = 512,
        hf_model_id: str | None = None,
    ) -> None:
        """Initialize the SemanticRouter.

        Args:
            model_path: Path to the trained model directory.
            device: Device for inference ('cpu', 'cuda', or 'mps').
            max_length: Maximum token length for input sequences.
            hf_model_id: Hugging Face model ID for auto-download if local
                model doesn't exist. If None, auto-download is disabled.

        Raises:
            FileNotFoundError: If model_path does not exist and hf_model_id is None.
            RuntimeError: If model loading fails.
        """
        self.model_path = Path(model_path)
        self.hf_model_id = hf_model_id
        self.device = torch.device(device)
        self.max_length = max_length

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._is_loaded = False

        logger.info(
            "SemanticRouter initialized",
            model_path=str(self.model_path),
            hf_model_id=hf_model_id,
            device=device,
            max_length=max_length,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        This should be called during application startup, not on first request,
        to avoid cold-start latency.

        If the model doesn't exist locally and hf_model_id is provided,
        it will be automatically downloaded from Hugging Face Hub.

        Raises:
            FileNotFoundError: If model path does not exist and hf_model_id is None.
            RuntimeError: If model loading fails.
        """
        if self._is_loaded:
            logger.warning("Model already loaded, skipping reload")
            return

        # Ensure model exists (download if necessary)
        if not self.model_path.exists() and self.hf_model_id:
            logger.info(
                "Model not found locally, attempting auto-download",
                local_path=str(self.model_path),
                hf_model_id=self.hf_model_id,
            )
            self.model_path = ensure_model_exists(self.model_path, self.hf_model_id)
        elif not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Either train a model locally or provide hf_model_id for auto-download."
            )

        logger.info("Loading model", path=str(self.model_path))

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self._model.to(self.device)  # type: ignore[arg-type]
            self._model.eval()

            self._is_loaded = True
            logger.info(
                "Model loaded successfully",
                num_labels=self._model.config.num_labels,
                device=str(self.device),
            )
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise RuntimeError(f"Failed to load model: {e}") from e

    def unload(self) -> None:
        """Release model resources.

        Should be called during graceful shutdown to free GPU memory.
        """
        if not self._is_loaded:
            return

        logger.info("Unloading model")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify a batch of texts.

        Args:
            texts: List of query texts to classify.

        Returns:
            List of ClassificationResult objects, one per input text.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not texts:
            return []

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Convert to results
        results = []
        for probs in probabilities:
            probs_list = probs.cpu().tolist()
            label = int(probs.argmax().item())
            confidence = float(probs[label].item())

            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    probabilities=probs_list,
                )
            )

        return results

    def predict_single(self, text: str) -> ClassificationResult:
        """Classify a single text.

        Convenience method for single predictions.

        Args:
            text: Query text to classify.

        Returns:
            ClassificationResult for the input text.
        """
        results = self.predict([text])
        return results[0]
