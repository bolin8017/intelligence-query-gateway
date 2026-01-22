"""Model downloader utility for fetching models from Hugging Face Hub.

This module provides functionality to download pre-trained models from
Hugging Face Model Hub if they don't exist locally.
"""

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.logging import get_logger

logger = get_logger(__name__)

# Default model repository on Hugging Face Hub
DEFAULT_HF_MODEL_ID = "bolin8017/query-gateway-router"


def download_model_from_hub(
    model_id: str = DEFAULT_HF_MODEL_ID,
    local_dir: str | Path | None = None,
) -> Path:
    """Download model from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID (e.g., 'username/model-name').
        local_dir: Local directory to save the model. If None, uses
            transformers cache directory.

    Returns:
        Path to the downloaded model directory.

    Raises:
        RuntimeError: If download fails.
    """
    logger.info("Downloading model from Hugging Face Hub", model_id=model_id)

    try:
        # Download tokenizer and model
        # These will be cached in ~/.cache/huggingface by default
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(local_dir) if local_dir else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            cache_dir=str(local_dir) if local_dir else None,
        )

        # If local_dir is specified, save the model there
        if local_dir:
            local_path = Path(local_dir)
            local_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(local_path))
            model.save_pretrained(str(local_path))
            logger.info("Model saved to local directory", path=str(local_path))
            return local_path
        else:
            # Return the cache directory path
            cache_path = Path(tokenizer.name_or_path)
            logger.info("Model cached", path=str(cache_path))
            return cache_path

    except Exception as e:
        logger.error("Failed to download model", error=str(e), model_id=model_id)
        raise RuntimeError(f"Failed to download model from {model_id}: {e}") from e


def ensure_model_exists(model_path: str | Path, hf_model_id: str | None = None) -> Path:
    """Ensure model exists locally, downloading if necessary.

    This function checks if the model exists at the specified path.
    If not, it downloads the model from Hugging Face Hub.

    Args:
        model_path: Path where the model should be located.
        hf_model_id: Hugging Face model ID to download if local model
            doesn't exist. If None, uses DEFAULT_HF_MODEL_ID.

    Returns:
        Path to the model directory (either existing or newly downloaded).

    Raises:
        RuntimeError: If model doesn't exist locally and download fails.
    """
    model_path = Path(model_path)

    # Check if model exists locally
    if model_path.exists() and (model_path / "config.json").exists():
        logger.info("Model found locally", path=str(model_path))
        return model_path

    # Model doesn't exist locally, download from Hub
    logger.warning(
        "Model not found locally, will download from Hugging Face Hub",
        local_path=str(model_path),
    )

    hf_model_id = hf_model_id or DEFAULT_HF_MODEL_ID
    return download_model_from_hub(model_id=hf_model_id, local_dir=model_path)
