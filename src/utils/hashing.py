"""Utility functions for cache key generation and text normalization."""

import hashlib
import re


def normalize_text(text: str) -> str:
    """Normalize text for consistent cache key generation.

    Applies the following normalizations:
    1. Strip leading/trailing whitespace
    2. Collapse multiple whitespace into single space
    3. Convert to lowercase

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    # Strip and collapse whitespace
    normalized = re.sub(r"\s+", " ", text.strip())
    # Convert to lowercase for case-insensitive matching
    return normalized.lower()


def generate_cache_key(text: str, prefix: str = "classify") -> str:
    """Generate a cache key from query text.

    Uses SHA256 hash of normalized text to create a fixed-length,
    safe key that works with any cache backend.

    Args:
        text: Query text to hash.
        prefix: Optional prefix for the cache key namespace.

    Returns:
        Cache key in format: {prefix}:{sha256_hash}
    """
    normalized = normalize_text(text)
    hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{hash_digest}"
