"""Test data generation utilities for load testing.

This module loads real queries directly from the Dolly dataset
for authentic load testing scenarios.
"""

import random
import string
from typing import List, Tuple, Optional

# Global cache for loaded dataset
_FAST_PATH_QUERIES: Optional[List[str]] = None
_SLOW_PATH_QUERIES: Optional[List[str]] = None
_DATASET_LOADED = False


def _load_dolly_dataset():
    """Load and cache queries from the Dolly dataset."""
    global _FAST_PATH_QUERIES, _SLOW_PATH_QUERIES, _DATASET_LOADED

    if _DATASET_LOADED:
        return

    try:
        from datasets import load_dataset

        print("Loading Dolly dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

        fast_path = []  # classification, summarization
        slow_path = []  # open_qa, general_qa, creative_writing

        for item in dataset:
            cat = item["category"]
            text = item["instruction"]

            # Truncate very long queries
            if len(text) > 500:
                text = text[:500]

            if cat in ["classification", "summarization"]:
                fast_path.append(text)
            elif cat in ["open_qa", "general_qa", "creative_writing"]:
                slow_path.append(text)

        _FAST_PATH_QUERIES = fast_path
        _SLOW_PATH_QUERIES = slow_path
        _DATASET_LOADED = True

        print(f"Loaded {len(fast_path)} Fast Path queries, {len(slow_path)} Slow Path queries")

    except ImportError:
        print("Warning: 'datasets' library not installed. Using fallback queries.")
        _use_fallback_queries()
    except Exception as e:
        print(f"Warning: Failed to load Dolly dataset: {e}. Using fallback queries.")
        _use_fallback_queries()


def _use_fallback_queries():
    """Use hardcoded fallback queries if dataset loading fails."""
    global _FAST_PATH_QUERIES, _SLOW_PATH_QUERIES, _DATASET_LOADED

    _FAST_PATH_QUERIES = [
        "Which is a species of fish? Black Mamba or Black Sea Bass",
        "Classify each of the following as inclined or flat: stairs, beach, mountain, lake",
        "Identify which instrument is string or percussion: Sabar, Sharud",
        "Tell me which of these countries has more than 1 billion population: Japan, China, India",
        "Which of the following exercises are push exercises: bench press, bicep curl, pull up",
    ]

    _SLOW_PATH_QUERIES = [
        "What is the most important gear to bring for backpacking?",
        "Why should you get out of your comfort zone?",
        "What is a bond in finance?",
        "How can we reduce the impact of global warming?",
        "Why do people sleep?",
    ]

    _DATASET_LOADED = True


def generate_random_string(length: int = 20) -> str:
    """Generate a random ASCII string."""
    return "".join(random.choices(string.ascii_letters, k=length))


def generate_unique_query() -> str:
    """Generate a unique query from the Dolly dataset.

    Returns:
        A query string from the Dolly dataset.
    """
    _load_dolly_dataset()

    # Mix of fast and slow path queries (matching dataset distribution ~33% fast, ~67% slow)
    if random.random() < 0.33:
        return random.choice(_FAST_PATH_QUERIES)
    else:
        return random.choice(_SLOW_PATH_QUERIES)


def generate_common_query() -> str:
    """Generate a common query for cache hit testing.

    Returns:
        A query from a smaller subset for higher cache hit rate.
    """
    _load_dolly_dataset()

    # Use first 20 queries from each category for cache testing
    common_fast = _FAST_PATH_QUERIES[:20]
    common_slow = _SLOW_PATH_QUERIES[:20]

    if random.random() < 0.33:
        return random.choice(common_fast)
    else:
        return random.choice(common_slow)


def generate_fast_path_query() -> str:
    """Generate a Fast Path query (classification/summarization).

    Returns:
        A query expected to return label 0.
    """
    _load_dolly_dataset()
    return random.choice(_FAST_PATH_QUERIES)


def generate_slow_path_query() -> str:
    """Generate a Slow Path query (open_qa/creative_writing).

    Returns:
        A query expected to return label 1.
    """
    _load_dolly_dataset()
    return random.choice(_SLOW_PATH_QUERIES)


def generate_query_with_expected_label() -> Tuple[str, int]:
    """Generate a query with its expected label.

    Returns:
        Tuple of (query, expected_label).
    """
    _load_dolly_dataset()

    if random.random() < 0.33:
        return (random.choice(_FAST_PATH_QUERIES), 0)
    else:
        return (random.choice(_SLOW_PATH_QUERIES), 1)


def generate_query_batch(
    size: int,
    cache_hit_ratio: float = 0.3,
) -> List[str]:
    """Generate a batch of queries with specified cache hit ratio.

    Args:
        size: Number of queries to generate.
        cache_hit_ratio: Ratio of common queries (0.0-1.0).

    Returns:
        List of query strings.
    """
    queries = []
    for _ in range(size):
        if random.random() < cache_hit_ratio:
            queries.append(generate_common_query())
        else:
            queries.append(generate_unique_query())
    return queries


def get_common_queries() -> List[str]:
    """Get the list of common queries for cache testing.

    Returns:
        List of common query strings (subset for cache hits).
    """
    _load_dolly_dataset()
    return _FAST_PATH_QUERIES[:10] + _SLOW_PATH_QUERIES[:10]


def get_fast_path_queries() -> List[str]:
    """Get all Fast Path queries from Dolly dataset.

    Returns:
        List of Fast Path query strings.
    """
    _load_dolly_dataset()
    return _FAST_PATH_QUERIES.copy()


def get_slow_path_queries() -> List[str]:
    """Get all Slow Path queries from Dolly dataset.

    Returns:
        List of Slow Path query strings.
    """
    _load_dolly_dataset()
    return _SLOW_PATH_QUERIES.copy()


def get_dataset_stats() -> dict:
    """Get statistics about the loaded dataset.

    Returns:
        Dictionary with dataset statistics.
    """
    _load_dolly_dataset()
    return {
        "fast_path_count": len(_FAST_PATH_QUERIES),
        "slow_path_count": len(_SLOW_PATH_QUERIES),
        "total_count": len(_FAST_PATH_QUERIES) + len(_SLOW_PATH_QUERIES),
        "fast_path_ratio": len(_FAST_PATH_QUERIES) / (len(_FAST_PATH_QUERIES) + len(_SLOW_PATH_QUERIES)),
    }


if __name__ == "__main__":
    # Quick test when run directly
    print("Loading dataset...")
    stats = get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"  Fast Path queries: {stats['fast_path_count']}")
    print(f"  Slow Path queries: {stats['slow_path_count']}")
    print(f"  Total: {stats['total_count']}")
    print(f"  Fast Path ratio: {stats['fast_path_ratio']:.1%}")

    print("\nFast Path Queries (Label 0) - Sample:")
    for query in get_fast_path_queries()[:3]:
        print(f"  - {query[:70]}...")

    print("\nSlow Path Queries (Label 1) - Sample:")
    for query in get_slow_path_queries()[:3]:
        print(f"  - {query[:70]}...")

    print("\nRandom unique queries:")
    for _ in range(3):
        print(f"  - {generate_unique_query()[:70]}...")
