"""Integration tests for CacheService with hashing utilities.

Tests validate cache behavior with realistic key generation
and cache hit/miss patterns.
"""

from unittest.mock import patch

import pytest

from src.utils.hashing import generate_cache_key


@pytest.mark.asyncio
class TestCacheServiceIntegration:
    """Integration tests for CacheService."""

    async def test_cache_hit_miss_pattern_with_real_keys(self, cache_service):
        """Cache correctly handles hit/miss with real cache keys."""
        query = "What is the capital of France?"
        cache_key = generate_cache_key(query)

        cached = await cache_service.get(cache_key)
        assert cached is None

        test_result = {"label": 0, "confidence": 0.95}
        await cache_service.set(cache_key, test_result)

        cached = await cache_service.get(cache_key)
        assert cached == test_result

    async def test_cache_key_normalization_consistency(self, cache_service):
        """Cache keys are consistent across normalized variations."""
        query1 = "What is AI?"
        query2 = "what is ai?"
        query3 = "  WHAT   IS   AI?  "

        key1 = generate_cache_key(query1)
        key2 = generate_cache_key(query2)
        key3 = generate_cache_key(query3)

        assert key1 == key2 == key3

        test_result = {"label": 0, "confidence": 0.9}
        await cache_service.set(key1, test_result)

        assert await cache_service.get(key2) == test_result
        assert await cache_service.get(key3) == test_result

    async def test_cache_stores_classification_results(self, cache_service):
        """Cache correctly stores and retrieves classification results."""
        query = "Write a story"
        cache_key = generate_cache_key(query)

        result = {"label": 1, "confidence": 0.85, "category": "slow_path"}

        await cache_service.set(cache_key, result)
        cached = await cache_service.get(cache_key)

        assert cached["label"] == 1
        assert cached["confidence"] == 0.85
        assert cached["category"] == "slow_path"

    async def test_cache_handles_multiple_queries(self, cache_service):
        """Cache handles multiple different queries correctly."""
        queries_and_results = [
            ("What is ML?", {"label": 0, "confidence": 0.92}),
            ("Write a poem", {"label": 1, "confidence": 0.88}),
            ("Summarize this", {"label": 0, "confidence": 0.90}),
        ]

        for query, result in queries_and_results:
            cache_key = generate_cache_key(query)
            await cache_service.set(cache_key, result)

        for query, expected_result in queries_and_results:
            cache_key = generate_cache_key(query)
            cached = await cache_service.get(cache_key)
            assert cached == expected_result

    async def test_cache_eviction_with_real_keys(self, cache_service):
        """Cache evicts LRU entries when full."""
        cache_service._max_size = 3

        queries = [f"Query {i}" for i in range(4)]
        results = [{"label": 0, "confidence": 0.9} for _ in range(4)]

        with patch("time.time", return_value=1000.0):
            for query, result in zip(queries, results):
                cache_key = generate_cache_key(query)
                await cache_service.set(cache_key, result)

        with patch("time.time", return_value=1100.0):
            key0 = generate_cache_key(queries[0])
            assert await cache_service.get(key0) is None

            key3 = generate_cache_key(queries[3])
            assert await cache_service.get(key3) is not None

    async def test_cache_ttl_expiration_with_real_keys(self, cache_service):
        """Cache entries expire after TTL with real keys."""
        cache_service._ttl = 100

        query = "Test query"
        cache_key = generate_cache_key(query)
        result = {"label": 0, "confidence": 0.9}

        with patch("time.time", return_value=1000.0):
            await cache_service.set(cache_key, result)

        with patch("time.time", return_value=1050.0):
            assert await cache_service.get(cache_key) == result

        with patch("time.time", return_value=1150.0):
            assert await cache_service.get(cache_key) is None

    async def test_cache_disabled_behavior(self):
        """Cache with max_size=0 always returns None."""
        from src.services.cache import CacheService

        disabled_cache = CacheService(max_size=0)

        query = "Test"
        cache_key = generate_cache_key(query)

        await disabled_cache.set(cache_key, {"label": 0})
        cached = await disabled_cache.get(cache_key)

        assert cached is None
        assert disabled_cache.size == 0

    async def test_cache_clear_removes_all_entries(self, cache_service):
        """Cache clear removes all cached results."""
        queries = [f"Query {i}" for i in range(5)]

        with patch("time.time", return_value=1000.0):
            for query in queries:
                cache_key = generate_cache_key(query)
                await cache_service.set(cache_key, {"label": 0})

            assert cache_service.size == 5

            cache_service.clear()

            assert cache_service.size == 0

        with patch("time.time", return_value=1100.0):
            for query in queries:
                cache_key = generate_cache_key(query)
                assert await cache_service.get(cache_key) is None

    async def test_cache_updates_existing_entry(self, cache_service):
        """Cache updates value and TTL when setting existing key."""
        query = "Test query"
        cache_key = generate_cache_key(query)

        with patch("time.time", return_value=1000.0):
            await cache_service.set(cache_key, {"label": 0, "confidence": 0.8})

        with patch("time.time", return_value=1050.0):
            await cache_service.set(cache_key, {"label": 1, "confidence": 0.9})

        with patch("time.time", return_value=1100.0):
            cached = await cache_service.get(cache_key)
            assert cached["label"] == 1
            assert cached["confidence"] == 0.9

    async def test_cache_handles_unicode_queries(self, cache_service):
        """Cache handles unicode characters in queries."""
        query = "Qu'est-ce que l'intelligence artificielle?"
        cache_key = generate_cache_key(query)
        result = {"label": 0, "confidence": 0.9}

        with patch("time.time", return_value=1000.0):
            await cache_service.set(cache_key, result)

            cached = await cache_service.get(cache_key)

        assert cached == result

    async def test_cache_different_queries_different_keys(self, cache_service):
        """Different queries produce different cache keys and separate entries."""
        query1 = "What is AI?"
        query2 = "What is ML?"

        key1 = generate_cache_key(query1)
        key2 = generate_cache_key(query2)

        assert key1 != key2

        with patch("time.time", return_value=1000.0):
            await cache_service.set(key1, {"label": 0})
            await cache_service.set(key2, {"label": 1})

            assert await cache_service.get(key1) != await cache_service.get(key2)
