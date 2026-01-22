"""Unit tests for CacheService.

Tests cover LRU eviction, TTL expiration, cache hit/miss behavior,
and size management without actual I/O or time dependencies.
"""

import time
from unittest.mock import patch

import pytest

from src.services.cache import CacheEntry, CacheService


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """CacheEntry stores value and expiration timestamp."""
        entry = CacheEntry(value="test_value", expires_at=1234567890.0)

        assert entry.value == "test_value"
        assert entry.expires_at == 1234567890.0

    def test_cache_entry_generic_type(self):
        """CacheEntry works with different value types."""
        entry_str = CacheEntry(value="text", expires_at=100.0)
        assert isinstance(entry_str.value, str)

        entry_dict = CacheEntry(value={"key": "val"}, expires_at=100.0)
        assert isinstance(entry_dict.value, dict)

        entry_int = CacheEntry(value=42, expires_at=100.0)
        assert isinstance(entry_int.value, int)


class TestCacheServiceInitialization:
    """Tests for CacheService initialization."""

    def test_cache_service_initialization_defaults(self):
        """CacheService initializes with default parameters."""
        cache = CacheService()

        assert cache._max_size == 10000
        assert cache._ttl == 300
        assert cache._level == "L1"
        assert cache._enabled is True
        assert cache.size == 0

    def test_cache_service_initialization_custom_parameters(self):
        """CacheService accepts custom parameters."""
        cache = CacheService(max_size=5000, ttl_seconds=600, level="L2")

        assert cache._max_size == 5000
        assert cache._ttl == 600
        assert cache._level == "L2"

    def test_cache_service_disabled_when_max_size_zero(self):
        """CacheService is disabled when max_size is 0."""
        cache = CacheService(max_size=0)

        assert cache.is_enabled is False

    def test_cache_service_enabled_when_max_size_positive(self):
        """CacheService is enabled when max_size > 0."""
        cache = CacheService(max_size=1)

        assert cache.is_enabled is True


@pytest.mark.asyncio
class TestCacheServiceGet:
    """Tests for CacheService.get method."""

    async def test_get_returns_none_when_cache_disabled(self):
        """get returns None when cache is disabled."""
        cache = CacheService(max_size=0)

        result = await cache.get("test_key")

        assert result is None

    async def test_get_returns_none_when_key_not_found(self):
        """get returns None when key does not exist."""
        cache = CacheService()

        result = await cache.get("nonexistent_key")

        assert result is None

    async def test_get_returns_value_when_key_exists_and_not_expired(self):
        """get returns cached value when key exists and not expired."""
        cache = CacheService(ttl_seconds=300)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "test_value")

        with patch("time.time", return_value=1100.0):
            result = await cache.get("test_key")

        assert result == "test_value"

    async def test_get_returns_none_when_entry_expired(self):
        """get returns None and removes entry when expired."""
        cache = CacheService(ttl_seconds=300)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "test_value")

        with patch("time.time", return_value=1400.0):
            result = await cache.get("test_key")

        assert result is None
        assert cache.size == 0

    async def test_get_moves_accessed_entry_to_end_for_lru(self):
        """get moves accessed entry to end for LRU tracking."""
        cache = CacheService(max_size=3, ttl_seconds=1000)

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")
            await cache.set("key3", "val3")

            await cache.get("key1")

            await cache.set("key4", "val4")

        with patch("time.time", return_value=1100.0):
            assert await cache.get("key1") == "val1"
            assert await cache.get("key2") is None
            assert await cache.get("key3") == "val3"
            assert await cache.get("key4") == "val4"

    async def test_get_handles_expiration_edge_case(self):
        """get correctly handles exact expiration timestamp."""
        cache = CacheService(ttl_seconds=100)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "test_value")

        with patch("time.time", return_value=1100.1):
            result = await cache.get("test_key")

        assert result is None


@pytest.mark.asyncio
class TestCacheServiceSet:
    """Tests for CacheService.set method."""

    async def test_set_does_nothing_when_cache_disabled(self):
        """set does nothing when cache is disabled."""
        cache = CacheService(max_size=0)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "test_value")

        assert cache.size == 0

    async def test_set_stores_value_with_ttl(self):
        """set stores value with correct expiration timestamp."""
        cache = CacheService(ttl_seconds=300)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "test_value")

        assert cache.size == 1

        with patch("time.time", return_value=1200.0):
            result = await cache.get("test_key")
            assert result == "test_value"

    async def test_set_updates_existing_key(self):
        """set updates value and resets TTL for existing key."""
        cache = CacheService(ttl_seconds=100)

        with patch("time.time", return_value=1000.0):
            await cache.set("test_key", "old_value")

        with patch("time.time", return_value=1050.0):
            await cache.set("test_key", "new_value")

        with patch("time.time", return_value=1140.0):
            result = await cache.get("test_key")

        assert result == "new_value"
        assert cache.size == 1

    async def test_set_evicts_lru_entry_when_cache_full(self):
        """set evicts least recently used entry when cache reaches max_size."""
        cache = CacheService(max_size=3, ttl_seconds=1000)

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")
            await cache.set("key3", "val3")

            assert cache.size == 3

            await cache.set("key4", "val4")

            assert cache.size == 3

        with patch("time.time", return_value=1100.0):
            assert await cache.get("key1") is None
            assert await cache.get("key2") == "val2"
            assert await cache.get("key3") == "val3"
            assert await cache.get("key4") == "val4"

    async def test_set_maintains_lru_order_with_access(self):
        """set respects LRU order modified by get accesses."""
        cache = CacheService(max_size=2, ttl_seconds=1000)

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")

            await cache.get("key1")

            await cache.set("key3", "val3")

        with patch("time.time", return_value=1100.0):
            assert await cache.get("key1") == "val1"
            assert await cache.get("key2") is None
            assert await cache.get("key3") == "val3"

    async def test_set_handles_different_value_types(self):
        """set handles various value types correctly."""
        cache = CacheService()

        with patch("time.time", return_value=1000.0):
            await cache.set("str_key", "string_value")
            await cache.set("dict_key", {"nested": "dict"})
            await cache.set("list_key", [1, 2, 3])
            await cache.set("int_key", 42)

        with patch("time.time", return_value=1100.0):
            assert await cache.get("str_key") == "string_value"
            assert await cache.get("dict_key") == {"nested": "dict"}
            assert await cache.get("list_key") == [1, 2, 3]
            assert await cache.get("int_key") == 42


class TestCacheServiceClear:
    """Tests for CacheService.clear method."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self):
        """clear removes all entries from cache."""
        cache = CacheService()

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            await cache.set("key2", "val2")
            await cache.set("key3", "val3")

            assert cache.size == 3

            cache.clear()

            assert cache.size == 0

    @pytest.mark.asyncio
    async def test_clear_allows_new_entries_after_clearing(self):
        """clear allows cache to accept new entries afterward."""
        cache = CacheService()

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            cache.clear()
            await cache.set("key2", "val2")

            result = await cache.get("key2")

        assert result == "val2"
        assert cache.size == 1

    def test_clear_on_empty_cache(self):
        """clear works correctly on empty cache."""
        cache = CacheService()

        cache.clear()

        assert cache.size == 0


class TestCacheServiceProperties:
    """Tests for CacheService property methods."""

    @pytest.mark.asyncio
    async def test_size_property_reflects_entry_count(self):
        """size property returns current number of entries."""
        cache = CacheService()

        assert cache.size == 0

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            assert cache.size == 1

            await cache.set("key2", "val2")
            assert cache.size == 2

            await cache.set("key3", "val3")
            assert cache.size == 3

    @pytest.mark.asyncio
    async def test_size_decreases_after_expiration(self):
        """size decreases when expired entry is accessed."""
        cache = CacheService(ttl_seconds=100)

        with patch("time.time", return_value=1000.0):
            await cache.set("key1", "val1")
            assert cache.size == 1

        with patch("time.time", return_value=1200.0):
            await cache.get("key1")
            assert cache.size == 0

    def test_is_enabled_property_when_disabled(self):
        """is_enabled returns False when max_size is 0."""
        cache = CacheService(max_size=0)

        assert cache.is_enabled is False

    def test_is_enabled_property_when_enabled(self):
        """is_enabled returns True when max_size > 0."""
        cache = CacheService(max_size=100)

        assert cache.is_enabled is True


@pytest.mark.asyncio
class TestCacheServiceIntegrationBehavior:
    """Integration-style tests validating complete cache behaviors."""

    async def test_cache_hit_miss_pattern(self):
        """Cache correctly handles hit/miss patterns."""
        cache = CacheService(ttl_seconds=1000)

        with patch("time.time", return_value=1000.0):
            assert await cache.get("key") is None

            await cache.set("key", "value")

            assert await cache.get("key") == "value"

    async def test_lru_eviction_with_multiple_accesses(self):
        """LRU eviction works correctly with complex access patterns."""
        cache = CacheService(max_size=3, ttl_seconds=1000)

        with patch("time.time", return_value=1000.0):
            await cache.set("a", "1")
            await cache.set("b", "2")
            await cache.set("c", "3")

            await cache.get("a")
            await cache.get("b")

            await cache.set("d", "4")

            assert await cache.get("a") == "1"
            assert await cache.get("b") == "2"
            assert await cache.get("c") is None
            assert await cache.get("d") == "4"

    async def test_ttl_expiration_across_multiple_keys(self):
        """TTL expiration works correctly for multiple keys."""
        cache = CacheService(ttl_seconds=100)

        with patch("time.time", return_value=1000.0):
            await cache.set("short_lived", "value1")

        with patch("time.time", return_value=1050.0):
            await cache.set("longer_lived", "value2")

        with patch("time.time", return_value=1110.0):
            assert await cache.get("short_lived") is None
            assert await cache.get("longer_lived") == "value2"

        with patch("time.time", return_value=1160.0):
            assert await cache.get("longer_lived") is None
