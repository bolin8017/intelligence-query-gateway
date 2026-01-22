"""Cache services for L1 (in-memory) and L2 (Redis) caching.

This module implements a two-level cache architecture:
- L1: In-memory LRU cache for ultra-fast lookups
- L2: Redis cache for distributed/persistent caching

The TwoLevelCache class coordinates both layers with write-through strategy.
"""

import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from src.core.logging import get_logger
from src.core.metrics import CACHE_HIT_COUNT, CACHE_MISS_COUNT, CACHE_SIZE

logger = get_logger(__name__)

T = TypeVar("T")


class CacheProtocol(ABC, Generic[T]):
    """Abstract base class defining the cache interface.

    All cache implementations (L1, L2, TwoLevel) must implement this protocol.
    """

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get cached value by key."""
        ...

    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        """Store value in cache."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        ...

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if cache is enabled."""
        ...


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and expiration timestamp.

    Attributes:
        value: Cached value.
        expires_at: Unix timestamp when this entry expires.
    """

    value: T
    expires_at: float


class CacheService(CacheProtocol[T]):
    """L1 in-memory LRU cache with TTL support.

    This cache uses OrderedDict for efficient LRU eviction and tracks
    expiration times for each entry. When the cache reaches max_size,
    the least recently used item is evicted.

    The cache is thread-safe for async operations since Python's GIL
    protects OrderedDict operations, and we only use simple atomic ops.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 300,
        level: str = "L1",
    ) -> None:
        """Initialize CacheService.

        Args:
            max_size: Maximum number of entries to store (0 to disable).
            ttl_seconds: Time-to-live for each entry in seconds.
            level: Cache level identifier for metrics (default "L1").
        """
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._level = level
        self._enabled = max_size > 0

        logger.info(
            "CacheService initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            level=level,
            enabled=self._enabled,
        )

    async def get(self, key: str) -> T | None:
        """Get cached value if it exists and hasn't expired.

        This method implements LRU by moving accessed items to the end
        of the OrderedDict.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        if not self._enabled:
            CACHE_MISS_COUNT.labels(level=self._level).inc()
            return None

        entry = self._cache.get(key)

        # Cache miss
        if entry is None:
            CACHE_MISS_COUNT.labels(level=self._level).inc()
            logger.debug("Cache miss", key=key, level=self._level)
            return None

        # Check if expired
        if time.time() > entry.expires_at:
            # Remove expired entry
            del self._cache[key]
            CACHE_SIZE.labels(level=self._level).set(len(self._cache))
            CACHE_MISS_COUNT.labels(level=self._level).inc()
            logger.debug("Cache expired", key=key, level=self._level)
            return None

        # Cache hit - move to end for LRU
        self._cache.move_to_end(key)
        CACHE_HIT_COUNT.labels(level=self._level).inc()
        logger.debug("Cache hit", key=key, level=self._level)
        return entry.value

    async def set(self, key: str, value: T) -> None:
        """Store value in cache with TTL.

        If cache is full, evicts the least recently used entry.
        If key already exists, updates the value and resets TTL.

        Args:
            key: Cache key to store.
            value: Value to cache.
        """
        if not self._enabled:
            return

        expires_at = time.time() + self._ttl
        entry = CacheEntry(value=value, expires_at=expires_at)

        # If key exists, remove it first (we'll re-add at the end)
        if key in self._cache:
            del self._cache[key]

        # Add new entry at the end
        self._cache[key] = entry

        # Evict LRU entry if over capacity
        if len(self._cache) > self._max_size:
            # Remove the first (oldest) item
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(
                "Cache eviction",
                evicted_key=evicted_key,
                level=self._level,
            )

        # Update metrics
        CACHE_SIZE.labels(level=self._level).set(len(self._cache))
        logger.debug(
            "Cache set",
            key=key,
            level=self._level,
            cache_size=len(self._cache),
        )

    def clear(self) -> None:
        """Clear all entries from the cache.

        Useful for testing or manual cache invalidation.
        """
        self._cache.clear()
        CACHE_SIZE.labels(level=self._level).set(0)
        logger.info("Cache cleared", level=self._level)

    @property
    def size(self) -> int:
        """Get current cache size (number of entries)."""
        return len(self._cache)

    @property
    def is_enabled(self) -> bool:
        """Check if cache is enabled."""
        return self._enabled


class RedisCacheService(CacheProtocol[T]):
    """L2 Redis cache with graceful degradation.

    This cache connects to Redis for distributed caching. When Redis is
    unavailable, operations fail silently and log warnings (graceful degradation).

    Features:
    - Connection pooling via redis-py async client
    - JSON serialization for cross-language compatibility
    - TTL-based expiration handled by Redis
    - Automatic reconnection on transient failures
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        serializer: type | None = None,
    ) -> None:
        """Initialize RedisCacheService.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0).
            ttl_seconds: Time-to-live for each entry in seconds.
            serializer: Optional class with to_dict/from_dict for custom serialization.
        """
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._serializer = serializer
        self._client: Any = None  # redis.asyncio.Redis
        self._enabled = True
        self._healthy = True

        logger.info(
            "RedisCacheService initialized",
            redis_url=self._mask_url(redis_url),
            ttl_seconds=ttl_seconds,
        )

    def _mask_url(self, url: str) -> str:
        """Mask password in Redis URL for logging."""
        if "@" in url:
            # redis://:password@host:port -> redis://***@host:port
            prefix, suffix = url.split("@", 1)
            return f"{prefix.split(':')[0]}://***@{suffix}"
        return url

    async def connect(self) -> None:
        """Establish connection to Redis.

        Should be called during application startup.
        """
        try:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            self._healthy = True
            logger.info("Redis connection established")
        except ImportError:
            logger.error(
                "redis package not installed. Install with: pip install redis"
            )
            self._enabled = False
            self._healthy = False
        except Exception as e:
            logger.warning(
                "Redis connection failed, degrading gracefully",
                error=str(e),
            )
            self._healthy = False

    async def disconnect(self) -> None:
        """Close Redis connection.

        Should be called during application shutdown.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    async def get(self, key: str) -> T | None:
        """Get cached value from Redis.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached value if found, None otherwise.
            Returns None on Redis errors (graceful degradation).
        """
        if not self._enabled or self._client is None:
            CACHE_MISS_COUNT.labels(level="L2").inc()
            return None

        try:
            data = await self._client.get(key)

            if data is None:
                CACHE_MISS_COUNT.labels(level="L2").inc()
                logger.debug("L2 cache miss", key=key)
                return None

            # Deserialize JSON
            value_dict = json.loads(data)

            # Use custom deserializer if provided
            if self._serializer is not None and hasattr(self._serializer, "from_dict"):
                value = self._serializer.from_dict(value_dict)
            else:
                value = value_dict

            CACHE_HIT_COUNT.labels(level="L2").inc()
            self._healthy = True
            logger.debug("L2 cache hit", key=key)
            return value  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning(
                "Redis get failed, degrading gracefully",
                key=key,
                error=str(e),
            )
            self._healthy = False
            CACHE_MISS_COUNT.labels(level="L2").inc()
            return None

    async def set(self, key: str, value: T) -> None:
        """Store value in Redis with TTL.

        Args:
            key: Cache key to store.
            value: Value to cache (must be JSON serializable or have to_dict method).
        """
        if not self._enabled or self._client is None:
            return

        try:
            # Serialize to JSON
            if hasattr(value, "to_dict"):
                value_dict = value.to_dict()
            elif hasattr(value, "__dict__"):
                value_dict = value.__dict__
            else:
                value_dict = value

            data = json.dumps(value_dict)

            await self._client.setex(key, self._ttl, data)
            self._healthy = True
            logger.debug("L2 cache set", key=key)

        except Exception as e:
            logger.warning(
                "Redis set failed, degrading gracefully",
                key=key,
                error=str(e),
            )
            self._healthy = False

    def clear(self) -> None:
        """Clear is not implemented for L2 (would affect other services).

        Use Redis CLI or management tools for cache clearing.
        """
        logger.warning("L2 cache clear not implemented (use Redis CLI)")

    @property
    def is_enabled(self) -> bool:
        """Check if Redis cache is enabled."""
        return self._enabled

    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return self._healthy


class TwoLevelCache(CacheProtocol[T]):
    """Coordinates L1 (in-memory) and L2 (Redis) caches.

    Implements write-through caching strategy:
    - Read: Check L1 first, then L2 on miss, populate L1 from L2 hit
    - Write: Write to both L1 and L2 simultaneously

    If L2 is unavailable, falls back to L1-only operation (graceful degradation).
    """

    def __init__(
        self,
        l1_cache: CacheService[T],
        l2_cache: RedisCacheService[T] | None = None,
    ) -> None:
        """Initialize TwoLevelCache.

        Args:
            l1_cache: L1 in-memory cache (required).
            l2_cache: L2 Redis cache (optional, None for L1-only mode).
        """
        self._l1 = l1_cache
        self._l2 = l2_cache

        logger.info(
            "TwoLevelCache initialized",
            l1_enabled=l1_cache.is_enabled,
            l2_enabled=l2_cache is not None and l2_cache.is_enabled,
        )

    async def get(self, key: str) -> T | None:
        """Get cached value, checking L1 first then L2.

        On L2 hit, the value is promoted to L1 for faster subsequent access.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached value if found in either level, None otherwise.
        """
        # Try L1 first (fastest)
        value = await self._l1.get(key)
        if value is not None:
            return value

        # Try L2 if available
        if self._l2 is not None and self._l2.is_enabled:
            value = await self._l2.get(key)
            if value is not None:
                # Promote to L1 for faster future access
                await self._l1.set(key, value)
                logger.debug("Promoted L2 hit to L1", key=key)
                return value

        return None

    async def set(self, key: str, value: T) -> None:
        """Store value in both L1 and L2 (write-through).

        Args:
            key: Cache key to store.
            value: Value to cache.
        """
        # Write to L1 (always)
        await self._l1.set(key, value)

        # Write to L2 if available (fire-and-forget on failure)
        if self._l2 is not None and self._l2.is_enabled:
            await self._l2.set(key, value)

    def clear(self) -> None:
        """Clear L1 cache. L2 is not cleared (use Redis CLI)."""
        self._l1.clear()
        if self._l2 is not None:
            self._l2.clear()

    @property
    def is_enabled(self) -> bool:
        """Check if at least L1 is enabled."""
        return self._l1.is_enabled

    @property
    def l1(self) -> CacheService[T]:
        """Get L1 cache instance."""
        return self._l1

    @property
    def l2(self) -> RedisCacheService[T] | None:
        """Get L2 cache instance (may be None)."""
        return self._l2
