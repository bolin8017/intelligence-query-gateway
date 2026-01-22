"""FastAPI dependencies for dependency injection.

This module implements constructor injection pattern using FastAPI's
dependency injection system.
"""

from typing import Annotated

from fastapi import Depends

from src.core.config import Settings, get_settings
from src.models.semantic_router import SemanticRouter
from src.services.batching import BatchingService
from src.services.cache import CacheProtocol, CacheService, RedisCacheService, TwoLevelCache
from src.services.classifier import ClassifierService, ClassifyResult

# Global service instances (initialized in lifespan)
_semantic_router: SemanticRouter | None = None
_classifier_service: ClassifierService | None = None
_batching_service: BatchingService | None = None
_cache_service: TwoLevelCache[ClassifyResult] | None = None
_l2_cache: RedisCacheService[ClassifyResult] | None = None


def get_semantic_router() -> SemanticRouter:
    """Get the SemanticRouter instance.

    Returns:
        SemanticRouter instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _semantic_router is None:
        raise RuntimeError("SemanticRouter not initialized")
    return _semantic_router


def get_classifier_service() -> ClassifierService:
    """Get the ClassifierService instance.

    Returns:
        ClassifierService instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _classifier_service is None:
        raise RuntimeError("ClassifierService not initialized")
    return _classifier_service


def get_batching_service() -> BatchingService:
    """Get the BatchingService instance.

    Returns:
        BatchingService instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _batching_service is None:
        raise RuntimeError("BatchingService not initialized")
    return _batching_service


def get_cache_service() -> CacheProtocol[ClassifyResult]:
    """Get the CacheService instance (TwoLevelCache or CacheService).

    Returns:
        Cache instance implementing CacheProtocol.

    Raises:
        RuntimeError: If not initialized.
    """
    if _cache_service is None:
        raise RuntimeError("CacheService not initialized")
    return _cache_service


def get_l2_cache() -> RedisCacheService[ClassifyResult] | None:
    """Get the L2 Redis cache instance.

    Returns:
        RedisCacheService instance or None if not configured.
    """
    return _l2_cache


async def init_services(settings: Settings) -> None:
    """Initialize all services during application startup.

    Args:
        settings: Application settings.
    """
    global _semantic_router, _classifier_service, _batching_service, _cache_service, _l2_cache

    # Initialize model
    _semantic_router = SemanticRouter(
        model_path=settings.model_path,
        device=settings.model_device,
        max_length=settings.model_max_length,
    )
    _semantic_router.load()

    # Initialize classifier service
    _classifier_service = ClassifierService(model=_semantic_router)

    # Initialize L1 cache (in-memory)
    l1_cache = CacheService[ClassifyResult](
        max_size=settings.cache_l1_size,
        ttl_seconds=settings.cache_l1_ttl_sec,
        level="L1",
    )

    # Initialize L2 cache (Redis) if configured
    if settings.is_redis_enabled:
        _l2_cache = RedisCacheService[ClassifyResult](
            redis_url=settings.redis_url,
            ttl_seconds=settings.cache_l2_ttl_sec,
            serializer=ClassifyResult,
        )
        await _l2_cache.connect()

    # Create two-level cache coordinator
    _cache_service = TwoLevelCache(
        l1_cache=l1_cache,
        l2_cache=_l2_cache,
    )

    # Initialize batching service
    _batching_service = BatchingService(
        classifier=_classifier_service,
        max_batch_size=settings.batch_max_size,
        max_wait_ms=settings.batch_max_wait_ms,
    )

    # Start batching service background task
    await _batching_service.start()


async def cleanup_services() -> None:
    """Cleanup services during application shutdown."""
    global _semantic_router, _classifier_service, _batching_service, _cache_service, _l2_cache

    # Stop batching service gracefully
    if _batching_service is not None:
        await _batching_service.stop()
        _batching_service = None

    # Clear cache
    if _cache_service is not None:
        _cache_service.clear()
        _cache_service = None

    # Disconnect L2 Redis cache
    if _l2_cache is not None:
        await _l2_cache.disconnect()
        _l2_cache = None

    # Unload model
    if _semantic_router is not None:
        _semantic_router.unload()
        _semantic_router = None

    _classifier_service = None


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
ClassifierDep = Annotated[ClassifierService, Depends(get_classifier_service)]
BatchingDep = Annotated[BatchingService, Depends(get_batching_service)]
CacheDep = Annotated[CacheProtocol[ClassifyResult], Depends(get_cache_service)]
L2CacheDep = Annotated[RedisCacheService[ClassifyResult] | None, Depends(get_l2_cache)]
