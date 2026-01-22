"""Health check endpoints for Kubernetes probes.

GET /health/live - Liveness probe (is the process running?)
GET /health/ready - Readiness probe (is the service ready to accept traffic?)
GET /health/deep - Deep health check with detailed metrics (not for K8s probes)
"""

from typing import Any, Dict

from fastapi import APIRouter, Response, status

from src.api.dependencies import (
    get_batching_service,
    get_cache_service,
    get_classifier_service,
    get_l2_cache,
)
from src.api.schemas import HealthResponse
from src.core.logging import get_logger
from src.core.metrics import (
    ACTIVE_REQUESTS,
    BATCH_QUEUE_SIZE,
    CACHE_SIZE,
    MODEL_LOADED,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is alive"},
    },
)
async def liveness() -> HealthResponse:
    """Liveness probe for Kubernetes.

    This endpoint should return quickly and only check if the process
    is running. It should NOT check external dependencies.

    Returns:
        HealthResponse indicating the service is alive.
    """
    return HealthResponse(
        status="healthy",
        checks={"process": True},
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness(response: Response) -> HealthResponse:
    """Readiness probe for Kubernetes.

    This endpoint checks if the service is ready to accept traffic.
    It verifies that the model is loaded and all dependencies are available.

    Returns:
        HealthResponse with detailed check results.
    """
    checks = {}

    # Check model readiness
    try:
        classifier = get_classifier_service()
        checks["model"] = classifier.is_ready
    except RuntimeError:
        checks["model"] = False

    # Determine overall status
    is_healthy = all(checks.values())

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("Readiness check failed", checks=checks)

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        checks=checks,
    )


@router.get(
    "/deep",
    responses={
        200: {"description": "Detailed health check with metrics"},
    },
)
async def deep_health_check() -> Dict[str, Any]:
    """Deep health check with detailed system metrics.

    This endpoint provides comprehensive system diagnostics including:
    - Model status and device information
    - Cache statistics (size, hit rate)
    - Batch processing metrics (queue depth, batch sizes)
    - Current system load (active requests)

    Note: This endpoint should NOT be used for Kubernetes probes as it
    may be slower than live/ready endpoints. Use it for manual debugging
    or monitoring dashboards.

    Returns:
        Dictionary with detailed health and performance metrics.
    """
    checks: Dict[str, Any] = {}

    # Model information
    try:
        classifier = get_classifier_service()
        checks["model"] = {
            "loaded": classifier.is_ready,
            "device": str(classifier._model.device),
        }
    except RuntimeError as e:
        checks["model"] = {
            "loaded": False,
            "error": str(e),
        }

    # Cache information
    try:
        cache = get_cache_service()
        # Get L1 cache size from Prometheus gauge
        l1_size = CACHE_SIZE.labels(level="L1")._value.get()

        cache_info: Dict[str, Any] = {
            "l1_enabled": cache.l1.is_enabled if hasattr(cache, "l1") else cache.is_enabled,
            "l1_size": l1_size,
        }

        # Add L1 max_size if available
        if hasattr(cache, "l1"):
            cache_info["l1_max_size"] = cache.l1._max_size
        elif hasattr(cache, "_max_size"):
            cache_info["l1_max_size"] = cache._max_size

        # L2 (Redis) cache status
        l2_cache = get_l2_cache()
        if l2_cache is not None:
            cache_info["l2_enabled"] = l2_cache.is_enabled
            cache_info["l2_healthy"] = l2_cache.is_healthy
            # Get L2 cache size from Prometheus gauge (if available)
            try:
                l2_size = CACHE_SIZE.labels(level="L2")._value.get()
                cache_info["l2_size"] = l2_size
            except Exception:
                pass
        else:
            cache_info["l2_enabled"] = False

        checks["cache"] = cache_info
    except RuntimeError as e:
        checks["cache"] = {
            "error": str(e),
        }

    # Batch processor information
    try:
        batching = get_batching_service()
        queue_size = BATCH_QUEUE_SIZE._value.get()
        checks["batch_processor"] = {
            "running": batching.is_running,
            "queue_depth": queue_size,
            "max_batch_size": batching._max_batch_size,
            "max_wait_sec": batching._max_wait_sec,
        }
    except RuntimeError as e:
        checks["batch_processor"] = {
            "error": str(e),
        }

    # Current system metrics
    checks["metrics"] = {
        "active_requests": ACTIVE_REQUESTS._value.get(),
        "model_loaded": MODEL_LOADED._value.get(),
    }

    # Determine overall status
    model_ok = checks.get("model", {}).get("loaded", False)
    batch_ok = checks.get("batch_processor", {}).get("running", False)
    overall_healthy = model_ok and batch_ok

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "checks": checks,
    }
