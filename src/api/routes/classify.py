"""Classification endpoint for the Intelligence Query Gateway.

POST /v1/query-classify - Classify a query into Fast Path or Slow Path.
"""

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import BatchingDep, CacheDep, get_settings
from src.api.schemas import ClassifyRequest, ClassifyResponseSpec
from src.core.logging import get_logger
from src.core.metrics import (
    ACTIVE_REQUESTS,
    CONFIDENCE_ROUTING_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.utils.hashing import generate_cache_key

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["classification"])


@router.post(
    "/query-classify",
    response_model=ClassifyResponseSpec,
    responses={
        200: {
            "description": "Successful classification",
            "headers": {
                "x-router-latency": {
                    "description": "Router latency in milliseconds",
                    "schema": {"type": "string"},
                }
            },
        },
        400: {"description": "Invalid request"},
        503: {"description": "Service unavailable (model not ready)"},
    },
)
async def classify_query(
    request_body: ClassifyRequest,
    batching: BatchingDep,
    cache: CacheDep,
    request: Request,
) -> JSONResponse:
    """Classify a query into Fast Path (0) or Slow Path (1).

    This endpoint integrates with CacheService (L1) and BatchingService
    to provide optimized classification with caching and batching.

    Flow:
    1. Generate cache key from normalized query text
    2. Check L1 cache for existing result
    3. If cache miss, submit to BatchingService for inference
    4. Cache the result and return

    Args:
        request_body: Classification request with text field.
        batching: Injected BatchingService for dynamic batching.
        cache: Injected CacheService for L1 caching.
        request: FastAPI request object for metadata.

    Returns:
        JSONResponse with label and x-router-latency header.
    """
    start_time = time.perf_counter()
    ACTIVE_REQUESTS.inc()

    try:
        # Generate request ID if not provided
        request_id = request_body.request_id or str(uuid.uuid4())

        logger.info(
            "Classification request received",
            request_id=request_id,
            text_length=len(request_body.text),
        )

        # Generate cache key
        cache_key = generate_cache_key(request_body.text)

        # Try to get from cache
        cached_result = await cache.get(cache_key)

        if cached_result is not None:
            # Cache hit
            cache_hit = True
            result = cached_result
            logger.debug(
                "Cache hit",
                request_id=request_id,
                cache_key=cache_key,
            )
        else:
            # Cache miss - go through batching service
            cache_hit = False
            logger.debug(
                "Cache miss",
                request_id=request_id,
                cache_key=cache_key,
            )

            # Perform batched classification
            result = await batching.classify(request_body.text)

            # Cache the result
            await cache.set(cache_key, result)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        REQUEST_COUNT.labels(
            status="success",
            cache_hit=str(cache_hit).lower(),
        ).inc()
        REQUEST_LATENCY.labels(endpoint="/v1/query-classify").observe(
            time.perf_counter() - start_time
        )

        # Confidence-aware routing: if confidence is below threshold and
        # the model predicts Fast Path (0), route to Slow Path (1) instead.
        # This ensures low-confidence predictions are handled more carefully.
        settings = get_settings()
        final_label = result.label
        confidence_routed = False

        if (
            result.label == 0
            and result.confidence < settings.confidence_threshold
        ):
            final_label = 1
            confidence_routed = True
            CONFIDENCE_ROUTING_COUNT.labels(
                original_label="0", routed_label="1"
            ).inc()
            logger.info(
                "Low confidence routing triggered",
                request_id=request_id,
                original_label=result.label,
                routed_label=final_label,
                confidence=result.confidence,
                threshold=settings.confidence_threshold,
            )

        logger.info(
            "Classification completed",
            request_id=request_id,
            label=final_label,
            original_label=result.label,
            confidence=result.confidence,
            confidence_routed=confidence_routed,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

        # Return response with label and confidence (Bonus requirement)
        response_data = ClassifyResponseSpec(
            label=str(final_label),
            confidence=round(result.confidence, 4),
        )

        return JSONResponse(
            content=response_data.model_dump(),
            headers={"x-router-latency": str(int(latency_ms))},
        )

    finally:
        ACTIVE_REQUESTS.dec()
