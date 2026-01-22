"""FastAPI application entry point with lifespan management.

This module initializes the FastAPI application, configures middleware,
and manages the application lifecycle (startup/shutdown).
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.dependencies import cleanup_services, init_services
from src.api.routes import classify, health
from src.core.config import get_settings
from src.core.exceptions import ServiceError
from src.core.logging import configure_logging, get_logger
from src.core.metrics import MODEL_LOADED

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    This context manager handles:
    1. Startup: Load configuration, initialize logging, load model
    2. Shutdown: Cleanup resources, unload model

    Args:
        app: FastAPI application instance.

    Yields:
        None during application runtime.
    """
    # Startup
    settings = get_settings()

    # Configure logging
    configure_logging(level=settings.log_level, format=settings.log_format)
    logger.info(
        "Starting Intelligence Query Gateway",
        env=settings.app_env.value,
        host=settings.app_host,
        port=settings.app_port,
    )

    # Initialize services
    try:
        await init_services(settings)
        MODEL_LOADED.set(1)
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        MODEL_LOADED.set(0)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Intelligence Query Gateway")
    await cleanup_services()
    MODEL_LOADED.set(0)
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Intelligence Query Gateway",
        description="Semantic Router Gateway for query classification",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
    )

    # Register exception handlers
    @app.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        """Handle ServiceError exceptions with structured response."""
        logger.warning(
            "Service error",
            error_type=type(exc).__name__,
            message=exc.message,
            status=exc.status.value,
        )
        return JSONResponse(
            status_code=exc.code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected error",
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "An internal error occurred",
                    "status": "INTERNAL",
                    "details": [],
                }
            },
        )

    # Register middleware for request ID tracking
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        """Add request ID to logs and response headers for tracing.

        This middleware:
        1. Extracts or generates a unique request ID
        2. Binds it to the logging context (available in all logs)
        3. Adds it to response headers for client-side tracing
        4. Clears the context after request completes
        """
        # Extract from header or generate new ID
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        # Bind to logging context (will appear in all logs for this request)
        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            response = await call_next(request)
            # Add to response headers
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Clean up context
            structlog.contextvars.clear_contextvars()

    # Include routers
    app.include_router(classify.router)
    app.include_router(health.router)

    # Mount Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
