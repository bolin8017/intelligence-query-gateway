"""Structured logging configuration using structlog.

Provides JSON-formatted logs in production and colored console logs
in development, following Google Cloud Logging best practices.
"""

import logging
import sys
from typing import Literal

import structlog
from structlog.types import Processor


def configure_logging(
    level: str = "INFO",
    format: Literal["json", "console"] = "json",
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        format: Output format ('json' for production, 'console' for development).
    """
    # Shared processors for both formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if format == "json":
        # Production: JSON output for log aggregation systems
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output for readability
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name (typically __name__).

    Returns:
        Configured structlog logger instance.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
