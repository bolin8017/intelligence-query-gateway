"""Custom exception classes following Google Cloud API error model.

This module defines a hierarchy of exceptions that map to HTTP status codes
and provide structured error responses.

Reference: https://cloud.google.com/apis/design/errors
"""

from enum import Enum
from typing import Any


class ErrorStatus(str, Enum):
    """Standard error status codes following Google API conventions."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


class ServiceError(Exception):
    """Base exception for all service errors.

    Attributes:
        message: Human-readable error description.
        code: HTTP status code.
        status: Error status following Google API conventions.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        code: int = 500,
        status: ErrorStatus = ErrorStatus.INTERNAL,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize ServiceError.

        Args:
            message: Human-readable error description.
            code: HTTP status code.
            status: Error status enum value.
            details: Optional list of additional error details.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.details = details or []

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to Google API error response format.

        Returns:
            Dictionary following Google Cloud API error format.
        """
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status": self.status.value,
                "details": self.details,
            }
        }


class ValidationError(ServiceError):
    """Exception for request validation failures.

    Raised when the request body fails validation (e.g., empty text,
    text too long, invalid format).
    """

    def __init__(
        self,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Description of the validation failure.
            details: Optional field-level error details.
        """
        super().__init__(
            message=message,
            code=400,
            status=ErrorStatus.INVALID_ARGUMENT,
            details=details,
        )


class ModelNotReadyError(ServiceError):
    """Exception when model is not loaded or ready for inference.

    Raised during startup before model loading completes, or if
    model loading fails.
    """

    def __init__(
        self,
        message: str = "Model is not ready for inference",
    ) -> None:
        """Initialize ModelNotReadyError.

        Args:
            message: Description of why model is not ready.
        """
        super().__init__(
            message=message,
            code=503,
            status=ErrorStatus.UNAVAILABLE,
        )


class CacheError(ServiceError):
    """Exception for cache operation failures.

    Note: This exception is typically caught internally and not exposed
    to clients. The system degrades gracefully when cache fails.
    """

    def __init__(
        self,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize CacheError.

        Args:
            message: Description of the cache failure.
            details: Optional additional details about the failure.
        """
        super().__init__(
            message=message,
            code=500,
            status=ErrorStatus.INTERNAL,
            details=details,
        )


class RateLimitError(ServiceError):
    """Exception when request exceeds rate limits.

    Includes retry information to help clients back off appropriately.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after_seconds: int = 60,
    ) -> None:
        """Initialize RateLimitError.

        Args:
            message: Description of the rate limit.
            retry_after_seconds: Suggested retry delay in seconds.
        """
        super().__init__(
            message=message,
            code=429,
            status=ErrorStatus.RESOURCE_EXHAUSTED,
            details=[{"retry_after_seconds": retry_after_seconds}],
        )
        self.retry_after_seconds = retry_after_seconds


class InternalError(ServiceError):
    """Exception for unexpected internal errors.

    Used as a catch-all for unhandled exceptions. Internal details
    should not be exposed to clients.
    """

    def __init__(
        self,
        message: str = "An internal error occurred",
    ) -> None:
        """Initialize InternalError.

        Args:
            message: Generic error message (avoid exposing internals).
        """
        super().__init__(
            message=message,
            code=500,
            status=ErrorStatus.INTERNAL,
        )
