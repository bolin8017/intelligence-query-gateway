"""Unit tests for custom exception classes.

Tests cover exception initialization, error response formatting,
HTTP status codes, and error status enums following Google API conventions.
"""


from src.core.exceptions import (
    CacheError,
    ErrorStatus,
    InternalError,
    ModelNotReadyError,
    RateLimitError,
    ServiceError,
    ValidationError,
)


class TestErrorStatus:
    """Tests for ErrorStatus enumeration."""

    def test_error_status_values(self):
        """ErrorStatus enum has expected Google API conventional values."""
        assert ErrorStatus.INVALID_ARGUMENT.value == "INVALID_ARGUMENT"
        assert ErrorStatus.NOT_FOUND.value == "NOT_FOUND"
        assert ErrorStatus.ALREADY_EXISTS.value == "ALREADY_EXISTS"
        assert ErrorStatus.PERMISSION_DENIED.value == "PERMISSION_DENIED"
        assert ErrorStatus.RESOURCE_EXHAUSTED.value == "RESOURCE_EXHAUSTED"
        assert ErrorStatus.FAILED_PRECONDITION.value == "FAILED_PRECONDITION"
        assert ErrorStatus.UNAVAILABLE.value == "UNAVAILABLE"
        assert ErrorStatus.INTERNAL.value == "INTERNAL"


class TestServiceError:
    """Tests for ServiceError base exception."""

    def test_service_error_initialization_with_defaults(self):
        """ServiceError initializes with default code and status."""
        error = ServiceError(message="Something went wrong")

        assert error.message == "Something went wrong"
        assert error.code == 500
        assert error.status == ErrorStatus.INTERNAL
        assert error.details == []

    def test_service_error_initialization_with_all_parameters(self):
        """ServiceError initializes with all parameters."""
        details = [{"field": "user_id", "reason": "not_found"}]
        error = ServiceError(
            message="Resource not found",
            code=404,
            status=ErrorStatus.NOT_FOUND,
            details=details,
        )

        assert error.message == "Resource not found"
        assert error.code == 404
        assert error.status == ErrorStatus.NOT_FOUND
        assert error.details == details

    def test_service_error_to_dict_basic(self):
        """ServiceError.to_dict() formats error as Google API response."""
        error = ServiceError(
            message="Test error",
            code=400,
            status=ErrorStatus.INVALID_ARGUMENT,
        )

        result = error.to_dict()

        assert result == {
            "error": {
                "code": 400,
                "message": "Test error",
                "status": "INVALID_ARGUMENT",
                "details": [],
            }
        }

    def test_service_error_to_dict_with_details(self):
        """ServiceError.to_dict() includes details when present."""
        details = [
            {"field": "email", "reason": "invalid_format"},
            {"field": "age", "reason": "out_of_range"},
        ]
        error = ServiceError(
            message="Validation failed",
            code=400,
            status=ErrorStatus.INVALID_ARGUMENT,
            details=details,
        )

        result = error.to_dict()

        assert result["error"]["details"] == details

    def test_service_error_is_exception(self):
        """ServiceError is a standard Python exception."""
        error = ServiceError(message="Test")

        assert isinstance(error, Exception)
        assert str(error) == "Test"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_defaults(self):
        """ValidationError sets correct code and status."""
        error = ValidationError(message="Invalid input")

        assert error.message == "Invalid input"
        assert error.code == 400
        assert error.status == ErrorStatus.INVALID_ARGUMENT
        assert error.details == []

    def test_validation_error_with_details(self):
        """ValidationError accepts field-level details."""
        details = [{"field": "text", "reason": "too_long"}]
        error = ValidationError(message="Validation failed", details=details)

        assert error.details == details

    def test_validation_error_to_dict(self):
        """ValidationError formats as 400 error response."""
        error = ValidationError(message="Empty text not allowed")

        result = error.to_dict()

        assert result["error"]["code"] == 400
        assert result["error"]["status"] == "INVALID_ARGUMENT"


class TestModelNotReadyError:
    """Tests for ModelNotReadyError exception."""

    def test_model_not_ready_error_default_message(self):
        """ModelNotReadyError has sensible default message."""
        error = ModelNotReadyError()

        assert error.message == "Model is not ready for inference"
        assert error.code == 503
        assert error.status == ErrorStatus.UNAVAILABLE

    def test_model_not_ready_error_custom_message(self):
        """ModelNotReadyError accepts custom message."""
        error = ModelNotReadyError(message="Model loading failed")

        assert error.message == "Model loading failed"
        assert error.code == 503

    def test_model_not_ready_error_to_dict(self):
        """ModelNotReadyError formats as 503 unavailable response."""
        error = ModelNotReadyError()

        result = error.to_dict()

        assert result["error"]["code"] == 503
        assert result["error"]["status"] == "UNAVAILABLE"


class TestCacheError:
    """Tests for CacheError exception."""

    def test_cache_error_defaults(self):
        """CacheError sets correct code and status."""
        error = CacheError(message="Redis connection failed")

        assert error.message == "Redis connection failed"
        assert error.code == 500
        assert error.status == ErrorStatus.INTERNAL
        assert error.details == []

    def test_cache_error_with_details(self):
        """CacheError accepts additional details."""
        details = [{"cache_layer": "L2", "backend": "redis"}]
        error = CacheError(message="Cache write failed", details=details)

        assert error.details == details

    def test_cache_error_to_dict(self):
        """CacheError formats as 500 internal error response."""
        error = CacheError(message="Cache error")

        result = error.to_dict()

        assert result["error"]["code"] == 500
        assert result["error"]["status"] == "INTERNAL"


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_defaults(self):
        """RateLimitError has sensible defaults."""
        error = RateLimitError()

        assert error.message == "Rate limit exceeded"
        assert error.code == 429
        assert error.status == ErrorStatus.RESOURCE_EXHAUSTED
        assert error.retry_after_seconds == 60

    def test_rate_limit_error_custom_retry_after(self):
        """RateLimitError accepts custom retry_after value."""
        error = RateLimitError(
            message="Too many requests",
            retry_after_seconds=120,
        )

        assert error.message == "Too many requests"
        assert error.retry_after_seconds == 120

    def test_rate_limit_error_details_include_retry_after(self):
        """RateLimitError.details includes retry_after_seconds."""
        error = RateLimitError(retry_after_seconds=90)

        assert {"retry_after_seconds": 90} in error.details

    def test_rate_limit_error_to_dict(self):
        """RateLimitError formats as 429 rate limit response."""
        error = RateLimitError(retry_after_seconds=30)

        result = error.to_dict()

        assert result["error"]["code"] == 429
        assert result["error"]["status"] == "RESOURCE_EXHAUSTED"
        assert {"retry_after_seconds": 30} in result["error"]["details"]


class TestInternalError:
    """Tests for InternalError exception."""

    def test_internal_error_default_message(self):
        """InternalError has generic default message."""
        error = InternalError()

        assert error.message == "An internal error occurred"
        assert error.code == 500
        assert error.status == ErrorStatus.INTERNAL

    def test_internal_error_custom_message(self):
        """InternalError accepts custom message."""
        error = InternalError(message="Unexpected failure")

        assert error.message == "Unexpected failure"

    def test_internal_error_to_dict(self):
        """InternalError formats as 500 internal error response."""
        error = InternalError()

        result = error.to_dict()

        assert result["error"]["code"] == 500
        assert result["error"]["status"] == "INTERNAL"
        assert result["error"]["details"] == []


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_custom_exceptions_inherit_from_service_error(self):
        """All custom exceptions inherit from ServiceError."""
        assert issubclass(ValidationError, ServiceError)
        assert issubclass(ModelNotReadyError, ServiceError)
        assert issubclass(CacheError, ServiceError)
        assert issubclass(RateLimitError, ServiceError)
        assert issubclass(InternalError, ServiceError)

    def test_all_custom_exceptions_are_exceptions(self):
        """All custom exceptions inherit from Exception."""
        assert issubclass(ServiceError, Exception)
        assert issubclass(ValidationError, Exception)
        assert issubclass(ModelNotReadyError, Exception)
        assert issubclass(CacheError, Exception)
        assert issubclass(RateLimitError, Exception)
        assert issubclass(InternalError, Exception)

    def test_custom_exceptions_can_be_caught_as_service_error(self):
        """Custom exceptions can be caught using ServiceError."""
        try:
            raise ValidationError(message="Test")
        except ServiceError as e:
            assert e.message == "Test"

        try:
            raise ModelNotReadyError()
        except ServiceError as e:
            assert e.code == 503
