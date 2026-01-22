"""Unit tests for API Pydantic schemas.

Tests cover request/response validation, field constraints,
and schema serialization behavior.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ClassifyData,
    ClassifyMetadata,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResponseSpec,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)


class TestClassifyRequest:
    """Tests for ClassifyRequest schema."""

    def test_classify_request_valid(self):
        """ClassifyRequest accepts valid text."""
        request = ClassifyRequest(text="What is the capital of France?")

        assert request.text == "What is the capital of France?"
        assert request.request_id is None

    def test_classify_request_with_request_id(self):
        """ClassifyRequest accepts optional request_id."""
        request = ClassifyRequest(
            text="Test query",
            request_id="req-abc-123",
        )

        assert request.text == "Test query"
        assert request.request_id == "req-abc-123"

    def test_classify_request_rejects_empty_text(self):
        """ClassifyRequest rejects empty text."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequest(text="")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) for e in errors)

    def test_classify_request_rejects_text_too_long(self):
        """ClassifyRequest rejects text exceeding max_length."""
        text = "a" * 2049

        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequest(text=text)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) for e in errors)

    def test_classify_request_accepts_max_length_text(self):
        """ClassifyRequest accepts text at exactly max_length."""
        text = "a" * 2048

        request = ClassifyRequest(text=text)

        assert len(request.text) == 2048

    def test_classify_request_rejects_request_id_too_long(self):
        """ClassifyRequest rejects request_id exceeding 128 chars."""
        request_id = "x" * 129

        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequest(text="test", request_id=request_id)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("request_id",) for e in errors)

    def test_classify_request_missing_text_field(self):
        """ClassifyRequest requires text field."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequest()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) for e in errors)


class TestClassifyData:
    """Tests for ClassifyData schema."""

    def test_classify_data_valid(self):
        """ClassifyData accepts valid label, confidence, category."""
        data = ClassifyData(
            label=0,
            confidence=0.95,
            category="fast_path",
        )

        assert data.label == 0
        assert data.confidence == 0.95
        assert data.category == "fast_path"

    def test_classify_data_label_validation_minimum(self):
        """ClassifyData rejects label below 0."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyData(label=-1, confidence=0.5, category="test")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("label",) for e in errors)

    def test_classify_data_label_validation_maximum(self):
        """ClassifyData rejects label above 1."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyData(label=2, confidence=0.5, category="test")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("label",) for e in errors)

    def test_classify_data_confidence_validation_minimum(self):
        """ClassifyData rejects confidence below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyData(label=0, confidence=-0.1, category="test")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence",) for e in errors)

    def test_classify_data_confidence_validation_maximum(self):
        """ClassifyData rejects confidence above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyData(label=0, confidence=1.1, category="test")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence",) for e in errors)

    def test_classify_data_boundary_values(self):
        """ClassifyData accepts boundary values."""
        data_min = ClassifyData(label=0, confidence=0.0, category="test")
        assert data_min.label == 0
        assert data_min.confidence == 0.0

        data_max = ClassifyData(label=1, confidence=1.0, category="test")
        assert data_max.label == 1
        assert data_max.confidence == 1.0


class TestClassifyMetadata:
    """Tests for ClassifyMetadata schema."""

    def test_classify_metadata_valid(self):
        """ClassifyMetadata accepts valid fields."""
        metadata = ClassifyMetadata(
            request_id="req-123",
            latency_ms=15.5,
            cache_hit=True,
            batch_size=4,
        )

        assert metadata.request_id == "req-123"
        assert metadata.latency_ms == 15.5
        assert metadata.cache_hit is True
        assert metadata.batch_size == 4

    def test_classify_metadata_defaults(self):
        """ClassifyMetadata uses defaults for optional fields."""
        metadata = ClassifyMetadata(
            request_id="req-123",
            latency_ms=10.0,
        )

        assert metadata.cache_hit is False
        assert metadata.batch_size == 1

    def test_classify_metadata_latency_non_negative(self):
        """ClassifyMetadata rejects negative latency."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyMetadata(request_id="req-123", latency_ms=-1.0)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("latency_ms",) for e in errors)

    def test_classify_metadata_batch_size_minimum(self):
        """ClassifyMetadata rejects batch_size below 1."""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyMetadata(
                request_id="req-123",
                latency_ms=10.0,
                batch_size=0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("batch_size",) for e in errors)


class TestClassifyResponse:
    """Tests for ClassifyResponse schema."""

    def test_classify_response_valid(self):
        """ClassifyResponse composes data and metadata correctly."""
        data = ClassifyData(label=0, confidence=0.95, category="fast_path")
        metadata = ClassifyMetadata(request_id="req-123", latency_ms=12.3)

        response = ClassifyResponse(data=data, metadata=metadata)

        assert response.data.label == 0
        assert response.metadata.request_id == "req-123"

    def test_classify_response_serialization(self):
        """ClassifyResponse serializes to dict correctly."""
        data = ClassifyData(label=1, confidence=0.85, category="slow_path")
        metadata = ClassifyMetadata(
            request_id="req-456",
            latency_ms=20.0,
            cache_hit=True,
            batch_size=8,
        )
        response = ClassifyResponse(data=data, metadata=metadata)

        result = response.model_dump()

        assert result["data"]["label"] == 1
        assert result["data"]["confidence"] == 0.85
        assert result["data"]["category"] == "slow_path"
        assert result["metadata"]["request_id"] == "req-456"
        assert result["metadata"]["latency_ms"] == 20.0
        assert result["metadata"]["cache_hit"] is True
        assert result["metadata"]["batch_size"] == 8


class TestClassifyResponseSpec:
    """Tests for ClassifyResponseSpec schema."""

    def test_classify_response_spec_valid(self):
        """ClassifyResponseSpec accepts string label and confidence."""
        response = ClassifyResponseSpec(label="0", confidence=0.95)

        assert response.label == "0"
        assert response.confidence == 0.95

    def test_classify_response_spec_label_values(self):
        """ClassifyResponseSpec accepts both label values."""
        response_0 = ClassifyResponseSpec(label="0", confidence=0.8)
        assert response_0.label == "0"

        response_1 = ClassifyResponseSpec(label="1", confidence=0.9)
        assert response_1.label == "1"

    def test_classify_response_spec_serialization(self):
        """ClassifyResponseSpec serializes to dict with label and confidence."""
        response = ClassifyResponseSpec(label="1", confidence=0.73)

        result = response.model_dump()

        assert result == {"label": "1", "confidence": 0.73}

    def test_classify_response_spec_confidence_bounds(self):
        """ClassifyResponseSpec validates confidence is between 0 and 1."""
        # Valid confidence values
        response = ClassifyResponseSpec(label="0", confidence=0.0)
        assert response.confidence == 0.0

        response = ClassifyResponseSpec(label="0", confidence=1.0)
        assert response.confidence == 1.0


class TestErrorDetail:
    """Tests for ErrorDetail schema."""

    def test_error_detail_valid(self):
        """ErrorDetail accepts all required fields."""
        detail = ErrorDetail(
            code=400,
            message="Invalid input",
            status="INVALID_ARGUMENT",
        )

        assert detail.code == 400
        assert detail.message == "Invalid input"
        assert detail.status == "INVALID_ARGUMENT"
        assert detail.details == []

    def test_error_detail_with_details(self):
        """ErrorDetail accepts additional details list."""
        detail = ErrorDetail(
            code=400,
            message="Validation failed",
            status="INVALID_ARGUMENT",
            details=[{"field": "text", "reason": "too_long"}],
        )

        assert len(detail.details) == 1
        assert detail.details[0]["field"] == "text"


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_error_response_valid(self):
        """ErrorResponse wraps ErrorDetail correctly."""
        detail = ErrorDetail(
            code=503,
            message="Service unavailable",
            status="UNAVAILABLE",
        )
        response = ErrorResponse(error=detail)

        assert response.error.code == 503
        assert response.error.message == "Service unavailable"

    def test_error_response_serialization(self):
        """ErrorResponse serializes to Google API error format."""
        detail = ErrorDetail(
            code=500,
            message="Internal error",
            status="INTERNAL",
            details=[{"trace_id": "abc123"}],
        )
        response = ErrorResponse(error=detail)

        result = response.model_dump()

        assert result["error"]["code"] == 500
        assert result["error"]["message"] == "Internal error"
        assert result["error"]["status"] == "INTERNAL"
        assert result["error"]["details"] == [{"trace_id": "abc123"}]


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_health_response_healthy(self):
        """HealthResponse represents healthy state."""
        response = HealthResponse(
            status="healthy",
            checks={"model": True, "cache": True},
        )

        assert response.status == "healthy"
        assert response.checks["model"] is True
        assert response.checks["cache"] is True

    def test_health_response_unhealthy(self):
        """HealthResponse represents unhealthy state."""
        response = HealthResponse(
            status="unhealthy",
            checks={"model": False, "cache": True},
        )

        assert response.status == "unhealthy"
        assert response.checks["model"] is False

    def test_health_response_default_checks(self):
        """HealthResponse defaults to empty checks dict."""
        response = HealthResponse(status="healthy")

        assert response.checks == {}

    def test_health_response_serialization(self):
        """HealthResponse serializes correctly."""
        response = HealthResponse(
            status="healthy",
            checks={"model": True},
        )

        result = response.model_dump()

        assert result == {
            "status": "healthy",
            "checks": {"model": True},
        }
