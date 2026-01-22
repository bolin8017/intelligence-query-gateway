"""Pydantic schemas for API request and response models.

Following Google API style guide for response structure.
"""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request body for the classification endpoint.

    Attributes:
        text: Query text to classify (1-2048 characters).
        request_id: Optional client-provided request ID for tracing.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Query text to classify",
        examples=["What is the capital of France?"],
    )
    request_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional client-provided request ID for tracing",
        examples=["req-abc-123"],
    )


class ClassifyData(BaseModel):
    """Classification result data.

    Attributes:
        label: Classification label (0=Fast Path, 1=Slow Path).
        confidence: Confidence score (0.0 to 1.0).
        category: Human-readable category name.
    """

    label: int = Field(
        ...,
        ge=0,
        le=1,
        description="Classification label",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score",
    )
    category: str = Field(
        ...,
        description="Human-readable category name",
    )


class ClassifyMetadata(BaseModel):
    """Metadata about the classification request.

    Attributes:
        request_id: Request ID for tracing.
        latency_ms: Processing latency in milliseconds.
        cache_hit: Whether result came from cache.
        batch_size: Size of the inference batch.
    """

    request_id: str = Field(
        ...,
        description="Request ID for tracing",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing latency in milliseconds",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether result came from cache",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Size of the inference batch",
    )


class ClassifyResponse(BaseModel):
    """Full response for classification endpoint (Google API style).

    This is the rich response format used internally. The actual HTTP response
    follows the spec format with just {"label": "0"} and x-router-latency header.
    """

    data: ClassifyData
    metadata: ClassifyMetadata


class ClassifyResponseSpec(BaseModel):
    """Response body following the homework specification.

    The spec requires: {"label": "0"} (string label)
    Bonus: includes confidence score for confidence-aware routing.
    """

    label: str = Field(
        ...,
        description="Classification label as string (0=Fast Path, 1=Slow Path)",
        examples=["0", "1"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the prediction",
        examples=[0.73, 0.95],
    )


class ErrorDetail(BaseModel):
    """Error detail following Google API error model."""

    code: int
    message: str
    status: str
    details: list[dict] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response following Google API error model."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status ('healthy' or 'unhealthy').
        checks: Individual check results.
    """

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy", "unhealthy"],
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual health check results",
    )
