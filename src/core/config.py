"""Application configuration using Pydantic Settings.

This module implements the 12-Factor App configuration pattern,
loading settings from environment variables with validation.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class Settings(BaseSettings):
    """Application settings with validation.

    All settings can be overridden via environment variables.
    Environment variables should be prefixed appropriately or match
    the field names exactly (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Environment = Field(
        default=Environment.DEV,
        description="Application environment",
    )
    app_host: str = Field(
        default="0.0.0.0",
        description="Server bind address",
    )
    app_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port",
    )
    app_debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Model
    model_path: str = Field(
        default="./models/router",
        description="Path to the trained model directory",
    )
    model_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for model inference",
    )
    model_max_length: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum token length for input",
    )

    # Batching
    batch_max_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Maximum batch size for inference",
    )
    batch_max_wait_ms: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum wait time in milliseconds before processing batch",
    )

    # Cache (L1 - Local LRU)
    cache_l1_size: int = Field(
        default=10000,
        ge=0,
        description="Maximum entries in L1 cache (0 to disable)",
    )
    cache_l1_ttl_sec: int = Field(
        default=300,
        ge=0,
        description="L1 cache TTL in seconds",
    )

    # Cache (L2 - Redis, Optional)
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL (optional)",
    )
    cache_l2_ttl_sec: int = Field(
        default=3600,
        ge=0,
        description="L2 cache TTL in seconds",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )

    # Confidence Routing
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for routing decisions",
    )

    # Rate Limiting (Phase 5)
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting",
    )
    rate_limit_requests_per_second: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per second",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        """Ensure log level is uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == Environment.PROD

    @property
    def is_redis_enabled(self) -> bool:
        """Check if Redis L2 cache is configured."""
        return self.redis_url is not None


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings: Application settings instance.
    """
    return Settings()
