"""Unit tests for configuration module.

Tests cover settings validation, type coercion, property methods,
and environment variable loading behavior.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import Environment, Settings, get_settings


class TestEnvironmentEnum:
    """Tests for Environment enumeration."""

    def test_environment_values(self):
        """Environment enum has expected values."""
        assert Environment.DEV.value == "dev"
        assert Environment.STAGING.value == "staging"
        assert Environment.PROD.value == "prod"

    def test_environment_from_string(self):
        """Environment can be constructed from string values."""
        assert Environment("dev") == Environment.DEV
        assert Environment("staging") == Environment.STAGING
        assert Environment("prod") == Environment.PROD


class TestSettings:
    """Tests for Settings configuration class."""

    def test_default_settings(self):
        """Settings initializes with valid defaults when no .env file."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            assert settings.app_env == Environment.DEV
            assert settings.app_host == "0.0.0.0"
            assert settings.app_port == 8000
            assert settings.app_debug is False

            assert settings.model_path == "./models/router"
            assert settings.model_device == "cpu"
            assert settings.model_max_length == 512

            assert settings.batch_max_size == 32
            assert settings.batch_max_wait_ms == 10

            assert settings.cache_l1_size == 10000
            assert settings.cache_l1_ttl_sec == 300

            assert settings.redis_url is None
            assert settings.cache_l2_ttl_sec == 3600

            assert settings.log_level == "INFO"
            assert settings.log_format == "json"

            assert settings.confidence_threshold == 0.7

    def test_app_port_validation_minimum(self):
        """App port rejects values below 1."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(app_port=0)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("app_port",) for e in errors)

    def test_app_port_validation_maximum(self):
        """App port rejects values above 65535."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(app_port=65536)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("app_port",) for e in errors)

    def test_app_port_validation_valid_range(self):
        """App port accepts values in valid range."""
        settings_min = Settings(app_port=1)
        assert settings_min.app_port == 1

        settings_max = Settings(app_port=65535)
        assert settings_max.app_port == 65535

        settings_common = Settings(app_port=8080)
        assert settings_common.app_port == 8080

    def test_model_max_length_validation(self):
        """Model max length validates range constraints."""
        with pytest.raises(ValidationError):
            Settings(model_max_length=0)

        with pytest.raises(ValidationError):
            Settings(model_max_length=2049)

        settings = Settings(model_max_length=512)
        assert settings.model_max_length == 512

    def test_batch_max_size_validation(self):
        """Batch max size validates range constraints."""
        with pytest.raises(ValidationError):
            Settings(batch_max_size=0)

        with pytest.raises(ValidationError):
            Settings(batch_max_size=257)

        settings = Settings(batch_max_size=64)
        assert settings.batch_max_size == 64

    def test_batch_max_wait_ms_validation(self):
        """Batch max wait validates range constraints."""
        with pytest.raises(ValidationError):
            Settings(batch_max_wait_ms=0)

        with pytest.raises(ValidationError):
            Settings(batch_max_wait_ms=1001)

        settings = Settings(batch_max_wait_ms=50)
        assert settings.batch_max_wait_ms == 50

    def test_confidence_threshold_validation(self):
        """Confidence threshold validates range 0.0-1.0."""
        with pytest.raises(ValidationError):
            Settings(confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            Settings(confidence_threshold=1.1)

        settings_min = Settings(confidence_threshold=0.0)
        assert settings_min.confidence_threshold == 0.0

        settings_max = Settings(confidence_threshold=1.0)
        assert settings_max.confidence_threshold == 1.0

    def test_log_level_uppercase_validation(self):
        """Log level is converted to uppercase."""
        settings_lower = Settings(log_level="debug")
        assert settings_lower.log_level == "DEBUG"

        settings_mixed = Settings(log_level="InFo")
        assert settings_mixed.log_level == "INFO"

        settings_upper = Settings(log_level="ERROR")
        assert settings_upper.log_level == "ERROR"

    def test_is_production_property(self):
        """is_production property correctly identifies production environment."""
        dev_settings = Settings(app_env=Environment.DEV)
        assert dev_settings.is_production is False

        staging_settings = Settings(app_env=Environment.STAGING)
        assert staging_settings.is_production is False

        prod_settings = Settings(app_env=Environment.PROD)
        assert prod_settings.is_production is True

    def test_is_redis_enabled_property(self):
        """is_redis_enabled property checks Redis URL presence."""
        settings_no_redis = Settings(redis_url=None)
        assert settings_no_redis.is_redis_enabled is False

        settings_with_redis = Settings(redis_url="redis://localhost:6379/0")
        assert settings_with_redis.is_redis_enabled is True

    def test_model_device_literal_validation(self):
        """Model device only accepts cpu, cuda, or mps."""
        cpu_settings = Settings(model_device="cpu")
        assert cpu_settings.model_device == "cpu"

        cuda_settings = Settings(model_device="cuda")
        assert cuda_settings.model_device == "cuda"

        mps_settings = Settings(model_device="mps")
        assert mps_settings.model_device == "mps"

        with pytest.raises(ValidationError):
            Settings(model_device="gpu")

    def test_log_format_literal_validation(self):
        """Log format only accepts json or console."""
        json_settings = Settings(log_format="json")
        assert json_settings.log_format == "json"

        console_settings = Settings(log_format="console")
        assert console_settings.log_format == "console"

        with pytest.raises(ValidationError):
            Settings(log_format="text")

    def test_settings_from_environment_variables(self):
        """Settings loads values from environment variables."""
        env_vars = {
            "APP_ENV": "prod",
            "APP_PORT": "9000",
            "APP_DEBUG": "true",
            "MODEL_DEVICE": "cuda",
            "BATCH_MAX_SIZE": "64",
            "CACHE_L1_SIZE": "5000",
            "LOG_LEVEL": "warning",
            "CONFIDENCE_THRESHOLD": "0.85",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert settings.app_env == Environment.PROD
            assert settings.app_port == 9000
            assert settings.app_debug is True
            assert settings.model_device == "cuda"
            assert settings.batch_max_size == 64
            assert settings.cache_l1_size == 5000
            assert settings.log_level == "WARNING"
            assert settings.confidence_threshold == 0.85


class TestGetSettings:
    """Tests for get_settings cached factory function."""

    def test_get_settings_returns_settings_instance(self):
        """get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_caches_result(self):
        """get_settings returns the same instance on multiple calls."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_cache_clear_creates_new_instance(self):
        """Clearing cache causes get_settings to create a new instance."""
        get_settings.cache_clear()
        settings1 = get_settings()

        get_settings.cache_clear()
        settings2 = get_settings()

        assert settings1 is not settings2
