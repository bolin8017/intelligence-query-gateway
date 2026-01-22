# ============================================================================
# Multi-stage Dockerfile for Intelligence Query Gateway
# Optimized for production deployment with security best practices
# ============================================================================

# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim AS builder

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (for layer caching)
# Note: README.md is required by pyproject.toml metadata
COPY pyproject.toml README.md ./

# Install Python dependencies
# Note: For CPU-only deployment, you can add --index-url for PyTorch:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
# This reduces image size from ~5GB to ~1GB
RUN pip install --user --no-warn-script-location -e .

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    APP_HOME=/app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for health checks
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR $APP_HOME

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
COPY --chown=appuser:appuser pyproject.toml ./

# Create directories for models and logs (will be mounted or populated)
RUN mkdir -p models logs && chown -R appuser:appuser models logs

# Switch to non-root user
USER appuser

# Set entrypoint
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Expose application port
EXPOSE 8000

# Health check (using liveness endpoint)
# This is compatible with Kubernetes liveness/readiness probes
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command: Run the application with uvicorn
# Can be overridden in docker compose or k8s manifests
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
