#!/bin/bash
# Docker entrypoint script for Intelligence Query Gateway
#
# This script handles:
# 1. Model availability check
# 2. Auto-download from Hugging Face Hub if needed
# 3. Health checks before starting the service

set -e

echo "======================================"
echo "Intelligence Query Gateway - Starting"
echo "======================================"

# Environment variables (with defaults)
MODEL_PATH="${MODEL_PATH:-/app/models/router}"
HF_MODEL_ID="${HF_MODEL_ID:-bolin8017/query-gateway-router}"

echo "Configuration:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  HF_MODEL_ID: $HF_MODEL_ID"
echo ""

# Check if model exists locally
if [ -f "$MODEL_PATH/config.json" ]; then
    echo "✓ Model found at $MODEL_PATH"
else
    echo "⚠ Model not found locally at $MODEL_PATH"
    echo "  The application will auto-download from Hugging Face Hub:"
    echo "  → $HF_MODEL_ID"
    echo ""
    echo "  Note: First startup may take longer due to model download."
    echo "  Subsequent starts will use the cached model."
fi

echo ""
echo "Starting application..."
echo "======================================"

# Execute the main command (passed as arguments to this script)
exec "$@"
