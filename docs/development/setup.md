# Development Environment Setup

Guide for setting up a local development environment.

## Prerequisites

- **Python 3.11 or higher**
- **Git** for version control
- **(Optional) Docker** for containerized development

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd Intelligence-Query-Gateway-Microservices

# 2. Create Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Train model (or download pre-trained)
python scripts/train_router.py --output-dir ./models/router

# 5. Run service
python -m src.main

# 6. Test API
curl http://localhost:8000/health/ready
```

## Python Environment Setup

### Option 1: venv (Recommended)

Standard Python virtual environment:

```bash
# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install project
pip install -e ".[dev]"

# Verify installation
python -c "import fastapi, torch, transformers; print('All dependencies installed')"
```

**Deactivate**: `deactivate`

### Option 2: conda

For users preferring conda:

```bash
# Create environment
conda create -n query-gateway python=3.11 -y

# Activate
conda activate query-gateway

# Install dependencies
pip install -e ".[dev]"

# Verify
python -c "import fastapi; print(fastapi.__version__)"
```

**Deactivate**: `conda deactivate`

### Option 3: Docker Dev Container

For containerized development:

```bash
# Build dev container
docker build -f Dockerfile.dev -t query-gateway-dev .

# Run with source mounted
docker run -it \
  -v $(pwd):/app \
  -p 8000:8000 \
  query-gateway-dev bash

# Inside container
pip install -e ".[dev]"
python -m src.main
```

## Dependencies

### Core Dependencies

Automatically installed with `pip install -e ".[dev]"`:

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **torch**: PyTorch (CPU version)
- **transformers**: HuggingFace models
- **pydantic**: Data validation
- **pydantic-settings**: Configuration management
- **prometheus-client**: Metrics
- **structlog**: Structured logging

### Development Dependencies

Included in `[dev]` extra:

- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **httpx**: HTTP client for tests
- **black**: Code formatter
- **ruff**: Linter
- **mypy**: Type checker
- **locust**: Load testing

### Installing Individual Packages

```bash
# Only core dependencies
pip install -e .

# With development tools
pip install -e ".[dev]"

# Specific package
pip install pytest
```

## Model Setup

### Training the Model

```bash
# Basic training
python scripts/train_router.py \
  --output-dir ./models/router

# With custom parameters
python scripts/train_router.py \
  --output-dir ./models/router \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5

# GPU training
python scripts/train_router.py \
  --output-dir ./models/router \
  --device cuda
```

**Training Time**: ~10-15 minutes on CPU, ~3-5 minutes on GPU

**Output**: Creates `models/router/` with:
- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer_config.json`
- `vocab.txt`
- Other tokenizer files

### Using Pre-trained Model

If a pre-trained model is available:

```bash
# Extract model files to models/router/
tar -xzf router-model.tar.gz -C models/

# Verify
ls models/router/
# Should show: config.json, pytorch_model.bin, tokenizer files
```

## Running the Service

### Development Mode

With auto-reload on file changes:

```bash
# Using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python module
python -m src.main
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs (Swagger UI)
- Health: http://localhost:8000/health/ready

### Production Mode

Without auto-reload:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### With Environment Variables

```bash
# Create .env file for development
cat > .env <<EOF
# Application
APP_ENV=dev
APP_DEBUG=true

# Model
MODEL_PATH=./models/router
MODEL_DEVICE=cpu

# Batching
BATCH_MAX_SIZE=32
BATCH_MAX_WAIT_MS=10

# L1 Cache (In-Memory)
CACHE_L1_SIZE=10000
CACHE_L1_TTL_SEC=300

# L2 Cache (Redis - Optional for local dev)
# Uncomment to enable Redis L2 cache:
# REDIS_URL=redis://localhost:6379/0
# CACHE_L2_TTL_SEC=3600

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=console
EOF

# Load and run
export $(cat .env | xargs)
python -m src.main
```

**To test with Redis L2 cache locally**:
```bash
# Start Redis using Docker
docker run -d \
  --name dev-redis \
  -p 6379:6379 \
  redis:7-alpine

# Update .env to include:
# REDIS_URL=redis://localhost:6379/0

# Restart application
python -m src.main

# Verify Redis connection
curl -s http://localhost:8000/health/deep | jq '.checks.cache'
# Should show: l2_enabled: true, l2_healthy: true
```

## IDE Configuration

### VSCode

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

**Recommended Extensions**:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Ruff (charliermarsh.ruff)

### PyCharm

1. **Set Interpreter**:
   - File → Settings → Project → Python Interpreter
   - Add → Existing Environment → Select `venv/bin/python`

2. **Enable pytest**:
   - Settings → Tools → Python Integrated Tools
   - Default test runner: pytest

3. **Configure Black**:
   - Settings → Tools → Black
   - Enable: "Run Black on save"

## Running Tests

### Unit Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_cache.py -v

# Specific test
pytest tests/unit/test_cache.py::test_cache_hit -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests

Requires running service:

```bash
# Terminal 1: Start service
python -m src.main

# Terminal 2: Run integration tests
pytest tests/integration/ -v
```

### Load Tests

```bash
# Start service first
docker compose up -d gateway

# Run load test
cd tests/load
locust -f scenarios/baseline.py \
  --host=http://localhost:8000 \
  --users=10 --spawn-rate=2 --run-time=60s \
  --headless --html=reports/baseline.html

# View report
open reports/baseline.html
```

## Code Quality Tools

### Formatting with Black

```bash
# Format all code
black src/ tests/

# Check without modifying
black --check src/ tests/

# Single file
black src/services/cache.py
```

### Linting with Ruff

```bash
# Lint all code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Specific file
ruff check src/services/cache.py
```

### Type Checking with mypy

```bash
# Type check all code
mypy src/

# Ignore missing imports
mypy src/ --ignore-missing-imports

# Specific module
mypy src/services/
```

### Running All Checks

```bash
# Create a check script
cat > scripts/check.sh <<'EOF'
#!/bin/bash
set -e

echo "Running Black..."
black --check src/ tests/

echo "Running Ruff..."
ruff check src/ tests/

echo "Running mypy..."
mypy src/ --ignore-missing-imports

echo "Running tests..."
pytest tests/unit/ -v

echo "All checks passed!"
EOF

chmod +x scripts/check.sh
./scripts/check.sh
```

## Common Development Tasks

### Adding a New Dependency

```bash
# Install package
pip install package-name

# Add to pyproject.toml
# Edit pyproject.toml and add to dependencies

# Reinstall in editable mode
pip install -e ".[dev]"
```

### Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in (Python 3.7+)
breakpoint()

# Run with debugger
python -m pdb -m src.main
```

### Viewing Logs

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m src.main

# Pretty-print JSON logs
python -m src.main 2>&1 | jq '.'

# Filter by level
python -m src.main 2>&1 | grep '"level":"error"'
```

### Testing API Manually

```bash
# Health check
curl http://localhost:8000/health/ready

# Classification request
curl -X POST http://localhost:8000/v1/query-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize this article"}'

# With custom request ID
curl -X POST http://localhost:8000/v1/query-classify \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-123" \
  -d '{"text": "What is machine learning?"}'

# View metrics
curl http://localhost:8000/metrics
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -e ".[dev]"
```

### Model Loading Errors

**Problem**: `Model file not found` or `OSError: [Errno 2]`

**Solution**:
```bash
# Verify model files exist
ls -la models/router/

# If missing, train model
python scripts/train_router.py --output-dir ./models/router

# Check MODEL_PATH environment variable
echo $MODEL_PATH
```

### Port Already in Use

**Problem**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
# or
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.main:app --port 8001
```

### Memory Issues

**Problem**: `RuntimeError: [enforce fail at alloc_cpu.cpp:114]`

**Solution**:
```bash
# Reduce batch size
export BATCH_MAX_SIZE=16

# Reduce cache size
export CACHE_L1_SIZE=5000

# Use CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Tests Failing

**Problem**: Tests fail with connection errors

**Solution**:
```bash
# For integration tests, ensure service is running
python -m src.main &

# Wait for startup
sleep 5

# Run tests
pytest tests/integration/

# Cleanup
pkill -f "python -m src.main"
```

## Project Structure Reference

```
Intelligence-Query-Gateway-Microservices/
├── src/
│   ├── api/              # API routes and schemas
│   ├── core/             # Core configuration and utilities
│   ├── models/           # ML model wrappers
│   ├── services/         # Business logic
│   └── utils/            # Helper functions
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── load/             # Load tests
│   └── fixtures/         # Test data
├── scripts/              # Utility scripts
├── models/               # Trained models
├── docs/                 # Documentation
├── monitoring/           # Prometheus/Grafana configs
├── pyproject.toml        # Project metadata
├── Dockerfile            # Production container
└── docker compose.yml    # Multi-container setup
```

## Next Steps

After setup:

1. **Read Architecture**: See [architecture.md](../architecture.md) for system design
2. **Write Tests**: Follow [testing.md](testing.md) for testing guidelines
3. **Review Code Style**: Run `black` and `ruff` before committing
4. **Check Performance**: Run load tests to verify optimizations

## Related Documentation

- [Testing Guide](testing.md) - Test strategy and execution
- [Architecture](../architecture.md) - System design
- [Deployment Guide](../operations/deployment.md) - Production deployment

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-22
**Review cycle**: Quarterly
**Recent changes**: Added Redis L2 cache configuration for local development
