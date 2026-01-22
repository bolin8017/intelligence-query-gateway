# CLAUDE.md - Project Context for AI Assistants

This document provides context and guidelines for AI assistants working on this project.

## Project Overview

**Intelligence Query Gateway** - A semantic router microservice that classifies queries into Fast Path or Slow Path using a fine-tuned DistilBERT model.

- **Fast Path (Label 0)**: Simple tasks (classification, summarization)
- **Slow Path (Label 1)**: Complex tasks (creative_writing, open_qa)

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI + uvicorn (async)
- **ML**: PyTorch + Transformers (DistilBERT)
- **Cache**: Two-level (L1 in-memory LRU + L2 Redis)
- **Monitoring**: Prometheus + Grafana
- **Testing**: pytest + Locust (load testing)
- **Container**: Docker + Docker Compose

## Project Structure

```
src/
├── api/           # FastAPI routes and schemas
├── services/      # Business logic (cache, batching, classifier)
├── models/        # ML model layer (SemanticRouter)
├── core/          # Config, logging, metrics, exceptions
└── utils/         # Utilities

tests/
├── unit/          # Unit tests
├── integration/   # Integration tests
└── load/          # Locust load tests

monitoring/
├── prometheus/    # Prometheus config + alerts
└── grafana/       # Dashboards + provisioning
```

## Development Environment Setup

```bash
# Create and activate conda environment
conda create -n query-gateway python=3.11 -y
conda activate query-gateway

# Install dependencies (editable with dev extras)
pip install -e ".[dev]"
```

## Model Management

### Quick Start (No Training Required)

The project supports **automatic model download** from Hugging Face Hub:

```bash
# Just start the service - model downloads automatically
docker compose up -d

# Or run locally (model downloads on first run if not found)
python -m src.main
```

**Default model**: `bolin8017/query-gateway-router` on Hugging Face Hub (~500MB)

### Using Your Own Model

**Option 1: Train Locally**
```bash
python scripts/train_router.py --output-dir ./models/router
```

**Option 2: Use Different HF Model**
```bash
# Set environment variable
export HF_MODEL_ID=your-username/your-model-name

# Or in docker-compose.yml
environment:
  - HF_MODEL_ID=your-username/your-model-name
```

### Upload Model to Hugging Face Hub

Share your trained model:

```bash
# 1. Login (one-time setup)
huggingface-cli login

# 2. Upload your model
python scripts/upload_model_to_hub.py
```

Edit [scripts/upload_model_to_hub.py](scripts/upload_model_to_hub.py) to change the target repository ID.

## Model Training

Train the SemanticRouter model with early stopping and LR scheduling:

```bash
# Basic training (uses early stopping, patience=3)
python scripts/train_router.py --output-dir ./models/router

# With Prometheus monitoring (requires Pushgateway at localhost:9091)
python scripts/train_router.py \
  --output-dir ./models/router \
  --pushgateway-url http://localhost:9091

# Custom hyperparameters
python scripts/train_router.py \
  --output-dir ./models/router \
  --max-epochs 20 \
  --patience 5 \
  --learning-rate 3e-5 \
  --batch-size 16 \
  --warmup-ratio 0.1 \
  --log-interval 50
```

Training features:
- **Early stopping**: Stops when validation loss doesn't improve for `patience` epochs
- **LR scheduler**: Warmup + linear decay (industry standard for BERT models)
- **Step-level metrics**: Real-time loss/LR/gradient monitoring via Pushgateway
- **Grafana dashboard**: http://localhost:3000/d/model-training

## Running the Service

### Local Development
```bash
# Direct run
python -m src.main

# With uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose (Recommended)
```bash
# Start all services (Gateway + Redis + Prometheus + Grafana)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f gateway
```

### Service Endpoints
- **API**: http://localhost:8000 (or 8080 with Docker Compose)
- **Prometheus**: http://localhost:9090
- **Pushgateway**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
  - Overview Dashboard: `/d/query-gateway-overview`
  - Training Dashboard: `/d/model-training`

## Testing Commands

### Unit & Integration Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/api/test_schemas.py -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

### Load Tests
```bash
# Run all load tests
./tests/load/run_tests.sh

# Run specific scenario
locust -f tests/load/scenarios/cache_test.py \
  --host=http://localhost:8000 \
  --users=50 --spawn-rate=10 --run-time=60s \
  --headless --html=reports/cache_test.html
```

## Key Configuration (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model | ./models/router |
| `MODEL_DEVICE` | Inference device (cpu/cuda/mps) | cpu |
| `BATCH_MAX_SIZE` | Maximum batch size | 32 |
| `BATCH_MAX_WAIT_MS` | Batch wait time (ms) | 10 |
| `CACHE_L1_SIZE` | L1 cache max entries | 10000 |
| `CACHE_L1_TTL_SEC` | L1 cache TTL (seconds) | 300 |
| `REDIS_URL` | Redis URL for L2 cache | None |
| `CONFIDENCE_THRESHOLD` | Routing confidence threshold | 0.7 |

## API Quick Reference

### Classify Query
```bash
curl -X POST http://localhost:8000/v1/query-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize this article"}'

# Response: {"label": "0", "confidence": 0.95}
# Header: x-router-latency: 5.23
```

### Health Checks
```bash
curl http://localhost:8000/health/live   # Liveness
curl http://localhost:8000/health/ready  # Readiness
curl http://localhost:8000/health/deep   # Deep health (model, cache, batch)
```

## Architecture Highlights

1. **Dynamic Batching**: Requests aggregated within 10ms window (max 32)
2. **Adaptive Batching**: Auto-adjusts batch size (8-64) and wait time (5-15ms) based on load
3. **Two-Level Cache**: L1 (in-memory, <1ms) + L2 (Redis, <5ms)
4. **Confidence-aware Routing**: Low-confidence Fast Path predictions routed to Slow Path

## Common Tasks

### Rebuild Docker Image
```bash
docker build -t query-gateway:latest .
docker compose up -d --build gateway
```

### View Metrics
```bash
curl http://localhost:8000/metrics/
```

### Check Model Status
```bash
curl -s http://localhost:8000/health/deep | jq '.checks.model'
```

### Clear Cache (Restart Service)
```bash
docker compose restart gateway
```

## Known Considerations

1. **Model Training**: The model is trained on databricks-dolly-15k dataset. Uses early stopping (patience=3) and warmup+linear decay LR scheduler.

2. **GPU Support**: Set `MODEL_DEVICE=cuda` for GPU inference (requires CUDA-enabled PyTorch).

3. **Redis Optional**: L2 cache is optional. Without Redis, only L1 in-memory cache is used.

4. **Training Metrics**: Each training run has a unique `run_id`. Use Grafana's "Training Run" selector to filter or compare runs.

## Documentation

- [Architecture](docs/architecture.md) - System design details
- [Deployment Guide](docs/operations/deployment.md) - Production deployment
- [Monitoring Guide](docs/operations/monitoring.md) - Metrics and dashboards
- [Runbook](docs/operations/runbook.md) - Incident response
- [Testing Guide](docs/development/testing.md) - Testing strategy
