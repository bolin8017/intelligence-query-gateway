# Intelligence Query Gateway

A semantic router gateway microservice for classifying queries into Fast Path or Slow Path.

## Overview

This service implements a Semantic Router that classifies user queries based on semantic complexity:

- **Fast Path (Label 0)**: Simple tasks like classification and summarization
- **Slow Path (Label 1)**: Complex tasks like creative writing and open Q&A

## Features

- **High Concurrency**: Async I/O with FastAPI and uvicorn
- **Dynamic Batching**: Request aggregation within configurable time windows
- **Two-Level Caching**: L1 in-memory LRU + L2 Redis for distributed deployments
- **Confidence Routing**: Optional confidence-based routing decisions
- **Observability**: Structured logging (structlog) + Prometheus metrics + Grafana dashboards

## Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended for environment management)

### Installation

```bash
# Create and activate conda environment
conda create -n query-gateway python=3.11 -y
conda activate query-gateway

# Install dependencies
pip install -e ".[dev]"
```

### Train the Model

```bash
# Basic training with early stopping
python scripts/train_router.py --output-dir ./models/router

# With real-time monitoring (requires Docker Compose stack)
python scripts/train_router.py \
  --output-dir ./models/router \
  --pushgateway-url http://localhost:9091
```

Training features:
- **Early stopping**: Automatically stops when validation loss converges (patience=3)
- **LR scheduler**: Warmup + linear decay for optimal convergence
- **Real-time monitoring**: View training progress in Grafana at `/d/model-training`

### Run the Service

```bash
# Development mode
python -m src.main

# Or with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Deployment

### Docker (Recommended)

The project includes a complete Docker Compose stack with:
- **Gateway**: Main application service (port 8080)
- **Redis**: L2 distributed cache (port 6379)
- **Prometheus**: Metrics collection (port 9090)
- **Pushgateway**: Batch job metrics for training (port 9091)
- **Grafana**: Visualization dashboards (port 3000)

#### Quick Start with Docker Compose

```bash
# 1. Build and start all services (Gateway, Redis, Prometheus, Grafana)
docker compose up -d

# 2. Check service status
docker compose ps

# 3. View logs
docker compose logs -f gateway

# 4. Test API endpoints
curl http://localhost:8080/health/ready
curl -s http://localhost:8080/health/deep | jq '.checks.cache'

# 5. Access monitoring dashboards
# Grafana: http://localhost:3000 (admin/admin)
#   - Overview: /d/query-gateway-overview
#     * Pie chart for Fast/Slow Path distribution
#     * Color-coded confidence score trends (P50/P95/P99)
#   - Training: /d/model-training
# Prometheus: http://localhost:9090
```

#### Using Docker Directly

**Without Redis (L1 cache only):**
```bash
# 1. Build the image
docker build -t query-gateway:latest .

# 2. Run the container
docker run -d \
  --name query-gateway \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/router \
  query-gateway:latest

# 3. Check health
curl http://localhost:8000/health/live
```

**With Redis (L1 + L2 cache):**
```bash
# 1. Start Redis
docker run -d \
  --name query-gateway-redis \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

# 2. Run gateway with Redis connection
docker run -d \
  --name query-gateway \
  -p 8080:8000 \
  --link query-gateway-redis:redis \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/router \
  -e REDIS_URL=redis://redis:6379/0 \
  query-gateway:latest

# 3. Verify cache is working
curl -s http://localhost:8080/health/deep | jq '.checks.cache'
```

#### Production Deployment

For production deployments, see the comprehensive [Deployment Guide](docs/operations/deployment.md).

### Local Development

For local development setup, see [Development Setup Guide](docs/development/setup.md).

## API Specification

### POST /v1/query-classify

Classify a query into Fast Path or Slow Path.

**Request:**
```json
{"text": "User input string..."}
```

**Response:**
```json
{"label": "0"}
```

**Response Headers:**
- `x-router-latency`: Router latency in milliseconds

### Health Endpoints

- `GET /health/live` - Liveness probe (basic health check)
- `GET /health/ready` - Readiness probe (service readiness)
- `GET /health/deep` - Deep health check (model, cache, batch processor status)
- `GET /metrics` - Prometheus metrics

## Configuration

Configuration is managed via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (dev/staging/prod) | dev |
| `APP_HOST` | Server bind address | 0.0.0.0 |
| `APP_PORT` | Server port | 8000 |
| `MODEL_PATH` | Path to trained model | ./models/router |
| `MODEL_DEVICE` | Device for inference (cpu/cuda/mps) | cpu |
| `BATCH_MAX_SIZE` | Maximum batch size | 32 |
| `BATCH_MAX_WAIT_MS` | Batch wait time (ms) | 10 |
| `CACHE_L1_SIZE` | L1 cache max entries | 10000 |
| `CACHE_L1_TTL_SEC` | L1 cache TTL in seconds | 300 |
| `REDIS_URL` | Redis connection URL (optional) | None |
| `CACHE_L2_TTL_SEC` | L2 cache TTL in seconds | 3600 |
| `LOG_LEVEL` | Logging level | INFO |
| `LOG_FORMAT` | Log format (json/console) | json |
| `CONFIDENCE_THRESHOLD` | Confidence threshold for routing | 0.7 |

**Redis Configuration**:

To enable L2 Redis cache, set the `REDIS_URL` environment variable:
```bash
# Docker Compose (already configured)
REDIS_URL=redis://redis:6379/0

# External Redis
REDIS_URL=redis://username:password@host:port/db

# Redis with SSL
REDIS_URL=rediss://host:port/db
```

Redis is configured with cache-optimized settings:
- Max memory: 256MB
- Eviction policy: allkeys-lru (removes least recently used keys)
- Persistence: Disabled (pure cache, no AOF/RDB)

**Docker-specific**:
- All environment variables can be set via `docker run -e` or in `docker compose.yml`
- See [.env.production.example](.env.production.example) for complete configuration reference

## System Design

### Batching Strategy

- **Batch Window**: ≤10ms (configurable via `BATCH_MAX_WAIT_MS`)
- **Max Batch Size**: 32 (configurable via `BATCH_MAX_SIZE`)
- Requests are aggregated using `asyncio.Queue` with time-based and size-based triggers

### Cache Strategy

The service implements a sophisticated two-level caching system:

- **L1 Cache (In-Memory LRU)**:
  - Local in-memory cache with LRU eviction
  - Default size: 10,000 entries
  - Default TTL: 300 seconds (5 minutes)
  - Ultra-low latency: < 1ms

- **L2 Cache (Redis)**:
  - Optional distributed cache for multi-instance deployments
  - Default TTL: 3,600 seconds (1 hour)
  - Configured via `REDIS_URL` environment variable
  - Redis settings: 256MB maxmemory, allkeys-lru eviction policy

- **Cache Key Generation**: SHA256 hash of normalized query text
- **Cache Coordinator**: `TwoLevelCache` manages L1/L2 lookups and write-through

### Confidence-aware Routing

The router implements confidence-aware routing to handle uncertain predictions conservatively:

**How it works**:
- The model outputs a confidence score (0.0-1.0) representing prediction certainty
- Confidence is calculated using softmax probabilities: `confidence = max(P(class_0), P(class_1))`
- When confidence is below the threshold AND the prediction is Fast Path (0), the request is routed to Slow Path (1) instead

**Threshold Selection** (default: 0.7):
- Analyzed confidence score distribution on the test set
- 0.7 threshold balances precision (avoiding false Fast Path) vs recall (not over-routing to Slow Path)
- At 0.7 threshold: ~95% of correct predictions have confidence ≥ 0.7
- Configurable via `CONFIDENCE_THRESHOLD` environment variable

**Response Format**:
```json
{"label": "0", "confidence": 0.95}
```

**Rationale**: Low-confidence Fast Path predictions are more risky than low-confidence Slow Path predictions. Routing uncertain queries to Slow Path ensures they receive more thorough processing.

### Adaptive Batching

The batching system dynamically adjusts based on current load:

- **Low Load**: Shorter wait times (down to 5ms) to minimize latency
- **High Load**: Longer wait times (up to 15ms) to maximize throughput
- **Batch Size**: Adjusts between 8-64 based on queue depth

This ensures optimal performance across varying traffic patterns.

## Performance

### Phase 4 - Load Testing Results

**Latest Benchmark** (2026-01-21):

| Test Scenario | Users | P50 Latency | P95 Latency | P99 Latency | RPS | Error Rate |
|--------------|-------|-------------|-------------|-------------|-----|-----------|
| **Cache Test** | 50 | 3ms | 5ms | 7ms | 365.7 | 0% |
| **Baseline** | 10 | 3ms | 89ms | 140ms | 12.3 | 0% |
| **Stress Test** | 150 | 330ms | 5100ms | 7200ms | 69.1 | 0% |

**Key Achievements**:
- ✅ **Cache Performance**: P99 latency < 7ms (target: < 100ms)
- ✅ **High Throughput**: 365 RPS under cache-heavy load
- ✅ **Zero Errors**: 0% failure rate across all tests
- ✅ **Cache Hit Rate**: 92.1% in realistic scenarios
- ✅ **Two-Level Cache**: L1 (< 1ms) + L2 Redis (< 5ms) for distributed deployments

**Test Environment**:
- Platform: WSL2 Linux, Docker containerized
- CPU: 2 cores limited
- Memory: 4GB limited
- Model Device: CPU (no GPU)

For detailed performance analysis, see [Architecture Document](docs/architecture.md#performance-characteristics).

### Load Testing

Run load tests with Locust:

```bash
# Activate conda environment (adjust path as needed)
conda activate query-gateway

# Run all tests
./tests/load/run_tests.sh

# Or run specific scenarios
locust -f tests/load/scenarios/cache_test.py \
  --host=http://localhost:8000 \
  --users=50 --spawn-rate=10 --run-time=60s \
  --headless --html=reports/cache_test.html

# Interactive Web UI
locust -f tests/load/locustfile.py --host=http://localhost:8000
# Visit http://localhost:8089
```

See [Testing Guide](docs/development/testing.md) for detailed instructions.

### Historical Performance

**Phase 1 - Basic Implementation**:
- Router Latency: ~25-30ms (NVIDIA GeForce RTX 2060)
- Status: ✅ Phase 1 Complete

**Phase 2 - Optimization**:
- Cache Hit Latency: < 1ms
- Batching Speedup: 7-11x
- Status: ✅ Phase 2 Complete

## Classification Performance

**Model**: Fine-tuned DistilBERT on Databricks Dolly 15k dataset

| Metric | Test Set Performance |
|--------|---------------------|
| Accuracy | **98.60%** |
| F1 Score | **98.95%** |
| Precision | **98.22%** |
| Recall | **99.70%** |
| Dataset Size | 9,966 samples (4 categories) |

**Training Configuration**:
- Early stopping with patience=3 epochs
- Warmup + linear decay LR scheduler
- Real-time metrics via Prometheus Pushgateway

**Category Mapping**:
- Fast Path (0): `classification`, `summarization`
- Slow Path (1): `creative_writing`, `open_qa`

## Documentation

Complete documentation is available in the [docs/](docs/) directory:

### For Operators
- **[Deployment Guide](docs/operations/deployment.md)** - Production deployment procedures
- **[Monitoring Guide](docs/operations/monitoring.md)** - Metrics, dashboards, and alerting
- **[Runbook](docs/operations/runbook.md)** - Incident response and troubleshooting

### For Developers
- **[Development Setup](docs/development/setup.md)** - Environment configuration
- **[Testing Guide](docs/development/testing.md)** - Testing strategy and execution

### Architecture
- **[System Architecture](docs/architecture.md)** - Design decisions and performance characteristics
- **[Design History](docs/design-history/)** - Historical design documents

## Project Structure

```
intelligence-query-gateway/
├── src/                       # Application source code
│   ├── api/                  # API layer (routes, schemas)
│   ├── services/             # Business logic layer
│   │   ├── cache.py          # Two-level cache (L1 + L2 Redis)
│   │   ├── batching.py       # Dynamic request batching
│   │   └── classifier.py     # Query classification service
│   ├── models/               # ML model layer
│   ├── core/                 # Core components (config, logging, metrics)
│   └── utils/                # Utility functions
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── load/                 # Load testing with Locust
├── scripts/                   # Training and utility scripts
├── docs/                      # Documentation
│   ├── README.md             # Documentation index
│   ├── architecture.md       # System design
│   ├── operations/           # Operations guides
│   ├── development/          # Development guides
│   └── design-history/       # Archived design docs
├── models/                    # Trained model files (not in git)
├── monitoring/                # Observability stack
│   ├── prometheus/           # Prometheus configuration & alerts
│   └── grafana/              # Grafana dashboards & provisioning
├── Dockerfile                 # Multi-stage Docker build
├── docker compose.yml         # Complete stack (Gateway + Redis + Monitoring)
└── pyproject.toml             # Project dependencies
```

## License

MIT
