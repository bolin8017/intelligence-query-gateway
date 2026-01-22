# Testing Guide

Comprehensive testing strategy and procedures for the Intelligence Query Gateway.

## Overview

The project follows the **test pyramid** approach recommended by Google:

- **70% Unit Tests**: Fast, isolated component testing
- **20% Integration Tests**: Component interaction testing
- **10% Load Tests**: Performance and scalability validation

## Test Structure

```
tests/
├── unit/                   # Unit tests (fast, isolated)
│   ├── test_cache.py
│   ├── test_batching.py
│   ├── test_classifier.py
│   └── test_api.py
├── integration/            # Integration tests (requires running service)
│   ├── test_full_flow.py
│   └── test_health_checks.py
├── load/                   # Load tests (Locust)
│   ├── locustfile.py
│   ├── scenarios/
│   │   ├── baseline.py
│   │   ├── cache_test.py
│   │   ├── batch_test.py
│   │   └── stress_test.py
│   └── run_tests.sh
└── fixtures/               # Test data
    └── sample_queries.json
```

## Running Tests

### Quick Test Commands

```bash
# All unit tests
pytest tests/unit/ -v

# All tests with coverage
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term

# Specific test file
pytest tests/unit/test_cache.py -v

# Specific test function
pytest tests/unit/test_cache.py::test_cache_hit -v

# Integration tests (requires running service)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

### With Different Verbosity

```bash
# Minimal output
pytest tests/unit/ -q

# Detailed output
pytest tests/unit/ -vv

# Show print statements
pytest tests/unit/ -s

# Stop on first failure
pytest tests/unit/ -x
```

## Unit Testing

### Testing Philosophy

Unit tests should be:
- **Fast**: < 100ms per test
- **Isolated**: No external dependencies (use mocks)
- **Deterministic**: Same input → same output
- **Focused**: Test one component at a time

### Cache Service Tests

Example unit test structure:

```python
# tests/unit/test_cache.py
import pytest
from src.services.cache import CacheService
from src.api.schemas import ClassifyResult

@pytest.fixture
def cache_service():
    """Create a cache service for testing."""
    return CacheService(max_size=100, ttl_seconds=60)

def test_cache_hit(cache_service):
    """Test cache retrieval with valid key."""
    # Arrange
    key = "test_key"
    result = ClassifyResult(label=0, confidence=0.95, category="classification")

    # Act
    cache_service.set(key, result)
    retrieved = cache_service.get(key)

    # Assert
    assert retrieved is not None
    assert retrieved.label == 0
    assert retrieved.confidence == 0.95

def test_cache_miss(cache_service):
    """Test cache retrieval with non-existent key."""
    result = cache_service.get("nonexistent_key")
    assert result is None

def test_cache_expiration(cache_service):
    """Test cache TTL expiration."""
    import time

    cache_service._ttl = 0.1  # 100ms TTL
    cache_service.set("key", result)

    time.sleep(0.2)  # Wait for expiration

    assert cache_service.get("key") is None

def test_cache_lru_eviction(cache_service):
    """Test LRU eviction when cache is full."""
    # Fill cache to capacity
    for i in range(100):
        cache_service.set(f"key_{i}", result)

    # Add one more item
    cache_service.set("new_key", result)

    # Oldest item should be evicted
    assert cache_service.get("key_0") is None
    assert cache_service.get("new_key") is not None
```

### Batching Service Tests

```python
# tests/unit/test_batching.py
import pytest
import asyncio
from src.services.batching import BatchingService

@pytest.mark.asyncio
async def test_batch_aggregation():
    """Test that multiple requests are batched together."""
    # Mock classifier
    mock_classifier = MockClassifier()
    batching = BatchingService(mock_classifier, max_batch_size=4, max_wait_ms=50)

    # Send 3 requests concurrently
    tasks = [
        batching.classify("query 1"),
        batching.classify("query 2"),
        batching.classify("query 3"),
    ]

    results = await asyncio.gather(*tasks)

    # Verify all requests processed in single batch
    assert mock_classifier.call_count == 1
    assert len(results) == 3

@pytest.mark.asyncio
async def test_batch_timeout_trigger():
    """Test batch processes after timeout even if not full."""
    mock_classifier = MockClassifier()
    batching = BatchingService(mock_classifier, max_batch_size=10, max_wait_ms=100)

    # Send only 2 requests (less than max_batch_size)
    start = time.time()
    result = await batching.classify("query")
    elapsed = time.time() - start

    # Should process after timeout, not wait for full batch
    assert elapsed >= 0.1  # At least 100ms
    assert elapsed < 0.2   # But not too long
```

### Mocking Strategy

Use mocks to isolate components:

```python
from unittest.mock import Mock, AsyncMock

# Mock model inference
mock_model = Mock()
mock_model.predict = Mock(return_value=[0])  # Returns label 0

# Mock async operations
mock_cache = AsyncMock()
mock_cache.get = AsyncMock(return_value=None)
mock_cache.set = AsyncMock()

# Mock external services
@pytest.fixture
def mock_redis():
    with patch('redis.Redis') as mock:
        yield mock
```

## Integration Testing

### Setup

Integration tests require a running service:

```bash
# Terminal 1: Start service
python -m src.main

# Terminal 2: Run integration tests
pytest tests/integration/ -v
```

Or use pytest fixtures to manage lifecycle:

```python
# tests/integration/conftest.py
import pytest
import subprocess
import time
import requests

@pytest.fixture(scope="module")
def running_service():
    """Start service for integration tests."""
    # Start service
    process = subprocess.Popen(
        ["python", "-m", "src.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for service to be ready
    for _ in range(30):
        try:
            response = requests.get("http://localhost:8000/health/ready")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.5)

    yield process

    # Cleanup
    process.terminate()
    process.wait()
```

### Full Flow Tests

```python
# tests/integration/test_full_flow.py
import pytest
import httpx

@pytest.mark.asyncio
async def test_classification_request(running_service):
    """Test complete classification flow."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/query-classify",
            json={"text": "Summarize this article"}
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "data" in data
    assert "metadata" in data

    # Verify data fields
    assert data["data"]["label"] in [0, 1]
    assert 0 <= data["data"]["confidence"] <= 1
    assert data["data"]["category"] in ["classification", "summarization", "creative_writing", "open_qa"]

    # Verify metadata
    assert "request_id" in data["metadata"]
    assert "latency_ms" in data["metadata"]
    assert "cache_hit" in data["metadata"]

@pytest.mark.asyncio
async def test_cache_behavior(running_service):
    """Test cache hit on repeated queries."""
    async with httpx.AsyncClient() as client:
        # First request - cache miss
        response1 = await client.post(
            "http://localhost:8000/v1/query-classify",
            json={"text": "What is machine learning?"}
        )
        latency1 = response1.json()["metadata"]["latency_ms"]
        cache_hit1 = response1.json()["metadata"]["cache_hit"]

        # Second request - should hit cache
        response2 = await client.post(
            "http://localhost:8000/v1/query-classify",
            json={"text": "What is machine learning?"}
        )
        latency2 = response2.json()["metadata"]["latency_ms"]
        cache_hit2 = response2.json()["metadata"]["cache_hit"]

    assert cache_hit1 is False
    assert cache_hit2 is True
    assert latency2 < latency1  # Cache hit should be faster
```

### Health Check Tests

```python
# tests/integration/test_health_checks.py
import httpx

def test_liveness_probe():
    """Test liveness endpoint."""
    response = httpx.get("http://localhost:8000/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness_probe():
    """Test readiness endpoint."""
    response = httpx.get("http://localhost:8000/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["model"] == "loaded"

def test_deep_health_check():
    """Test detailed health endpoint."""
    response = httpx.get("http://localhost:8000/health/deep")
    assert response.status_code == 200
    data = response.json()

    # Verify detailed checks
    assert data["checks"]["model"]["loaded"] is True
    assert data["checks"]["cache"]["l1_enabled"] is True
    assert data["checks"]["batch_processor"]["running"] is True
```

## Load Testing

### Using Locust

Locust provides Python-based load testing:

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import random

class QueryGatewayUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests

    @task(3)  # Weight: 3x more common than unique queries
    def common_query(self):
        """Simulate repeated queries (tests cache)."""
        queries = [
            "Summarize this article",
            "What is machine learning?",
            "Classify this text",
        ]
        self.client.post("/v1/query-classify", json={
            "text": random.choice(queries)
        })

    @task(1)
    def unique_query(self):
        """Simulate unique queries (cache miss)."""
        text = f"Unique query {random.randint(1, 10000)}"
        self.client.post("/v1/query-classify", json={
            "text": text
        })
```

### Running Load Tests

```bash
# Start service
docker compose up -d gateway

# Run baseline test
cd tests/load
locust -f scenarios/baseline.py \
  --host=http://localhost:8000 \
  --users=10 --spawn-rate=2 --run-time=60s \
  --headless --html=reports/baseline.html

# Run cache performance test
locust -f scenarios/cache_test.py \
  --host=http://localhost:8000 \
  --users=50 --spawn-rate=10 --run-time=60s \
  --headless --html=reports/cache.html

# Run stress test
locust -f scenarios/stress_test.py \
  --host=http://localhost:8000 \
  --users=150 --spawn-rate=30 --run-time=90s \
  --headless --html=reports/stress.html

# View reports
open reports/baseline.html
```

### Load Test Scenarios

**Baseline Test** (`scenarios/baseline.py`):
- Users: 10
- Duration: 60s
- Goal: Establish performance baseline

**Cache Test** (`scenarios/cache_test.py`):
- Users: 50
- High cache hit rate (90%+)
- Goal: Validate cache performance

**Batch Test** (`scenarios/batch_test.py`):
- Users: 100
- All unique queries
- Goal: Test batching efficiency

**Stress Test** (`scenarios/stress_test.py`):
- Users: 150
- Duration: 90s
- Goal: Find performance limits

### Interpreting Results

**Key Metrics**:
- **RPS**: Requests per second (throughput)
- **P50/P95/P99 Latency**: Response time percentiles
- **Failure Rate**: Percentage of failed requests

**Target Performance**:
| Scenario | P99 Latency | RPS | Failure Rate |
|----------|-------------|-----|--------------|
| Baseline | < 150ms | > 10 | 0% |
| Cache Hit | < 10ms | > 300 | 0% |
| Batch Processing | < 5000ms* | > 50 | 0% |
| Stress | < 10000ms* | > 60 | 0% |

*High latency due to batch wait time is expected

## Test Coverage

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/unit/ --cov=src --cov-report=html

# View report
open htmlcov/index.html

# Generate terminal report
pytest tests/unit/ --cov=src --cov-report=term

# Fail if coverage below threshold
pytest tests/unit/ --cov=src --cov-fail-under=80
```

### Coverage Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| **API Routes** | 90%+ | Critical user-facing code |
| **Services** | 85%+ | Core business logic |
| **Models** | 70%+ | Mostly third-party wrapper |
| **Utils** | 90%+ | Reusable helpers |
| **Overall** | 80%+ | Balanced coverage |

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run linters
      run: |
        black --check src/ tests/
        ruff check src/ tests/
        mypy src/

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

### Writing Good Tests

**DO**:
- Use descriptive test names: `test_cache_returns_none_on_miss`
- Follow Arrange-Act-Assert pattern
- Test one thing per test
- Use fixtures for common setup
- Mock external dependencies

**DON'T**:
- Test implementation details
- Create test interdependencies
- Use hardcoded sleep() for timing (use timeouts)
- Ignore flaky tests
- Skip writing tests for "simple" code

### Test Data Management

```python
# tests/fixtures/sample_queries.json
{
  "classification": [
    "Is this email spam?",
    "Classify this sentiment"
  ],
  "summarization": [
    "Summarize this article",
    "Give me a brief overview"
  ],
  "creative_writing": [
    "Write a poem about the ocean",
    "Create a short story"
  ],
  "open_qa": [
    "What is machine learning?",
    "Explain quantum computing"
  ]
}
```

Load in tests:
```python
import json
import pytest

@pytest.fixture
def sample_queries():
    with open("tests/fixtures/sample_queries.json") as f:
        return json.load(f)

def test_classification_queries(sample_queries):
    for query in sample_queries["classification"]:
        # Test with query
        pass
```

### Debugging Failed Tests

```bash
# Run with Python debugger
pytest tests/unit/test_cache.py --pdb

# Print test output
pytest tests/unit/test_cache.py -s

# Show local variables on failure
pytest tests/unit/test_cache.py -l

# Re-run only failed tests
pytest --lf

# Run last failed, then all
pytest --ff
```

## Performance Benchmarking

### Benchmark Tests

```python
# tests/unit/test_performance.py
import pytest
import time

def test_cache_performance():
    """Ensure cache operations are fast."""
    cache = CacheService(max_size=10000)

    # Benchmark set operation
    start = time.perf_counter()
    for i in range(1000):
        cache.set(f"key_{i}", result)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1  # 1000 sets should take < 100ms

    # Benchmark get operation
    start = time.perf_counter()
    for i in range(1000):
        cache.get(f"key_{i}")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05  # 1000 gets should take < 50ms
```

## Related Documentation

- [Development Setup](setup.md) - Environment configuration
- [Architecture](../architecture.md) - System design and components
- [Monitoring Guide](../operations/monitoring.md) - Performance metrics

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-21
**Review cycle**: Quarterly
