# Load Testing Guide

This directory contains load testing scenarios for the Intelligence Query Gateway using Locust.

## Prerequisites

1. **Activate Conda environment**:
   ```bash
   conda activate query-gateway
   ```

2. **Install Locust** (if not already installed):
   ```bash
   pip install locust
   ```

3. **Start the service**:
   ```bash
   # Option 1: Docker Compose (recommended)
   docker compose up -d

   # Option 2: Direct Docker
   docker run -d \
     --name query-gateway \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     -e MODEL_PATH=/app/models/router \
     query-gateway:latest
   ```

4. **Verify service is running**:
   ```bash
   curl http://localhost:8000/health/ready
   ```

## Test Scenarios

### 1. General Load Test (Mixed Workload)

Tests realistic traffic with mixed cache hits and unique queries:

```bash
# Headless mode
locust -f locustfile.py \
  --host=http://localhost:8000 \
  --users=50 \
  --spawn-rate=5 \
  --run-time=120s \
  --headless \
  --html=reports/general_report.html \
  --csv=reports/general

# Web UI mode
locust -f locustfile.py --host=http://localhost:8000
# Then visit http://localhost:8089
```

### 2. Baseline Test

Establishes performance baselines with minimal load:

```bash
locust -f scenarios/baseline.py \
  --host=http://localhost:8000 \
  --users=10 \
  --spawn-rate=2 \
  --run-time=60s \
  --headless \
  --html=reports/baseline_report.html \
  --csv=reports/baseline
```

**Expected Results**:
- P99 latency < 100ms
- Error rate < 0.1%
- Establishes performance baseline

### 3. Cache Performance Test

Tests L1 cache with high cache hit rates:

```bash
locust -f scenarios/cache_test.py \
  --host=http://localhost:8000 \
  --users=50 \
  --spawn-rate=10 \
  --run-time=60s \
  --headless \
  --html=reports/cache_test_report.html \
  --csv=reports/cache_test
```

**Expected Results**:
- Cache hit rate > 90%
- P99 latency < 5ms (Phase 2 achieved < 1ms)
- Very low CPU usage

### 4. Batch Processing Test

Tests dynamic batching efficiency:

```bash
locust -f scenarios/batch_test.py \
  --host=http://localhost:8000 \
  --users=100 \
  --spawn-rate=20 \
  --run-time=120s \
  --headless \
  --html=reports/batch_test_report.html \
  --csv=reports/batch_test
```

**Expected Results**:
- Average batch size > 8 under load
- Speedup: 7-11x (Phase 2 benchmark)
- P99 latency < 100ms

### 5. Stress Test

Pushes system to find limits:

```bash
locust -f scenarios/stress_test.py \
  --host=http://localhost:8000 \
  --users=200 \
  --spawn-rate=20 \
  --run-time=120s \
  --headless \
  --html=reports/stress_test_report.html \
  --csv=reports/stress_test
```

**Expected Results**:
- Error rate < 0.1%
- Service remains stable
- Identifies bottlenecks

## Performance Targets (SLO)

Based on design document and Phase 2 results:

| Metric | Target | Source |
|--------|--------|--------|
| P50 Latency | < 30ms | Design doc |
| P95 Latency | < 50ms | Design doc |
| P99 Latency | < 100ms | Design doc |
| Cache Hit Latency | < 5ms | Phase 2: < 1ms |
| RPS (10 users) | > 100 | Baseline |
| RPS (100 users) | > 500 | High load |
| Cache Hit Rate | > 30% | Mixed workload |
| Error Rate | < 0.1% | Reliability |

## Directory Structure

```
tests/load/
├── __init__.py
├── locustfile.py           # Main test scenarios
├── scenarios/
│   ├── __init__.py
│   ├── baseline.py         # Baseline test
│   ├── cache_test.py       # Cache performance
│   ├── batch_test.py       # Batch processing
│   └── stress_test.py      # Stress test
├── utils/
│   ├── __init__.py
│   ├── data_generator.py   # Test data generation
│   └── metrics_collector.py # Prometheus metrics
├── reports/                # Test report output
│   ├── README.md
│   └── .gitkeep
└── README.md               # This file
```

## Monitoring During Tests

### Docker Stats

Monitor container resources during tests:

```bash
docker stats query-gateway
```

### Prometheus Metrics

Collect metrics manually:

```bash
curl http://localhost:8000/metrics
```

Or use the utility:

```bash
conda activate query-gateway
python -m tests.load.utils.metrics_collector
```

### Real-time Monitoring

If using Web UI mode, visit:
- http://localhost:8089 - Locust dashboard
- http://localhost:8000/metrics - Prometheus metrics

## Analyzing Results

### Locust HTML Report

Open `reports/*_report.html` in a browser to see:
- Response time distribution
- RPS over time
- Failure statistics
- Charts and visualizations

### CSV Data

Use `reports/*_stats.csv` for detailed analysis:

```python
import pandas as pd

# Load statistics
df = pd.read_csv('reports/stress_test_stats.csv')

# Analyze latency percentiles
print(df[['Name', '50%', '95%', '99%']].to_string())

# Calculate RPS
print(f"Average RPS: {df['Requests/s'].mean():.2f}")
```

### Prometheus Metrics

Key metrics to check:

- `query_gateway_requests_total` - Total requests
- `query_gateway_cache_hits_total` - Cache hits
- `query_gateway_cache_misses_total` - Cache misses
- `query_gateway_batch_size` - Batch size distribution
- `query_gateway_request_duration_seconds` - Latency

## Troubleshooting

### Service Not Responding

```bash
# Check if service is running
docker ps | grep query-gateway

# Check service logs
docker logs query-gateway

# Restart service
docker compose restart
```

### Locust Import Errors

Make sure you're in the Conda environment:

```bash
conda activate query-gateway
python --version  # Should show Python 3.11.x
```

### High Error Rates

If error rates exceed targets:

1. Check Docker logs for errors
2. Reduce user count and spawn rate
3. Increase `BATCH_MAX_WAIT_MS` in config
4. Verify model files are accessible

## Best Practices

1. **Warm-up**: Run a short test first to warm up caches
2. **Incremental load**: Start with low users, increase gradually
3. **Monitor resources**: Watch CPU/memory during tests
4. **Document results**: Save reports and metrics for comparison
5. **Clean state**: Restart service between major test runs

## References

- [Locust Documentation](https://docs.locust.io/)
- [Phase 4 Prompt](../../docs/phase4-prompt.md)
- [Design Document](../../docs/plans/2026-01-21-semantic-router-gateway-design.md)
- [Phase 2 Results](../../docs/PHASE2_OPTIMIZATION_REPORT.md)
