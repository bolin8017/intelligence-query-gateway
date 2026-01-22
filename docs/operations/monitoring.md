# Monitoring & Observability

Complete monitoring and observability guide for the Intelligence Query Gateway.

## Overview

The service implements a production-grade observability stack based on the three pillars:

- **Metrics**: Prometheus for time-series data collection
- **Logs**: Structured JSON logging with contextual binding
- **Traces**: Request ID propagation for distributed tracing

## Monitoring Stack

### Architecture

```
Application (FastAPI)
    │
    ├─→ Metrics (/metrics endpoint)
    │       ↓
    │   Prometheus (scrapes every 10s)
    │       ↓
    │   Grafana (visualizes)
    │
    ├─→ Logs (stdout/stderr)
    │       ↓
    │   JSON structured format
    │
    └─→ Traces (X-Request-ID header)
        ↓
    Request correlation
```

### Components

| Component | Port | Purpose |
|-----------|------|---------|
| **Query Gateway** | 8080 | Application server |
| **Redis** | 6379 | L2 distributed cache |
| **Prometheus** | 9090 | Metrics collection and alerting |
| **Pushgateway** | 9091 | Batch job metrics (model training) |
| **Grafana** | 3000 | Visualization dashboards |

## Quick Start

### Launch Monitoring Stack

```bash
# Start all services
docker compose up -d

# Verify status
docker compose ps

# Access interfaces
# - Application: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Redis: localhost:6379 (via redis-cli)
```

### Verify Metrics Collection

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'
# Should return: "up"

# View raw metrics
curl http://localhost:8080/metrics

# Query specific metric
curl -s 'http://localhost:9090/api/v1/query?query=query_gateway_model_loaded' \
  | jq '.data.result[0].value[1]'
# Should return: "1"
```

## Metrics

### Available Metrics

#### Request Metrics

```promql
# Total requests by status
query_gateway_requests_total{status="success|error"}

# Request latency histogram
query_gateway_request_latency_seconds_bucket

# Active concurrent requests
query_gateway_active_requests
```

#### Inference Metrics

```promql
# Model inference latency
query_gateway_inference_latency_seconds_bucket

# Batch size distribution
query_gateway_inference_batch_size_bucket

# Classification results by label
query_gateway_classifications_total{label="0|1"}

# Confidence score distribution
query_gateway_confidence_score_bucket
```

#### Cache Metrics

```promql
# Cache hits by level (L1 = in-memory, L2 = Redis)
query_gateway_cache_hits_total{level="L1"}
query_gateway_cache_hits_total{level="L2"}

# Cache misses by level
query_gateway_cache_misses_total{level="L1"}
query_gateway_cache_misses_total{level="L2"}

# Current cache size by level
query_gateway_cache_size{level="L1"}
query_gateway_cache_size{level="L2"}

# Cache operation latency
query_gateway_cache_latency_seconds{level="L1|L2",operation="get|set"}
```

#### Batch Processing Metrics

```promql
# Batch queue depth
query_gateway_batch_queue_size

# Batch wait time
query_gateway_batch_wait_time_seconds
```

#### System Health

```promql
# Model loaded status (0 or 1)
query_gateway_model_loaded
```

#### Training Metrics (via Pushgateway)

Step-level metrics (updated every N steps during training):

```promql
# Current training step
router_training_global_step{run_id="..."}

# Real-time training loss
router_training_step_train_loss{run_id="..."}

# Current learning rate
router_training_step_learning_rate{run_id="..."}

# Gradient norm (for detecting explosion/vanishing)
router_training_step_gradient_norm{run_id="..."}
```

Epoch-level metrics (updated after each epoch):

```promql
# Epoch number
router_training_epoch{run_id="..."}

# Training and validation loss
router_training_epoch_train_loss{run_id="..."}
router_training_val_loss{run_id="..."}
router_training_best_val_loss{run_id="..."}

# Validation performance
router_training_val_accuracy{run_id="..."}
router_training_val_f1{run_id="..."}
router_training_val_precision{run_id="..."}
router_training_val_recall{run_id="..."}

# Training status
router_training_training_active{run_id="..."}  # 1=running, 0=stopped
router_training_early_stopped{run_id="..."}    # 1=yes, 0=no
router_training_patience_counter{run_id="..."}
```

### Common PromQL Queries

```promql
# Request rate (RPS)
rate(query_gateway_requests_total[1m])

# Error rate percentage
100 * sum(rate(query_gateway_requests_total{status="error"}[5m]))
  / sum(rate(query_gateway_requests_total[5m]))

# P50/P95/P99 latency
histogram_quantile(0.50, rate(query_gateway_request_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(query_gateway_request_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(query_gateway_request_latency_seconds_bucket[5m]))

# Overall cache hit rate (L1 + L2 combined)
sum(rate(query_gateway_cache_hits_total[5m]))
  / (sum(rate(query_gateway_cache_hits_total[5m]))
     + sum(rate(query_gateway_cache_misses_total[5m])))

# L1 cache hit rate
sum(rate(query_gateway_cache_hits_total{level="L1"}[5m]))
  / sum(rate(query_gateway_cache_misses_total{level="L1"}[5m]) + rate(query_gateway_cache_hits_total{level="L1"}[5m]))

# L2 cache hit rate (for L1 misses)
sum(rate(query_gateway_cache_hits_total{level="L2"}[5m]))
  / sum(rate(query_gateway_cache_misses_total{level="L1"}[5m]))

# Average batch size
histogram_quantile(0.5, rate(query_gateway_inference_batch_size_bucket[5m]))

# Requests per classification label
sum by (label) (rate(query_gateway_classifications_total[5m]))
```

## Grafana Dashboards

### Overview Dashboard

The main dashboard provides a comprehensive view with 12 panels:

**Access**: http://localhost:3000/d/query-gateway-overview

#### Key Metrics Cards (Top Row)
- Model Status: Green/Red indicator
- RPS: Real-time request rate
- Error Rate: Gauge with threshold markers
- P99 Latency: Milliseconds
- Cache Hit Rate: Percentage gauge
- Active Requests: Current concurrency

#### Time-Series Panels
- **Latency Analysis**: P50/P95/P99 over time
- **Cache Performance**: Hit/miss rate trends by level (L1/L2)
- **Batch Efficiency**: Batch size distribution (P50/P95)
- **Model Inference**: Inference latency percentiles (P50/P95/P99)
- **Classification Distribution**: Pie chart (donut) showing Fast Path vs Slow Path ratio
- **Confidence Score Distribution**: Smooth time-series with P50/P95/P99 percentiles
  - Color-coded: P50 (blue), P95 (green), P99 (orange)
  - Red threshold line at 0.7 for quick anomaly detection

**Auto-refresh**: 10 seconds
**Default time range**: Last 15 minutes

### Model Training Dashboard

Monitors the SemanticRouter model training process in real-time.

**Access**: http://localhost:3000/d/model-training

#### Training Run Selection

Use the **Training Run** dropdown at the top to:
- Select a single run to view
- Select multiple runs for comparison
- Select "All" to view historical data

#### Dashboard Sections

**Training Status (Top Row)**
- Status: Running/Stopped indicator
- Global Step: Current training step
- Epoch: Current epoch number
- Current Loss: Real-time training loss
- Best Val Loss: Lowest validation loss achieved
- Val F1: Latest validation F1 score
- Patience: Early stopping counter (gauge)
- Early Stop: Whether training was early stopped

**Real-Time Training (Step-Level)**
- Training Loss: Step-by-step loss curve
- Learning Rate: LR schedule visualization
- Gradient Norm: Monitors gradient health (threshold at 5.0)

**Epoch-Level Metrics**
- Epoch Loss Curves: Train/Val/Best Val loss per epoch
- Validation Metrics: Accuracy, F1, Precision, Recall

**Auto-refresh**: 5 seconds
**Default time range**: Last 30 minutes

### Custom Dashboards

Create additional dashboards for specific needs:

```bash
# Export existing dashboard
curl -u admin:admin http://localhost:3000/api/dashboards/uid/query-gateway-overview \
  | jq '.dashboard' > my-dashboard.json

# Import dashboard
curl -X POST -u admin:admin \
  -H "Content-Type: application/json" \
  -d @my-dashboard.json \
  http://localhost:3000/api/dashboards/db
```

## Alerting

### Alert Rules

Located in `monitoring/prometheus/alerts.yml`, the service includes 12 production-ready alert rules.

#### SLO-Based Alerts

```yaml
# High P99 Latency
- alert: HighP99Latency
  expr: histogram_quantile(0.99, rate(query_gateway_request_latency_seconds_bucket[5m])) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High P99 latency detected"
    description: "P99 latency is {{ $value | humanizeDuration }}, exceeds 100ms SLO"

# High Error Rate
- alert: HighErrorRate
  expr: |
    sum(rate(query_gateway_requests_total{status="error"}[5m]))
    / sum(rate(query_gateway_requests_total[5m])) > 0.001
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Error rate exceeds threshold"
    description: "Error rate is {{ $value | humanizePercentage }}, exceeds 0.1% SLO"

# Low Cache Hit Rate
- alert: LowCacheHitRate
  expr: |
    sum(rate(query_gateway_cache_hits_total[10m]))
    / (sum(rate(query_gateway_cache_hits_total[10m]))
       + sum(rate(query_gateway_cache_misses_total[10m]))) < 0.3
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Cache hit rate below target"
    description: "Hit rate is {{ $value | humanizePercentage }}, target is 30%"
```

#### Availability Alerts

```yaml
# Service Down
- alert: ServiceDown
  expr: up{job="query-gateway"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Service is down"

# Model Not Ready
- alert: ModelNotReady
  expr: query_gateway_model_loaded != 1
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Model not loaded"
```

#### Performance Alerts

```yaml
# High Batch Queue Depth
- alert: HighBatchQueueDepth
  expr: query_gateway_batch_queue_size > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Batch queue depth is high"
    description: "Queue depth: {{ $value }}, may indicate processing bottleneck"

# Inefficient Batching
- alert: IneffecientBatching
  expr: histogram_quantile(0.5, rate(query_gateway_inference_batch_size_bucket[10m])) < 4
  for: 10m
  labels:
    severity: info
  annotations:
    summary: "Low average batch size"
    description: "Median batch size: {{ $value }}, consider tuning BATCH_MAX_WAIT_MS"
```

### Viewing Active Alerts

```bash
# Via Prometheus API
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts'

# Filter by state
curl -s http://localhost:9090/api/v1/alerts \
  | jq '.data.alerts[] | select(.state=="firing")'

# Or visit Prometheus UI
# http://localhost:9090/alerts
```

### Alert Configuration

Edit `monitoring/prometheus/alerts.yml` to customize:

```bash
# Validate alert rules
docker exec query-gateway-prometheus \
  promtool check rules /etc/prometheus/alerts.yml

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload
```

## Structured Logging

### Log Format

Production logs use JSON format for programmatic parsing:

```json
{
  "timestamp": "2026-01-21T10:30:00.123Z",
  "level": "info",
  "event": "query_classified",
  "request_id": "abc-123-def-456",
  "query_hash": "sha256:...",
  "label": 1,
  "confidence": 0.95,
  "latency_ms": 25.3,
  "cache_hit": false,
  "batch_size": 8
}
```

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| **DEBUG** | Development debugging | Request/response bodies, internal state |
| **INFO** | Normal operations | Request processed, cache hit |
| **WARNING** | Degraded performance | High latency, cache near capacity |
| **ERROR** | Operational errors | Model inference failed, validation error |

### Viewing Logs

```bash
# Real-time logs
docker logs -f query-gateway

# Filter by level
docker logs query-gateway 2>&1 | grep '"level":"error"'

# Pretty-print JSON logs
docker logs query-gateway 2>&1 | tail -20 | jq '.'

# Search by request ID
docker logs query-gateway 2>&1 | grep '"request_id":"abc-123"'
```

### Log Aggregation

For production, integrate with log aggregation systems:

**Fluentd Example**:
```xml
<source>
  @type tail
  path /var/lib/docker/containers/*/*.log
  pos_file /var/log/fluentd/docker.pos
  tag docker.*
  format json
</source>

<filter docker.**>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

<match docker.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name query-gateway-${Time.at(time).strftime("%Y.%m.%d")}
</match>
```

## Request Tracing

### X-Request-ID Header

Every request is assigned a unique ID for correlation:

```bash
# Client provides request ID
curl -X POST http://localhost:8000/v1/query-classify \
  -H "X-Request-ID: my-custom-id-123" \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'

# Response includes same ID
# Response headers: X-Request-ID: my-custom-id-123

# Logs also include this ID
# {"event": "...", "request_id": "my-custom-id-123", ...}
```

### Tracing Across Services

For distributed tracing:

```bash
# Generate request ID at edge service
REQUEST_ID=$(uuidgen)

# Propagate through service chain
curl -H "X-Request-ID: $REQUEST_ID" service-1:8000/api
# service-1 forwards to service-2 with same header
# All logs tagged with REQUEST_ID for correlation
```

## SLO Definitions

Service Level Objectives based on production requirements:

| Metric | Target | Measurement Window | Alert Threshold |
|--------|--------|-------------------|-----------------|
| **Availability** | 99.9% | 30 days | < 99% (1 hour) |
| **P99 Latency** | < 100ms | 5 minutes | > 100ms (5 min) |
| **Error Rate** | < 0.1% | 5 minutes | > 0.1% (2 min) |
| **Cache Hit Rate** | > 30% | 10 minutes | < 30% (10 min) |

### Calculating SLO Compliance

```promql
# Availability SLO
100 * (
  sum(rate(query_gateway_requests_total{status="success"}[30d]))
  / sum(rate(query_gateway_requests_total[30d]))
)

# Latency SLO (percentage under threshold)
100 * (
  sum(rate(query_gateway_request_latency_seconds_bucket{le="0.1"}[30d]))
  / sum(rate(query_gateway_request_latency_seconds_count[30d]))
)
```

## Performance Impact

Monitoring overhead is minimal:

| Operation | Latency | Frequency |
|-----------|---------|-----------|
| **Counter increment** | < 100 ns | Per request |
| **Histogram observation** | < 1 μs | Per request |
| **Gauge update** | < 100 ns | Per state change |
| **Metrics export** | 1-5 ms | Every 10s (Prometheus scrape) |

**Total overhead**: < 0.01% of request latency

## Troubleshooting

### Prometheus Not Scraping Metrics

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets \
  | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Verify gateway metrics endpoint
curl http://localhost:8000/metrics

# Check network connectivity
docker exec query-gateway-prometheus wget -O- http://gateway:8000/metrics

# Review Prometheus logs
docker logs query-gateway-prometheus | grep -i error
```

### Grafana Shows "No Data"

```bash
# Test Prometheus datasource
curl -u admin:admin http://localhost:3000/api/datasources

# Verify Prometheus is reachable from Grafana
docker exec query-gateway-grafana wget -O- http://prometheus:9090/api/v1/query?query=up

# Check dashboard time range
# Ensure "Last 15 minutes" is selected in Grafana UI
```

### Training Dashboard Shows "No Data"

```bash
# Verify Pushgateway is running
curl http://localhost:9091/metrics

# Check Prometheus scrapes Pushgateway
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="pushgateway")'

# Reload Prometheus config if needed
curl -X POST http://localhost:9090/-/reload

# Verify training metrics exist
curl -s 'http://localhost:9090/api/v1/query?query=router_training_training_active' | jq '.data.result'
```

### High Cardinality Warnings

If Prometheus warnings about high cardinality:

```bash
# Check metric cardinality
curl -s http://localhost:9090/api/v1/status/tsdb \
  | jq '.data.seriesCountByMetricName'

# Reduce label dimensions if needed
# Avoid labels with unbounded values (user IDs, timestamps, etc.)
```

## Best Practices

### Metric Naming
- Use `query_gateway_` prefix for all custom metrics
- Follow Prometheus naming conventions (`_total` for counters, `_seconds` for durations)
- Use lowercase with underscores, not camelCase

### Label Usage
- Keep label cardinality low (< 10 values per label)
- Use labels for dimensions you'll query/aggregate by (status, level, label)
- Avoid labels for high-cardinality data (request_id, query content)

### Dashboard Design
- Group related panels together
- Use consistent time ranges across panels
- Set appropriate refresh intervals (10-30s for real-time)
- Add thresholds to gauges for quick visual assessment

### Alert Hygiene
- Every alert must be actionable
- Include remediation steps in annotations
- Use appropriate severity levels
- Set `for` duration to avoid flapping
- Test alerts before deploying (use `amtool` or manual triggers)

## Related Documentation

- [Runbook](runbook.md) - Alert response procedures
- [Deployment Guide](deployment.md) - Health check configuration
- [Architecture](../architecture.md) - System design and performance targets

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-22
**Review cycle**: Quarterly
**Recent changes**: Added model training dashboard with run_id filtering
