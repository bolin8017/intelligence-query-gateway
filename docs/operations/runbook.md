# Operations Runbook

Incident response and troubleshooting procedures for on-call engineers.

**Target Audience**: SRE, DevOps, On-call Engineers

## Quick Reference

### Service Information

| Item | Value |
|------|-------|
| **Service Name** | Intelligence Query Gateway |
| **Endpoint** | http://localhost:8080 |
| **Health Checks** | /health/live, /health/ready, /health/deep |
| **Metrics** | http://localhost:8080/metrics |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 (admin/admin) |
| **Containers** | query-gateway, query-gateway-prometheus, query-gateway-grafana |

### SLO Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Availability** | 99.9% | < 99% |
| **P99 Latency (cache hit)** | < 10ms | > 10ms |
| **P99 Latency (overall)** | < 100ms | > 100ms |
| **Error Rate** | < 0.1% | > 0.1% |
| **Cache Hit Rate** | > 30% | < 30% |

### Emergency Contacts

```
# On-Call
Primary: [Your team's primary on-call]
Secondary: [Your team's secondary on-call]

# Escalation
Team Lead: [Team lead contact]
Engineering Manager: [Manager contact]
```

## Alert Response Guide

### Critical Alerts

#### ServiceDown

**Symptoms**: Service completely unavailable, all requests failing

**Impact**: Total service outage, users cannot access the service

**Diagnosis**:
```bash
# 1. Check container status
docker ps -a | grep query-gateway

# 2. View recent logs
docker logs query-gateway --tail=100

# 3. Check for errors
docker logs query-gateway 2>&1 | grep -i error | tail -20

# 4. Check resource usage
docker stats query-gateway --no-stream
```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| Container crashed | `docker compose up -d gateway` |
| OOM (out of memory) | Increase memory limit or reduce cache size |
| Model loading failed | Verify model files: `ls -lh models/router/` |
| Port conflict | `lsof -i :8000` to find conflicting process |

**Recovery Steps**:
```bash
# Quick restart
docker compose restart gateway

# If restart fails, rebuild
docker compose down
docker compose up -d

# Verify recovery
curl http://localhost:8080/health/ready
```

**Prevention**:
- Configure appropriate memory limits
- Implement circuit breaker patterns
- Add retry mechanisms with exponential backoff

---

#### ModelNotReady

**Symptoms**: Model not loaded, service returns 503 for classification requests

**Impact**: /v1/query-classify endpoint unavailable

**Diagnosis**:
```bash
# 1. Check model status
curl http://localhost:8080/health/deep | jq '.checks.model'

# 2. Verify model files
docker exec query-gateway ls -lh /app/models/router/

# 3. Check loading logs
docker logs query-gateway 2>&1 | grep -i "model"

# 4. Check memory usage
docker stats query-gateway --no-stream
```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| Model files missing | Mount correct directory: `./models:/app/models:ro` |
| Insufficient memory | Increase Docker memory limit |
| File permissions | `chmod -R 755 models/` |
| Corrupted model files | Retrain or re-download model |

**Recovery Steps**:
```bash
# 1. Verify model file integrity
ls -lh models/router/
# Expected: config.json, pytorch_model.bin, tokenizer_config.json

# 2. Restart service if files exist
docker compose restart gateway

# 3. Monitor loading progress
docker logs -f query-gateway | grep "model"

# 4. Verify recovery
curl http://localhost:8080/health/ready
```

---

#### HighErrorRate

**Symptoms**: Error rate exceeds 0.1%

**Impact**: Degraded user experience, partial request failures

**Diagnosis**:
```bash
# 1. Check error rate trend
curl -s 'http://localhost:9090/api/v1/query?query=rate(query_gateway_requests_total{status="error"}[5m])'

# 2. View error logs
docker logs query-gateway 2>&1 | grep '"level":"error"' | tail -20

# 3. Analyze error types
docker logs query-gateway 2>&1 | grep '"level":"error"' | jq -r '.error_type' | sort | uniq -c

# 4. Check model status
curl http://localhost:8080/health/deep | jq '.checks.model'
```

**Common Error Types**:

| Error Type | Likely Cause | Solution |
|-----------|-------------|----------|
| `ValidationError` | Invalid request format | Check client implementation |
| `ModelNotReady` | Model not loaded | See ModelNotReady section |
| `TimeoutError` | Inference timeout | Check model performance, increase timeout |
| `MemoryError` | Insufficient memory | Reduce batch size or increase memory |

**Recovery Steps**:
```bash
# 1. Monitor for auto-recovery
watch -n 10 'curl -s http://localhost:9090/api/v1/query?query=rate(query_gateway_requests_total{status=\"error\"}[1m]) | jq'

# 2. If errors persist, restart
docker compose restart gateway

# 3. Check upstream dependencies (if any)
# e.g., Redis, external APIs
```

---

### Warning Alerts

#### HighP99Latency

**Symptoms**: P99 latency exceeds 100ms

**Impact**: Degraded user experience, slow responses for 1% of requests

**Diagnosis**:
```bash
# 1. Check current latency distribution
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99, rate(query_gateway_request_latency_seconds_bucket[5m]))'

# 2. Check batch efficiency
curl http://localhost:8080/health/deep | jq '.checks.batch_processor'

# 3. Check current load
curl -s 'http://localhost:9090/api/v1/query?query=query_gateway_active_requests'

# 4. Check cache hit rate
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total[5m]))/(sum(rate(query_gateway_cache_hits_total[5m]))+sum(rate(query_gateway_cache_misses_total[5m])))'
```

**Common Causes**:

| Cause | Indicator | Solution |
|-------|-----------|----------|
| Low batch efficiency | Avg batch size < 4 | Adjust `BATCH_MAX_WAIT_MS` |
| Low cache hit rate | < 30% | Increase `CACHE_L1_SIZE` |
| High concurrent load | active_requests > 50 | Scale horizontally or add rate limiting |
| Slow model inference | Inference P95 > 50ms | Check CPU/GPU utilization |

**Tuning Steps**:
```bash
# 1. Check batch size distribution
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.5, rate(query_gateway_inference_batch_size_bucket[10m]))'

# If median < 4, increase wait time
# Edit docker compose.yml:
# BATCH_MAX_WAIT_MS=15  (from 10)

# 2. Check cache status
curl http://localhost:8080/health/deep | jq '.checks.cache'

# If l1_size near l1_max_size, increase cache
# CACHE_L1_SIZE=20000  (from 10000)

# 3. Apply changes
docker compose up -d gateway

# 4. Monitor improvement
watch -n 30 'curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.99, rate(query_gateway_request_latency_seconds_bucket[5m]))"'
```

---

#### LowCacheHitRate

**Symptoms**: Cache hit rate below 30%

**Impact**: More requests require model inference, increased latency

**Diagnosis**:
```bash
# 1. Check current cache status (L1 + L2)
curl http://localhost:8080/health/deep | jq '.checks.cache'
# Look for: l1_enabled, l1_size, l2_enabled, l2_healthy, l2_size

# 2. View cache hit rate trend (overall)
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total[10m]))/(sum(rate(query_gateway_cache_hits_total[10m]))+sum(rate(query_gateway_cache_misses_total[10m])))'

# 3. Check cache eviction rate
curl -s 'http://localhost:9090/api/v1/query?query=query_gateway_cache_size{level="L1"}'
```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| L1 cache capacity too small | Increase `CACHE_L1_SIZE` (10K → 20K) |
| L2 Redis not enabled | Enable Redis with `REDIS_URL` env var |
| High query diversity | Normal behavior, L2 cache helps with this |
| TTL too short | Increase `CACHE_L1_TTL_SEC` or `CACHE_L2_TTL_SEC` |
| Redis connection failed | Check Redis health, verify `REDIS_URL` |
| Traffic pattern changed | Analyze query patterns, adjust strategy |

**Recovery Steps**:
```bash
# Option 1: Enable Redis L2 cache (recommended for production)
# Edit docker compose.yml, ensure REDIS_URL is set:
# REDIS_URL=redis://redis:6379/0
# CACHE_L2_TTL_SEC=3600

# Restart to pick up Redis
docker compose restart gateway

# Verify Redis is connected
curl -s http://localhost:8080/health/deep | jq '.checks.cache.l2_enabled'
# Should return: true

# Option 2: Increase L1 cache size
# Edit docker compose.yml:
# CACHE_L1_SIZE=20000
# CACHE_L1_TTL_SEC=600  (from 300)

docker compose up -d gateway

# Option 3: Monitor hit rate improvement
watch -n 60 'curl -s "http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total[5m]))/(sum(rate(query_gateway_cache_hits_total[5m]))+sum(rate(query_gateway_cache_misses_total[5m])))"'

# Check L1 and L2 hit rates separately
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total{level="L1"}[5m]))'
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total{level="L2"}[5m]))'
```

---

#### RedisConnectionFailure

**Symptoms**: L2 cache disabled, `l2_healthy: false` in health check

**Impact**: Loss of distributed caching, lower cache hit rate across instances

**Diagnosis**:
```bash
# 1. Check Redis connection status
curl -s http://localhost:8080/health/deep | jq '.checks.cache'
# Look for: l2_enabled: false or l2_healthy: false

# 2. Verify Redis is running
docker ps | grep redis
docker logs query-gateway-redis --tail 50

# 3. Test Redis connectivity from gateway
docker exec query-gateway ping redis
docker exec query-gateway-redis redis-cli ping
# Expected: PONG

# 4. Check REDIS_URL configuration
docker exec query-gateway env | grep REDIS_URL

# 5. Check Redis authentication (if password protected)
docker exec query-gateway-redis redis-cli AUTH your_password
```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| Redis container down | Restart: `docker compose restart redis` |
| Wrong REDIS_URL | Fix connection string in docker compose.yml |
| Network isolation | Check Docker networks, ensure same network |
| Redis authentication failed | Verify password in REDIS_URL |
| Redis maxmemory exceeded | Increase maxmemory or flush old keys |
| Redis port conflict | Check port 6379 availability |

**Recovery Steps**:
```bash
# 1. Restart Redis service
docker compose restart redis

# 2. Wait for Redis to be healthy
docker exec query-gateway-redis redis-cli ping

# 3. Restart gateway to reconnect
docker compose restart gateway

# 4. Verify L2 cache is working
curl -s http://localhost:8080/health/deep | jq '.checks.cache.l2_healthy'
# Should return: true

# 5. Monitor cache metrics
curl -s http://localhost:8080/metrics | grep cache_hits_total

# 6. If Redis data corrupted, flush and restart
docker exec query-gateway-redis redis-cli FLUSHALL
docker compose restart gateway
```

**Prevention**:
- Monitor Redis health with Prometheus alerting
- Set up Redis persistence if needed (enable AOF/RDB)
- Use Redis Sentinel/Cluster for high availability
- Regular Redis backups if using persistence

---

#### HighBatchQueueDepth

**Symptoms**: Batch queue depth exceeds 100

**Impact**: Increased request wait time, potential timeouts

**Diagnosis**:
```bash
# 1. Check current queue depth
curl -s 'http://localhost:9090/api/v1/query?query=query_gateway_batch_queue_size'

# 2. Check request rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(query_gateway_requests_total[1m])'

# 3. Check batch processing speed
curl -s 'http://localhost:9090/api/v1/query?query=rate(query_gateway_inference_batch_size_count[1m])'

# 4. Check model inference time
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, rate(query_gateway_inference_latency_seconds_bucket[5m]))'
```

**Common Causes**:

| Cause | Solution |
|-------|----------|
| Request spike | Scale horizontally |
| Slow model inference | Check CPU/GPU, optimize model |
| Batch size too small | Increase `BATCH_MAX_SIZE` |
| Batch wait time too long | Decrease `BATCH_MAX_WAIT_MS` |

**Recovery Steps**:
```bash
# 1. If transient spike, wait for auto-recovery

# 2. If sustained high load, adjust batch config
# Edit docker compose.yml:
# BATCH_MAX_SIZE=64  (from 32)
# BATCH_MAX_WAIT_MS=5   (from 10, faster processing)

# 3. Restart service
docker compose up -d gateway

# 4. For Kubernetes, scale horizontally
# kubectl scale deployment query-gateway --replicas=3
```

---

## Common Failure Scenarios

### Scenario 1: Service Startup Failure

**Symptoms**: Container exits immediately after `docker compose up`

**Troubleshooting**:
```bash
# 1. View full logs
docker compose logs gateway

# 2. Common errors:
# - "Model file not found" → Check model directory mount
# - "Port already in use" → Kill process using port 8000
# - "Permission denied" → Check file permissions

# 3. Validate configuration
docker compose config

# 4. Check environment variables
docker compose run --rm gateway env | grep -E "(MODEL|BATCH|CACHE)"
```

---

### Scenario 2: Prometheus Not Scraping Metrics

**Symptoms**: Prometheus UI shows target DOWN

**Troubleshooting**:
```bash
# 1. Verify gateway metrics endpoint
curl http://localhost:8080/metrics

# 2. Check network connectivity
docker exec query-gateway-prometheus wget -O- http://gateway:8000/metrics

# 3. Check Prometheus configuration
docker exec query-gateway-prometheus cat /etc/prometheus/prometheus.yml

# 4. View Prometheus logs
docker logs query-gateway-prometheus | grep -i error
```

**Common Issues**:

| Problem | Solution |
|---------|----------|
| Network isolation | Ensure all services in same `monitoring` network |
| DNS resolution failure | Use `gateway:8000` not `localhost:8080` |
| Configuration error | Validate YAML syntax |

---

### Scenario 3: Grafana Shows "No Data"

**Symptoms**: All dashboard panels empty

**Troubleshooting**:
```bash
# 1. Check datasource configuration
curl -u admin:admin http://localhost:3000/api/datasources

# 2. Test Prometheus query
curl 'http://localhost:9090/api/v1/query?query=up'

# 3. Test datasource in Grafana
# UI: Configuration → Data Sources → Prometheus → Save & Test

# 4. Check time range
# Dashboard top right: select "Last 15 minutes"
```

---

### Scenario 4: High Memory Usage

**Symptoms**: Container OOM or system slowdown

**Troubleshooting**:
```bash
# 1. Check memory usage
docker stats query-gateway --no-stream

# 2. Check cache size
curl http://localhost:8080/health/deep | jq '.checks.cache'

# 3. Check batch queue
curl http://localhost:8080/health/deep | jq '.checks.batch_processor'

# 4. Check model memory usage
docker exec query-gateway python3 -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Optimization**:
```bash
# 1. Reduce cache size
# CACHE_L1_SIZE=5000

# 2. Reduce batch size
# BATCH_MAX_SIZE=16

# 3. Increase Docker memory limit
# docker compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 6G  (increase from 4G)

# 4. Restart service
docker compose up -d gateway
```

---

## Diagnostic Tools

### Quick Diagnostic Script

Create `scripts/diagnose.sh`:

```bash
#!/bin/bash
# Quick diagnostic script

echo "=== Query Gateway Diagnostics ==="
echo "Timestamp: $(date)"
echo ""

echo "1. Container Status:"
docker compose ps
echo ""

echo "2. Gateway Health:"
curl -s http://localhost:8080/health/deep | jq '.'
echo ""

echo "3. Current Metrics:"
echo "  Active Requests:"
curl -s 'http://localhost:9090/api/v1/query?query=query_gateway_active_requests' | jq -r '.data.result[0].value[1]'

echo "  Request Rate (1m):"
curl -s 'http://localhost:9090/api/v1/query?query=rate(query_gateway_requests_total[1m])' | jq -r '.data.result[0].value[1]'

echo "  Error Rate:"
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_requests_total{status="error"}[5m]))/sum(rate(query_gateway_requests_total[5m]))' | jq -r '.data.result[0].value[1]'

echo "  Cache Hit Rate:"
curl -s 'http://localhost:9090/api/v1/query?query=sum(rate(query_gateway_cache_hits_total[5m]))/(sum(rate(query_gateway_cache_hits_total[5m]))+sum(rate(query_gateway_cache_misses_total[5m])))' | jq -r '.data.result[0].value[1]'

echo ""
echo "4. Recent Errors (last 10):"
docker logs query-gateway 2>&1 | grep '"level":"error"' | tail -10 | jq -r '.event'

echo ""
echo "5. Active Alerts:"
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | {alert: .labels.alertname, severity: .labels.severity, state: .state}'
```

Usage:
```bash
chmod +x scripts/diagnose.sh
./scripts/diagnose.sh
```

---

## Maintenance Procedures

### Routine Checks

**Daily** (automated monitoring):
- Service availability
- Alert status
- Error rate trends

**Weekly**:
- Review Grafana dashboards
- Check disk space (Prometheus data)
- Review logs for anomalies
- Verify backup strategy

**Monthly**:
- Review SLO compliance
- Capacity planning assessment
- Update documentation
- Security patch review

### Log Cleanup

```bash
# Clean Docker logs
docker compose down
sudo truncate -s 0 $(docker inspect --format='{{.LogPath}}' query-gateway)

# Clean old Prometheus data (keep last 7 days)
docker exec query-gateway-prometheus \
  promtool tsdb delete --time-range="$(date -d '30 days ago' +%s)000-$(date -d '7 days ago' +%s)000" /prometheus
```

### Backup & Recovery

**Backup Prometheus Data**:
```bash
# 1. Stop Prometheus
docker compose stop prometheus

# 2. Backup data
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data

# 3. Restart Prometheus
docker compose start prometheus
```

**Restore Prometheus Data**:
```bash
# 1. Stop Prometheus
docker compose stop prometheus

# 2. Restore data
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/prometheus-backup-20260121.tar.gz -C /

# 3. Restart Prometheus
docker compose start prometheus
```

## Escalation Procedures

```
Level 1 (0-15 min): On-call Engineer
    ↓ (unable to resolve)
Level 2 (15-30 min): Senior SRE
    ↓ (requires dev support)
Level 3 (30+ min): Development Team Lead
    ↓ (major incident)
Level 4: Engineering Manager
```

## Related Documentation

- [Deployment Guide](deployment.md) - Deployment procedures and configuration
- [Monitoring Guide](monitoring.md) - Metrics, dashboards, and alerting details
- [Architecture](../architecture.md) - System design and components

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-22
**Review cycle**: Quarterly
**Recent changes**: Added Redis L2 cache troubleshooting, updated port to 8080
