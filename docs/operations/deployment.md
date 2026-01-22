# Deployment Guide

Production deployment procedures for the Intelligence Query Gateway.

## Prerequisites

### Required
- Docker 20.10+ and docker compose 2.x
- Trained model artifacts (see [Model Training](#model-training))
- 4GB+ RAM, 2+ CPU cores

### Optional
- Kubernetes cluster (for production orchestration)
- Redis 7+ instance (for L2 distributed cache)
- Prometheus + Grafana (for monitoring)

## Quick Start

### Using Docker Compose (Recommended)

The Docker Compose stack includes:
- **Gateway**: Main application (port 8080)
- **Redis**: L2 distributed cache (port 6379)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization dashboards (port 3000)

```bash
# 1. Ensure model files exist
ls -la models/router/
# Should contain: config.json, pytorch_model.bin, tokenizer_config.json

# 2. Start all services (Gateway + Redis + Monitoring)
docker compose up -d

# 3. Verify services are running
docker compose ps

# 4. Verify health
curl http://localhost:8080/health/ready

# 5. Check cache status (includes L1 + L2 Redis)
curl -s http://localhost:8080/health/deep | jq '.checks.cache'

# 6. Test classification
curl -X POST http://localhost:8080/v1/query-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize this document"}'

# 7. Access monitoring dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Using Docker

```bash
# 1. Build image
docker build -t query-gateway:latest .

# 2. Run container
docker run -d \
  --name query-gateway \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/router \
  -e APP_ENV=prod \
  -e LOG_FORMAT=json \
  query-gateway:latest

# 3. Check logs
docker logs -f query-gateway

# 4. Verify health
curl http://localhost:8000/health/live
```

## Model Training

Before deployment, train the semantic router model:

```bash
# Using Python directly
python scripts/train_router.py \
  --output-dir ./models/router \
  --epochs 3 \
  --batch-size 16

# Or use pre-trained model (if available)
# Download and extract to ./models/router/
```

Expected output in `models/router/`:
```
config.json
pytorch_model.bin
tokenizer_config.json
special_tokens_map.json
tokenizer.json
vocab.txt
```

## Production Deployment

### Docker Image Building

#### Standard Build
```bash
# Build with version tag
docker build -t query-gateway:v1.0.0 .

# Tag as latest
docker tag query-gateway:v1.0.0 query-gateway:latest

# Push to registry
docker tag query-gateway:v1.0.0 your-registry.com/query-gateway:v1.0.0
docker push your-registry.com/query-gateway:v1.0.0
```

#### Multi-Platform Build
```bash
# Build for both AMD64 and ARM64
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry.com/query-gateway:v1.0.0 \
  --push .
```

### Kubernetes Deployment

Create production-ready Kubernetes manifests:

#### Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: query-gateway
  labels:
    app: query-gateway
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: query-gateway
  template:
    metadata:
      labels:
        app: query-gateway
        version: v1.0.0
    spec:
      containers:
      - name: gateway
        image: your-registry.com/query-gateway:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: APP_ENV
          value: "prod"
        - name: MODEL_PATH
          value: "/app/models/router"
        - name: LOG_FORMAT
          value: "json"
        - name: BATCH_MAX_SIZE
          value: "32"
        - name: BATCH_MAX_WAIT_MS
          value: "10"
        - name: CACHE_L1_SIZE
          value: "10000"
        - name: CACHE_L1_TTL_SEC
          value: "300"
        # Redis L2 Cache (optional but recommended for multi-instance)
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: CACHE_L2_TTL_SEC
          value: "3600"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
```

#### Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: query-gateway
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: query-gateway
```

#### Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Verify deployment
kubectl get pods -l app=query-gateway
kubectl logs -l app=query-gateway -f

# Check service
kubectl get svc query-gateway
```

### Load Balancing with Nginx

For multi-instance deployments without Kubernetes:

```nginx
# nginx.conf
upstream query_gateway {
    least_conn;
    server gateway-1:8000 max_fails=3 fail_timeout=30s;
    server gateway-2:8000 max_fails=3 fail_timeout=30s;
    server gateway-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://query_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Request-ID $request_id;

        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health/ {
        proxy_pass http://query_gateway;
        access_log off;
    }

    location /metrics {
        proxy_pass http://query_gateway;
        allow 10.0.0.0/8;  # Restrict to internal network
        deny all;
    }
}
```

## Environment Variables

### Required Configuration

```bash
# Application
APP_ENV=prod                 # Environment: dev, staging, prod
MODEL_PATH=/app/models/router  # Path to trained model
MODEL_DEVICE=cpu             # Inference device: cpu, cuda, mps
```

### Performance Tuning

```bash
# Batching Configuration
BATCH_MAX_SIZE=32            # Max requests per batch (16-64)
BATCH_MAX_WAIT_MS=10         # Max wait time in ms (5-15)

# L1 Cache (In-Memory LRU)
CACHE_L1_SIZE=10000          # Number of cache entries (5000-50000)
CACHE_L1_TTL_SEC=300         # Cache TTL in seconds (300-3600)

# L2 Cache (Redis, Optional but Recommended for Production)
REDIS_URL=redis://redis:6379/0  # Connection URL
CACHE_L2_TTL_SEC=3600            # L2 TTL in seconds (3600-86400)
```

### Logging and Monitoring

```bash
# Logging
LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json              # json, console

# Monitoring (automatic)
# Metrics exposed at /metrics endpoint
```

### Complete Production Example

```bash
# .env.production
APP_ENV=prod
APP_HOST=0.0.0.0
APP_PORT=8000

MODEL_PATH=/app/models/router
MODEL_DEVICE=cpu

BATCH_MAX_SIZE=32
BATCH_MAX_WAIT_MS=10

CACHE_L1_SIZE=10000
CACHE_L1_TTL_SEC=300

LOG_LEVEL=INFO
LOG_FORMAT=json

# Redis L2 Cache (optional, comment out if not using)
REDIS_URL=redis://redis:6379/0
CACHE_L2_TTL_SEC=3600
```

## Redis L2 Cache Deployment

### Why Use Redis L2 Cache?

**Benefits**:
- **Distributed caching**: Share cache across multiple gateway instances
- **Higher hit rate**: Combined L1 + L2 improves overall cache effectiveness
- **Faster warm-up**: New instances immediately benefit from warm cache
- **Better scaling**: Reduces duplicate inference work across replicas

**When to use**:
- ✅ Multi-instance deployments (2+ replicas)
- ✅ High-traffic production environments
- ✅ When cache hit rate is > 30%
- ❌ Single instance development/testing (L1 is sufficient)

### Docker Compose (Included)

Redis is pre-configured in `docker compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  command: >
    redis-server
    --maxmemory 256mb
    --maxmemory-policy allkeys-lru
    --save ""
    --appendonly no
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
```

Simply start with `docker compose up -d` to get Redis automatically.

### Standalone Redis Deployment

#### Using Docker

```bash
# Start Redis container
docker run -d \
  --name query-gateway-redis \
  -p 6379:6379 \
  redis:7-alpine \
  redis-server \
    --maxmemory 256mb \
    --maxmemory-policy allkeys-lru \
    --save "" \
    --appendonly no

# Verify Redis is running
docker exec query-gateway-redis redis-cli ping
# Expected: PONG

# Connect gateway to Redis
docker run -d \
  --name query-gateway \
  --link query-gateway-redis:redis \
  -e REDIS_URL=redis://redis:6379/0 \
  query-gateway:latest
```

#### Kubernetes Redis Deployment

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        args:
          - redis-server
          - --maxmemory
          - 512mb
          - --maxmemory-policy
          - allkeys-lru
        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "500m"
            memory: "768Mi"
        livenessProbe:
          exec:
            command:
              - redis-cli
              - ping
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
    targetPort: 6379
  selector:
    app: redis
```

Deploy and configure gateway:

```bash
# Deploy Redis
kubectl apply -f redis-deployment.yaml

# Create Redis connection secret
kubectl create secret generic redis-credentials \
  --from-literal=url=redis://redis:6379/0

# Gateway will automatically use Redis via REDIS_URL env var
```

### Redis Configuration Options

#### Cache-Optimized Settings (Recommended)

```bash
# Redis command arguments
redis-server \
  --maxmemory 256mb              # Limit memory usage
  --maxmemory-policy allkeys-lru # Evict least recently used keys
  --save ""                       # Disable RDB snapshots
  --appendonly no                 # Disable AOF persistence
```

**Rationale**:
- **No persistence**: Pure cache, no disk I/O overhead
- **LRU eviction**: Automatically removes cold entries
- **Memory limit**: Prevents OOM, predictable resource usage

#### Production Settings with Authentication

```bash
redis-server \
  --maxmemory 512mb \
  --maxmemory-policy allkeys-lru \
  --requirepass YOUR_STRONG_PASSWORD \
  --save "" \
  --appendonly no

# Gateway connection URL
REDIS_URL=redis://:YOUR_STRONG_PASSWORD@redis:6379/0
```

#### Redis with TLS (Secure)

```bash
# Use rediss:// scheme for TLS
REDIS_URL=rediss://username:password@redis-hostname:6380/0
```

### Verifying Redis L2 Cache

```bash
# Check deep health endpoint
curl -s http://localhost:8080/health/deep | jq '.checks.cache'

# Expected output with Redis enabled:
{
  "l1_enabled": true,
  "l1_size": 245,
  "l1_max_size": 10000,
  "l2_enabled": true,      # ← Redis is enabled
  "l2_healthy": true,      # ← Redis is connected
  "l2_size": 1523          # ← Current entries in Redis
}

# Monitor Redis directly
docker exec query-gateway-redis redis-cli INFO stats | grep keys
docker exec query-gateway-redis redis-cli INFO memory | grep used_memory
```

### Redis Performance Tuning

#### For High-Traffic Environments

```bash
# Increase Redis memory
--maxmemory 1gb

# Increase gateway L2 TTL
CACHE_L2_TTL_SEC=7200  # 2 hours
```

#### For Low-Latency Requirements

```bash
# Use larger L1 cache to reduce Redis lookups
CACHE_L1_SIZE=50000
CACHE_L1_TTL_SEC=600   # 10 minutes
```

### Redis Troubleshooting

#### Redis Connection Failed

```bash
# Check Redis is accessible
docker exec query-gateway-redis redis-cli ping

# Check network connectivity from gateway
docker exec query-gateway ping redis

# Verify REDIS_URL environment variable
docker exec query-gateway env | grep REDIS_URL
```

#### High Redis Memory Usage

```bash
# Check current memory usage
docker exec query-gateway-redis redis-cli INFO memory

# Check key count
docker exec query-gateway-redis redis-cli DBSIZE

# Force eviction
docker exec query-gateway-redis redis-cli CONFIG SET maxmemory 128mb
```

#### Redis Performance Issues

```bash
# Check Redis latency
docker exec query-gateway-redis redis-cli --latency

# Monitor Redis operations
docker exec query-gateway-redis redis-cli MONITOR

# Check slow queries
docker exec query-gateway-redis redis-cli SLOWLOG GET 10
```

## Health Checks

The service provides three health check endpoints:

### Liveness Probe
```bash
curl http://localhost:8000/health/live
```

**Purpose**: Determine if container should be restarted
**Response Time**: < 1ms
**Use Case**: Kubernetes liveness probe

```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:30:00Z"
}
```

### Readiness Probe
```bash
curl http://localhost:8000/health/ready
```

**Purpose**: Determine if container can receive traffic
**Response Time**: < 5ms
**Use Case**: Kubernetes readiness probe, load balancer health checks

```json
{
  "status": "ready",
  "checks": {
    "model": "loaded",
    "cache": "operational"
  }
}
```

Returns 503 if model not loaded or critical services unavailable.

### Deep Health Check
```bash
curl http://localhost:8000/health/deep
```

**Purpose**: Detailed diagnostics for troubleshooting
**Response Time**: < 10ms
**Use Case**: Manual debugging, monitoring dashboards

```json
{
  "status": "healthy",
  "checks": {
    "model": {
      "loaded": true,
      "device": "cpu"
    },
    "cache": {
      "l1_enabled": true,
      "l1_size": 1234,
      "l1_max_size": 10000,
      "l2_enabled": true,
      "l2_healthy": true,
      "l2_size": 5678
    },
    "batch_processor": {
      "running": true,
      "queue_depth": 5,
      "max_batch_size": 32,
      "max_wait_sec": 0.01
    },
    "metrics": {
      "active_requests": 3,
      "model_loaded": 1
    }
  }
}
```

## Monitoring Integration

The service exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

### Key Metrics

- `query_gateway_requests_total` - Total requests by status
- `query_gateway_request_latency_seconds` - Request latency histogram
- `query_gateway_cache_hits_total` - Cache hits by level
- `query_gateway_cache_misses_total` - Cache misses
- `query_gateway_inference_batch_size` - Batch size distribution
- `query_gateway_model_loaded` - Model status (0 or 1)

For complete monitoring setup, see [Monitoring Guide](monitoring.md).

## Verification

### Smoke Tests

```bash
# 1. Health checks
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# 2. Classification request
curl -X POST http://localhost:8000/v1/query-classify \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'

# Expected response:
{
  "data": {
    "label": 1,
    "confidence": 0.95,
    "category": "open_qa"
  },
  "metadata": {
    "request_id": "...",
    "latency_ms": 25.3,
    "cache_hit": false
  }
}

# 3. Metrics endpoint
curl http://localhost:8000/metrics | grep query_gateway_model_loaded
# Should show: query_gateway_model_loaded 1.0
```

### Performance Validation

```bash
# Run baseline load test
cd tests/load
locust -f scenarios/baseline.py \
  --host=http://localhost:8000 \
  --users=10 --spawn-rate=2 --run-time=60s \
  --headless --html=reports/baseline.html

# Verify:
# - P99 latency < 150ms
# - Error rate = 0%
# - Cache hit rate > 30%
```

## Troubleshooting

### Container Fails to Start

**Symptoms**: Container exits immediately

```bash
# Check logs
docker logs query-gateway --tail=100

# Common causes:
# 1. Model files missing
docker exec query-gateway ls -la /app/models/router/

# 2. Memory insufficient
docker stats query-gateway --no-stream

# 3. Port conflict
netstat -tulpn | grep 8000
```

### Model Loading Errors

**Symptoms**: Readiness probe fails, 503 responses

```bash
# Verify model files
docker exec query-gateway ls -lh /app/models/router/

# Expected files:
# - config.json
# - pytorch_model.bin or model.safetensors
# - tokenizer files

# Check memory
docker stats query-gateway
# Ensure at least 2GB available
```

### High Latency

**Symptoms**: P99 > 200ms

```bash
# Check batch efficiency
curl http://localhost:8000/health/deep | jq '.checks.batch_processor'

# Tuning options:
# Option 1: Reduce wait time (lower latency)
# BATCH_MAX_WAIT_MS=5

# Option 2: Increase cache size (higher hit rate)
# CACHE_L1_SIZE=20000

# Option 3: GPU acceleration
# MODEL_DEVICE=cuda
```

## Security Best Practices

### Container Security

```yaml
# docker compose.yml
services:
  gateway:
    image: query-gateway:latest
    read_only: true  # Read-only filesystem
    tmpfs:
      - /tmp
      - /app/logs
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### Network Isolation

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  gateway:
    networks:
      - frontend
      - backend
```

## Scaling Strategies

### Horizontal Scaling

**Docker Compose**:
```bash
docker compose up -d --scale gateway=3
```

**Kubernetes**:
```bash
kubectl scale deployment query-gateway --replicas=5

# Or use HorizontalPodAutoscaler
kubectl autoscale deployment query-gateway \
  --cpu-percent=70 \
  --min=3 \
  --max=10
```

### Vertical Scaling

Increase resources for existing instances:

```yaml
resources:
  limits:
    cpu: "4"      # Increase from 2
    memory: "8Gi" # Increase from 4Gi
```

## Rollback Procedures

### Docker Compose

```bash
# Tag current version before upgrade
docker tag query-gateway:latest query-gateway:stable

# If upgrade fails, rollback
docker compose down
docker tag query-gateway:stable query-gateway:latest
docker compose up -d
```

### Kubernetes

```bash
# View rollout history
kubectl rollout history deployment/query-gateway

# Rollback to previous version
kubectl rollout undo deployment/query-gateway

# Rollback to specific revision
kubectl rollout undo deployment/query-gateway --to-revision=2
```

## Related Documentation

- [Monitoring Guide](monitoring.md) - Metrics, dashboards, and alerting
- [Runbook](runbook.md) - Incident response procedures
- [Architecture](../architecture.md) - System design and performance characteristics
- [Development Setup](../development/setup.md) - Local development environment

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-22
**Review cycle**: Quarterly
**Recent changes**: Added comprehensive Redis L2 cache deployment guide
