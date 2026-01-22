#!/bin/bash
# =============================================================================
# Monitoring Stack Verification Script
# =============================================================================
# This script verifies that the complete monitoring stack is running correctly:
# - Gateway service health
# - Prometheus metrics collection
# - Grafana configuration
# - Alert rules loading
# =============================================================================
#
# Usage:
#   ./verify_monitoring.sh                    # Auto-detect environment
#   GATEWAY_PORT=8000 ./verify_monitoring.sh  # Local development
#   GATEWAY_PORT=8080 ./verify_monitoring.sh  # Docker Compose
#
# Environment Variables:
#   GATEWAY_PORT       - Gateway API port (default: auto-detect)
#   PROMETHEUS_PORT    - Prometheus port (default: 9090)
#   GRAFANA_PORT       - Grafana port (default: 3000)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
# Auto-detect environment: check if running in Docker Compose
if docker compose ps gateway 2>/dev/null | grep -q "Up"; then
    # Docker Compose environment (gateway exposed on host:8080 -> container:8000)
    DEFAULT_GATEWAY_PORT=8080
    ENVIRONMENT="Docker Compose"
elif curl -sf http://localhost:8000/health/live >/dev/null 2>&1; then
    # Local development (direct access to port 8000)
    DEFAULT_GATEWAY_PORT=8000
    ENVIRONMENT="Local"
else
    # Fallback to Docker Compose port
    DEFAULT_GATEWAY_PORT=8080
    ENVIRONMENT="Unknown (assuming Docker Compose)"
fi

# Allow environment variable override
GATEWAY_PORT="${GATEWAY_PORT:-$DEFAULT_GATEWAY_PORT}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

echo "============================================="
echo " Intelligence Query Gateway"
echo " Monitoring Stack Verification"
echo "============================================="
echo " Environment: $ENVIRONMENT"
echo " Gateway Port: $GATEWAY_PORT"
echo "============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# =============================================================================
# 1. Check Docker Containers
# =============================================================================
echo "[1] Checking Docker Containers..."
if docker compose ps | grep -q "Up"; then
    GATEWAY_STATUS=$(docker compose ps gateway | grep "Up" || echo "")
    PROMETHEUS_STATUS=$(docker compose ps prometheus | grep "Up" || echo "")
    GRAFANA_STATUS=$(docker compose ps grafana | grep "Up" || echo "")

    if [ -n "$GATEWAY_STATUS" ]; then
        success "Gateway container is running"
    else
        error "Gateway container is not running"
    fi

    if [ -n "$PROMETHEUS_STATUS" ]; then
        success "Prometheus container is running"
    else
        error "Prometheus container is not running"
    fi

    if [ -n "$GRAFANA_STATUS" ]; then
        success "Grafana container is running"
    else
        error "Grafana container is not running"
    fi
else
    error "No containers are running. Please run 'docker compose up -d' first."
fi
echo ""

# =============================================================================
# 2. Check Gateway Health
# =============================================================================
echo "[2] Checking Gateway Health..."
HEALTH_RESPONSE=$(curl -s http://localhost:$GATEWAY_PORT/health/ready)
if echo "$HEALTH_RESPONSE" | grep -q "\"status\":\"healthy\""; then
    success "Gateway is healthy and ready"
else
    error "Gateway health check failed: $HEALTH_RESPONSE"
fi

# Check deep health endpoint
DEEP_HEALTH=$(curl -s http://localhost:$GATEWAY_PORT/health/deep)
if echo "$DEEP_HEALTH" | grep -q "\"model\""; then
    success "Deep health check endpoint is working"
else
    warning "Deep health check endpoint may have issues"
fi
echo ""

# =============================================================================
# 3. Check Metrics Endpoint
# =============================================================================
echo "[3] Checking Metrics Endpoint..."
# Note: FastAPI mount requires trailing slash
METRICS_RESPONSE=$(curl -sL http://localhost:$GATEWAY_PORT/metrics/)
if echo "$METRICS_RESPONSE" | grep -q "query_gateway"; then
    success "Metrics endpoint is exposing Gateway metrics"

    # Count number of metrics
    METRIC_COUNT=$(echo "$METRICS_RESPONSE" | grep "^query_gateway" | wc -l)
    success "Found $METRIC_COUNT Gateway metrics"
else
    error "Metrics endpoint is not working correctly"
fi
echo ""

# =============================================================================
# 4. Check Prometheus
# =============================================================================
echo "[4] Checking Prometheus..."
PROM_HEALTH=$(curl -s http://localhost:$PROMETHEUS_PORT/-/healthy)
if echo "$PROM_HEALTH" | grep -q "Healthy"; then
    success "Prometheus is healthy"
else
    error "Prometheus health check failed"
fi

# Check if Prometheus can scrape Gateway
UP_QUERY=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=up")
if echo "$UP_QUERY" | grep -q "\"job\":\"query-gateway\""; then
    UP_VALUE=$(echo "$UP_QUERY" | python3 -c "import sys, json; data=json.load(sys.stdin); result=[r for r in data['data']['result'] if r['metric']['job']=='query-gateway']; print(result[0]['value'][1] if result else '0')" 2>/dev/null || echo "0")

    if [ "$UP_VALUE" = "1" ]; then
        success "Prometheus is successfully scraping Gateway (up=1)"
    else
        error "Prometheus cannot scrape Gateway (up=$UP_VALUE)"
    fi
else
    warning "Gateway target not found in Prometheus yet (may need to wait for first scrape)"
fi
echo ""

# =============================================================================
# 5. Check Alert Rules
# =============================================================================
echo "[5] Checking Alert Rules..."
RULES_RESPONSE=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/rules")
GROUPS_COUNT=$(echo "$RULES_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data['data']['groups']))" 2>/dev/null || echo "0")

if [ "$GROUPS_COUNT" -ge "5" ]; then
    success "Alert rules loaded: $GROUPS_COUNT groups found"

    # List alert groups
    echo "$RULES_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for group in data['data']['groups']:
    print(f\"    - {group['name']}: {len(group['rules'])} rules\")
" 2>/dev/null || true
else
    error "Alert rules not loaded correctly (expected 5 groups, found $GROUPS_COUNT)"
fi
echo ""

# =============================================================================
# 6. Check Grafana
# =============================================================================
echo "[6] Checking Grafana..."
GRAFANA_HEALTH=$(curl -s http://localhost:$GRAFANA_PORT/api/health)
if echo "$GRAFANA_HEALTH" | grep -q "\"database\":\"ok\""; then
    success "Grafana is healthy"
else
    error "Grafana health check failed"
fi

# Check datasource
DATASOURCES=$(curl -s -u admin:admin http://localhost:$GRAFANA_PORT/api/datasources)
if echo "$DATASOURCES" | grep -q "\"name\":\"Prometheus\""; then
    success "Prometheus datasource is configured"
else
    warning "Prometheus datasource may not be configured correctly"
fi
echo ""

# =============================================================================
# 7. Generate Test Metrics
# =============================================================================
echo "[7] Generating Test Metrics..."
echo "    Sending 5 test requests to Gateway..."
for i in {1..5}; do
    curl -s -X POST http://localhost:$GATEWAY_PORT/v1/query-classify \
      -H "Content-Type: application/json" \
      -d "{\"text\": \"Verification test query number $i\"}" > /dev/null
done
success "Sent 5 test requests"

echo "    Waiting 10 seconds for metrics to be scraped..."
sleep 10

# Verify metrics were collected
REQUESTS_TOTAL=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=query_gateway_requests_total" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    result = data['data']['result']
    if result:
        print(result[0]['value'][1])
    else:
        print('0')
except:
    print('0')
" 2>/dev/null || echo "0")

if [ "$REQUESTS_TOTAL" != "0" ]; then
    success "Metrics are being collected (total requests: $REQUESTS_TOTAL)"
else
    warning "Metrics may not be collected yet (try again in a few seconds)"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "============================================="
echo " Verification Complete! ✓"
echo "============================================="
echo ""
echo "Access URLs:"
echo "  - Gateway API:  http://localhost:$GATEWAY_PORT"
echo "  - Swagger Docs: http://localhost:$GATEWAY_PORT/docs"
echo "  - Metrics:      http://localhost:$GATEWAY_PORT/metrics/"
echo "  - Prometheus:   http://localhost:$PROMETHEUS_PORT"
echo "  - Grafana:      http://localhost:$GRAFANA_PORT (admin/admin)"
echo ""
echo "Environment Details:"
echo "  - Detected:     $ENVIRONMENT"
echo "  - Gateway Port: $GATEWAY_PORT"
echo ""
echo "Next steps:"
echo "  1. Open Grafana: http://localhost:$GRAFANA_PORT"
echo "  2. Navigate to Dashboards → Query Gateway - Overview"
echo "  3. Generate more load: tests/load/locustfile.py"
echo "  4. Check alerts: http://localhost:$PROMETHEUS_PORT/alerts"
echo ""
echo "For more information, see:"
echo "  - monitoring/README.md"
echo "  - docs/operations/monitoring.md"
echo "  - docs/operations/runbook.md"
echo ""
