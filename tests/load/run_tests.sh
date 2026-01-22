#!/bin/bash
# Load testing execution script for Query Gateway
# This script runs all test scenarios sequentially and generates reports

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HOST="http://localhost:8000"
REPORTS_DIR="tests/load/reports"

# Ensure we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

# Activate conda environment
echo -e "${YELLOW}Activating Conda environment...${NC}"
# Note: Ensure conda is initialized in your shell (run 'conda init' if needed)
conda activate query-gateway

# Check if service is running
echo -e "${YELLOW}Checking if service is running...${NC}"
if ! curl -s -f "${HOST}/health/ready" > /dev/null; then
    echo -e "${RED}Error: Service not responding at ${HOST}${NC}"
    echo "Please start the service with: docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Service is ready${NC}"

# Create reports directory
mkdir -p "${REPORTS_DIR}"

# Function to run a test scenario
run_test() {
    local name=$1
    local file=$2
    local users=$3
    local spawn_rate=$4
    local duration=$5

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Running: ${name}${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo "Users: ${users}, Spawn Rate: ${spawn_rate}/s, Duration: ${duration}"

    locust -f "${file}" \
        --host="${HOST}" \
        --users="${users}" \
        --spawn-rate="${spawn_rate}" \
        --run-time="${duration}" \
        --headless \
        --html="${REPORTS_DIR}/${name}_report.html" \
        --csv="${REPORTS_DIR}/${name}" \
        --loglevel INFO

    echo -e "${GREEN}✓ ${name} completed${NC}"
    echo "Report: ${REPORTS_DIR}/${name}_report.html"
}

# Run all test scenarios
echo -e "${YELLOW}Starting load test suite...${NC}"
echo "Target: ${HOST}"
echo "Reports will be saved to: ${REPORTS_DIR}"

# 1. Baseline Test (10 users, 60s)
run_test "baseline" "tests/load/scenarios/baseline.py" 10 2 "60s"

# 2. Cache Test (50 users, 60s)
run_test "cache_test" "tests/load/scenarios/cache_test.py" 50 10 "60s"

# 3. Batch Test (100 users, 120s)
run_test "batch_test" "tests/load/scenarios/batch_test.py" 100 20 "120s"

# 4. General Load Test (50 users, 120s)
run_test "general" "tests/load/locustfile.py" 50 5 "120s"

# 5. Stress Test (200 users, 120s)
run_test "stress_test" "tests/load/scenarios/stress_test.py" 200 20 "120s"

# Collect final metrics
echo ""
echo -e "${YELLOW}Collecting final metrics...${NC}"
python -m tests.load.utils.metrics_collector || echo "Warning: Failed to collect metrics"

# Summary
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All tests completed!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Reports generated:"
ls -lh "${REPORTS_DIR}"/*.html 2>/dev/null || echo "No HTML reports found"

echo ""
echo "Next steps:"
echo "1. Open HTML reports in browser"
echo "2. Analyze CSV data: ${REPORTS_DIR}/*_stats.csv"
echo "3. Review metrics: curl ${HOST}/metrics"
echo ""
