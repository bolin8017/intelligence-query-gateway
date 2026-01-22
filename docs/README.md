# Intelligence Query Gateway - Documentation

Complete documentation for the Intelligence Query Gateway microservice.

## Quick Links

- **Production Operations**: Start with [Deployment Guide](operations/deployment.md)
- **Development Setup**: See [Development Setup](development/setup.md)
- **System Design**: Read [Architecture](architecture.md)
- **Incident Response**: Use [Runbook](operations/runbook.md)

## For Operators

### Deployment & Operations
- [Deployment Guide](operations/deployment.md) - Production deployment procedures
- [Monitoring & Observability](operations/monitoring.md) - Metrics, dashboards, and alerting
- [Runbook](operations/runbook.md) - Incident response and troubleshooting

### Quick Reference
```bash
# Health checks
curl http://localhost:8000/health/live      # Liveness
curl http://localhost:8000/health/ready     # Readiness
curl http://localhost:8000/health/deep      # Detailed diagnostics

# Monitoring
http://localhost:9090   # Prometheus
http://localhost:3000   # Grafana (admin/admin)
```

## For Developers

### Getting Started
- [Development Setup](development/setup.md) - Environment configuration and dependencies
- [Testing Guide](development/testing.md) - Unit, integration, and load testing

### Development Workflow
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run locally
python -m src.main

# Run tests
pytest tests/unit/ -v
```

## Architecture & Design

- [System Architecture](architecture.md) - Design decisions, components, and performance characteristics
- [Design History](design-history/) - Historical design documents (archived)

## Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── architecture.md              # System design and architectural decisions
├── operations/                  # For SRE and DevOps teams
│   ├── deployment.md           # Production deployment guide
│   ├── monitoring.md           # Observability and alerting
│   └── runbook.md              # Incident response procedures
├── development/                 # For developers
│   ├── setup.md                # Development environment setup
│   └── testing.md              # Testing strategy and procedures
└── design-history/             # Archived design documents
    └── semantic-router-gateway-2026-01-21.md
```

## Service Overview

The Intelligence Query Gateway is a semantic router microservice that classifies user queries into Fast Path (simple tasks) or Slow Path (complex tasks) using a fine-tuned DistilBERT model.

**Key Features**:
- High-performance async request processing with dynamic batching
- Two-level caching (L1 LRU + optional L2 Redis)
- Production-grade observability (Prometheus + Grafana)
- Docker-based deployment with health checks

**Performance**:
- P99 latency: < 100ms (< 10ms with cache hit)
- Throughput: > 1000 requests/second
- Model accuracy: 98.6%

## SLO Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Availability | 99.9% | < 99% |
| P99 Latency | < 100ms | > 100ms |
| Error Rate | < 0.1% | > 0.1% |
| Cache Hit Rate | > 30% | < 30% |

## Support & Contribution

- **Issues**: Report bugs and request features via GitHub Issues
- **Pull Requests**: Follow the testing guide and ensure all tests pass
- **Questions**: Refer to the appropriate guide above or contact the platform team

---

**Maintained by**: Platform Team
**Last updated**: 2026-01-21
**Review cycle**: Quarterly
