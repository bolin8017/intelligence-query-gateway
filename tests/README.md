# Test Suite Documentation

This document describes the complete testing architecture for the Intelligence Query Gateway Microservices project.

## Test Philosophy

- **Tests are executable specifications**: Each test describes a specific behavior the system must exhibit
- **Fail fast with clarity**: Test failures immediately point to the problem location and nature
- **Independence**: Each test is hermetic—no shared state, deterministic execution, order-independent
- **Prefer real over fake**: Use actual implementations except when external dependencies introduce non-determinism
- **Layer separation**: Unit tests validate logic, integration tests validate composition

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/core/test_config.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "cache"
```

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── unit/                                # Unit tests (no I/O)
│   ├── conftest.py                      # Unit-specific fixtures
│   ├── core/                            # Core module tests
│   │   ├── test_config.py               # Settings validation
│   │   └── test_exceptions.py           # Exception hierarchy
│   ├── utils/                           # Utility tests
│   │   └── test_hashing.py              # Text normalization, cache keys
│   ├── models/                          # Model tests (with mocked torch)
│   │   └── test_semantic_router.py      # [To be added]
│   ├── services/                        # Service logic tests
│   │   ├── test_classifier.py           # Classification logic
│   │   ├── test_cache.py                # LRU cache behavior
│   │   └── test_batching.py             # Queue and batching mechanics
│   └── api/                             # API schema tests
│       └── test_schemas.py              # Pydantic validation
└── integration/                         # Integration tests
    ├── conftest.py                      # Integration fixtures
    ├── test_classifier_service.py       # Classifier + Model
    ├── test_cache_integration.py        # Cache + Hashing
    └── test_batching_integration.py     # Batching + Classifier + Queue
```

## Test Layers

### Unit Tests (`tests/unit/`)

**Purpose**: Validate pure business logic in isolation

**Characteristics**:
- No I/O (no filesystem, network, database)
- No time dependencies (mock `time.time()`, `time.perf_counter()`)
- No external services (Redis, model files)
- Fast (<10ms per test)
- Mock only external boundaries

**Coverage**:
- Configuration validation and property methods
- Exception hierarchy and error formatting
- Cache key generation and text normalization
- Service business logic (LRU eviction, TTL expiration, batch triggering)
- Pydantic schema validation

### Integration Tests (`tests/integration/`)

**Purpose**: Validate component interactions with controlled dependencies

**Characteristics**:
- Use real implementations where feasible
- Isolate external dependencies (mock model inference)
- Deterministic and repeatable
- Fast (<100ms per test)
- No network calls to external services

**Coverage**:
- SemanticRouter + ClassifierService integration
- CacheService + hashing utilities workflow
- BatchingService + ClassifierService + asyncio queue
- Complete classification workflows

## Test Coverage Goals

| Module | Unit Tests | Integration Tests | Target Coverage |
|--------|-----------|------------------|----------------|
| `src/core/config.py` | ✅ Complete | N/A | 100% |
| `src/core/exceptions.py` | ✅ Complete | Via API tests | 100% |
| `src/utils/hashing.py` | ✅ Complete | ✅ Complete | 100% |
| `src/services/cache.py` | ✅ Complete | ✅ Complete | 95%+ |
| `src/services/classifier.py` | ✅ Complete | ✅ Complete | 95%+ |
| `src/services/batching.py` | ✅ Complete | ✅ Complete | 90%+ |
| `src/api/schemas.py` | ✅ Complete | Via API tests | 100% |

## Key Testing Patterns

### 1. Mocking Time for Deterministic Tests

```python
from unittest.mock import patch

async def test_cache_expiration():
    cache = CacheService(ttl_seconds=100)

    with patch("time.time", return_value=1000.0):
        await cache.set("key", "value")

    with patch("time.time", return_value=1150.0):
        assert await cache.get("key") is None  # Expired
```

### 2. Testing Async Services with Real Event Loops

```python
@pytest.mark.asyncio
async def test_batching_service():
    service = BatchingService(classifier=mock_classifier)
    await service.start()

    result = await service.classify("test")

    await service.stop()
```

### 3. Parametrized Tests for Edge Cases

```python
@pytest.mark.parametrize("port,should_fail", [
    (0, True),      # Below minimum
    (1, False),     # Minimum valid
    (8080, False),  # Normal
    (65535, False), # Maximum valid
    (65536, True),  # Above maximum
])
def test_port_validation(port, should_fail):
    if should_fail:
        with pytest.raises(ValidationError):
            Settings(app_port=port)
    else:
        settings = Settings(app_port=port)
        assert settings.app_port == port
```

### 4. Mocking External Dependencies at Boundaries

```python
@pytest.fixture
def mock_semantic_router():
    mock = Mock(spec=SemanticRouter)
    mock.is_loaded = True
    mock.predict.return_value = [
        ClassificationResult(label=0, confidence=0.9, probabilities=[0.9, 0.1])
    ]
    return mock
```

## Adding New Tests

When adding functionality, follow this checklist:

1. **Write unit tests first**:
   - Test pure logic without I/O
   - Mock external boundaries (time, filesystem, network)
   - Cover edge cases and error conditions

2. **Add integration tests**:
   - Test component interactions
   - Use real implementations with controlled dependencies
   - Validate complete workflows

3. **Verify coverage**:
   ```bash
   pytest --cov=src/path/to/new/module --cov-report=term-missing
   ```

4. **Ensure tests are fast**:
   - Unit tests: <10ms each
   - Integration tests: <100ms each
   - Full suite: <10s total

## Common Issues and Solutions

### Issue: Tests fail with "event loop is closed"

**Solution**: Use `@pytest.mark.asyncio` decorator:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

### Issue: Cache tests are flaky due to timing

**Solution**: Mock `time.time()` for deterministic control:
```python
with patch("time.time", return_value=1000.0):
    await cache.set("key", "value")
```

### Issue: Model loading errors in tests

**Solution**: Use mocked SemanticRouter from fixtures:
```python
def test_classification(mock_semantic_router):
    service = ClassifierService(model=mock_semantic_router)
    result = service.classify("test")
```

## Test Metrics

Track these metrics to maintain test quality:

- **Coverage**: Aim for 90%+ overall, 100% for critical paths
- **Speed**: Full suite should complete in <10 seconds
- **Flakiness**: Zero flaky tests (all deterministic)
- **Clarity**: Test failures should pinpoint exact problem
- **Maintenance**: Tests should survive refactoring when behavior preserved

## Future Enhancements

1. **Add model unit tests**: Test SemanticRouter with mocked PyTorch tensors
2. **Add API endpoint tests**: Test FastAPI routes with TestClient
3. **Add performance benchmarks**: Track inference latency and throughput
4. **Add property-based tests**: Use Hypothesis for edge case generation
5. **Add mutation testing**: Verify test suite catches bugs (using mutmut)
