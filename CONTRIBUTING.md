# Contributing to Intelligence Query Gateway

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

1. **Prerequisites**
   - Python 3.11+
   - Conda or virtualenv
   - Docker and Docker Compose (for full stack testing)

2. **Environment Setup**
   ```bash
   # Create conda environment
   conda create -n query-gateway python=3.11 -y
   conda activate query-gateway

   # Install dependencies in editable mode with dev extras
   pip install -e ".[dev]"
   ```

3. **Train the Model** (if models/router doesn't exist)
   ```bash
   python scripts/train_router.py --output-dir ./models/router
   ```

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
   - Add tests for new features
   - Update documentation as needed

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src --cov-report=html

   # Run specific test suite
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   ```

4. **Run Load Tests** (optional but recommended for performance changes)
   ```bash
   ./tests/load/run_tests.sh
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for function signatures
- Write docstrings for public APIs (Google style)
- Keep functions focused and small
- Use meaningful variable names

Example:
```python
def classify_query(text: str, threshold: float = 0.7) -> tuple[int, float]:
    """Classify a query into Fast Path or Slow Path.

    Args:
        text: The query text to classify.
        threshold: Confidence threshold for routing (default: 0.7).

    Returns:
        A tuple of (label, confidence) where label is 0 (Fast) or 1 (Slow).

    Raises:
        ValueError: If text is empty or threshold is out of range.
    """
    # Implementation
```

## Testing Guidelines

1. **Unit Tests**
   - Test individual functions and classes in isolation
   - Mock external dependencies (cache, model, etc.)
   - Aim for 80%+ code coverage

2. **Integration Tests**
   - Test API endpoints end-to-end
   - Use TestClient from FastAPI
   - Verify response schemas and status codes

3. **Load Tests**
   - Use Locust for performance testing
   - Test cache effectiveness, batching behavior
   - Measure P50/P95/P99 latencies

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add a clear description of changes in PR
4. Link related issues (if any)
5. Wait for code review and address feedback

## Reporting Issues

When reporting bugs or requesting features:
- Use a clear and descriptive title
- Provide steps to reproduce (for bugs)
- Include environment details (OS, Python version, etc.)
- Attach relevant logs or error messages

## Questions?

- Check [README.md](README.md) and [docs/](docs/) for documentation
- Review [CLAUDE.md](CLAUDE.md) for project context
- Open an issue for questions or discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
