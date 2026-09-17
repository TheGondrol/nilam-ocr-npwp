# Quick Start Guide - Testing

## Installation

Install test dependencies:
```bash
pip install -e ".[test]"
```

Or install manually:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

## Running Tests

### Quick Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run only unit tests
pytest test/unit

# Run only integration tests
pytest test/integration

# Run specific file
pytest test/unit/test_routes.py

# Run specific test
pytest test/unit/test_routes.py::TestPredictEndpoint::test_predict_success

# Run tests matching pattern
pytest -k "test_predict"

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Watch Mode (requires pytest-watch)
```bash
pip install pytest-watch
ptw test/
```

## Test Results Summary

Current Status: ✅ **157/160 tests passing (79% coverage)**

- Unit Tests: 121 passed, 3 skipped
- Integration Tests: 36 passed

## Common Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run unit and integration tests
pytest -m "unit or integration"
```

## Viewing Coverage

After running with `--cov`, open the HTML report:
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Open report (Windows)
start htmlcov/index.html

# Open report (Linux/Mac)
xdg-open htmlcov/index.html
```

## Troubleshooting

### Tests fail with ImportError
Make sure dependencies are installed:
```bash
pip install -e ".[test]"
```

### Tests are slow
Run only unit tests:
```bash
pytest test/unit -v
```

### Need to debug a test
Add `-vv` for extra verbose and `--pdb` to drop into debugger on failure:
```bash
pytest test/unit/test_routes.py -vv --pdb
```

### See print statements
Use `-s` to disable output capture:
```bash
pytest test/unit/test_routes.py -s
```

## Test Structure

```
test/
├── conftest.py                    # Shared fixtures
├── unit/                          # Fast, isolated tests
│   └── test_*.py
└── integration/                   # Component interaction tests
    └── test_*_integration.py
```

## Writing New Tests

### Unit Test Template
```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyFeature:
    def test_basic_functionality(self):
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
```

### Integration Test Template
```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
class TestMyIntegration:
    @pytest.mark.asyncio
    async def test_full_flow(self):
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.post("/api/endpoint", json={...})
        
        # Assert
        assert response.status_code == 200
```

## CI/CD Integration

### GitHub Actions
Add to `.github/workflows/tests.yml`:
```yaml
- name: Run tests
  run: pytest test/ -v --cov=src --cov-report=xml
```

### GitLab CI
Add to `.gitlab-ci.yml`:
```yaml
test:
  script:
    - pip install -e ".[test]"
    - pytest test/ -v --cov=src --cov-report=xml
```

## Links

- [Full Test Documentation](README.md)
- [Integration Test Guide](integration/README.md)
- [Test Summary](TEST_SUMMARY.md)
- [Pytest Documentation](https://docs.pytest.org/)
