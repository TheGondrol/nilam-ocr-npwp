# Test Suite for KTP Detection API

This directory contains comprehensive unit and integration tests for the KTP Detection API project.

## Structure

```
tests/
├── __init__.py                  # Test package initialization
├── conftest.py                  # Pytest fixtures and configuration
├── test_config.py              # Tests for core.config module
├── test_device.py              # Tests for core.device module
├── test_predictor.py           # Tests for services.predictor module
├── test_database_service.py    # Tests for services.database_service module
├── test_minio_service.py       # Tests for services.minio_service module
├── test_routes.py              # Tests for api.routes module
├── test_schemas.py             # Tests for api.schemas module
├── test_middleware.py          # Tests for middleware module
└── test_integration.py         # Integration tests for full API
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_config.py
```

### Run tests with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test class or function
```bash
pytest tests/test_config.py::TestConfig::test_config_singleton
```

## Test Categories

### Unit Tests (marked with `@pytest.mark.unit`)
- **test_config.py**: Configuration management tests
- **test_device.py**: Device detection and CPU counting tests
- **test_predictor.py**: Model loading and prediction tests
- **test_database_service.py**: Async database operations tests
- **test_minio_service.py**: MinIO file download tests
- **test_routes.py**: API endpoint tests
- **test_schemas.py**: Pydantic schema validation tests
- **test_middleware.py**: Request ID middleware tests

### Integration Tests (marked with `@pytest.mark.integration`)
- **test_integration.py**: Full API flow tests with mocked dependencies

## Key Features

### Fixtures (conftest.py)
- `mock_config`: Mock configuration dictionary
- `mock_yolo_model`: Mock YOLO model for testing
- `sample_image`: Sample PIL Image for testing
- `sample_image_bytes`: Sample image as bytes
- `temp_config_file`: Temporary config file
- `mock_database_session`: Mock async database session
- `mock_minio_client`: Mock MinIO client
- `clean_env`: Clean environment variables
- `app_client`: FastAPI test client
- `mock_torch_cuda`: Mock GPU availability
- `mock_torch_cpu`: Mock CPU-only environment

### Test Coverage

The test suite covers:
- ✅ Configuration loading and management
- ✅ Device detection (CPU/GPU)
- ✅ Model loading (PyTorch and OpenVINO)
- ✅ Image prediction and detection
- ✅ Database operations (async)
- ✅ MinIO file operations
- ✅ API endpoints (root, health, predict)
- ✅ Pydantic schemas validation
- ✅ Request ID middleware
- ✅ Error handling and edge cases
- ✅ End-to-end API flows

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

Or install all project dependencies:
```bash
pip install -e .
```

## Coverage Report

After running tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

View the HTML report:
```bash
# On Windows
start htmlcov/index.html

# On Linux/Mac
open htmlcov/index.html
```

## Writing New Tests

### Unit Test Template
```python
import pytest

@pytest.mark.unit
class TestMyModule:
    """Test MyModule class"""
    
    def test_my_function(self):
        """Test specific functionality"""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_output
```

### Async Test Template
```python
import pytest

@pytest.mark.unit
@pytest.mark.asyncio
class TestMyAsyncModule:
    """Test async operations"""
    
    async def test_my_async_function(self):
        """Test async functionality"""
        result = await my_async_function()
        assert result is not None
```

## CI/CD Integration

The test suite is designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (database, MinIO, model)
3. **Fixtures**: Reuse common test data via fixtures
4. **Naming**: Use descriptive test names that explain what is being tested
5. **Coverage**: Aim for >80% code coverage
6. **Fast**: Unit tests should run quickly
7. **Clear**: Tests should be easy to understand and maintain

## Troubleshooting

### Tests fail with module not found
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Async tests not running
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

### Coverage reports not generated
```bash
# Install pytest-cov
pip install pytest-cov
```

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain test coverage above 80%
4. Update this README if adding new test files
