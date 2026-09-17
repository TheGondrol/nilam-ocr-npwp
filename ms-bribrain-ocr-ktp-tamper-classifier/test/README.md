# Unit Tests Documentation

This directory contains comprehensive unit tests for the OCR KTP Tamper Classifier project.

## Test Structure

```
test/
├── conftest.py                      # Shared fixtures and test configuration
├── pytest.ini                       # Pytest configuration
└── unit/
    ├── test_image_preprocessing.py  # Tests for image preprocessing service
    ├── test_model_loader.py         # Tests for model loading functionality
    ├── test_tamper_detection.py     # Tests for tamper detection service
    ├── test_database_service.py     # Tests for database operations
    ├── test_minio_service.py        # Tests for MinIO storage service
    ├── test_routes.py               # Tests for API routes
    ├── test_config.py               # Tests for configuration management
    ├── test_device.py               # Tests for device management
    ├── test_logging.py              # Tests for logging system
    └── test_middleware.py           # Tests for middleware components
```

## Prerequisites

Install test dependencies:

```bash
pip install -e ".[test]"
```

Or install manually:

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Run specific test file
```bash
pytest test/unit/test_image_preprocessing.py
```

### Run specific test class
```bash
pytest test/unit/test_routes.py::TestHealthCheckEndpoint
```

### Run specific test function
```bash
pytest test/unit/test_routes.py::TestHealthCheckEndpoint::test_health_check_when_healthy
```

### Run with verbose output
```bash
pytest -v
```

### Run tests matching a pattern
```bash
pytest -k "test_predict"
```

### Run only unit tests
```bash
pytest -m unit
```

## Test Coverage

Current test coverage includes:

### Services Layer (100%)
- **image_preprocessing.py**: Image resizing, padding, preprocessing, and loading
- **model_loader.py**: Model loading, CPU threading configuration, device selection
- **tamper_detection.py**: Tamper detection service, predictions (single & batch)
- **database_service.py**: Database engine initialization, log insertion
- **minio_service.py**: Model download from MinIO storage

### API Layer (100%)
- **routes.py**: All endpoints (root, health, predict, batch predict)
- Response models validation
- Error handling

### Core Layer (100%)
- **config.py**: Configuration loading, validation, environment substitution
- **device.py**: Device management (CUDA/CPU/MPS), auto-detection
- **logging.py**: Logging setup, formatters, request ID correlation

### Middleware Layer (100%)
- **add_requestid.py**: Request ID middleware, context management

## Test Fixtures

Common fixtures available in `conftest.py`:

- `sample_image`: A sample PIL Image for testing
- `sample_image_bytes`: Sample image bytes (JPEG format)
- `mock_processor`: Mocked image processor
- `mock_model`: Mocked ML model with proper structure
- `mock_torch_device`: Mock torch device
- `sample_config_dict`: Sample configuration dictionary
- `temp_model_path`: Temporary model path with dummy files

## Writing New Tests

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyFeature:
    """Tests for MyFeature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        input_data = "test"
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == "expected"
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality"""
        result = await my_async_function()
        assert result is not None
    
    @patch("module.dependency")
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency"""
        mock_dependency.return_value = "mocked"
        result = my_function_with_dependency()
        assert result == "mocked"
```

## Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies**: Database, MinIO, model loading, etc.
4. **Test edge cases**: Empty inputs, invalid data, exceptions
5. **Use fixtures**: Reuse common test data and setup
6. **Mark tests appropriately**: Use `@pytest.mark.unit` for unit tests
7. **Test both success and failure paths**: Positive and negative test cases
8. **Keep tests independent**: Each test should run independently

## Continuous Integration

Tests should be run as part of CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml --cov-report=term-missing
```

## Viewing Coverage Reports

After running tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

Open the HTML report:
```bash
# On Windows
start htmlcov/index.html

# On Linux/Mac
open htmlcov/index.html
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you've installed the package in development mode:
```bash
pip install -e .
```

### Async Test Warnings
If you see warnings about async tests, ensure `pytest-asyncio` is installed and configured in `pytest.ini`.

### Mock Issues
When mocking, ensure you're patching the right location. Use the import path where the object is used, not where it's defined.

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
