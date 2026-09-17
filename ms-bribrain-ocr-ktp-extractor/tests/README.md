# Unit Tests for OCR KTP Extractor

This directory contains comprehensive unit tests for the OCR KTP Extractor project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures and configuration
├── test_core_config.py                  # Tests for src/core/config.py
├── test_core_device.py                  # Tests for src/core/device.py
├── test_core_exceptions.py              # Tests for src/core/exceptions.py
├── test_core_logging.py                 # Tests for src/core/logging.py
├── test_services_ocr_service.py         # Tests for src/services/ocr_service.py
├── test_services_database_service.py    # Tests for src/services/database_service.py
├── test_api_routes.py                   # Tests for src/api/routes.py
├── test_main.py                         # Tests for src/main.py
└── test_models_schemas.py               # Tests for src/models/schemas.py
```

## Installation

Install test dependencies:

```bash
# Using pip
pip install -e ".[test]"

# Using uv
uv pip install -e ".[test]"
```

This will install:
- pytest
- pytest-asyncio
- pytest-cov
- pytest-mock
- httpx

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_core_config.py
```

### Run specific test class
```bash
pytest tests/test_core_config.py::TestSettings
```

### Run specific test method
```bash
pytest tests/test_core_config.py::TestSettings::test_load_config_success
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests matching a pattern
```bash
pytest -k "config"
```

### Run only unit tests (marked)
```bash
pytest -m unit
```

### Run tests excluding slow tests
```bash
pytest -m "not slow"
```

## Test Markers

Tests are marked with the following markers:
- `unit`: Unit tests (most tests)
- `integration`: Integration tests
- `slow`: Slow running tests
- `requires_gpu`: Tests that require GPU
- `requires_db`: Tests that require database connection

## Coverage Reports

After running tests with coverage, view the HTML report:

```bash
# Open in browser (Linux/Mac)
open htmlcov/index.html

# Open in browser (Windows)
start htmlcov/index.html
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_settings`: Mock Settings object
- `temp_config_file`: Temporary config.yaml file
- `sample_image_bytes`: Sample image for testing
- `large_image_bytes`: Large image for size testing
- `mock_ocr_result`: Mock OCR result data
- `mock_paddle_ocr`: Mock PaddleOCR instance
- `mock_database_url`: Mock database URL
- `mock_sqlalchemy_engine`: Mock SQLAlchemy engine
- `mock_request_id`: Mock request ID

## Writing New Tests

### Test File Naming
- Test files must start with `test_`
- Match the module structure: `test_<module_path>.py`

### Test Class Naming
- Test classes must start with `Test`
- Group related tests in classes

### Test Method Naming
- Test methods must start with `test_`
- Use descriptive names: `test_function_does_what_when_condition`

### Example Test

```python
import pytest
from unittest.mock import patch

class TestMyFunction:
    """Test cases for my_function"""
    
    def test_my_function_success(self, mock_settings):
        """Test successful execution"""
        with patch('module.dependency', return_value="mocked"):
            result = my_function()
            assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_my_async_function(self):
        """Test async function"""
        result = await my_async_function()
        assert result is not None
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e ".[test]"
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import Errors
If you get import errors, ensure the project is installed in development mode:
```bash
pip install -e .
```

### Async Test Warnings
If async tests show warnings, ensure `pytest-asyncio` is installed and `asyncio_mode = auto` is set in `pytest.ini`.

### Mock Issues
If mocks aren't working as expected:
1. Check the patch path matches the import location in the code
2. Use `patch.object()` for patching attributes
3. Reset mocks in `setup_method()` or `teardown_method()`

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (database, file system, network)
3. **Fixtures**: Use fixtures for common setup
4. **Assertions**: Use specific assertions with descriptive messages
5. **Coverage**: Aim for >80% code coverage
6. **Documentation**: Document complex test scenarios
7. **Speed**: Keep unit tests fast (<1s each)

## Test Coverage Goals

Current coverage by module:
- Core modules: >90%
- Services: >85%
- API routes: >80%
- Models: >90%
- Overall: >85%

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if needed
