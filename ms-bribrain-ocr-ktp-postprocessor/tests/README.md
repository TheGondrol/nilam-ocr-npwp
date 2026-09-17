# Test Suite for OCR KTP Postprocessor

This directory contains the comprehensive test suite for the OCR KTP Postprocessor project.

## Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_api_routes.py   # API endpoint tests
│   ├── test_config.py       # Configuration tests
│   ├── test_remapper.py     # Remapper utility tests
│   ├── services/            # Service layer tests
│   │   ├── __init__.py
│   │   ├── test_ocr_processor.py
│   │   └── test_text_helpers.py
│   └── field_matchers/      # Field matcher tests
│       ├── __init__.py
│       ├── test_nik.py
│       ├── test_nama.py
│       └── test_rtrw.py
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_api_routes.py
```

### Run specific test class
```bash
pytest tests/unit/test_api_routes.py::TestOCRPostprocessEndpoint
```

### Run specific test function
```bash
pytest tests/unit/test_api_routes.py::TestOCRPostprocessEndpoint::test_successful_ocr_processing
```

### Run tests with specific marker
```bash
pytest -m unit
pytest -m integration
```

### Run tests in verbose mode
```bash
pytest -v
```

### Run tests and show output even for passing tests
```bash
pytest -s
```

## Test Coverage

After running tests with coverage, view the HTML report:
```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test
```python
import pytest

def test_example():
    """Test description."""
    # Arrange
    expected = "result"
    
    # Act
    actual = function_to_test()
    
    # Assert
    assert actual == expected
```

### Using Fixtures
```python
def test_with_fixture(sample_ocr_data):
    """Test using shared fixture."""
    result = process_ocr(sample_ocr_data)
    assert result is not None
```

### Mocking
```python
from unittest.mock import patch

def test_with_mock():
    """Test with mocked dependency."""
    with patch('module.function') as mock_func:
        mock_func.return_value = "mocked"
        result = call_function_that_uses_it()
        assert result == "mocked"
        mock_func.assert_called_once()
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocking for external dependencies
- Fast execution

### Integration Tests (To be added)
- Test interactions between components
- May use test databases or external services
- Slower execution

## Fixtures

Common fixtures available in `conftest.py`:
- `test_client`: FastAPI test client
- `sample_ocr_data`: Sample OCR data for testing
- `sample_cleaned_data`: Cleaned OCR data
- `sample_nik_data`: Sample NIK data
- `sample_nama_data`: Sample nama data
- `sample_ttl_data`: Sample tempat/tanggal lahir data
- `sample_rtrw_data`: Sample RT/RW data
- `mock_config`: Mocked configuration

## Coverage Goals

- Aim for >80% code coverage
- Critical paths should have 100% coverage
- All edge cases should be tested

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Descriptive names**: Test names should describe what they test
4. **Independent tests**: Tests should not depend on each other
5. **Use fixtures**: Share common setup code
6. **Mock external dependencies**: Isolate unit tests
7. **Test edge cases**: Include boundary conditions and error cases

## Continuous Integration

Tests are automatically run in CI/CD pipeline on:
- Pull requests
- Commits to main branch
- Scheduled builds

## Troubleshooting

### Import Errors
If you encounter import errors, ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Database Tests
If database tests fail, ensure:
- Database connection string is configured
- Test database is accessible
- Migrations are up to date

### Async Tests
For async test failures:
- Ensure `pytest-asyncio` is installed
- Use `@pytest.mark.asyncio` decorator
- Check `asyncio_mode = auto` in pytest.ini
