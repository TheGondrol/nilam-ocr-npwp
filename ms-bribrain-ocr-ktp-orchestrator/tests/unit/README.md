# Unit Tests for OCR KTP Orchestrator

This directory contains comprehensive unit tests for the OCR KTP Orchestrator project.

## Test Structure

```
tests/
├── __init__.py                   # Test package initialization
├── conftest.py                   # Pytest fixtures and configuration
├── test_routes.py                # API routes tests
├── test_ocr_service.py          # OCR service tests
├── test_classifier_service.py   # Classifier service tests
├── test_quality_service.py      # Quality check service tests
├── test_spoof_service.py        # Spoof detection service tests
├── test_manage_service.py       # Orchestration logic tests
├── test_core.py                 # Core modules tests (config, logging, device)
├── test_database_service.py     # Database service tests
└── test_rate_limiter.py         # Rate limiter middleware tests
```

## Installation

Install test dependencies:

```bash
# Using pip
pip install -e ".[test]"

# Or for development (includes additional tools)
pip install -e ".[dev]"
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_routes.py
```

### Run specific test class or function
```bash
pytest tests/test_routes.py::TestOCREndpoint::test_extract_text_success
```

### Run tests with specific markers
```bash
# Run only unit tests
pytest -m unit

# Run only async tests
pytest -m asyncio

# Skip slow tests
pytest -m "not slow"
```

### Run tests in parallel (faster)
```bash
pip install pytest-xdist
pytest -n auto
```

### Run tests with verbose output
```bash
pytest -v
```

## Test Coverage

After running tests with coverage, open the HTML report:
```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

## Test Categories

### 1. API Routes Tests (`test_routes.py`)
- Test the `/v1/ppocr` endpoint
- Validate request handling and response formats
- Test error scenarios (invalid files, service errors)
- Test quality check and spoof detection rejections

### 2. Service Tests
- **OCR Service** (`test_ocr_service.py`): OCR extraction functionality
- **Classifier Service** (`test_classifier_service.py`): KTP classification
- **Quality Service** (`test_quality_service.py`): Image quality checks (blur, glare, rotation)
- **Spoof Service** (`test_spoof_service.py`): Lamination, recapture, and graycopy detection

### 3. Orchestration Tests (`test_manage_service.py`)
- Test the complete OCR processing pipeline
- Test parallel quality checks
- Test service error handling
- Test business logic rejections

### 4. Core Module Tests (`test_core.py`)
- Configuration loading and environment variable substitution
- Logging setup
- Device detection (CPU/GPU)
- API models and response structures

### 5. Database Tests (`test_database_service.py`)
- Database initialization and disposal
- Log insertion functionality
- Error handling

### 6. Middleware Tests (`test_rate_limiter.py`)
- Rate limiting logic
- Client identification
- Request throttling
- Cleanup of old entries

## Writing New Tests

### Basic Test Template
```python
import pytest
from unittest.mock import AsyncMock, patch

class TestYourFeature:
    """Tests for your feature."""

    @pytest.mark.asyncio
    async def test_success_case(self, mock_settings):
        """Test successful execution."""
        # Arrange
        # ... setup test data
        
        # Act
        result = await your_function()
        
        # Assert
        assert result is not None
```

### Using Fixtures
Fixtures are defined in `conftest.py` and can be used in tests:

```python
def test_with_fixtures(
    sample_jpeg_bytes,
    sample_filename,
    request_id,
    mock_settings
):
    # Fixtures are automatically provided
    assert sample_filename == "test_ktp_image.jpg"
```

### Mocking External Services
```python
@pytest.mark.asyncio
async def test_with_mocked_service(mock_aiohttp_session):
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"result": "success"})
    
    mock_aiohttp_session.post.return_value.__aenter__ = AsyncMock(
        return_value=mock_response
    )
    
    # Your test code here
```

## Continuous Integration

Add this to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Best Practices

1. **Keep tests independent**: Each test should be able to run in isolation
2. **Use descriptive names**: Test names should clearly describe what they test
3. **Mock external dependencies**: Don't make real HTTP calls or database connections
4. **Test edge cases**: Include tests for error conditions and boundary cases
5. **Maintain test data**: Use fixtures for common test data
6. **Keep tests fast**: Use mocks to avoid slow operations

## Troubleshooting

### ImportError: No module named 'src'
```bash
# Make sure you're in the project root and install in editable mode
pip install -e .
```

### Async tests not running
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

### Coverage not showing all files
```bash
# Make sure pytest.ini is properly configured
# Run from project root directory
```

## Contributing

When adding new features, please also add corresponding tests:
1. Create test file in `tests/` directory
2. Add fixtures to `conftest.py` if needed
3. Write unit tests covering success and error cases
4. Ensure coverage remains above 80%
5. Run all tests before submitting PR

## Contact

For questions about tests, please contact the development team.
