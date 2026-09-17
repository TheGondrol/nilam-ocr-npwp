# Unit Tests for ms-bribrain-ocr-ktp-recapture-classifier

This directory contains comprehensive unit tests for the Screen Recapture Detection API.

## Test Structure

```
tests/
├── conftest.py                          # Shared pytest fixtures
└── unit/
    ├── __init__.py
    ├── test_config.py                   # Tests for configuration management
    ├── test_schemas.py                  # Tests for Pydantic API schemas
    ├── test_recapture_service.py        # Tests for recapture detection service
    ├── test_routes.py                   # Tests for API endpoints
    ├── test_middleware_and_utils.py     # Tests for middleware and utilities
    └── test_database_services.py        # Tests for database operations
```

## Running Tests

### Install Test Dependencies

```bash
uv sync --extra test
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/unit/test_config.py
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html --cov-report=term
```

### Run with Verbose Output

```bash
pytest -v
```

## Test Coverage

The test suite covers:

- **Configuration Management** (`test_config.py`)
  - Singleton pattern
  - Nested value access with dot notation
  - Default values and fallbacks
  - YAML loading and error handling

- **API Schemas** (`test_schemas.py`)
  - Pydantic model validation
  - Field constraints (e.g., confidence 0-1)
  - Required fields
  - Response models for all endpoints

- **Recapture Service** (`test_recapture_service.py`)
  - Model initialization
  - Image processing (RGB conversion, resizing)
  - Prediction logic for both classes
  - Aspect ratio preservation
  - Error handling

- **API Routes** (`test_routes.py`)
  - Root endpoint
  - Health check (healthy/unhealthy states)
  - Predict endpoint (success and error cases)
  - File type validation
  - Different image formats (JPEG, PNG)

- **Middleware & Utilities** (`test_middleware_and_utils.py`)
  - Request ID middleware
  - Device selection (CPU/CUDA)
  - Model loading
  - CPU threading configuration
  - Model warmup

- **Database Services** (`test_database_services.py`)
  - Log insertion
  - Error handling
  - Engine initialization and disposal
  - MinIO model download

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_config_dict`: Sample configuration dictionary
- `sample_image_bytes`: Sample image in bytes
- `small_image_bytes`: Small test image
- `mock_model`: Mock PyTorch model
- `mock_device`: Mock PyTorch device
- `mock_transform`: Mock transform function
- `mock_database_session`: Mock async database session
- `mock_request_id`: Mock request ID

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on other tests
2. **Mocking**: External dependencies (database, MinIO, model) are mocked
3. **Coverage**: Tests cover both success and error paths
4. **Clarity**: Test names clearly describe what is being tested
5. **Fixtures**: Shared setup is extracted to reusable fixtures

## Adding New Tests

When adding new tests:

1. Place them in the appropriate test file based on the module being tested
2. Use descriptive test names starting with `test_`
3. Use fixtures from `conftest.py` when applicable
4. Mock external dependencies
5. Test both success and error cases
6. Add docstrings explaining what the test validates

## Continuous Integration

These tests should be run in CI/CD pipeline before deployment to ensure code quality and prevent regressions.
