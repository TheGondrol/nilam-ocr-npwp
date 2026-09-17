# Unit Tests

This directory contains comprehensive unit tests for the OCR Quality Service.

## Test Structure

- `conftest.py` - Shared pytest fixtures and test utilities
- `test_blur_detection.py` - Tests for blur detection service
- `test_glare_detection.py` - Tests for glare detection service
- `test_rotation_detection.py` - Tests for rotation detection service
- `test_image_quality.py` - Tests for main image quality orchestration
- `test_api_routes.py` - Tests for FastAPI endpoints
- `test_database_service.py` - Tests for database operations

## Running Tests

### Install test dependencies

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install test dependencies
pip install -e ".[test]"
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_blur_detection.py
```

### Run specific test class or function

```bash
pytest tests/test_blur_detection.py::TestBlurDetection
pytest tests/test_blur_detection.py::TestBlurDetection::test_sharp_image
```

### Run tests with coverage

```bash
pytest --cov=src --cov-report=html
```

Coverage report will be generated in `htmlcov/` directory.

### Run tests in parallel

```bash
pip install pytest-xdist
pytest -n auto
```

### Run tests with markers

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

The test suite covers:

- **Blur Detection**: Tests for Laplacian variance calculation and confidence checking
- **Glare Detection**: Tests for brightness analysis, threshold determination, and text region overlap
- **Rotation Detection**: Tests for face detection, orientation checking, and position verification
- **Image Quality**: Tests for the main orchestration function with various image formats
- **API Routes**: Tests for FastAPI endpoints with various input scenarios
- **Database Service**: Tests for async database operations with mocking

## Writing New Tests

When adding new functionality, follow these guidelines:

1. Create test file matching the module name: `test_<module_name>.py`
2. Organize tests into classes: `class Test<FunctionName>`
3. Use descriptive test names: `test_<scenario>_<expected_result>`
4. Use fixtures from `conftest.py` for common test data
5. Mock external dependencies (database, file I/O, external APIs)
6. Test both success and failure cases
7. Test edge cases and boundary conditions

## Example Test

```python
def test_blur_detection_with_sharp_image(sample_grayscale_image):
    """Test blur detection with a sharp image."""
    is_blurry, variance = blur_detection(sample_grayscale_image)
    
    assert is_blurry is False, "Sharp image should not be detected as blurry"
    assert variance > 100, "Sharp image should have high variance"
```

## Continuous Integration

These tests can be integrated with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```
