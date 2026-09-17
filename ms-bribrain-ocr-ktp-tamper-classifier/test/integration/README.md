# Integration Tests

Integration tests verify that multiple components work correctly together. Unlike unit tests that test individual components in isolation, integration tests ensure proper interaction between different parts of the system.

## Test Files

### test_api_integration.py
Tests the full API request/response cycle:
- Root and health endpoints
- Prediction endpoint with various scenarios
- Error handling and validation
- Response format validation
- CORS and middleware integration
- Concurrent request handling

### test_service_integration.py
Tests interactions between service layer components:
- Complete image preprocessing pipeline
- Tamper detection service initialization and prediction flow
- Database service initialization and logging
- MinIO service model download flow
- End-to-end prediction flow

### test_middleware_integration.py
Tests middleware components with API:
- Request ID middleware integration
- Middleware stack interaction
- Logging with request ID context

### test_config_integration.py
Tests configuration system integration:
- Configuration loading and validation
- Environment variable substitution
- Configuration usage across services
- Singleton pattern and reload functionality

## Running Integration Tests

### Run all integration tests
```powershell
pytest test/integration -v
```

### Run specific integration test file
```powershell
pytest test/integration/test_api_integration.py -v
```

### Run with coverage
```powershell
pytest test/integration --cov=src --cov-report=html
```

### Run only integration tests (exclude unit tests)
```powershell
pytest -m integration -v
```

### Run both unit and integration tests
```powershell
pytest test/ -v
```

## Test Markers

Integration tests are marked with `@pytest.mark.integration`:

```python
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    pass
```

## Test Fixtures

Integration tests use shared fixtures from `conftest.py`:
- `sample_image`: Sample PIL Image
- `sample_image_bytes`: Sample image bytes
- `mock_tamper_service`: Mocked tamper detection service
- `app_client`: FastAPI TestClient with mocked services

## Key Differences from Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|-------------------|
| Scope | Single component | Multiple components |
| Dependencies | Heavily mocked | Real or minimal mocking |
| Speed | Fast | Slower |
| Isolation | High | Lower |
| Purpose | Verify logic | Verify interactions |

## Best Practices

1. **Test realistic scenarios**: Use realistic data and workflows
2. **Minimize external dependencies**: Mock external services (DB, MinIO, etc.)
3. **Test error paths**: Ensure proper error handling across components
4. **Keep tests independent**: Each test should run independently
5. **Clean up resources**: Ensure proper cleanup after tests
6. **Use descriptive names**: Test names should describe the integration scenario

## Common Integration Test Patterns

### API Integration Test Pattern
```python
def test_api_endpoint(self, app_client, sample_image_bytes):
    """Test API endpoint with realistic request"""
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = app_client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
```

### Service Integration Test Pattern
```python
@pytest.mark.asyncio
async def test_service_flow(self, sample_image_bytes):
    """Test complete service flow"""
    # Step 1: Initialize service
    service = initialize_service()
    
    # Step 2: Process data
    result = await service.process(sample_image_bytes)
    
    # Step 3: Verify result
    assert result is not None
```

### Database Integration Test Pattern
```python
@pytest.mark.asyncio
async def test_database_flow(self):
    """Test database operations"""
    # Initialize
    await init_engine()
    
    # Perform operations
    await insert_log(...)
    
    # Cleanup
    await dispose_engine()
```

## Troubleshooting

### Tests fail with import errors
- Ensure all dependencies are installed: `pip install -e ".[test]"`
- Check that the package is installed in development mode

### Tests fail with async warnings
- Ensure `pytest-asyncio` is installed
- Use `@pytest.mark.asyncio` decorator for async tests

### Tests are too slow
- Reduce the number of integration scenarios
- Use smaller test data
- Consider parallel test execution: `pytest -n auto`

## Continuous Integration

Integration tests should be part of your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run integration tests
  run: |
    pytest test/integration -v --cov=src --cov-report=xml
```

## Coverage Goals

Integration tests should focus on:
- ✅ API endpoint flows (80%+ coverage)
- ✅ Service interactions (70%+ coverage)
- ✅ Error handling paths (60%+ coverage)
- ✅ Configuration loading (80%+ coverage)
- ✅ Middleware integration (70%+ coverage)
