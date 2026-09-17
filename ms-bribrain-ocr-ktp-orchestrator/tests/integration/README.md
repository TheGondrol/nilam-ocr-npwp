# Integration Tests

This directory contains integration tests for the OCR KTP Orchestrator service.

## Overview

Integration tests verify the interaction between multiple components and services:

- **API Integration**: End-to-end API endpoint testing with service orchestration
- **Database Integration**: Database operations and lifecycle testing
- **Service Orchestration**: Multi-service workflow and error handling

## Running Integration Tests

Run all integration tests:
```bash
pytest tests/integration/ -v -m integration
```

Run specific test file:
```bash
pytest tests/integration/test_api_integration.py -v
```

Run with coverage:
```bash
pytest tests/integration/ --cov=src --cov-report=html -m integration
```

## Test Structure

### test_api_integration.py
Tests for API endpoints including:
- Full OCR workflow from upload to result
- Quality rejection scenarios
- Spoof detection scenarios
- Invalid file type handling
- Concurrent request handling
- Error handling and recovery

### test_database_integration.py
Tests for database operations including:
- Database lifecycle (init, insert, query, dispose)
- Multiple log entries
- Date range queries
- Status-based queries
- Error handling

### test_service_orchestration.py
Tests for service orchestration including:
- Complete OCR process flow with all services
- Parallel quality checks
- Service timeout handling
- Poor quality image handling
- Classifier and spoof detection rejections

## Markers

Integration tests are marked with `@pytest.mark.integration`:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_example():
    ...
```

## Fixtures

Integration tests use fixtures from `conftest.py`:

- `test_app`: FastAPI test application
- `client`: Test client for API calls
- `aiohttp_session`: Async HTTP session
- `mock_settings`: Mock configuration settings
- `in_memory_db`: SQLite in-memory database
- `sample_ktp_image_bytes`: Sample KTP image
- `sample_ocr_response`: Sample OCR service response
- `sample_postprocess_response`: Sample postprocess response

## Best Practices

1. **Use Mocks for External Services**: Mock external API calls to avoid dependencies
2. **Test Service Integration**: Focus on component interaction, not individual functions
3. **Test Error Scenarios**: Include timeout, connection errors, and rejections
4. **Use In-Memory Database**: Use SQLite in-memory for database tests
5. **Test Concurrent Operations**: Verify system handles concurrent requests
6. **Verify Complete Workflows**: Test end-to-end flows from input to output

## Notes

- Integration tests may be slower than unit tests
- External service mocks simulate real service behavior
- Database tests use in-memory SQLite for speed and isolation
- Tests are independent and can run in any order
