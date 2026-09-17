# Integration Tests - README

## Overview
Integration tests have been created to test how different components of the OCR KTP Postprocessor work together. These tests verify end-to-end workflows, API integration, and component interactions.

## Test Structure

### 1. API Integration Tests (`test_api_integration.py`)
Tests the complete API flow from HTTP request to response:
- Complete KTP processing with all fields
- Partial KTP data with remapping
- Noisy OCR data with character correction
- Low confidence data handling
- Multiple format variations
- Concurrent request handling
- Performance testing

**Note**: These tests require the test client fixture to be properly configured for the HTTPX/FastAPI version being used.

### 2. OCR Pipeline Tests (`test_ocr_pipeline.py`)
Tests the complete processing pipeline:
- End-to-end data processing
- Character correction and data cleaning
- Field remapping integration
- Confidence threshold enforcement
- Empty and malformed data handling
- Performance with large datasets

**Note**: The `mappingnext()` function returns a tuple `(dict, image_box)`, not just a dict.

### 3. Field Extraction Tests (`test_field_extraction.py`)
Tests integration between field matchers:
- NIK and Nama extraction
- TTL (birth date/place) extraction
- Address components (alamat, RT/RW, kel/desa, kecamatan)
- Gender and religion fields
- Handling noise and special characters
- Consistency and independence of matchers

**Note**: Field matcher functions take a single `data` list argument, not separate key, data, and next_data arguments. Example:
```python
# Correct usage:
result = matching_nik(data)  # data = [text, confidence, box]

# Not: matching_nik("nik", data, next_data)
```

## Known Issues & Required Fixes

### 1. AsyncClient Fixture
The `integration_test_client` fixture needs to be updated for the version of HTTPX being used:
```python
# Current (incorrect):
async with AsyncClient(app=app, base_url="http://test") as client:

# Should be (check HTTPX version):
from httpx import ASGITransport
async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
```

### 2. Function Signatures
Most field matchers take a single `data` list as argument:
```python
def matching_nik(data: list) -> str
def matching_nama(data: list) -> str
def matching_alamat(data: list) -> str
```

### 3. Return Types
- `mappingnext()` returns `tuple[dict, Any]` not just `dict`
- Unpack as: `result, nik_box = mappingnext(data)`

### 4. Data Format
OCR data format for `mappingnext`:
```python
[
    [[[x1, y1], [x2, y2]], ["text", confidence]],
    ...
]
```

Field matcher data format:
```python
["text", confidence, [[x1, y1], [x2, y2]]]
```

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_ocr_pipeline.py -v

# Run with coverage
pytest tests/integration/ --cov=src --cov-report=term-missing
```

## Test Coverage Goals

Integration tests complement unit tests by:
- Testing component interactions
- Verifying end-to-end workflows  
- Testing with realistic data scenarios
- Validating API contracts
- Performance and concurrency testing

## Next Steps

1. Fix AsyncClient fixture for proper HTTPX version compatibility
2. Update field extraction tests to use correct function signatures
3. Update pipeline tests to handle tuple return from `mappingnext()`
4. Add more realistic OCR data scenarios
5. Add database integration tests (currently mocked)

## Test Data Fixtures

The `tests/integration/conftest.py` provides realistic test data:
- `realistic_ktp_ocr_data`: Complete KTP with all fields
- `partial_ktp_ocr_data`: Minimal KTP data
- `noisy_ktp_ocr_data`: Data with OCR errors
- `low_confidence_ktp_data`: Low quality OCR data
- `multiple_format_variations`: Different KTP format styles

These fixtures can be reused across different integration test files.
