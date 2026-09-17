"""Unit tests for src.models.schemas module"""

import pytest
from pydantic import ValidationError
from src.models.schemas import (
    OCRTextRegion,
    OCRSuccessResponse,
    ErrorResponse,
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
    RootResponse
)


class TestOCRTextRegion:
    """Test cases for OCRTextRegion schema"""

    def test_valid_ocr_text_region(self):
        """Test creating valid OCRTextRegion"""
        region = OCRTextRegion(
            coordinates=[[10, 10], [100, 10], [100, 30], [10, 30]],
            text="Sample Text",
            confidence=0.95
        )
        
        assert region.text == "Sample Text"
        assert region.confidence == pytest.approx(0.95)
        assert len(region.coordinates) == 4

    def test_confidence_validation_max(self):
        """Test confidence cannot exceed 1.0"""
        with pytest.raises(ValidationError):
            OCRTextRegion(
                coordinates=[[10, 10], [100, 10]],
                text="Text",
                confidence=1.5  # Invalid: > 1.0
            )

    def test_confidence_validation_min(self):
        """Test confidence cannot be less than 0.0"""
        with pytest.raises(ValidationError):
            OCRTextRegion(
                coordinates=[[10, 10], [100, 10]],
                text="Text",
                confidence=-0.1  # Invalid: < 0.0
            )

    def test_missing_required_fields(self):
        """Test that required fields are enforced"""
        with pytest.raises(ValidationError):
            OCRTextRegion(coordinates=[[10, 10]])  # Missing text and confidence


class TestOCRSuccessResponse:
    """Test cases for OCRSuccessResponse schema"""

    def test_valid_success_response(self):
        """Test creating valid OCRSuccessResponse"""
        response = OCRSuccessResponse(
            ocr_result=[
                ([[10, 10], [100, 10], [100, 30], [10, 30]], ("Text 1", 0.95)),
                ([[10, 40], [100, 40], [100, 60], [10, 60]], ("Text 2", 0.88))
            ],
            processing_time=1.5,
            text_regions_count=2
        )
        
        assert len(response.ocr_result) == 2
        assert response.processing_time == pytest.approx(1.5)
        assert response.text_regions_count == 2

    def test_negative_processing_time(self):
        """Test that negative processing time is invalid"""
        with pytest.raises(ValidationError):
            OCRSuccessResponse(
                ocr_result=[],
                processing_time=-1.0,  # Invalid
                text_regions_count=0
            )

    def test_negative_text_regions_count(self):
        """Test that negative text regions count is invalid"""
        with pytest.raises(ValidationError):
            OCRSuccessResponse(
                ocr_result=[],
                processing_time=1.0,
                text_regions_count=-1  # Invalid
            )

    def test_empty_ocr_result(self):
        """Test success response with empty results"""
        response = OCRSuccessResponse(
            ocr_result=[],
            processing_time=0.5,
            text_regions_count=0
        )
        
        assert len(response.ocr_result) == 0
        assert response.text_regions_count == 0


class TestErrorResponse:
    """Test cases for ErrorResponse schema"""

    def test_valid_error_response(self):
        """Test creating valid ErrorResponse"""
        response = ErrorResponse(
            error="Invalid file type",
            request_id="req-123"
        )
        
        assert response.error == "Invalid file type"
        assert response.request_id == "req-123"

    def test_error_response_without_request_id(self):
        """Test ErrorResponse with optional request_id"""
        response = ErrorResponse(error="Test error")
        
        assert response.error == "Test error"
        # request_id should be optional


class TestHealthResponse:
    """Test cases for HealthResponse schema"""

    def test_valid_health_response_healthy(self):
        """Test creating healthy HealthResponse"""
        response = HealthResponse(
            status="healthy",
            ocr_loaded=True,
            version="1.0.0",
            checks={"ocr": {"status": "up"}, "database": {"status": "up"}},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.ocr_loaded is True

    def test_valid_health_response_degraded(self):
        """Test creating degraded HealthResponse"""
        response = HealthResponse(
            status="degraded",
            ocr_loaded=True,
            version="1.0.0",
        )

        assert response.status == "degraded"
        assert response.ocr_loaded is True

    def test_valid_health_response_unhealthy(self):
        """Test creating unhealthy HealthResponse"""
        response = HealthResponse(
            status="unhealthy",
            ocr_loaded=False,
            version="1.0.0",
        )

        assert response.status == "unhealthy"
        assert response.ocr_loaded is False


class TestLivenessResponse:
    """Test cases for LivenessResponse schema"""

    def test_valid_liveness_response(self):
        response = LivenessResponse(status="alive", version="1.0.0")
        assert response.status == "alive"
        assert response.version == "1.0.0"


class TestReadinessResponse:
    """Test cases for ReadinessResponse schema"""

    def test_valid_readiness_response(self):
        response = ReadinessResponse(
            status="ready",
            checks={"ocr": {"status": "up"}, "database": {"status": "up"}},
        )
        assert response.status == "ready"
        assert response.checks is not None


class TestRootResponse:
    """Test cases for RootResponse schema"""

    def test_valid_root_response(self):
        """Test creating valid RootResponse"""
        response = RootResponse(
            message="OCR Extract API",
            version="1.0.0",
            endpoints={
                "ocr_extract": "/v1/ocr_extract",
                "health": "/health",
                "docs": "/docs"
            }
        )
        
        assert response.message == "OCR Extract API"
        assert response.version == "1.0.0"
        assert len(response.endpoints) == 3
        assert response.endpoints["health"] == "/health"

    def test_root_response_serialization(self):
        """Test that RootResponse can be serialized to JSON"""
        response = RootResponse(
            message="Test API",
            version="1.0.0",
            endpoints={"test": "/test"}
        )
        
        json_data = response.model_dump()
        
        assert json_data["message"] == "Test API"
        assert json_data["version"] == "1.0.0"
        assert "endpoints" in json_data


class TestSchemaValidation:
    """Test cross-schema validation and edge cases"""

    def test_ocr_success_response_json_schema(self):
        """Test OCRSuccessResponse JSON schema generation"""
        schema = OCRSuccessResponse.model_json_schema()
        
        assert "properties" in schema
        assert "ocr_result" in schema["properties"]
        assert "processing_time" in schema["properties"]
        assert "text_regions_count" in schema["properties"]

    def test_error_response_json_schema(self):
        """Test ErrorResponse JSON schema generation"""
        schema = ErrorResponse.model_json_schema()
        
        assert "properties" in schema
        assert "error" in schema["properties"]

    def test_health_response_json_schema(self):
        """Test HealthResponse JSON schema generation"""
        schema = HealthResponse.model_json_schema()

        assert "properties" in schema
        assert "status" in schema["properties"]
        assert "version" in schema["properties"]
        assert "ocr_loaded" in schema["properties"]
