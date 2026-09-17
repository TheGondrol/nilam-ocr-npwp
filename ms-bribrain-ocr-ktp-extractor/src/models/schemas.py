"""Pydantic schemas for API request/response models"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class OCRTextRegion(BaseModel):
    """
    Represents a single detected text region from OCR.
    
    Attributes:
        coordinates: List of [x, y] coordinate pairs defining the bounding polygon
        text: The detected text content
        confidence: Confidence score between 0 and 1
    """
    coordinates: List[List[float]] = Field(
        ..., 
        description="Bounding polygon coordinates as [[x1,y1], [x2,y2], ...]"
    )
    text: str = Field(..., description="Detected text content")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0-1)"
    )


class OCRSuccessResponse(BaseModel):
    """
    Successful OCR extraction response.
    
    Attributes:
        ocr_result: List of detected text regions with coordinates and confidence
        processing_time: Time taken to process the image in seconds
        text_regions_count: Number of text regions detected
    """
    ocr_result: List[Any] = Field(
        ..., 
        description="List of OCR results with coordinates and text"
    )
    processing_time: float = Field(
        ..., 
        ge=0, 
        description="Processing time in seconds"
    )
    text_regions_count: int = Field(
        ..., 
        ge=0, 
        description="Number of detected text regions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "ocr_result": [
                    [[[100.0, 50.0], [200.0, 50.0], [200.0, 80.0], [100.0, 80.0]], ("Sample Text", 0.95)]
                ],
                "processing_time": 1.23,
                "text_regions_count": 1
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Attributes:
        error: Human-readable error message
        request_id: Optional request ID for debugging/tracking
    """
    error: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(
        None, 
        description="Request ID for tracking"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Invalid file type. Only image/jpeg, image/png are supported.",
                "request_id": "abc123-def456"
            }
        }
    }


class HealthResponse(BaseModel):
    """Full health check response model."""
    status: str = Field(..., description="Health status")
    ocr_loaded: bool = Field(..., description="OCR engine readiness")
    version: str = Field(..., description="API version")
    checks: Optional[dict] = Field(None, description="Component health checks")

    model_config = {"protected_namespaces": ()}


class LivenessResponse(BaseModel):
    """Liveness probe response model."""
    status: str = Field(..., description="Liveness status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Readiness probe response model."""
    status: str = Field(..., description="Readiness status")
    checks: Optional[dict] = Field(None, description="Component health checks")


class RootResponse(BaseModel):
    """Root endpoint response model."""
    message: str
    version: str
    endpoints: dict
