"""
API request and response models.

Defines Pydantic models for API validation and documentation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Generic, TypeVar, Literal, List
from pydantic import BaseModel, Field


# ============================================================================
# Error Codes
# ============================================================================


class ErrorCode:
    """Machine-readable error codes for API responses."""

    # Client input errors
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    INVALID_REQUEST_ID = "INVALID_REQUEST_ID"
    BODY_NOT_ALLOWED = "BODY_NOT_ALLOWED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    REQUEST_NOT_FOUND = "REQUEST_NOT_FOUND"
    REQUEST_ALREADY_PROCESSED = "REQUEST_ALREADY_PROCESSED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Quality rejections
    QUALITY_BLUR = "QUALITY_BLUR"
    QUALITY_GLARE = "QUALITY_GLARE"
    QUALITY_ROTATION = "QUALITY_ROTATION"
    QUALITY_MULTIPLE = "QUALITY_MULTIPLE"
    QUALITY_DL_POOR = "QUALITY_DL_POOR"

    # Spoof rejections
    SPOOF_UNLAMINATED = "SPOOF_UNLAMINATED"
    SPOOF_RECAPTURE = "SPOOF_RECAPTURE"
    SPOOF_GRAYCOPY = "SPOOF_GRAYCOPY"
    NOT_KTP = "NOT_KTP"
    IMAGE_REJECTED = "IMAGE_REJECTED"

    # Tamper rejection
    TAMPERED = "TAMPERED"

    # Server errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PIPELINE_TIMEOUT = "PIPELINE_TIMEOUT"
    SERVICE_ERROR = "SERVICE_ERROR"


# ============================================================================
# Pipeline Error
# ============================================================================


@dataclass
class PipelineError:
    """Structured error returned from the OCR pipeline."""

    error_code: str
    message: str
    http_status: int = 400
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# API Response Models (External - Keep Unchanged)
# ============================================================================


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message describing what went wrong")


class OCRSuccessResponse(BaseModel):
    """Successful OCR response model."""

    results: Dict[str, Any] = Field(
        ..., description="OCR extraction results containing KTP fields"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status (healthy / degraded / unhealthy)")
    version: str = Field(..., description="Application version")
    device: Optional[str] = Field(None, description="Compute device being used")
    checks: Optional[Dict[str, Any]] = Field(None, description="Individual component checks")


# ============================================================================
# Internal Service Models (Type-Safe Service Returns)
# ============================================================================

T = TypeVar("T")


class ServiceError(BaseModel):
    """Represents a service error (timeout, HTTP error, connection failure)."""

    error_type: str = Field(
        ..., description="Type of error (timeout, http_error, connection_error, etc.)"
    )
    message: str = Field(..., description="Error message")
    status_code: Optional[int] = Field(
        None, description="HTTP status code if applicable"
    )
    details: Optional[str] = Field(None, description="Additional error details")


class ServiceRejection(BaseModel):
    """Represents a business logic rejection (spoof detected, low quality, etc.)."""

    rejection_type: str = Field(
        ..., description="Type of rejection (spoof, quality, temper, etc.)"
    )
    message: str = Field(..., description="Human-readable rejection message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional rejection metadata"
    )


class ServiceResult(BaseModel, Generic[T]):
    """Generic service result that can be success, rejection, or error."""

    status: Literal["success", "rejection", "error"] = Field(
        ..., description="Result status"
    )
    data: Optional[T] = Field(None, description="Success data")
    rejection: Optional[ServiceRejection] = Field(
        None, description="Rejection details if status=rejection"
    )
    error: Optional[ServiceError] = Field(
        None, description="Error details if status=error"
    )

    @classmethod
    def success(cls, data: T) -> "ServiceResult[T]":
        """Create a success result."""
        return cls(status="success", data=data, rejection=None, error=None)

    @classmethod
    def reject(
        cls, rejection_type: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> "ServiceResult[T]":
        """Create a rejection result."""
        return cls(
            status="rejection",
            data=None,
            rejection=ServiceRejection(
                rejection_type=rejection_type, message=message, details=details
            ),
            error=None,
        )

    @classmethod
    def fail(
        cls,
        error_type: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[str] = None,
    ) -> "ServiceResult[T]":
        """Create an error result."""
        return cls(
            status="error",
            data=None,
            rejection=None,
            error=ServiceError(
                error_type=error_type,
                message=message,
                status_code=status_code,
                details=details,
            ),
        )


# ============================================================================
# Service-Specific Data Models
# ============================================================================


class ClassifierData(BaseModel):
    """Data returned by classifier service."""

    raw_response: Dict[str, Any] = Field(
        ..., description="Full JSON response from classifier API"
    )


class OCRData(BaseModel):
    """Data returned by OCR service."""

    text: Optional[List] = Field(
        None, description="Raw OCR output with bounding boxes and confidence scores"
    )


class PostprocessData(BaseModel):
    """Data returned by postprocess service."""

    results: Dict[str, Any] = Field(..., description="Postprocessed OCR results")


class QualityData(BaseModel):
    """Data returned by quality service (when quality is good)."""

    passed: bool = Field(True, description="Quality check passed")


class SpoofData(BaseModel):
    """Data returned by spoof service (when no spoof detected)."""

    passed: bool = Field(True, description="Spoof check passed")


class TemperData(BaseModel):
    """Data returned by temper service (when no tampering detected)."""

    passed: bool = Field(True, description="Temper check passed")
