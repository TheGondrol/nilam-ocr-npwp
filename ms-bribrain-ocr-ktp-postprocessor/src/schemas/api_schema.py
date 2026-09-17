"""Pydantic models for API requests and responses."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class OCRExtract(BaseModel):
    """Request model for OCR text extraction."""

    ocr_text: str = Field(
        ...,
        description="OCR text data as a string representation of a list",
        json_schema_extra={"example": '[[[0, 0], [100, 0], [100, 50], [0, 50]], ("PROVINSI", 0.95)]'}
    )


class OCRResult(BaseModel):
    """Individual OCR field result."""

    nik: Optional[str] = None
    nama: Optional[str] = None
    tempat_lahir: Optional[str] = None
    tanggal_lahir: Optional[str] = None
    jenis_kelamin: Optional[str] = None
    alamat: Optional[str] = None
    rt: Optional[str] = None
    rw: Optional[str] = None
    kel_desa: Optional[str] = None
    kecamatan: Optional[str] = None
    agama: Optional[str] = None
    status_perkawinan: Optional[str] = None
    pekerjaan: Optional[str] = None


class OCRResponse(BaseModel):
    """Response model for successful OCR processing."""

    ocr_result: dict = Field(
        ...,
        description="Extracted KTP fields"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(
        ...,
        description="Error message"
    )


class LogEntry(BaseModel):
    """Model for database log entries."""

    request_id: str = Field(..., description="Unique request identifier")
    response_code: int = Field(..., description="HTTP response code")
    payload: Optional[Any] = Field(None, description="Request payload data")
    error_message: str = Field("", description="Error message if any")
    result: Optional[Any] = Field(None, description="Processing result")
    processing_time: float = Field(..., description="Processing time in seconds")


# ---------------------------------------------------------------------------
# Health check response models
# ---------------------------------------------------------------------------


class LivenessResponse(BaseModel):
    """Response model for liveness probe."""

    status: str = Field(..., description="Liveness status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Response model for readiness probe."""

    status: str = Field(..., description="Readiness status")
    checks: Optional[dict] = Field(None, description="Component health checks")


class HealthResponse(BaseModel):
    """Response model for full health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    checks: Optional[dict] = Field(None, description="Component health checks")

