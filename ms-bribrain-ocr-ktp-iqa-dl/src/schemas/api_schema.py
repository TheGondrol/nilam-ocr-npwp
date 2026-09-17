"""API request and response schemas using Pydantic."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Crop(BaseModel):
    """Crop information from OCR result."""

    bbox: list[list[int]] = Field(..., description="Bounding box with 4 points [[x,y], ...]")
    text: Optional[str] = Field(None, description="OCR text content")
    confidence: Optional[float] = Field(None, description="OCR confidence score")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[list[int]]) -> list[list[int]]:
        """Validate bbox has exactly 4 points with [x, y] coordinates."""
        if len(value) != 4:
            raise ValueError("bbox must have exactly 4 points")
        for point in value:
            if len(point) != 2:
                raise ValueError("each bbox point must be [x, y]")
        return value


class CropPrediction(BaseModel):
    """Prediction result for a single crop."""

    bbox: list[list[int]] = Field(..., description="Bounding box of the crop")
    prediction: int = Field(..., ge=0, le=1, description="Predicted class (0=bad, 1=good)")
    label: Literal["good", "bad"] = Field(..., description="Human-readable label")
    score: Optional[float] = Field(None, description="Prediction confidence score")
    text: Optional[str] = Field(None, description="OCR text (debug mode only)")
    confidence: Optional[float] = Field(None, description="OCR confidence (debug mode only)")


class ClassificationResponse(BaseModel):
    """Response from quality classification endpoint."""

    label: Literal["good", "bad"] = Field(..., description="Overall image quality label")
    num_bad: int = Field(..., description="Number of crops predicted as bad")
    num_filtered: int = Field(..., description="Number of crops after filtering")
    total_crops: int = Field(..., description="Total number of input crops")
    num_failed_crops: int = Field(..., description="Number of crops that failed extraction")
    bad_crop_threshold: int = Field(..., description="Threshold used for bad classification")
    predictions: list[CropPrediction] = Field(..., description="Individual crop predictions")


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(..., description="Process status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str = Field(..., description="Readiness status")
    checks: dict = Field(..., description="Component checks")


class HealthResponse(BaseModel):
    """Full health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used (cuda/cpu)")
    version: str = Field(..., description="API version")
    checks: Optional[dict] = Field(None, description="Component checks")
