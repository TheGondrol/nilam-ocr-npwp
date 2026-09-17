"""
API Schema Models for OCR Graycopy Detection Service
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    filename: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probability_graycopy: float = Field(ge=0.0, le=1.0)
    probability_original: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    timestamp: str


class LivenessResponse(BaseModel):
    """Response model for liveness probe"""
    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Response model for readiness probe"""
    model_config = {"protected_namespaces": ()}

    status: str
    checks: dict


class HealthResponse(BaseModel):
    """Response model for full health check"""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    device: str
    timestamp: str
    checks: Optional[dict] = None
