"""
API Response and Request Schemas
Pydantic models for API validation and documentation
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    filename: str
    prediction: str
    prob: float
    threshold: float
    timestamp: str


class LivenessResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field(..., description="Process status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    status: str = Field(..., description="Readiness status")
    checks: dict = Field(..., description="Component checks")


class HealthResponse(BaseModel):
    """Response model for health check"""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    device: str
    version: str
    checks: Optional[dict] = Field(None, description="Component checks")


class RootResponse(BaseModel):
    """Response model for root endpoint"""
    message: str
    description: str
    version: str
    endpoints: Dict[str, str]


class BatchResultItem(BaseModel):
    """Individual result in batch prediction"""
    filename: str
    result: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    results: list[BatchResultItem]

