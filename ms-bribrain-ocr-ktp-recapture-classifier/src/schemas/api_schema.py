"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    
    filename: str = Field(..., description="Name of the uploaded file")
    prediction: str = Field(..., description="Prediction result: ORIGINAL or RECAPTURED")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction")
    probability_recaptured: float = Field(..., ge=0.0, le=1.0, description="Probability that image is recaptured")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Classification threshold used")
    timestamp: str = Field(..., description="Timestamp of prediction in ISO format")


class HealthResponse(BaseModel):
    """Response model for full health check endpoint"""
    model_config = {"protected_namespaces": ()}  # Allow field names starting with 'model_'

    status: str = Field(..., description="Service status: healthy, degraded, or unhealthy")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    device: str = Field(..., description="Device being used: cuda or cpu")
    version: str = Field(..., description="API version")
    checks: Optional[dict] = Field(None, description="Component health checks")


class LivenessResponse(BaseModel):
    """Response model for liveness probe"""

    status: str = Field(..., description="Liveness status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Response model for readiness probe"""

    status: str = Field(..., description="Readiness status")
    checks: Optional[dict] = Field(None, description="Component health checks")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response"""
    
    filename: str = Field(..., description="Name of the file")
    result: Optional[PredictionResponse] = Field(None, description="Prediction result if successful")
    error: Optional[str] = Field(None, description="Error message if prediction failed")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    
    results: List[BatchPredictionItem] = Field(..., description="List of prediction results")
    total: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
