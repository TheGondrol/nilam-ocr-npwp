"""
API Response and Request Schemas
Pydantic models for API validation and documentation
"""

from pydantic import BaseModel
from typing import Optional, List


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectionBox(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max]

class KTPDetectionResponse(BaseModel):
    filename: str
    detected: bool
    num_detected: int
    detections: List[Detection]
    status: str
    reason: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for full health check"""
    status: str
    model_loaded: bool
    device: str
    device_info: dict
    timestamp: str
    checks: Optional[dict] = None


class LivenessResponse(BaseModel):
    """Response model for liveness probe"""
    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Response model for readiness probe"""
    status: str
    checks: dict


class RootResponse(BaseModel):
    """Response model for root endpoint"""
    message: str
    description: str
    version: str
    endpoints: dict
