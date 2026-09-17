"""API response schemas using Pydantic."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


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
    version: str = Field(..., description="API version")
    checks: Optional[dict] = Field(None, description="Component checks")
