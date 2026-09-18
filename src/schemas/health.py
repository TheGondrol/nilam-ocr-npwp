from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    version: str = Field(..., examples=["1.0.0"])
    device: str = Field(..., examples=["cpu"])
    backends: dict[str, str] = Field(..., description="Backend model aktif per app", examples=[{"guardrails": "mock"}])
