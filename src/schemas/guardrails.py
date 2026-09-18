from pydantic import BaseModel, Field

from src.schemas.common import SuccessEnvelope


class CheckResult(BaseModel):
    name: str = Field(..., examples=["image_quality"])
    passed: bool = Field(..., examples=[True])
    score: float | None = Field(None, ge=0, le=1, examples=[0.92])
    message: str = Field(..., examples=["Image quality is acceptable"])


class GuardrailReport(BaseModel):
    passed: bool = Field(..., description="True hanya jika semua check lolos", examples=[True])
    document_type: str | None = Field(None, examples=["npwp"])
    confidence: float | None = Field(None, ge=0, le=1, examples=[0.97])
    checks: list[CheckResult]


class GuardrailCheckResponse(SuccessEnvelope):
    data: GuardrailReport
