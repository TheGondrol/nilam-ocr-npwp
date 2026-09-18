from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.common import SuccessEnvelope


class FieldValue(BaseModel):
    value: str | None = Field(None, examples=["12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, examples=[0.96])


class ScoreRequest(BaseModel):
    document_type: str = Field("npwp", examples=["npwp"])
    fields: dict[str, FieldValue] = Field(
        ...,
        description="Field terstruktur (mis. `data.fields` dari /v1/structuring/structure)",
        examples=[{"nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96}}],
    )


class FieldScore(BaseModel):
    name: str = Field(..., examples=["nomor_npwp"])
    score: float = Field(..., ge=0, le=1, examples=[0.96])
    issues: list[str] = Field(default_factory=list, examples=[["missing"]])


class ScoreReport(BaseModel):
    score: float = Field(..., ge=0, le=1, examples=[0.91])
    decision: Literal["approve", "review", "reject"] = Field(..., examples=["approve"])
    field_scores: list[FieldScore]
    reasons: list[str] = Field(default_factory=list)


class ScoreResponse(SuccessEnvelope):
    data: ScoreReport
