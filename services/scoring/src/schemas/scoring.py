from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ocr_common.schemas import REQUEST_ID_EXAMPLE, SuccessEnvelope


class FieldValue(BaseModel):
    value: str | None = Field(None, examples=["12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, examples=[0.96])


class ScoreRequest(BaseModel):
    document_type: str = Field("npwp", examples=["npwp"])
    fields: dict[str, FieldValue] = Field(
        ...,
        description="Field terstruktur (`data.fields` dari service structuring)",
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


class StructuredField(FieldValue):
    # extra="allow": `source` dan field lain dari ServiceStructuring diterima apa adanya.
    model_config = ConfigDict(extra="allow")


class StructuringPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    fields: dict[str, StructuredField] = Field(..., description="Field terstruktur (`structuring.results`)")


class ScoringJobRequest(BaseModel):
    request_id: str = Field(..., min_length=1, examples=[REQUEST_ID_EXAMPLE])
    document_type: str = Field("npwp", examples=["npwp"])
    guardrails: dict[str, Any] | None = Field(
        None, description="Hasil guardrails dari orkestrator; ikut masuk ke hasil akhir"
    )
    ocr: dict[str, Any] | None = Field(
        None, description="Hasil tahap OCR. Belum dipakai scorer heuristic; tersedia untuk model scoring nanti"
    )
    structuring: StructuringPayload = Field(..., description="Hasil tahap structuring")
