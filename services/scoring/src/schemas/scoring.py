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


class ConfidenceRequest(BaseModel):
    """Payload dari ML engineer. Semua nilai adalah hasil berantai tahap sebelumnya; null = tidak tersedia."""

    npwp: str | None = Field(None, examples=["123456789012000"])
    npwp_score: float | None = Field(None, ge=0, le=1, examples=[0.98])
    npwp_has_homoglyph: bool | None = Field(None, examples=[False])
    npwp_candidate_count: int | None = Field(None, ge=0, examples=[1])
    name: str | None = Field(None, examples=["PT CONTOH INDONESIA"])
    name_score: float | None = Field(None, ge=0, le=1, examples=[0.96])
    name_corrected: bool | None = Field(None, examples=[True])
    n_boxes: int | None = Field(None, ge=0, examples=[34])
    num_pages: int | None = Field(None, ge=0, examples=[1])
    avg_doc_score: float | None = Field(None, ge=0, le=1, examples=[0.912])
    min_doc_score: float | None = Field(None, ge=0, le=1, examples=[0.62])
    flag: bool | None = Field(None, examples=[False])
    guardrail_probability: float | None = Field(None, ge=0, le=1, examples=[0.9821])


class ConfidenceResult(BaseModel):
    npwp_confidence: float | None = Field(
        ..., ge=0, le=1, examples=[0.93], description="P(nomor NPWP benar); null kalau `npwp` kosong"
    )
    name_confidence: float | None = Field(
        ..., ge=0, le=1, examples=[0.88], description="P(nama benar); null kalau `name` kosong"
    )


class ConfidenceResponse(SuccessEnvelope):
    data: ConfidenceResult


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
        None,
        description="Hasil tahap OCR: n_boxes, num_pages, avg/min_doc_score di payload model dihitung dari `blocks`",
    )
    structuring: StructuringPayload = Field(..., description="Hasil tahap structuring")
