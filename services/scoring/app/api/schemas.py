from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ocr_common.pipeline import SCORING, checked_sequence
from ocr_common.pipeline.schemas import FieldConfidences, GuardrailsResult, OcrPayload, StructuringPayload
from ocr_common.web.schemas import PIPELINE_SEQUENCE_DESCRIPTION, REQUEST_ID_EXAMPLE, JobStatusBase, SuccessEnvelope


class ConfidenceRequest(BaseModel):
    """The ML team's scoring payload (nilamnpwp `scoring/schemas.py`, model retrained 21 Sep 2026)."""

    npwp: str | None = Field(
        None,
        description="Extracted NPWP number, digits or printed form; null when not found",
        examples=["12.345.678.9-012.000"],
    )
    npwp_score: float | None = Field(
        None, ge=0, le=1, description="OCR score of the line the number came from", examples=[0.98]
    )
    npwp_candidate_count: int | None = Field(
        None,
        ge=0,
        description="NPWP-shaped numbers found in the document (a card may print a 15- and a 16-digit one)",
        examples=[1],
    )
    name_base: str | None = Field(
        None,
        description=(
            "The name as OCR read it, BEFORE any normalisation or correction (`signals.name_base` of the "
            "structuring result); null when not found. The model measures the name's shape on this"
        ),
        examples=["PT CONTOH INDONESIA"],
    )
    name_score: float | None = Field(
        None, ge=0, le=1, description="OCR score of the line the name came from", examples=[0.95]
    )
    avg_doc_score: float | None = Field(None, ge=0, le=1, description="Mean OCR score over all lines", examples=[0.91])
    min_doc_score: float | None = Field(
        None, ge=0, le=1, description="Lowest OCR score over all lines", examples=[0.62]
    )
    flag: bool | None = Field(
        None,
        description="Review flag of the structuring rules (`flag` of the structuring result); null counts as false",
        examples=[False],
    )
    guardrail_probability: float | None = Field(
        None,
        ge=0,
        le=1,
        description="`document.confidence` of the guardrails result when the document was accepted",
        examples=[0.98],
    )


class ConfidenceResponse(SuccessEnvelope):
    data: FieldConfidences


class ScoringJobRequest(BaseModel):
    request_id: str = Field(
        ..., min_length=1, description="request_id of the pipeline run", examples=[REQUEST_ID_EXAMPLE]
    )
    document_type: str = Field(
        "npwp", description="Only `npwp` is supported; anything else fails the job", examples=["npwp"]
    )
    guardrails: GuardrailsResult | None = Field(
        None, description="Guardrails result submitted with the OCR job; returned unchanged in the final result"
    )
    ocr: OcrPayload | None = Field(
        None, description="Result of the OCR stage; `avg_doc_score` / `min_doc_score` are computed from `blocks`"
    )
    structuring: StructuringPayload | None = Field(
        None,
        description=(
            "Result of the structuring stage. Left out when the structuring service hands off by reference "
            "(`PIPELINE_HANDOFF_BY_REFERENCE`): this service then reads `structuring_results` (and `ocr_results`) "
            "of the shared database"
        ),
    )
    pipeline_name_sequence: list[str] | None = Field(
        None,
        description=PIPELINE_SEQUENCE_DESCRIPTION,
        examples=[["guardrails", "extraction", "structuring", "scoring"]],
    )

    @field_validator("pipeline_name_sequence")
    @classmethod
    def _sequence_includes_this_stage(cls, value: list[str] | None) -> list[str] | None:
        return checked_sequence(value, SCORING)


class ScoringJobResult(FieldConfidences):
    payload: ConfidenceRequest = Field(
        ..., description="Exactly what was scored, built from the chained results: an audit trail for the two numbers"
    )


class ScoringJobStatus(JobStatusBase):
    stage: Literal["SCORING"] = Field("SCORING", description="Always `SCORING` on this service", examples=["SCORING"])
    result: ScoringJobResult | None = Field(None, description="Confidences once `status` is `DONE`; null otherwise")


class ScoringJobStatusResponse(SuccessEnvelope):
    data: ScoringJobStatus


class FieldValue(BaseModel):
    value: str | None = Field(None, description="null when the field was not found", examples=["12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, description="OCR score of the line the value came from", examples=[0.96])


class ScoreRequest(BaseModel):
    document_type: str = Field("npwp", description="Only `npwp` is supported", examples=["npwp"])
    fields: dict[str, FieldValue] = Field(
        ...,
        description="Structured fields (`data.fields` of the structuring service)",
        examples=[{"nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96}}],
    )


class FieldScore(BaseModel):
    name: str = Field(..., description="Field name", examples=["nomor_npwp"])
    score: float = Field(
        ..., ge=0, le=1, description="OCR confidence, or 0 when missing / badly formatted", examples=[0.96]
    )
    issues: list[str] = Field(default_factory=list, description="`missing`, or `invalid format: ...`", examples=[[]])


class ScoreReport(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Weighted mean of the field scores", examples=[0.91])
    decision: Literal["approve", "review", "reject"] = Field(
        ..., description="`score` against the configured thresholds (0.8 / 0.5 by default)", examples=["approve"]
    )
    field_scores: list[FieldScore] = Field(..., description="One entry per scored field")
    reasons: list[str] = Field(
        default_factory=list, description="Why a required field lowered the score", examples=[[]]
    )


class ScoreResponse(SuccessEnvelope):
    data: ScoreReport
