from typing import Any, Literal

from pydantic import BaseModel, Field

from ocr_common.pipeline_schemas import GuardrailsResult, OcrPayload
from ocr_common.schemas import REQUEST_ID_EXAMPLE, JobStatusBase, SuccessEnvelope


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left", examples=[35])
    y1: float = Field(..., description="Top", examples=[389])
    x2: float = Field(..., description="Right", examples=[637])
    y2: float = Field(..., description="Bottom", examples=[437])


class TextLine(BaseModel):
    text: str = Field(..., description="One OCR text line", examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, description="Recognition score of this line", examples=[0.9992])
    bbox: BoundingBox | None = Field(
        None,
        description=(
            "Position of the line (`blocks[].bbox` of the OCR result). The `npwp_rules` backend finds the name by "
            "its distance to the NPWP number, because real cards print the name without a label. Without `bbox` "
            "the position is derived from the order of the lines"
        ),
    )
    page: int = Field(0, ge=0, description="0-based page number", examples=[0])


class StructureRequest(BaseModel):
    lines: list[TextLine] = Field(..., min_length=1, description="OCR text lines in reading order (top to bottom)")


class StructuredField(BaseModel):
    value: str | None = Field(None, description="null when the field was not found", examples=["12.345.678.9-012.345"])
    confidence: float = Field(
        ..., ge=0, le=1, description="OCR score of the line the value came from; 0 when not found", examples=[0.9992]
    )
    source: str | None = Field(
        None, description="Raw OCR line the value came from", examples=["NPWP : 12.345.678.9-012.345"]
    )
    signals: dict[str, Any] | None = Field(
        None,
        description=(
            "Inputs for the scoring model (backend `npwp_rules` only; null otherwise). nomor_npwp: `has_homoglyph` "
            "(an OCR letter/digit mix-up such as O->0 was corrected) and `candidate_count` (NPWP-shaped numbers "
            "found in the document). nama / nama_badan: `corrected` (the name-master correction changed the name)"
        ),
        examples=[{"has_homoglyph": False, "candidate_count": 1}],
    )


class StructuredDocument(BaseModel):
    document_type: str = Field(..., description="Document type the fields were read as", examples=["npwp"])
    fields: dict[str, StructuredField] = Field(
        ...,
        description=(
            "Always `nomor_npwp`, `nama`, `nama_badan`; a field that was not found has `value: null`. A person's "
            "card fills `nama`, a company's (PT, CV, ...) fills `nama_badan`. A 15-digit number is formatted "
            "`XX.XXX.XXX.X-XXX.XXX`; a 16-digit NIK-based number is 16 plain digits, and wins when a card prints both"
        ),
        examples=[
            {
                "nomor_npwp": {
                    "value": "12.345.678.9-012.345",
                    "confidence": 0.9992,
                    "source": "NPWP : 12.345.678.9-012.345",
                    "signals": {"has_homoglyph": False, "candidate_count": 1},
                },
                "nama": {
                    "value": "BUDI SANTOSO",
                    "confidence": 0.9773,
                    "source": "BUDI SANTOSO",
                    "signals": {"corrected": False},
                },
                "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
            }
        ],
    )


class StructureResponse(SuccessEnvelope):
    data: StructuredDocument


class StructuringJobRequest(BaseModel):
    request_id: str = Field(
        ..., min_length=1, description="request_id of the pipeline run", examples=[REQUEST_ID_EXAMPLE]
    )
    document_type: str = Field("npwp", description="Only `npwp` is supported", examples=["npwp"])
    guardrails: GuardrailsResult | None = Field(
        None, description="Guardrails result submitted with the OCR job; only forwarded to the next stage"
    )
    ocr: OcrPayload = Field(..., description="Result of the OCR stage")


class StructuringJobStatus(JobStatusBase):
    stage: Literal["STRUCTURING"] = Field(
        "STRUCTURING", description="Always `STRUCTURING` on this service", examples=["STRUCTURING"]
    )
    result: StructuredDocument | None = Field(
        None, description="Structured fields once `status` is `DONE`; null otherwise"
    )


class StructuringJobStatusResponse(SuccessEnvelope):
    data: StructuringJobStatus
