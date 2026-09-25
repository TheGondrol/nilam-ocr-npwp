from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ocr_common.pipeline import STRUCTURING, checked_sequence
from ocr_common.pipeline.schemas import GuardrailsResult, OcrPayload
from ocr_common.web.schemas import PIPELINE_SEQUENCE_DESCRIPTION, REQUEST_ID_EXAMPLE, JobStatusBase, SuccessEnvelope


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
            "Inputs for the scoring model and review signals (backend `npwp_rules` only; null otherwise). "
            "nomor_npwp: `candidate_count` (NPWP-shaped numbers found in the document), `has_homoglyph` (OCR read a "
            "letter where a digit belongs; the letter is dropped, not corrected), `invalid_province_prefix` / "
            "`invalid_kecamatan_prefix` / `invalid_birthdate` (a 16-digit NIK-based number fails the Kode Wilayah "
            "or birthdate check), `invalid_kpp_prefix` (a 15-digit number's KPP office code is unknown). "
            "nama / nama_badan: `name_base` (the read before any normalisation; the trust model measures the name "
            "on this), `corrected` (`value` differs from `name_base`)"
        ),
        examples=[{"candidate_count": 1, "has_homoglyph": False}],
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
                    "signals": {"candidate_count": 1, "has_homoglyph": False},
                },
                "nama": {
                    "value": "BUDI SANTOSO",
                    "confidence": 0.9773,
                    "source": "BUDI SANTOSO",
                    "signals": {"name_base": "BUDI SANTOSO", "corrected": False},
                },
                "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
            }
        ],
    )
    flag: bool = Field(
        ...,
        description=(
            "Flag of the ML team's rules, a feature of the trust model. True when: another document is bundled "
            "in (KTP, KK, Akta), the page is a CAPTCHA or a screenshot of the DJP lookup, the number or the name "
            "was not found, the name is a single word, the number contains a letter, its province / kecamatan / "
            "birthdate / KPP code is invalid, or the upload has more than 2 pages"
        ),
        examples=[False],
    )
    flag_reason: str | None = Field(
        None,
        description="The first reason `flag` is true, in Indonesian; null when not flagged",
        examples=[None],
    )
    reject_reason: str | None = Field(
        None,
        description=(
            "The first reason that rejects the document, in Indonesian; null when the document is accepted. Every "
            "check rejects except a single-word name and a letter in the number, which only raise `flag`. In the "
            "pipeline a rejected document stops here (no scoring) and the client gets a 400 with this message"
        ),
        examples=[None],
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
    ocr: OcrPayload | None = Field(
        None,
        description=(
            "Result of the OCR stage. Left out when the OCR service hands off by reference "
            "(`PIPELINE_HANDOFF_BY_REFERENCE`): this service then reads `ocr_results` of the shared database"
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
        return checked_sequence(value, STRUCTURING)


class StructuringJobStatus(JobStatusBase):
    stage: Literal["STRUCTURING"] = Field(
        "STRUCTURING", description="Always `STRUCTURING` on this service", examples=["STRUCTURING"]
    )
    result: StructuredDocument | None = Field(
        None, description="Structured fields once `status` is `DONE`; null otherwise"
    )


class StructuringJobStatusResponse(SuccessEnvelope):
    data: StructuringJobStatus
