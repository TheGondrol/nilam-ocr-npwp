from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ocr_common.schemas import REQUEST_ID_EXAMPLE, Stage

Verdict = Literal["accepted", "reject"]


class _Forwarded(BaseModel):
    model_config = ConfigDict(extra="allow")


class GuardrailsPage(_Forwarded):
    page_index: int | None = Field(None, ge=0, description="0-based page number", examples=[0])
    proba_approve: float | None = Field(
        None, ge=0, le=1, description="Probability that this page is an acceptable document", examples=[0.9821]
    )
    proba_reject: float | None = Field(None, ge=0, le=1, description="1 - `proba_approve`", examples=[0.0179])
    verdict: Verdict | None = Field(None, description="Verdict for this page alone", examples=["accepted"])


class GuardrailsDocument(_Forwarded):
    verdict: Verdict | None = Field(None, description="Document verdict", examples=["accepted"])
    confidence: float | None = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Confidence in the verdict: accepted -> lowest `proba_approve` over the pages. "
            "Sent to the scoring model as `guardrail_probability`"
        ),
        examples=[0.9821],
    )
    n_pages: int | None = Field(None, ge=0, description="Pages that were checked", examples=[1])
    n_approve: int | None = Field(None, ge=0, description="Pages with verdict `accepted`", examples=[1])
    n_reject: int | None = Field(None, ge=0, description="Pages with verdict `rejected`", examples=[0])


class GuardrailsResult(_Forwarded):
    model_config = ConfigDict(
        json_schema_extra={"description": "`data` of POST /v1/guardrails/check, forwarded unchanged down the pipeline."}
    )

    passed: bool | None = Field(None, description="true: the document may proceed to OCR", examples=[True])
    reason: str | None = Field(None, description="Why it was rejected; null when passed", examples=[None])
    document: GuardrailsDocument | None = Field(None, description="Verdict for the document as a whole")
    pages: list[GuardrailsPage] = Field(default_factory=list, description="One entry per page, in page order")


class BoundingBoxPayload(_Forwarded):
    x1: float = Field(..., description="Left edge, pixels", examples=[271])
    y1: float = Field(..., description="Top edge, pixels", examples=[358])
    x2: float = Field(..., description="Right edge, pixels", examples=[1347])
    y2: float = Field(..., description="Bottom edge, pixels", examples=[511])


class OcrBlockPayload(_Forwarded):
    text: str = Field(..., description="One OCR text line", examples=["12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, description="Recognition score of this line", examples=[0.9999])
    bbox: BoundingBoxPayload | None = Field(
        None,
        description=(
            "Upright box around the line, in pixels of the ORIENTATION-CORRECTED image the OCR model worked on "
            "(not of the uploaded photo). Structuring finds the name by its distance to the NPWP number"
        ),
    )
    page: int = Field(0, ge=0, description="0-based page number", examples=[0])


class OcrPayload(_Forwarded):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Result of the OCR stage (`ocr_results`), forwarded to structuring and scoring."
        }
    )

    engine: str | None = Field(None, description="OCR backend that produced the blocks", examples=["paddle"])
    model: str | None = Field(
        None,
        description="Model identity reported by the OCR model service",
        examples=["PP-OCRv6_medium_det+PP-OCRv6_medium_rec"],
    )
    elapsed_ms: float | None = Field(None, description="Time spent in the OCR model", examples=[412.5])
    full_text: str | None = Field(None, description="All blocks joined by newline")
    blocks: list[OcrBlockPayload] = Field(..., description="Text lines in reading order (top to bottom)")


class StructuredFieldPayload(_Forwarded):
    value: str | None = Field(None, description="null when the field was not found", examples=["3201234567890001"])
    confidence: float = Field(
        1.0, ge=0, le=1, description="OCR score of the line the value came from", examples=[0.9762]
    )
    source: str | None = Field(
        None, description="Raw OCR line the value came from", examples=["NPWP16:3201 2345 6789 0001"]
    )
    signals: dict[str, Any] | None = Field(
        None,
        description=(
            "Inputs for the scoring model. nomor_npwp: `has_homoglyph` (an OCR letter/digit mix-up such as O->0 "
            "was corrected), `candidate_count` (NPWP-shaped numbers found in the document). "
            "nama / nama_badan: `corrected` (the name was changed by the name-master correction)"
        ),
        examples=[{"has_homoglyph": False, "candidate_count": 2}],
    )


class StructuringPayload(_Forwarded):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Result of the structuring stage (`structuring_results`), forwarded to scoring."
        }
    )

    document_type: str | None = Field(None, description="Document type the fields were read as", examples=["npwp"])
    fields: dict[str, StructuredFieldPayload] = Field(
        ...,
        description="Always `nomor_npwp`, `nama`, `nama_badan`. A person's card fills `nama`, a company's `nama_badan`",
    )


class FinalField(BaseModel):
    value: str | None = Field(None, description="null when the field was not found", examples=["3201234567890001"])
    confidence: float = Field(
        ..., ge=0, le=1, description="OCR score of the line the value came from", examples=[0.9762]
    )


class FieldConfidences(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Output of the ML team's trust model: probability that each extracted field is correct."
        }
    )

    npwp_confidence: float | None = Field(
        ..., ge=0, le=1, description="P(the NPWP number is correct); null when no number was found", examples=[0.7296]
    )
    name_confidence: float | None = Field(
        ..., ge=0, le=1, description="P(the name is correct); null when no name was found", examples=[0.9471]
    )


class FinalResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "What the pipeline produced for one request_id. Carried by the SCORING / DONE callback."
        }
    )

    document_type: str = Field(..., description="Document type the fields were read as", examples=["npwp"])
    fields: dict[str, FinalField] = Field(
        ...,
        description=(
            "`nomor_npwp`, `nama`, `nama_badan`. A 15-digit number is formatted `XX.XXX.XXX.X-XXX.XXX`; a 16-digit "
            "(NIK-based) number is 16 plain digits. A card printing both reports the 16-digit one"
        ),
        examples=[
            {
                "nomor_npwp": {"value": "3201234567890001", "confidence": 0.9762},
                "nama": {"value": "BUDI SANTOSO", "confidence": 0.9931},
                "nama_badan": {"value": None, "confidence": 0.0},
            }
        ],
    )
    scoring: FieldConfidences = Field(
        ...,
        description=(
            "Per-field trust. There is NO document-level score and NO approve / reject decision: "
            "thresholds belong to the caller"
        ),
    )
    guardrails: GuardrailsResult | None = Field(
        None, description="The guardrails result that was submitted with the job, returned unchanged"
    )


class StageCallback(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"description": "Body of the callback a stage POSTs to the orchestrator when it finishes."}
    )

    request_id: str = Field(
        ..., description="The request_id you submitted with `POST /v1/ekstraksi/jobs`", examples=[REQUEST_ID_EXAMPLE]
    )
    stage: Stage = Field(
        ...,
        description=(
            "Stage the status is about. Usually the sender's own stage; when the hand-off to the NEXT stage "
            "fails after retries, the sender reports `status: FAILED` with the next stage's name, because that "
            "stage never received the job and cannot report for itself"
        ),
        examples=["OCR"],
    )
    status: Literal["DONE", "FAILED"] = Field(
        ..., description="Outcome of the stage. `FAILED` of any stage ends the request", examples=["DONE"]
    )
    result: None = Field(
        None, description="Always null for OCR and STRUCTURING; read the stage result from GET .../jobs/{request_id}"
    )
    error_message: str | None = Field(None, description="Why it failed; null when `status` is `DONE`", examples=[None])


class ScoringStageCallback(StageCallback):
    stage: Literal["SCORING"] = Field("SCORING", description="Always `SCORING`: the last stage", examples=["SCORING"])
    result: FinalResult | None = Field(  # type: ignore[assignment]
        None, description="The final result of the request when `status` is `DONE`; null when `FAILED`"
    )
