from typing import Any, Literal

from pydantic import BaseModel, Field

from ocr_common.schemas import SuccessEnvelope

Verdict = Literal["accepted", "reject"]
JobStatus = Literal["pending", "processing", "completed", "failed"]


class PageResult(BaseModel):
    page_index: int = Field(..., ge=0, description="0-based page number; an image is always page 0", examples=[0])
    proba_approve: float = Field(..., ge=0, le=1, description="Model probability of `accepted`", examples=[0.9821])
    proba_reject: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model probability of `reject`; `proba_approve + proba_reject = 1`",
        examples=[0.0179],
    )
    verdict: Verdict = Field(
        ...,
        description="`reject` when `proba_reject` reaches the reject threshold (0.5 by default)",
        examples=["accepted"],
    )


class DocumentResult(BaseModel):
    verdict: Verdict = Field(
        ...,
        description="Document verdict. Default policy `all`: accepted only when every page is accepted",
        examples=["accepted"],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Confidence in the verdict. accepted -> the lowest `proba_approve` over the pages (the weakest page "
            "decides); reject -> the highest `proba_reject` among the rejected pages"
        ),
        examples=[0.9821],
    )
    n_pages: int = Field(..., ge=0, description="Pages judged (a PDF is capped at 20)", examples=[2])
    n_approve: int = Field(..., ge=0, description="Pages with verdict `accepted`", examples=[2])
    n_reject: int = Field(..., ge=0, description="Pages with verdict `reject`", examples=[0])


class GuardrailReport(BaseModel):
    passed: bool = Field(
        ...,
        description="true: the document may go on to the OCR stage. false: stop, and answer the client with `reason`",
        examples=[True],
    )
    reason: str | None = Field(
        None,
        description=(
            "Why the document was rejected; null when `passed`. The model is a binary classifier, so the reason "
            "can only state how many pages were rejected and how confidently, not *what* is wrong with them"
        ),
        examples=[None],
    )
    document: DocumentResult
    pages: list[PageResult] = Field(..., description="One entry per page, in page order")


class GuardrailReportResponse(SuccessEnvelope):
    message: str = Field("OK", description="Human-readable outcome", examples=["OK"])
    data: GuardrailReport


class ContractField(BaseModel):
    value: str | None = Field(
        None, description="The value read; null when the field was not found", examples=["12.345.678.9-012.345"]
    )
    confidence: Literal[0, 1] = Field(
        ...,
        description=(
            "1 when the ML team's trust model gives this value a probability of being correct of at least "
            "`FIELD_CONFIDENCE_THRESHOLD` (0.5 by default); 0 when it is lower, or when there is no value"
        ),
        examples=[1],
    )


class NpwpData(BaseModel):
    nomor_npwp: ContractField = Field(
        ...,
        description="15-digit number as `XX.XXX.XXX.X-XXX.XXX`, or a 16-digit NIK-based number as 16 plain digits",
    )
    nama: ContractField = Field(
        ..., description="The name on the card: the taxpayer's name, or the registered name on a company's card"
    )


class ExtractOcrResponse(BaseModel):
    status_code: int = Field(..., description="Same as the HTTP status code", examples=[200])
    status_desc: str = Field(..., description="Reason phrase of `status_code`", examples=["OK"])
    message: str = Field(
        ...,
        description="Human-readable; its wording may change, so branch on `errors` instead",
        examples=["OCR extraction completed successfully"],
    )
    data: NpwpData | None = Field(None, description="The OCR fields when `job_status` is `completed`; null otherwise")
    errors: str | None = Field(
        None, description="Failure code when the request failed or was refused; null otherwise", examples=[None]
    )
    request_id: str | None = Field(None, description="The request_id this response belongs to")
    document_type: str | None = Field(None, description="Document type of the request", examples=["npwp"])
    job_status: JobStatus | None = Field(
        None,
        description=(
            "`completed` (200), `processing` (202), or `failed`; null when the request was refused before "
            "anything was processed"
        ),
        examples=["completed"],
    )
    guardrails: Literal[0, 1] | None = Field(
        None,
        description=(
            "1: the document passed the guardrails model; 0: it was rejected. Null on 202, and when the request "
            "was refused before the check"
        ),
        examples=[1],
    )
    params: Any = Field(
        None,
        description="The `params` sent with this request, returned unchanged; null when not sent",
        examples=[{"nik": "3123456711950001", "refno": "PK19039Y8U"}],
    )
