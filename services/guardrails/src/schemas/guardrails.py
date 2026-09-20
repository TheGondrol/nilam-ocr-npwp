"""Skema service guardrails. Teks Field(...) tampil di Swagger yang dibaca tim gateway: bahasa Inggris."""

from typing import Literal

from pydantic import BaseModel, Field

from ocr_common.schemas import SuccessEnvelope

Verdict = Literal["accepted", "reject"]


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
        description=(
            "true: send the document on to the OCR stage. false: stop, and answer the client with `reason` "
            "(the sequence diagram uses 422 for this)"
        ),
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


class GuardrailCheckResponse(SuccessEnvelope):
    message: str = Field("OK", description="Human-readable outcome", examples=["OK"])
    data: GuardrailReport
