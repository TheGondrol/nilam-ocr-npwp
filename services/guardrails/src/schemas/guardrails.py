from typing import Literal

from pydantic import BaseModel, Field

from ocr_common.schemas import SuccessEnvelope

Verdict = Literal["accepted", "reject"]


class PageResult(BaseModel):
    page_index: int = Field(..., ge=0, examples=[0])
    proba_approve: float = Field(..., ge=0, le=1, examples=[0.9821])
    proba_reject: float = Field(..., ge=0, le=1, examples=[0.0179])
    verdict: Verdict = Field(..., examples=["accepted"])


class DocumentResult(BaseModel):
    verdict: Verdict = Field(
        ..., description="accepted hanya kalau lolos kebijakan dokumen (GUARDRAILS_DOCUMENT_POLICY)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Keyakinan pada vonis: accepted -> proba_approve halaman terlemah; reject -> proba_reject tertinggi"
        ),
        examples=[0.9821],
    )
    n_pages: int = Field(..., ge=0, examples=[2])
    n_approve: int = Field(..., ge=0, examples=[2])
    n_reject: int = Field(..., ge=0, examples=[0])


class GuardrailReport(BaseModel):
    passed: bool = Field(
        ..., description="true -> dokumen boleh lanjut ke OCR; false -> orkestrator menolak (422) dengan `reason`"
    )
    reason: str | None = Field(
        None,
        description="Alasan penolakan; null kalau passed",
        examples=["Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)"],
    )
    document: DocumentResult
    pages: list[PageResult]


class GuardrailCheckResponse(SuccessEnvelope):
    message: str = Field("OK", examples=["OK"])
    data: GuardrailReport
