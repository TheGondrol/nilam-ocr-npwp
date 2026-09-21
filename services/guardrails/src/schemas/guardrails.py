from typing import Literal

from pydantic import BaseModel, Field

from ocr_common.pipeline_schemas import FinalResult
from ocr_common.schemas import JobAccepted, JobState, Stage, SuccessEnvelope

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


class PipelineProgress(BaseModel):
    stage: Stage = Field(
        ...,
        description=(
            "The stage this is about: `SCORING` when the pipeline is `DONE`, the stage that failed when `FAILED`, "
            "or the stage still running when the wait ran out"
        ),
        examples=["SCORING"],
    )
    status: JobState = Field(
        ...,
        description=(
            "`DONE`: `result` holds the final result. `FAILED`: the pipeline stopped at `stage`. `PROCESSING`: still "
            "running when the wait ran out (HTTP 202); the outcome follows in the stage callbacks"
        ),
        examples=["DONE"],
    )
    error_message: str | None = Field(
        None, description="Why `stage` failed; null unless `status` is `FAILED`", examples=[None]
    )


class GuardrailJobReport(GuardrailReport):
    job: JobAccepted | None = Field(
        None,
        description=(
            "The OCR job this service started on the ekstraksi service when `passed`; null when rejected (nothing "
            "runs and no callback follows) or when `handoff` was false"
        ),
    )
    pipeline: PipelineProgress | None = Field(
        None,
        description=(
            "Where the pipeline stood when this response was sent, after waiting up to `PIPELINE_WAIT_SECONDS` "
            "from the moment the request arrived. Null when rejected, when `handoff` was false, or when waiting is "
            "disabled (`PIPELINE_WAIT_SECONDS=0`)"
        ),
    )
    result: FinalResult | None = Field(
        None,
        description=(
            "The final result when `pipeline.status` is `DONE`: identical to the `result` of the SCORING callback, "
            "which is still sent. Null otherwise"
        ),
    )


class GuardrailCheckResponse(SuccessEnvelope):
    message: str = Field("OK", description="Human-readable outcome", examples=["OK"])
    data: GuardrailJobReport
