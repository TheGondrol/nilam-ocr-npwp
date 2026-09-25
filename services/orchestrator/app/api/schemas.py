from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "processing", "completed", "failed"]


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
            "0: the document passed the guardrails model (and the pipeline ran); 1: it was rejected by the "
            "guardrails model or by the structuring rules. Null on 202, and when the request was refused before "
            "the check"
        ),
        examples=[0],
    )
    params: Any = Field(
        None,
        description=(
            "The `params` sent with `POST /v1/extract-ocr`, returned unchanged; null when not sent. Not stored, so "
            "always null on `GET /v1/extract-ocr/{request_id}`"
        ),
        examples=[{"nik": "3123456711950001", "refno": "PK19039Y8U"}],
    )
