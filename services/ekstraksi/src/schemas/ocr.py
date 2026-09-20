"""Skema kontrak lama (generate-request-id -> extract-ocr -> get-ocr-result). Teks Field(...) tampil di Swagger."""

from typing import Any, Literal

from pydantic import BaseModel, Field

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"


class Confidence(BaseModel):
    """Satu field hasil OCR beserta seberapa yakin mesin membacanya.

    Nilainya selalu terbungkus, juga saat confidence-nya tinggi: bentuk yang
    berubah-ubah menuntut tiap konsumen memeriksa dulu ini dict atau skalar,
    dan yang lupa memeriksa baru ketahuan di produksi."""

    value: Any = Field(..., description="The value read; null when the field was not found", examples=["BUDI SANTOSO"])
    confidence: float | None = Field(
        None, ge=0, le=1, description="OCR score of the line the value came from; 0 when not found", examples=[0.97]
    )


class NpwpFields(BaseModel):
    nomor_npwp: Confidence = Field(
        ...,
        description="15-digit number as `XX.XXX.XXX.X-XXX.XXX`, or a 16-digit NIK-based number as 16 plain digits",
        examples=[{"value": "12.345.678.9-012.345", "confidence": 0.99}],
    )
    nama: Confidence = Field(
        ..., description="Taxpayer name on a person's card", examples=[{"value": "BUDI SANTOSO", "confidence": 0.97}]
    )
    nama_badan: Confidence = Field(
        ...,
        description="Registered name on a company's card (PT, CV, ...); null on a person's card",
        examples=[{"value": None, "confidence": 0.0}],
    )


class _LegacyEnvelope(BaseModel):
    status_code: int = Field(200, description="Same as the HTTP status code", examples=[200])
    status_desc: str = Field("OK", description="Reason phrase of `status_code`", examples=["OK"])
    message: str = Field("Success", description="Human-readable outcome", examples=["Success"])
    errors: None = Field(None, description="Always null on success")
    request_id: str = Field(..., description="The request_id this response belongs to", examples=[RID])


class GenerateRequestIdData(BaseModel):
    request_id: str = Field(..., description="Use it once in POST /v1/extract-ocr", examples=[RID])


class GenerateRequestIdResponse(_LegacyEnvelope):
    data: GenerateRequestIdData


GUARDRAILS_DESCRIPTION = (
    "Overall document score (0..1) from the legacy heuristic: OCR confidence per field combined with format "
    "checks. NOT produced by the ML team's trust model; the async pipeline reports per-field confidences instead."
)


class ExtractOcrResponse(_LegacyEnvelope):
    data: NpwpFields
    guardrails: float | None = Field(None, ge=0, le=1, examples=[0.98], description=GUARDRAILS_DESCRIPTION)


class GetOcrResultData(BaseModel):
    request_id: str = Field(..., description="request_id from /v1/generate-request-id", examples=[RID])
    status: Literal["pending", "completed", "failed"] = Field(
        ..., description="`pending` -> `completed` | `failed`", examples=["completed"]
    )
    result: NpwpFields | None = Field(None, description="Present once `status` is `completed`")
    error_message: str | None = Field(None, description="Present once `status` is `failed`", examples=[None])
    created_at: str = Field(..., description="ISO 8601, UTC", examples=["2026-03-06T07:01:59.976912+00:00"])
    updated_at: str = Field(..., description="ISO 8601, UTC", examples=["2026-03-06T07:07:04.809697+00:00"])


class GetOcrResultResponse(_LegacyEnvelope):
    data: GetOcrResultData
    guardrails: float | None = Field(
        None, ge=0, le=1, examples=[0.98], description=GUARDRAILS_DESCRIPTION + " Null until `status` is `completed`."
    )
