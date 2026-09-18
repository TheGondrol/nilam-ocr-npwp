from typing import Any

from pydantic import BaseModel, Field

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"


class Confidence(BaseModel):
    """Satu field hasil OCR beserta seberapa yakin mesin membacanya.

    Nilainya selalu terbungkus, juga saat confidence-nya tinggi: bentuk yang
    berubah-ubah menuntut tiap konsumen memeriksa dulu ini dict atau skalar,
    dan yang lupa memeriksa baru ketahuan di produksi."""

    value: Any = Field(..., description="Nilai yang terbaca; null kalau field tidak ditemukan")
    confidence: float | None = Field(None, ge=0, le=1, examples=[0.87])


class NpwpFields(BaseModel):
    nomor_npwp: Confidence = Field(..., examples=[{"value": "12.345.678.9-012.345", "confidence": 0.87}])
    nama: Confidence = Field(
        ..., description="Nama wajib pajak orang pribadi", examples=[{"value": "RIAN DERMAWAN", "confidence": 0.87}]
    )
    nama_badan: Confidence = Field(
        ...,
        description="Nama badan usaha pemilik NPWP",
        examples=[{"value": "PT NUSANTARA DIGITAL TEKNOLOGI", "confidence": 0.87}],
    )


class GenerateRequestIdData(BaseModel):
    request_id: str = Field(..., examples=[RID])


class GenerateRequestIdResponse(BaseModel):
    status_code: int = Field(200, examples=[200])
    status_desc: str = Field("OK", examples=["OK"])
    message: str = Field("Success", examples=["Success"])
    data: GenerateRequestIdData
    errors: None = None
    request_id: str = Field(..., examples=[RID])


GUARDRAILS_DESCRIPTION = (
    "Skor kepercayaan dokumen ini secara keseluruhan (0..1), dihitung service scoring dari "
    "confidence tiap field dan validasi formatnya. Orchestrator membandingkannya dengan "
    "ambang batas role pemanggil."
)


class ExtractOcrResponse(BaseModel):
    status_code: int = Field(200, examples=[200])
    status_desc: str = Field("OK", examples=["OK"])
    message: str = Field("Success", examples=["Success"])
    data: NpwpFields
    errors: None = None
    request_id: str = Field(..., examples=[RID])
    guardrails: float | None = Field(None, ge=0, le=1, examples=[0.72], description=GUARDRAILS_DESCRIPTION)


class GetOcrResultData(BaseModel):
    request_id: str = Field(..., examples=[RID])
    status: str = Field(..., description="pending | completed | failed", examples=["completed"])
    result: NpwpFields | None = Field(None, description="Present once status is 'completed'")
    error_message: str | None = Field(None, description="Present once status is 'failed'")
    created_at: str = Field(..., examples=["2026-03-06T07:01:59.976912+00:00"])
    updated_at: str = Field(..., examples=["2026-03-06T07:07:04.809697+00:00"])


class GetOcrResultResponse(BaseModel):
    status_code: int = Field(200, examples=[200])
    status_desc: str = Field("OK", examples=["OK"])
    message: str = Field("Success", examples=["Success"])
    data: GetOcrResultData
    errors: None = None
    request_id: str = Field(..., examples=[RID])
    guardrails: float | None = Field(
        None, ge=0, le=1, examples=[0.72], description=GUARDRAILS_DESCRIPTION + " Null selama status belum completed."
    )
