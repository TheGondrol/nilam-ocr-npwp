"""Skema response yang sama untuk semua service: envelope sukses, error (+ contoh), health."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ocr_common.envelope import envelope

REQUEST_ID_EXAMPLE = "REQ_9cb01af2-493d-446d-b191-af120333f6d0"


class SuccessEnvelope(BaseModel):
    """Bagian envelope yang sama untuk semua response sukses; subclass hanya menentukan `data`."""

    status_code: int = Field(200, examples=[200])
    status_desc: str = Field("OK", examples=["OK"])
    message: str = Field("Success", examples=["Success"])
    errors: None = None
    request_id: str | None = Field(None, examples=[REQUEST_ID_EXAMPLE])


class JobAccepted(BaseModel):
    request_id: str = Field(..., examples=[REQUEST_ID_EXAMPLE])
    stage: Literal["OCR", "STRUCTURING", "SCORING"] = Field(..., examples=["OCR"])
    status: Literal["PROCESSING", "DONE", "FAILED"] = Field(
        ..., description="PROCESSING untuk job baru; status terkini kalau request_id ini sudah pernah dikirim"
    )
    duplicate: bool = Field(
        False, description="true: request_id ini sudah pernah diterima, kerja TIDAK dijalankan ulang (idempoten)"
    )


class JobAcceptedResponse(SuccessEnvelope):
    status_code: int = Field(202, examples=[202])
    status_desc: str = Field("Accepted", examples=["Accepted"])
    message: str = Field("Accepted", examples=["Accepted"])
    data: JobAccepted


class JobStatus(BaseModel):
    request_id: str = Field(..., examples=[REQUEST_ID_EXAMPLE])
    stage: Literal["OCR", "STRUCTURING", "SCORING"] = Field(..., examples=["OCR"])
    status: Literal["PROCESSING", "DONE", "FAILED"] = Field(..., examples=["DONE"])
    result: dict[str, Any] | None = Field(None, description="Hasil tahap ini saat DONE; null selain itu")
    error_message: str | None = Field(None, description="Pesan kegagalan saat FAILED; null selain itu")
    created_at: str = Field(..., examples=["2026-09-18T04:00:00+00:00"])
    updated_at: str = Field(..., examples=["2026-09-18T04:00:01+00:00"])


class JobStatusResponse(SuccessEnvelope):
    data: JobStatus


class ErrorResponse(BaseModel):
    status_code: int = Field(..., examples=[400])
    status_desc: str = Field(..., examples=["Bad Request"])
    message: str = Field(..., examples=["Uploaded file is empty"])
    data: None = None
    errors: str = Field(..., examples=["Uploaded file is empty"])
    request_id: str | None = Field(None, description="Present when the error is tied to a known request_id")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    version: str = Field(..., examples=["1.0.0"])
    device: str = Field(..., examples=["cpu"])
    backends: dict[str, str] = Field(
        ..., description="Backend model aktif di service ini", examples=[{"ekstraksi": "paddle"}]
    )


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"] = Field(..., examples=["ready"])
    checks: dict[str, Literal["ok", "failed"]] = Field(
        ..., description="Dependensi wajib service ini, satu entri per dependensi", examples=[{"database": "ok"}]
    )


def error(
    status_code: int,
    description: str,
    message: str,
    *,
    request_id: str | None = None,
    errors: str | None = None,
) -> dict:
    """Satu entri `responses` lengkap dengan contohnya sendiri.

    Tanpa contoh per status, /docs memakai example bawaan ErrorResponse untuk
    semua kode: 401 pun tampil sebagai 400, dan pembacanya belajar kode yang
    salah."""
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "example": envelope(status_code, message, None, request_id, errors=errors or message),
            }
        },
    }


# Entri 401 yang sama di semua router; diimpor supaya kalimatnya satu sumber.
UNAUTHORIZED = error(401, "Missing or invalid X-API-Key", "Invalid or missing API key")
