from pydantic import BaseModel, Field

from src.core.envelope import envelope


class ErrorResponse(BaseModel):
    status_code: int = Field(..., examples=[400])
    status_desc: str = Field(..., examples=["Bad Request"])
    message: str = Field(..., examples=["Uploaded image is not recognized as an NPWP"])
    data: None = None
    errors: str = Field(..., examples=["Uploaded image is not recognized as an NPWP"])
    request_id: str | None = Field(None, description="Present when the error is tied to a known request_id")


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
    semua kode: 401 pun tampil sebagai 400 "bukan NPWP", dan pembacanya
    belajar kode yang salah."""
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {
                "example": envelope(status_code, message, None, request_id, errors=errors or message),
            }
        },
    }
