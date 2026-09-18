from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ocr_common.schemas import REQUEST_ID_EXAMPLE, SuccessEnvelope


class TextLine(BaseModel):
    text: str = Field(..., examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, examples=[0.96])


class StructureRequest(BaseModel):
    lines: list[TextLine] = Field(..., min_length=1, description="Baris teks hasil OCR, urut atas ke bawah")


class StructuredField(BaseModel):
    value: str | None = Field(None, examples=["12.345.678.9-012.345"], description="null kalau field tidak ditemukan")
    confidence: float = Field(..., ge=0, le=1, examples=[0.96])
    source: str | None = Field(
        None, description="Baris mentah asal nilai ini", examples=["NPWP : 12.345.678.9-012.345"]
    )


class StructuredDocument(BaseModel):
    document_type: str = Field(..., examples=["npwp"])
    fields: dict[str, StructuredField] = Field(
        ...,
        description=(
            "Selalu berisi semua field dokumen (nomor_npwp, nama, nama_badan); yang tidak ditemukan bernilai null"
        ),
    )


class StructureResponse(SuccessEnvelope):
    data: StructuredDocument


class OcrBlock(BaseModel):
    # extra="allow": bbox/page dan field lain dari ServiceOCR ikut diteruskan ke scoring apa adanya.
    model_config = ConfigDict(extra="allow")

    text: str = Field(..., examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(1.0, ge=0, le=1, examples=[0.96])


class OcrPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    blocks: list[OcrBlock] = Field(..., description="Blok teks hasil OCR, urut atas ke bawah")


class StructuringJobRequest(BaseModel):
    request_id: str = Field(..., min_length=1, examples=[REQUEST_ID_EXAMPLE])
    document_type: str = Field("npwp", examples=["npwp"])
    guardrails: dict[str, Any] | None = Field(
        None, description="Hasil guardrails dari orkestrator; hanya diteruskan ke tahap berikutnya"
    )
    ocr: OcrPayload = Field(..., description="Hasil tahap OCR (`ocr.results`)")
