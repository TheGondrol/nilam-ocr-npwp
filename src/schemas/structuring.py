from pydantic import BaseModel, Field

from src.schemas.common import SuccessEnvelope


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
