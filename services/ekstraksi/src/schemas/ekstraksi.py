from pydantic import BaseModel, Field

from ocr_common.schemas import SuccessEnvelope


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TextBlock(BaseModel):
    text: str = Field(..., examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.96])
    bbox: BoundingBox | None = Field(None, description="Kotak tegak yang melingkupi teks, piksel")
    page: int = Field(0, ge=0, description="Indeks halaman 0-based (PDF multi-halaman)", examples=[0])


class OcrResult(BaseModel):
    engine: str = Field(..., description="Backend yang dipakai (EKSTRAKSI_BACKEND)", examples=["paddle"])
    model: str | None = Field(
        None,
        description="Identitas model yang membaca, dari response model",
        examples=["PP-OCRv6_medium_det+PP-OCRv6_medium_rec"],
    )
    elapsed_ms: float = Field(..., examples=[412.5])
    full_text: str = Field(..., description="Semua blok digabung dengan newline")
    blocks: list[TextBlock]


class ExtractResponse(SuccessEnvelope):
    data: OcrResult
