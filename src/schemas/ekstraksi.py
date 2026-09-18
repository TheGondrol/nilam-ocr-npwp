from pydantic import BaseModel, Field

from src.schemas.common import SuccessEnvelope


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TextBlock(BaseModel):
    text: str = Field(..., examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.96])
    bbox: BoundingBox | None = None


class OcrResult(BaseModel):
    engine: str = Field(..., examples=["mock"])
    elapsed_ms: float = Field(..., examples=[12.5])
    full_text: str = Field(..., description="Semua blok digabung dengan newline")
    blocks: list[TextBlock]


class ExtractResponse(SuccessEnvelope):
    data: OcrResult
