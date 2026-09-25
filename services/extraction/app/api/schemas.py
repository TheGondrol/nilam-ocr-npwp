from typing import Literal

from pydantic import BaseModel, Field

from ocr_common.web.schemas import JobStatusBase, SuccessEnvelope


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left", examples=[190])
    y1: int = Field(..., description="Top", examples=[320])
    x2: int = Field(..., description="Right", examples=[760])
    y2: int = Field(..., description="Bottom", examples=[366])


class TextBlock(BaseModel):
    text: str = Field(..., description="One OCR text line, as read", examples=["NPWP : 12.345.678.9-012.345"])
    confidence: float = Field(..., ge=0, le=1, description="Recognition score of this line", examples=[0.9992])
    bbox: BoundingBox | None = Field(
        None,
        description=(
            "Upright box around the line, in pixels of the ORIENTATION-CORRECTED image the OCR model worked on. "
            "The model straightens and unwarps the card first, so these are NOT coordinates in the uploaded photo"
        ),
    )
    page: int = Field(0, ge=0, description="0-based page number (multi-page PDF)", examples=[0])


class OcrResult(BaseModel):
    engine: str = Field(
        ..., description="OCR backend that read the document (`EXTRACTION_BACKEND`)", examples=["paddle"]
    )
    model: str | None = Field(
        None,
        description="Model identity reported by the OCR model service; null when it does not report one",
        examples=["PP-OCRv6_medium_det+PP-OCRv6_medium_rec"],
    )
    elapsed_ms: float = Field(..., description="Time spent in the OCR model, milliseconds", examples=[412.5])
    full_text: str = Field(
        ...,
        description="All blocks joined by newline, in reading order",
        examples=["DIREKTORAT JENDERAL PAJAK\nNPWP : 12.345.678.9-012.345\nBUDI SANTOSO"],
    )
    blocks: list[TextBlock] = Field(..., description="Text lines in reading order (top to bottom)")


class ExtractResponse(SuccessEnvelope):
    data: OcrResult


class OcrJobStatus(JobStatusBase):
    stage: Literal["OCR"] = Field("OCR", description="Always `OCR` on this service", examples=["OCR"])
    result: OcrResult | None = Field(None, description="Raw OCR result once `status` is `DONE`; null otherwise")


class OcrJobStatusResponse(SuccessEnvelope):
    data: OcrJobStatus
