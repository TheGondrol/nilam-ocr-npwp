from ocr_common.errors import BadRequest
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.types import OcrBlock, StructuringResult

from app.ml.base import Structurer


class StructuringService:
    """Drops empty lines, runs the structurer, and shapes the stage result (`StructuringResult`)."""

    def __init__(self, structurer: Structurer):
        self._structurer = structurer

    def structure(self, lines: list[OcrBlock]) -> StructuringResult:
        cleaned = [line for line in lines if (line.get("text") or "").strip()]
        if not cleaned:
            raise BadRequest("No text lines to structure")

        document = self._structurer.structure(cleaned)
        return {
            "document_type": DOCUMENT_TYPE,
            "fields": document["fields"],
            "flag": document["flag"],
            "flag_reason": document["flag_reason"],
            "reject_reason": document.get("reject_reason"),
        }
