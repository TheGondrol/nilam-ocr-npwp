from typing import Protocol

from ocr_common.types import OcrBlock, StructuredField


class Structurer(Protocol):
    """Turns OCR lines ({text, confidence, bbox, page}) into the named NPWP fields.

    Returns one entry per field in `ocr_common.npwp.NPWP_FIELDS`, each
    {value, confidence, source, signals}; `value` is None when the field was not found.
    Raises `BadRequest` when the lines are not a lone NPWP card.
    """

    name: str

    def structure(self, lines: list[OcrBlock]) -> dict[str, StructuredField]: ...
