from typing import Protocol

from ocr_common.types import OcrBlock, StructuredDocument


class Structurer(Protocol):
    """Turns OCR lines ({text, confidence, bbox, page}) into the named NPWP fields.

    Returns `fields` with one entry per name in `ocr_common.npwp.NPWP_FIELDS`, each
    {value, confidence, source, signals} (`value` is None when the field was not found), plus the
    document-level `flag` / `flag_reason` and `reject_reason`. A structurer never raises for what it cannot
    read: that comes back as null values and a flag; `reject_reason` tells the pipeline to stop here.
    """

    name: str

    def structure(self, lines: list[OcrBlock]) -> StructuredDocument: ...
