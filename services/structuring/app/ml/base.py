from typing import Protocol

from ocr_common.types import OcrBlock, StructuredDocument


class Structurer(Protocol):
    """Turns OCR lines ({text, confidence, bbox, page}) into the named NPWP fields.

    Returns `fields` with one entry per name in `ocr_common.npwp.NPWP_FIELDS`, each
    {value, confidence, source, signals} (`value` is None when the field was not found), plus the
    document-level review `flag` / `flag_reason`. A structurer never rejects a document: what it cannot
    read comes back as null values and a flag.
    """

    name: str

    def structure(self, lines: list[OcrBlock]) -> StructuredDocument: ...
