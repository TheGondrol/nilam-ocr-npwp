from typing import Any, Protocol


class Structurer(Protocol):
    """Turns OCR lines ({text, confidence, bbox, page}) into the named NPWP fields.

    Returns one entry per field in `ocr_common.npwp.NPWP_FIELDS`, each
    {value, confidence, source, signals}; `value` is None when the field was not found.
    Raises `ServiceError(400, ...)` when the lines are not a lone NPWP card.
    """

    name: str

    def structure(self, lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]: ...
