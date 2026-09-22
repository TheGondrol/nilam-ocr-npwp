from typing import Any, Protocol


class OcrEngine(Protocol):
    """Reads the text lines of a document image or PDF.

    Returns {"blocks": [{text, confidence, bbox, page}, ...], "model": str | None} with the blocks in
    reading order. Raises `ServiceError` (503 / 504 / 500) when the model cannot be reached or answers
    something unexpected.
    """

    name: str

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> dict[str, Any]: ...
