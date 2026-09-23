from typing import Protocol

from ocr_common.types import OcrEngineResult


class OcrEngine(Protocol):
    """Reads the text lines of a document image or PDF.

    Returns the blocks in reading order plus the model's identity. Raises `UpstreamUnavailable`,
    `UpstreamTimeout` or `InternalError` when the model cannot be reached or answers something unexpected.
    """

    name: str

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> OcrEngineResult: ...
