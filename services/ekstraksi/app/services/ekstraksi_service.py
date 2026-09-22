import time
from typing import Any

from ocr_common.image_validation import validate_image

from app.config import Settings


class EkstraksiService:
    def __init__(self, engine, settings: Settings):
        self._engine = engine
        self._settings = settings

    async def extract(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        validate_image(content_type, content, self._settings)

        started = time.perf_counter()
        result = await self._engine.extract(filename, content, content_type)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        blocks = result["blocks"]
        return {
            "engine": self._engine.name,
            "model": result.get("model"),
            "elapsed_ms": elapsed_ms,
            "full_text": "\n".join(block["text"] for block in blocks),
            "blocks": blocks,
        }
