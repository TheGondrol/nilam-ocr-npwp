"""
Business logic Ekstraksi: jalankan OCR engine pada gambar dan kembalikan
blok teks mentah + metadata. Penafsiran teks menjadi field adalah tugas
StructuringService.
"""

import time
from typing import Any

from src.core.config import Settings
from src.services.image_validation import validate_image


class EkstraksiService:
    def __init__(self, engine, settings: Settings):
        self._engine = engine
        self._settings = settings

    def extract(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        validate_image(content_type, content, self._settings)

        started = time.perf_counter()
        blocks = self._engine.extract(filename, content)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        return {
            "engine": self._engine.name,
            "elapsed_ms": elapsed_ms,
            "full_text": "\n".join(block["text"] for block in blocks),
            "blocks": blocks,
        }
