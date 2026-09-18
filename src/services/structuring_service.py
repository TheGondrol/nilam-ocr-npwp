"""
Business logic Structuring: baris teks mentah (mis. output ekstraksi) ->
field bernama dengan confidence per field. Selalu mengembalikan semua field
dokumen; yang tidak ditemukan bernilai null.
"""

from typing import Any

from src.core.errors import ServiceError
from src.models.structuring import DOCUMENT_TYPE


class StructuringService:
    def __init__(self, structurer):
        self._structurer = structurer

    def structure(self, lines: list[dict]) -> dict[str, Any]:
        cleaned = [line for line in lines if (line.get("text") or "").strip()]
        if not cleaned:
            raise ServiceError(400, "No text lines to structure")

        fields = self._structurer.structure(cleaned)
        return {"document_type": DOCUMENT_TYPE, "fields": fields}
