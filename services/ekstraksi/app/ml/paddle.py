"""The `paddle` backend: the PaddleOCR model server (EKSTRAKSI_OCR_URL) and its /ocr contract."""

from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

from app.ml.utils import bbox, confidence, model_name


class PaddleOcrEngine:
    name = "paddle"

    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> dict[str, Any]:
        body = await self._client.post_multipart(
            "/ocr",
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
        )
        documents = body if isinstance(body, list) else [body]

        blocks: list[dict[str, Any]] = []
        model: str | None = None
        for document in documents:
            if not isinstance(document, dict):
                raise ServiceError(500, f"{self._client.name} returned an unexpected response shape")
            model = model or model_name(document.get("models"))
            for page in document.get("pages") or []:
                page_index = int(page.get("page_index", 0) or 0)
                for item in page.get("texts") or []:
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    blocks.append(
                        {
                            "text": text,
                            "confidence": confidence(item.get("score")),
                            "bbox": bbox(item.get("poly")),
                            "page": page_index,
                        }
                    )
        return {"blocks": blocks, "model": model}

    async def aclose(self) -> None:
        await self._client.aclose()
