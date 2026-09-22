"""The `remote` backend: the ML team's OCR model service and its /v1/predict/json contract
({data: [{rec_texts, rec_scores, rec_polys, page_index}, ...]})."""

from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

from app.ml.utils import bbox, confidence


class RemoteOcrEngine:
    name = "remote"
    PREDICT_PATH = "/v1/predict/json"

    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> dict[str, Any]:
        body = await self._client.post_multipart(
            self.PREDICT_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
        )
        return {"blocks": parse_rec_pages(body, self._client.name), "model": None}

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_rec_pages(body: Any, name: str) -> list[dict[str, Any]]:
    unexpected = ServiceError(500, f"{name} returned an unexpected response")
    pages = body.get("data") if isinstance(body, dict) else body
    if not isinstance(pages, list):
        raise unexpected

    blocks: list[dict[str, Any]] = []
    for position, page in enumerate(pages):
        if not isinstance(page, dict):
            raise unexpected
        texts, scores, polys = page.get("rec_texts"), page.get("rec_scores"), page.get("rec_polys")
        if not isinstance(texts, list) or not isinstance(scores, list) or len(scores) != len(texts):
            raise unexpected
        if polys is None:
            polys = [None] * len(texts)
        if not isinstance(polys, list) or len(polys) != len(texts):
            raise unexpected
        try:
            page_index = int(page.get("page_index", position))
        except (TypeError, ValueError):
            raise unexpected from None

        for text, score, poly in zip(texts, scores, polys, strict=True):
            text = str(text or "").strip()
            if not text:
                continue
            blocks.append({"text": text, "confidence": confidence(score), "bbox": bbox(poly), "page": page_index})
    return blocks
