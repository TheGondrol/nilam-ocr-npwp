"""The `remote` backend: the ML team's OCR model service and its /v1/predict/json contract.

Their service answers `{models: {detection, recognition}, num_pages, pages: [{page_index, rec_texts,
rec_scores, rec_polys}], n_boxes, avg_doc_score, min_doc_score}` (nilamnpwp `ocr/schemas.py`, 23 Sep
2026); the earlier shapes, a bare list of pages or `{data: [...]}`, are still accepted. Only `pages`
and `models` are used: the document-level scores are recomputed by scoring from the blocks.

In production one OCR model will serve several document types and take its parameters per call, so
`EKSTRAKSI_OCR_PARAMS` (a JSON object) is sent as extra form fields with every request."""

from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError
from ocr_common.types import OcrBlock, OcrEngineResult

from app.ml.utils import bbox, confidence, model_name


class RemoteOcrEngine:
    name = "remote"
    PREDICT_PATH = "/v1/predict/json"

    def __init__(self, client: RemoteModelClient, params: dict[str, Any] | None = None):
        self._client = client
        self._params = {key: str(value) for key, value in (params or {}).items()}

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> OcrEngineResult:
        body = await self._client.post_multipart(
            self.PREDICT_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
            data=self._params or None,
        )
        return {"blocks": parse_rec_pages(body, self._client.name), "model": parse_model(body)}

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_model(body: Any) -> str | None:
    """`models.detection+recognition` of the current contract; None for the older shapes."""
    return model_name(body.get("models")) if isinstance(body, dict) else None


def parse_rec_pages(body: Any, name: str) -> list[OcrBlock]:
    unexpected = InternalError(f"{name} returned an unexpected response")
    if isinstance(body, dict):
        pages = body.get("pages") if "pages" in body else body.get("data")
    else:
        pages = body
    if not isinstance(pages, list):
        raise unexpected

    blocks: list[OcrBlock] = []
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
