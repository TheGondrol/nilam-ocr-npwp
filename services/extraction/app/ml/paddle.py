"""The `paddle` backend: the PaddleOCR model server (EXTRACTION_OCR_URL) and its /ocr contract.

The same contract is served at /ocr by the old VM and at /v1/ocr by the PP-OCRv6 service in the shared cluster
(EXTRACTION_OCR_PATH)."""

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import InternalError
from ocr_common.types import OcrBlock, OcrEngineResult

from app.ml.utils import bbox, confidence, model_name

V1_OCR_PATH = "/v1/ocr"
# Fixed preprocessing for the PP-OCRv6 service's /v1/ocr, every call (agreed 22 Oct 2026). These names replace
# use_doc_orientation_classify / use_doc_unwarping / use_textline_orientation, which it answers with 400.
# The old VM's /ocr gets no query.
V1_OCR_PARAMS = {"doc_orientation": "true", "doc_unwarping": "true", "textline_orientation": "false"}


class PaddleOcrEngine:
    name = "paddle"

    def __init__(self, client: RemoteModelClient, path: str = "/ocr"):
        self._client = client
        self._path = path

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> OcrEngineResult:
        body = await self._client.post_multipart(
            self._path,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
            params=V1_OCR_PARAMS if self._path == V1_OCR_PATH else None,
        )
        documents = body if isinstance(body, list) else [body]

        blocks: list[OcrBlock] = []
        model: str | None = None
        for document in documents:
            if not isinstance(document, dict):
                raise InternalError(f"{self._client.name} returned an unexpected response shape")
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
