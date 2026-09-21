import hashlib
import random
from functools import lru_cache
from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.registry import Factory, build_backend
from ocr_common.remote import RemoteModelClient
from src.core.config import Settings, get_settings


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
            model = model or _model_name(document.get("models"))
            for page in document.get("pages") or []:
                page_index = int(page.get("page_index", 0) or 0)
                for item in page.get("texts") or []:
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    blocks.append(
                        {
                            "text": text,
                            "confidence": _confidence(item.get("score")),
                            "bbox": _bbox(item.get("poly")),
                            "page": page_index,
                        }
                    )
        return {"blocks": blocks, "model": model}

    async def aclose(self) -> None:
        await self._client.aclose()


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
        return {"blocks": _parse_rec_pages(body, self._client.name), "model": None}

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_rec_pages(body: Any, name: str) -> list[dict[str, Any]]:
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
            blocks.append({"text": text, "confidence": _confidence(score), "bbox": _bbox(poly), "page": page_index})
    return blocks


def _model_name(models: Any) -> str | None:
    if not isinstance(models, dict):
        return None
    detection, recognition = models.get("detection"), models.get("recognition")
    if detection and recognition:
        return f"{detection}+{recognition}"
    return detection or recognition or models.get("pipeline") or None


def _confidence(score: Any) -> float:
    try:
        return round(min(max(float(score), 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _bbox(poly: Any) -> dict[str, int] | None:
    try:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return {"x1": round(min(xs)), "y1": round(min(ys)), "x2": round(max(xs)), "y2": round(max(ys))}


NAMA_POOL = [
    "BUDI SANTOSO",
    "SITI AMINAH",
    "AGUS WIJAYA",
    "DEWI LESTARI",
    "RUDI HARTONO",
    "ANI SURYANI",
    "EKO PRASETYO",
    "RINA MARLINA",
    "JOKO SUSILO",
    "WATI RAHAYU",
]
NAMA_BADAN_POOL = [
    "PT SINAR ABADI SEJAHTERA",
    "PT CIPTA KARYA MANDIRI",
    "PT NUSANTARA DIGITAL TEKNOLOGI",
    "PT BUMI MAKMUR SENTOSA",
    "PT GLOBAL MITRA INDUSTRI",
]
LINE_HEIGHT = 40


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(count))


def _format_npwp(d: str) -> str:
    return f"{d[0:2]}.{d[2:5]}.{d[5:8]}.{d[8]}-{d[9:12]}.{d[12:15]}"


class MockOcrEngine:
    name = "mock"

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> dict[str, Any]:
        if "servererror" in (filename or "").lower():
            raise ServiceError(500, "Internal server error while processing OCR")

        rng = random.Random(hashlib.sha256(content).hexdigest())
        lines = [
            "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA",
            "DIREKTORAT JENDERAL PAJAK",
            f"NPWP : {_format_npwp(_digits(rng, 15))}",
            f"NAMA : {rng.choice(NAMA_POOL)}",
            f"NAMA BADAN : {rng.choice(NAMA_BADAN_POOL)}",
        ]
        blocks = [
            {
                "text": text,
                "confidence": round(rng.uniform(0.85, 0.99), 3),
                "bbox": {"x1": 20, "y1": 20 + i * LINE_HEIGHT, "x2": 20 + 12 * len(text), "y2": 50 + i * LINE_HEIGHT},
                "page": 0,
            }
            for i, text in enumerate(lines)
        ]
        return {"blocks": blocks, "model": None}


def _build_paddle(settings: Settings) -> PaddleOcrEngine:
    if not settings.ekstraksi_ocr_url:
        raise RuntimeError("EKSTRAKSI_OCR_URL is required when EKSTRAKSI_BACKEND=paddle")
    client = RemoteModelClient(
        settings.ekstraksi_ocr_url, settings.ekstraksi_ocr_timeout_seconds, name="ekstraksi OCR model"
    )
    return PaddleOcrEngine(client)


def _build_remote(settings: Settings) -> RemoteOcrEngine:
    if not settings.ekstraksi_ocr_url:
        raise RuntimeError("EKSTRAKSI_OCR_URL is required when EKSTRAKSI_BACKEND=remote")
    headers = {"X-API-Key": settings.ekstraksi_ocr_api_key} if settings.ekstraksi_ocr_api_key else None
    client = RemoteModelClient(
        settings.ekstraksi_ocr_url,
        settings.ekstraksi_ocr_timeout_seconds,
        name="ekstraksi OCR model",
        headers=headers,
    )
    return RemoteOcrEngine(client)


OCR_BACKENDS: dict[str, Factory] = {
    "mock": lambda settings: MockOcrEngine(),
    "paddle": _build_paddle,
    "remote": _build_remote,
}


@lru_cache
def get_ocr_engine():
    settings = get_settings()
    return build_backend(OCR_BACKENDS, settings.ekstraksi_backend, settings, "ekstraksi OCR")
