import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from ocr_common.errors import ServiceError
from ocr_common.registry import Factory, build_backend
from ocr_common.remote import RemoteModelClient
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

PagePrediction = tuple[float, float]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EfficientNetPageClassifier:
    name = "efficientnet"

    def __init__(self, model_path: str, device: str = "cpu", threads: int | None = None):
        import torch
        from torchvision.models import efficientnet_b0

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Guardrails model not found at {path} (GUARDRAILS_MODEL_PATH)")

        if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
            logger.warning("guardrails device %r not available in this build; falling back to cpu", device)
            device = "cpu"
        if threads:
            torch.set_num_threads(threads)

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        self.class_names: list[str] = list(checkpoint.get("class_names") or ["accepted", "reject"])
        self.image_size: int = int(checkpoint.get("image_size") or 224)
        self.reject_threshold: float = float(checkpoint.get("reject_threshold") or 0.5)
        self.metadata: dict[str, Any] = {
            "architecture": "efficientnet_b0",
            "device": device,
            "threads": torch.get_num_threads(),
            "epoch": checkpoint.get("epoch"),
            "val_macro_f1": checkpoint.get("val_macro_f1"),
            "class_names": self.class_names,
            "image_size": self.image_size,
            "reject_threshold": self.reject_threshold,
        }
        if [c.lower() for c in self.class_names] != ["accepted", "reject"]:
            raise RuntimeError(f"Unexpected class_names in checkpoint: {self.class_names}")

        model = efficientnet_b0(weights=None, num_classes=len(self.class_names))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        logger.info("guardrails model loaded from %s: %s", path, self.metadata)

    def _to_tensor(self, image: Image.Image):
        import numpy as np

        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self._mean) / self._std

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]:
        if not pages:
            return []
        batch = self._torch.stack([self._to_tensor(page) for page in pages]).to(self._device)
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self._model(batch), dim=1).cpu()
        return [(round(float(p[0]), 4), round(float(p[1]), 4)) for p in probabilities]


class RemoteGuardrailsModel:
    name = "remote"
    PREDICT_PATH = "/v1/predict/json"

    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
        body = await self._client.post_multipart(
            self.PREDICT_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
        )
        return _parse_report(body, self._client.name)

    async def aclose(self) -> None:
        await self._client.aclose()


def _verdict(value: Any) -> str:
    if value not in ("accepted", "reject"):
        raise ValueError(f"unknown verdict: {value!r}")
    return value


def _parse_report(body: Any, name: str) -> dict[str, Any]:
    try:
        data = body["data"]
        document = data["document"]
        return {
            "document": {
                "verdict": _verdict(document["verdict"]),
                "confidence": float(document["confidence"]),
                "n_pages": int(document["n_pages"]),
                "n_approve": int(document["n_approve"]),
                "n_reject": int(document["n_reject"]),
            },
            "pages": [
                {
                    "page_index": int(page["page_index"]),
                    "proba_approve": float(page["proba_approve"]),
                    "proba_reject": float(page["proba_reject"]),
                    "verdict": _verdict(page["verdict"]),
                }
                for page in data["pages"]
            ],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError(500, f"{name} returned an unexpected response") from exc


def _build_remote(settings: Settings) -> RemoteGuardrailsModel:
    if not settings.guardrails_model_url:
        raise RuntimeError("GUARDRAILS_MODEL_URL is required when GUARDRAILS_BACKEND=remote")
    headers = {"X-API-Key": settings.guardrails_model_api_key} if settings.guardrails_model_api_key else None
    return RemoteGuardrailsModel(
        RemoteModelClient(
            settings.guardrails_model_url,
            settings.guardrails_model_timeout_seconds,
            name="guardrails model",
            headers=headers,
        )
    )


class MockPageClassifier:
    name = "mock"
    reject_threshold = 0.5
    metadata: dict[str, Any] = {"architecture": "mock"}

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]:
        name = (filename or "").lower()
        rejected = "blur" in name or "invalid" in name or "notnpwp" in name
        prediction = (0.1179, 0.8821) if rejected else (0.9821, 0.0179)
        return [prediction for _ in pages]


CLASSIFIER_BACKENDS: dict[str, Factory] = {
    "mock": lambda settings: MockPageClassifier(),
    "efficientnet": lambda settings: EfficientNetPageClassifier(
        settings.guardrails_model_path, settings.guardrails_device, settings.guardrails_torch_threads
    ),
    "remote": _build_remote,
}


@lru_cache
def get_page_classifier():
    settings: Settings = get_settings()
    return build_backend(CLASSIFIER_BACKENDS, settings.guardrails_backend, settings, "guardrails classifier")
