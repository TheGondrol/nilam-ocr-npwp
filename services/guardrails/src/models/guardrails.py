"""
Pembungkus model Guardrails: pengklasifikasi halaman accepted / reject.

Dua kontrak method, tergantung di mana modelnya jalan:

    backend lokal (model di proses ini):
    classify(filename, pages: list[PIL.Image.Image]) -> list[PagePrediction]
        PagePrediction = (proba_approve, proba_reject), keduanya 0..1, jumlah 1.
        Service layer (src/services/guardrails_service.py) yang merender file
        menjadi halaman, menerapkan ambang batas, dan mengagregasi vonis dokumen.

    backend remote (model sebagai service HTTP):
    async check_document(filename, content, content_type) -> {"document", "pages"}
        Service model menerima berkas utuh dan sudah mengembalikan vonis dokumen.

Implementasi:
- remote: service model guardrails milik ML engineer (POST /v1/predict/json);
  lihat RemoteGuardrailsModel.
- efficientnet: checkpoint torchvision EfficientNet-B0 milik ML engineer
  (weights/best_model.pt). Checkpoint berisi `model_state_dict`,
  `class_names` ['accepted', 'reject'], `image_size` 224, `reject_threshold`
  0.5, `epoch`, `val_macro_f1`. torch di-import di dalam kelas supaya backend
  mock (dan test) tidak butuh torch terpasang.
- mock: vonis dari nama file, skenario sama dengan mock ocr-*.
"""

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

PagePrediction = tuple[float, float]  # (proba_approve, proba_reject)

# Normalisasi ImageNet, bawaan bobot pretrained torchvision yang di-fine-tune.
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

        # weights_only=True: checkpoint hanya berisi tensor + metadata skalar/list,
        # jadi pemuatan tanpa eksekusi pickle sembarang aman dan cukup.
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

        # Resize langsung ke image_size x image_size (bukan crop), supaya tepi
        # halaman yang ter-crop tetap terlihat oleh model.
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0  # HWC
        tensor = self._torch.from_numpy(array).permute(2, 0, 1)  # CHW
        return (tensor - self._mean) / self._std

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]:
        if not pages:
            return []
        batch = self._torch.stack([self._to_tensor(page) for page in pages]).to(self._device)
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self._model(batch), dim=1).cpu()
        return [(round(float(p[0]), 4), round(float(p[1]), 4)) for p in probabilities]


class RemoteGuardrailsModel:
    """
    Klien ke service model guardrails milik ML engineer. Kontraknya:

        POST {GUARDRAILS_MODEL_URL}/v1/predict/json
        header X-API-Key, multipart `file` (gambar atau PDF)
        -> {"status_code": 200, "message": "OK",
            "data": {"document": {"verdict", "confidence", "n_pages", "n_approve", "n_reject"},
                     "pages": [{"page_index", "proba_approve", "proba_reject", "verdict"}]}}

    Beda dengan backend lokal: service itu menerima BERKAS utuh (PDF dirender
    di sana) dan sudah mengembalikan vonis dokumen, jadi kontraknya
    check_document(), bukan classify(halaman). Vonisnya diteruskan apa adanya:
    ambang reject, kebijakan dokumen, dan render PDF adalah urusan service
    model, sehingga GUARDRAILS_REJECT_THRESHOLD / _DOCUMENT_POLICY / _PDF_DPI /
    _MAX_PAGES tidak berlaku untuk backend ini.
    """

    name = "remote"
    PREDICT_PATH = "/v1/predict/json"

    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
        """-> {"document": {...}, "pages": [...]}, bentuk yang sama dengan backend lokal."""
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
    """Ambil hanya field kontrak, dengan tipe yang dijanjikan schema kita. Bentuk lain -> 500,
    bukan diteruskan: vonis yang salah baca lebih berbahaya daripada request yang gagal."""
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
