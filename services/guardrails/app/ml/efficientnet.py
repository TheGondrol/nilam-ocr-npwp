"""The `efficientnet` backend: the trained EfficientNet-B0 checkpoint (weights/best_model.pt) run with
torch in this process."""

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.base import PagePrediction
from app.ml.utils import image_to_tensor

logger = logging.getLogger(__name__)


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
        logger.info("guardrails model loaded from %s: %s", path, self.metadata)

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]:
        if not pages:
            return []
        batch = self._torch.stack([image_to_tensor(page, self.image_size) for page in pages]).to(self._device)
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self._model(batch), dim=1).cpu()
        return [(round(float(p[0]), 4), round(float(p[1]), 4)) for p in probabilities]
