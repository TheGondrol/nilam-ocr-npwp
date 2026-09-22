"""The `mock` backend for local development and tests: the verdict comes from the file name."""

from typing import Any

from PIL import Image

from app.ml.base import PagePrediction


class MockPageClassifier:
    name = "mock"
    reject_threshold = 0.5
    metadata: dict[str, Any] = {"architecture": "mock"}

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]:
        name = (filename or "").lower()
        rejected = "blur" in name or "invalid" in name or "notnpwp" in name
        prediction = (0.1179, 0.8821) if rejected else (0.9821, 0.0179)
        return [prediction for _ in pages]
