"""Normalisation of what the OCR model services return into the block shape of this service."""

from typing import Any

from ocr_common.types import BoundingBox


def model_name(models: Any) -> str | None:
    if not isinstance(models, dict):
        return None
    detection, recognition = models.get("detection"), models.get("recognition")
    if detection and recognition:
        return f"{detection}+{recognition}"
    return detection or recognition or models.get("pipeline") or None


def confidence(score: Any) -> float:
    try:
        return round(min(max(float(score), 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return 0.0


def bbox(poly: Any) -> BoundingBox | None:
    """Upright box around a polygon of [x, y] points; None when the polygon is missing or malformed."""
    try:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return {"x1": round(min(xs)), "y1": round(min(ys)), "x2": round(max(xs)), "y2": round(max(ys))}
