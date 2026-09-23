import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ocr_common.types import FieldConfidences

logger = logging.getLogger(__name__)

FEATURES = (
    "field_is_npwp",
    "field_score",
    "field_corrected",
    "shape_confidence",
    "field_multiple_candidates",
    "n_boxes_per_page",
    "avg_doc_score",
    "min_doc_score",
    "flag",
    "guardrail_probability",
)
UNDEFINED_FEATURES = ("shape_confidence",)
MISSING = float("nan")


def _number(value: Any) -> float:
    if value is None:
        return MISSING
    try:
        return float(value)
    except (TypeError, ValueError):
        return MISSING


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class TrustModel:
    name = "trust_model"

    def __init__(self, model_path: str):
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Trust model not found at {path} (SCORING_MODEL_PATH)")

        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)

        if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_cols" not in bundle:
            raise RuntimeError(f"{path} is not the expected trust-model bundle (dict with pipeline + feature_cols)")
        columns = tuple(bundle["feature_cols"])
        if columns != FEATURES:
            raise RuntimeError(f"Unexpected feature_cols in {path}: {columns}")
        self._pipeline = bundle["pipeline"]
        classes = [int(c) for c in self._pipeline.named_steps["clf"].classes_]
        if classes != [0, 1]:
            raise RuntimeError(f"Unexpected classes in {path}: {classes}")

        self.metadata: dict[str, Any] = {
            "steps": [type(step).__name__ for _, step in self._pipeline.steps],
            "features": list(FEATURES),
            "undefined_features": list(UNDEFINED_FEATURES),
            "train_report": {key: _plain(value) for key, value in (bundle.get("train_report") or {}).items()},
        }
        logger.info("trust model loaded from %s: %s", path, self.metadata)

    @staticmethod
    def _shape_confidence(payload: dict[str, Any], *, is_npwp: bool) -> float:
        return MISSING

    def feature_rows(self, payload: dict[str, Any]) -> dict[str, list[float]]:
        n_boxes, num_pages = _number(payload.get("n_boxes")), _number(payload.get("num_pages"))
        boxes_per_page = n_boxes / num_pages if num_pages and num_pages > 0 else MISSING
        candidates = payload.get("npwp_candidate_count")
        multiple = MISSING if candidates is None else float(candidates > 1)
        document = [
            boxes_per_page,
            _number(payload.get("avg_doc_score")),
            _number(payload.get("min_doc_score")),
            _number(payload.get("flag")),
            _number(payload.get("guardrail_probability")),
        ]
        return {
            "npwp": [
                1.0,
                _number(payload.get("npwp_score")),
                _number(payload.get("npwp_has_homoglyph")),
                self._shape_confidence(payload, is_npwp=True),
                multiple,
                *document,
            ],
            "name": [
                0.0,
                _number(payload.get("name_score")),
                _number(payload.get("name_corrected")),
                self._shape_confidence(payload, is_npwp=False),
                multiple,
                *document,
            ],
        }

    def predict(self, payload: dict[str, Any]) -> FieldConfidences:
        rows = self.feature_rows(payload)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            probabilities = self._pipeline.predict_proba(np.array([rows["npwp"], rows["name"]], dtype=float))[:, 1]

        return {
            "npwp_confidence": round(float(probabilities[0]), 4) if _has_text(payload.get("npwp")) else None,
            "name_confidence": round(float(probabilities[1]), 4) if _has_text(payload.get("name")) else None,
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
