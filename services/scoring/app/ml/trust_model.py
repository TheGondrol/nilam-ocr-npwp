"""The ML team's trust model (weights/trust_model.joblib, retrained 21 Sep 2026): a pooled npwp+name
logistic regression that answers "how likely is this field's value correct". One feature row per
field, computed exactly as their `scoring/trust_scoring.py` computes it from a live request."""

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
    "shape_confidence",
    "field_multiple_candidates",
    "name_max_char_len",
    "avg_doc_score",
    "min_doc_score",
    "flag",
    "guardrail_probability",
)
PAYLOAD_KEYS = (
    "npwp",
    "npwp_score",
    "npwp_candidate_count",
    "name_base",
    "name_score",
    "avg_doc_score",
    "min_doc_score",
    "flag",
    "guardrail_probability",
)
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


def npwp_shape_confidence(npwp: str | None) -> int:
    """2 = 16-digit NIK-based format, 1 = legacy 15 digits, 0 = anything else or missing. Counted on the
    digits only, so both `12.345.678.9-012.000` and `123456789012000` are tier 1."""
    digits = "".join(ch for ch in npwp if ch.isdigit()) if npwp else ""
    return {16: 2, 15: 1}.get(len(digits), 0)


def name_shape_confidence(name_base: str | None) -> int:
    """2 = a normal multi-word name, 1 = exactly one word, 0 = empty or missing."""
    words = len(name_base.split()) if name_base else 0
    return 2 if words >= 2 else words


def name_max_char_len(name_base: str | None) -> int:
    """Longest whitespace-separated token of the base name read (0 when missing): an abnormally long token
    means OCR ran words together."""
    return max((len(token) for token in name_base.split()), default=0) if name_base else 0


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
            "train_report": {key: _plain(value) for key, value in (bundle.get("train_report") or {}).items()},
        }
        logger.info("trust model loaded from %s: %s", path, self.metadata)

    @staticmethod
    def feature_rows(payload: dict[str, Any]) -> dict[str, list[float]]:
        """The two feature rows (npwp, name) of one request, in `FEATURES` order. A missing input is NaN
        and the model's own median imputer fills it, as in training."""
        candidates = payload.get("npwp_candidate_count")
        document = [
            _number(payload.get("avg_doc_score")),
            _number(payload.get("min_doc_score")),
            float(bool(payload.get("flag"))),
            _number(payload.get("guardrail_probability")),
        ]
        name_base = payload.get("name_base")
        return {
            "npwp": [
                1.0,
                _number(payload.get("npwp_score")),
                float(npwp_shape_confidence(payload.get("npwp"))),
                MISSING if candidates is None else float(int(candidates) > 1),
                MISSING,  # name-only feature, imputed for the npwp row as in training
                *document,
            ],
            "name": [
                0.0,
                _number(payload.get("name_score")),
                float(name_shape_confidence(name_base)),
                MISSING,  # no name-candidate scorer exists; NaN in training too
                float(name_max_char_len(name_base)),
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
            "name_confidence": round(float(probabilities[1]), 4) if _has_text(payload.get("name_base")) else None,
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
