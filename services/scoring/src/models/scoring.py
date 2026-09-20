import re
from collections.abc import Callable
from functools import lru_cache

from ocr_common.registry import Factory, build_backend
from src.core.config import get_settings

WEIGHTS: dict[str, float] = {
    "nomor_npwp": 3.0,
    "nama": 2.0,
    "nama_badan": 2.0,
}
REQUIRED = ("nomor_npwp",)
ONE_OF = ("nama", "nama_badan")

Validator = Callable[[str], str | None]


def _validate_npwp(value: str) -> str | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) not in (15, 16):
        return f"expected 15 or 16 digits, got {len(digits)}"
    return None


def _validate_nama(value: str) -> str | None:
    if len(value.strip()) < 3:
        return "too short"
    if re.search(r"\d", value):
        return "contains digits"
    return None


def _validate_nama_badan(value: str) -> str | None:
    if len(value.strip()) < 3:
        return "too short"
    return None


VALIDATORS: dict[str, Validator] = {
    "nomor_npwp": _validate_npwp,
    "nama": _validate_nama,
    "nama_badan": _validate_nama_badan,
}


class HeuristicNpwpScorer:
    name = "heuristic"
    supported_document_types = ("npwp",)

    def score(self, fields: dict[str, dict]) -> dict:
        field_scores: list[dict] = []
        reasons: list[str] = []
        weighted_total = 0.0
        weight_total = 0.0

        present_one_of = [name for name in ONE_OF if self._has_value(fields.get(name))]
        for name, weight in WEIGHTS.items():
            field_score = self._score_field(name, fields.get(name))
            field_scores.append(field_score)
            if name in ONE_OF and present_one_of and field_score["issues"] == ["missing"]:
                continue
            weighted_total += weight * field_score["score"]
            weight_total += weight
            if name in REQUIRED and field_score["issues"]:
                reasons.append(f"required field {name}: {', '.join(field_score['issues'])}")

        if not present_one_of:
            reasons.append("required one of nama, nama_badan: missing")

        overall = weighted_total / weight_total if weight_total else 0.0
        return {"score": round(overall, 4), "field_scores": field_scores, "reasons": reasons}

    @staticmethod
    def _has_value(field: dict | None) -> bool:
        return bool(field and field.get("value") is not None and str(field["value"]).strip())

    @classmethod
    def _score_field(cls, name: str, field: dict | None) -> dict:
        if field is None or not cls._has_value(field):
            return {"name": name, "score": 0.0, "issues": ["missing"]}

        validator = VALIDATORS.get(name)
        issue = validator(str(field["value"])) if validator else None
        if issue:
            return {"name": name, "score": 0.0, "issues": [f"invalid format: {issue}"]}

        confidence = min(max(float(field.get("confidence", 1.0)), 0.0), 1.0)
        return {"name": name, "score": round(confidence, 4), "issues": []}


SCORER_BACKENDS: dict[str, Factory] = {
    "heuristic": lambda settings: HeuristicNpwpScorer(),
}


@lru_cache
def get_scorer():
    settings = get_settings()
    return build_backend(SCORER_BACKENDS, settings.scoring_backend, settings, "scoring")
