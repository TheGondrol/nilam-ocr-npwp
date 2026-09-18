"""
Business logic Scoring: nilai field terstruktur, lalu terapkan ambang batas
dari Settings untuk memutuskan approve / review / reject.
"""

from typing import Any

from ocr_common.errors import ServiceError
from src.core.config import Settings

DECISION_APPROVE = "approve"
DECISION_REVIEW = "review"
DECISION_REJECT = "reject"


class ScoringService:
    def __init__(self, scorer, settings: Settings):
        self._scorer = scorer
        self._settings = settings

    def score(self, document_type: str, fields: dict[str, dict]) -> dict[str, Any]:
        if document_type not in self._scorer.supported_document_types:
            raise ServiceError(
                400,
                f"Unsupported document_type: {document_type}. Supported: {list(self._scorer.supported_document_types)}",
            )
        if not fields:
            raise ServiceError(400, "No fields to score")

        result = self._scorer.score(fields)
        score = round(min(max(result["score"], 0.0), 1.0), 4)
        return {
            "score": score,
            "decision": self._decide(score),
            "field_scores": result["field_scores"],
            "reasons": result["reasons"],
        }

    def _decide(self, score: float) -> str:
        if score >= self._settings.scoring_approve_threshold:
            return DECISION_APPROVE
        if score >= self._settings.scoring_review_threshold:
            return DECISION_REVIEW
        return DECISION_REJECT
