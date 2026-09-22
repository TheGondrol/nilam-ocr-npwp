"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from ocr_common.pipeline import STAGE_SCORING, OutboxRelay, StagePipeline, build_outbox_relay, build_stage_pipeline
from ocr_common.registry import Factory, build_backend

from app.config import Settings, get_settings
from app.ml.base import Scorer
from app.ml.heuristic import HeuristicNpwpScorer
from app.ml.trust_model import TrustModel
from app.services.confidence_service import ConfidenceService
from app.services.job_service import ScoringJobService
from app.services.scoring_service import ScoringService

DB_TABLE_PREFIX = "scoring"

# SCORING_BACKEND -> how to build the legacy document scorer.
SCORER_BACKENDS: dict[str, Factory[Scorer]] = {
    "heuristic": lambda settings: HeuristicNpwpScorer(),
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_scorer() -> Scorer:
    settings: Settings = get_settings()
    return build_backend(SCORER_BACKENDS, settings.scoring_backend, settings, "scoring")


@lru_cache
def get_trust_model() -> TrustModel:
    return TrustModel(get_settings().scoring_model_path)


# --- pipeline -----------------------------------------------------------------------


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_SCORING, table_prefix=DB_TABLE_PREFIX)


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


# --- services (cheap to build: one per request) ------------------------------------------


def get_scoring_service() -> ScoringService:
    return ScoringService(get_scorer(), get_settings())


def get_confidence_service() -> ConfidenceService:
    return ConfidenceService(get_trust_model())


def get_job_service() -> ScoringJobService:
    return ScoringJobService(get_pipeline(), get_confidence_service(), get_settings().field_confidence_threshold)
