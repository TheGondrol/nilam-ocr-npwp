"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from ocr_common.pipeline import (
    STAGE_STRUCTURING,
    NextStageClient,
    OutboxRelay,
    SqlStageResults,
    StagePipeline,
    StaleJobReaper,
    build_next_stage_client,
    build_outbox_relay,
    build_stage_pipeline,
    build_stage_results,
    build_stale_job_reaper,
)
from ocr_common.registry import Factory, build_backend

from app.config import Settings, get_settings
from app.ml.base import Structurer
from app.ml.npwp_rules import NpwpRulesStructurer
from app.ml.rule_based import RuleBasedNpwpStructurer
from app.services.job_service import StructuringJobService
from app.services.structuring_service import StructuringService

DB_TABLE_PREFIX = "structuring"

# STRUCTURING_BACKEND -> how to build it. Add a backend here and, if it needs settings, in config.py.
STRUCTURER_BACKENDS: dict[str, Factory[Structurer]] = {
    "npwp_rules": lambda settings: NpwpRulesStructurer(page_guardrails=settings.structuring_page_guardrails),
    "rule_based": lambda settings: RuleBasedNpwpStructurer(),
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_structurer() -> Structurer:
    settings: Settings = get_settings()
    return build_backend(STRUCTURER_BACKENDS, settings.structuring_backend, settings, "structuring")


# --- pipeline -----------------------------------------------------------------------


@lru_cache
def get_next_stage() -> NextStageClient:
    settings = get_settings()
    return build_next_stage_client(
        settings,
        base_url=settings.scoring_service_url,
        api_key=settings.scoring_api_key,
        timeout=settings.scoring_timeout_seconds,
        path="/v1/scoring/jobs",
        name="scoring service",
    )


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(), stage=STAGE_STRUCTURING, table_prefix=DB_TABLE_PREFIX, next_stage=get_next_stage()
    )


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


@lru_cache
def get_results() -> SqlStageResults | None:
    return build_stage_results(get_settings())


# --- services (cheap to build: one per request) ------------------------------------------


def get_structuring_service() -> StructuringService:
    return StructuringService(get_structurer())


def get_job_service() -> StructuringJobService:
    return StructuringJobService(
        get_pipeline(),
        get_structuring_service(),
        results=get_results(),
        handoff_by_reference=get_settings().pipeline_handoff_by_reference,
    )


@lru_cache
def get_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_pipeline(), get_job_service().resume)
