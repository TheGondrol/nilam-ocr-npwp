"""Perakitan tahap STRUCTURING di pipeline async: tabel structuring.jobs/results, callback, handoff ke scoring."""

from functools import lru_cache

from ocr_common.jobs import (
    STAGE_STRUCTURING,
    NextStageClient,
    StagePipeline,
    build_next_stage_client,
    build_stage_pipeline,
)
from src.core.config import get_settings

DB_SCHEMA = "structuring"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_STRUCTURING, schema=DB_SCHEMA)


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
