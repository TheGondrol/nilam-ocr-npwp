"""Perakitan tahap OCR di pipeline async: tabel ocr.jobs/ocr.results, callback, handoff ke structuring."""

from functools import lru_cache

from ocr_common.jobs import (
    STAGE_OCR,
    NextStageClient,
    StagePipeline,
    build_next_stage_client,
    build_stage_pipeline,
)
from src.core.config import get_settings

DB_SCHEMA = "ocr"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_OCR, schema=DB_SCHEMA)


@lru_cache
def get_next_stage() -> NextStageClient:
    settings = get_settings()
    return build_next_stage_client(
        settings,
        base_url=settings.structuring_service_url,
        api_key=settings.structuring_api_key,
        timeout=settings.structuring_timeout_seconds,
        path="/v1/structuring/jobs",
        name="structuring service",
    )
