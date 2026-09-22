from functools import lru_cache

from ocr_common.jobs import (
    STAGE_OCR,
    NextStageClient,
    OutboxRelay,
    StagePipeline,
    build_next_stage_client,
    build_outbox_relay,
    build_stage_pipeline,
)
from src.core.config import get_settings

DB_TABLE_PREFIX = "ocr"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(), stage=STAGE_OCR, table_prefix=DB_TABLE_PREFIX, next_stage=get_next_stage()
    )


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


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
