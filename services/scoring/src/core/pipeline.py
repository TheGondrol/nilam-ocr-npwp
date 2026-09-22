from functools import lru_cache

from ocr_common.jobs import STAGE_SCORING, OutboxRelay, StagePipeline, build_outbox_relay, build_stage_pipeline
from src.core.config import get_settings

DB_TABLE_PREFIX = "scoring"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_SCORING, table_prefix=DB_TABLE_PREFIX)


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())
