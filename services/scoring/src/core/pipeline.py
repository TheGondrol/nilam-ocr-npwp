from functools import lru_cache

from ocr_common.jobs import STAGE_SCORING, StagePipeline, build_stage_pipeline
from src.core.config import get_settings

DB_TABLE_PREFIX = "scoring"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_SCORING, table_prefix=DB_TABLE_PREFIX)
