"""Perakitan tahap SCORING di pipeline async: tabel scoring.jobs/results dan callback hasil akhir."""

from functools import lru_cache

from ocr_common.jobs import STAGE_SCORING, StagePipeline, build_stage_pipeline
from src.core.config import get_settings

DB_SCHEMA = "scoring"


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(get_settings(), stage=STAGE_SCORING, schema=DB_SCHEMA)
