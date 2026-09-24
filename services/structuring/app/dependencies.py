"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from ocr_common.pipeline import (
    STAGE_STRUCTURING,
    NextStageClient,
    OutboxRelay,
    StagePipeline,
    StageResults,
    StaleJobReaper,
    build_next_stage_client,
    build_outbox_relay,
    build_stage_pipeline,
    build_stage_results,
    build_stale_job_reaper,
)
from ocr_common.registry import Factory, build_backend
from ocr_common.testing_endpoints import testing_path

from app.config import Settings, get_settings
from app.ml.base import Structurer
from app.ml.npwp_rules import NpwpRulesStructurer
from app.ml.rule_based import RuleBasedNpwpStructurer
from app.services.job_service import StructuringJobService
from app.services.structuring_service import StructuringService

logger = logging.getLogger(__name__)

DB_TABLE_PREFIX = "structuring"

# The vendored rules read their reference-data paths from these environment variables (the ML team's
# own convention, so a mounted volume + env var works unchanged). Settings may also come from `.env`,
# which pydantic does not export, so the values are pushed to the environment here, once, before the
# rules are first used.
REFERENCE_DATA_ENV = {
    "wilayah_codes_path": "WILAYAH_CODES_PATH",
    "kpp_codes_path": "KPP_CODES_PATH",
    "name_master_path": "NAME_MASTER_PATH",
    "npwp_name_list_path": "NPWP_NAME_LIST_PATH",
}


def _build_npwp_rules(settings: Settings) -> NpwpRulesStructurer:
    for attribute, variable in REFERENCE_DATA_ENV.items():
        value = getattr(settings, attribute)
        if value:
            os.environ[variable] = value

    from app.vendor.npwp_rules import kpp_codes, name_master, wilayah_codes

    for label, path in (
        ("kode_wilayah.json", wilayah_codes.default_data_path()),
        ("kpp_codes.json", kpp_codes.default_data_path()),
        ("name_lnmast.xlsx", name_master.default_master_path()),
    ):
        state = "found" if Path(path).is_file() else "MISSING (the check it feeds gives no signal)"
        logger.info("structuring reference data %s: %s at %s", label, state, path)
    return NpwpRulesStructurer()


# STRUCTURING_BACKEND -> how to build it. Add a backend here and, if it needs settings, in config.py.
STRUCTURER_BACKENDS: dict[str, Factory[Structurer]] = {
    "npwp_rules": _build_npwp_rules,
    "rule_based": lambda settings: RuleBasedNpwpStructurer(),
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_structurer() -> Structurer:
    settings: Settings = get_settings()
    return build_backend(STRUCTURER_BACKENDS, settings.structuring_backend, settings, "structuring")


# --- pipeline -----------------------------------------------------------------------


SCORING_JOBS_PATH = "/v1/scoring/jobs"


def _next_stage(path: str, name: str) -> NextStageClient:
    settings = get_settings()
    return build_next_stage_client(
        settings,
        base_url=settings.scoring_service_url,
        api_key=settings.scoring_api_key,
        timeout=settings.scoring_timeout_seconds,
        path=path,
        name=name,
    )


@lru_cache
def get_next_stage() -> NextStageClient:
    return _next_stage(SCORING_JOBS_PATH, "scoring service")


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(), stage=STAGE_STRUCTURING, table_prefix=DB_TABLE_PREFIX, next_stage=get_next_stage()
    )


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


@lru_cache
def get_results() -> StageResults | None:
    return build_stage_results(get_settings())


# The testing endpoints (TESTING_ENDPOINTS): the same pipeline on the testing_* tables, handing off to
# scoring's `-test` endpoint, without callbacks (see ocr_common.testing_endpoints).


@lru_cache
def get_testing_next_stage() -> NextStageClient:
    return _next_stage(testing_path(SCORING_JOBS_PATH), "scoring service (testing)")


@lru_cache
def get_testing_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(),
        stage=STAGE_STRUCTURING,
        table_prefix=DB_TABLE_PREFIX,
        next_stage=get_testing_next_stage(),
        testing=True,
    )


@lru_cache
def get_testing_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_testing_pipeline())


@lru_cache
def get_testing_results() -> StageResults | None:
    return build_stage_results(get_settings(), testing=True)


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


def get_testing_job_service() -> StructuringJobService:
    return StructuringJobService(
        get_testing_pipeline(),
        get_structuring_service(),
        results=get_testing_results(),
        handoff_by_reference=get_settings().pipeline_handoff_by_reference,
    )


@lru_cache
def get_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_pipeline(), get_job_service().resume)


@lru_cache
def get_testing_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_testing_pipeline(), get_testing_job_service().resume)
