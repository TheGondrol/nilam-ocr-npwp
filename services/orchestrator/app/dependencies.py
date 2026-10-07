"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from fastapi import Depends

from ocr_common.testing_endpoints import TESTING_TABLE_PREFIX

from app.clients.extraction import ExtractionJobClient, build_extraction_client
from app.clients.guardrails import GuardrailsClient, build_guardrails_client
from app.clients.stages import StageStatusClient, build_stage_status_clients
from app.config import Settings, get_settings
from app.services.extract_service import ExtractOcrService
from app.services.guardrails_log import GuardrailsLog, NoGuardrailsLog, SqlGuardrailsLog
from app.services.pipeline_waiter import PipelineWaiter
from app.services.response_log import NoResponseLog, ResponseLog, SqlResponseLog

# --- HTTP clients to the other services (one per process, closed in main.lifespan) -------------


@lru_cache
def get_guardrails_client() -> GuardrailsClient:
    return build_guardrails_client(get_settings())


@lru_cache
def get_extraction_client() -> ExtractionJobClient:
    return build_extraction_client(get_settings())


@lru_cache
def get_stage_status_clients() -> tuple[StageStatusClient, ...]:
    return build_stage_status_clients(get_settings())


@lru_cache
def get_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(get_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds)


def _guardrails_log(table_prefix: str = "") -> GuardrailsLog:
    settings = get_settings()
    if not settings.database_url:
        return NoGuardrailsLog()
    return SqlGuardrailsLog(
        settings.database_url, table_prefix=table_prefix, timeout=settings.guardrails_log_timeout_seconds
    )


@lru_cache
def get_guardrails_log() -> GuardrailsLog:
    return _guardrails_log()


@lru_cache
def get_response_logs() -> dict[str, ResponseLog]:
    """The log of the answers to POST /v1/extract-ocr (nilam_ocr_results) and to its -test twin
    (nilam_testing_ocr_results), keyed by table prefix; read by app.api.response_log.ResponseLogMiddleware."""
    settings = get_settings()
    if not settings.database_url:
        return {"": NoResponseLog(), TESTING_TABLE_PREFIX: NoResponseLog()}
    timeout = settings.guardrails_log_timeout_seconds
    return {
        prefix: SqlResponseLog(settings.database_url, table_prefix=prefix, timeout=timeout)
        for prefix in ("", TESTING_TABLE_PREFIX)
    }


# The testing endpoints (TESTING_ENDPOINTS): the same check and wait, on the stages' `-test` endpoints
# (see ocr_common.testing_endpoints). Guardrails keeps nothing, so its client is the live one; its verdicts
# go to testing_guardrails_results.


@lru_cache
def get_testing_extraction_client() -> ExtractionJobClient:
    return build_extraction_client(get_settings(), testing=True)


@lru_cache
def get_testing_stage_status_clients() -> tuple[StageStatusClient, ...]:
    return build_stage_status_clients(get_settings(), testing=True)


@lru_cache
def get_testing_guardrails_log() -> GuardrailsLog:
    return _guardrails_log(TESTING_TABLE_PREFIX)


@lru_cache
def get_testing_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(
        get_testing_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds
    )


# --- services (cheap to build: one per request) ------------------------------------------


def get_extract_service(
    guardrails: GuardrailsClient = Depends(get_guardrails_client),
    extraction: ExtractionJobClient = Depends(get_extraction_client),
    waiter: PipelineWaiter = Depends(get_pipeline_waiter),
    settings: Settings = Depends(get_settings),
    log: GuardrailsLog = Depends(get_guardrails_log),
) -> ExtractOcrService:
    return ExtractOcrService(guardrails, extraction, waiter, settings, log)


def get_testing_extract_service(
    guardrails: GuardrailsClient = Depends(get_guardrails_client),
    extraction: ExtractionJobClient = Depends(get_testing_extraction_client),
    waiter: PipelineWaiter = Depends(get_testing_pipeline_waiter),
    settings: Settings = Depends(get_settings),
    log: GuardrailsLog = Depends(get_testing_guardrails_log),
) -> ExtractOcrService:
    return ExtractOcrService(guardrails, extraction, waiter, settings, log)
