"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from fastapi import Depends

from app.clients.ekstraksi import EkstraksiJobClient, build_ekstraksi_client
from app.clients.guardrails import GuardrailsClient, build_guardrails_client
from app.clients.stages import StageStatusClient, build_stage_status_clients
from app.config import Settings, get_settings
from app.services.extract_service import ExtractOcrService
from app.services.pipeline_waiter import PipelineWaiter

# --- HTTP clients to the other services (one per process, closed in main.lifespan) -------------


@lru_cache
def get_guardrails_client() -> GuardrailsClient:
    return build_guardrails_client(get_settings())


@lru_cache
def get_ekstraksi_client() -> EkstraksiJobClient:
    return build_ekstraksi_client(get_settings())


@lru_cache
def get_stage_status_clients() -> tuple[StageStatusClient, ...]:
    return build_stage_status_clients(get_settings())


@lru_cache
def get_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(get_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds)


# The testing endpoints (TESTING_ENDPOINTS): the same check and wait, on the stages' `-test` endpoints
# (see ocr_common.testing_endpoints). Guardrails keeps nothing, so its client is the live one.


@lru_cache
def get_testing_ekstraksi_client() -> EkstraksiJobClient:
    return build_ekstraksi_client(get_settings(), testing=True)


@lru_cache
def get_testing_stage_status_clients() -> tuple[StageStatusClient, ...]:
    return build_stage_status_clients(get_settings(), testing=True)


@lru_cache
def get_testing_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(
        get_testing_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds
    )


# --- services (cheap to build: one per request) ------------------------------------------


def get_extract_service(
    guardrails: GuardrailsClient = Depends(get_guardrails_client),
    ekstraksi: EkstraksiJobClient = Depends(get_ekstraksi_client),
    waiter: PipelineWaiter = Depends(get_pipeline_waiter),
    settings: Settings = Depends(get_settings),
) -> ExtractOcrService:
    return ExtractOcrService(guardrails, ekstraksi, waiter, settings)


def get_testing_extract_service(
    guardrails: GuardrailsClient = Depends(get_guardrails_client),
    ekstraksi: EkstraksiJobClient = Depends(get_testing_ekstraksi_client),
    waiter: PipelineWaiter = Depends(get_testing_pipeline_waiter),
    settings: Settings = Depends(get_settings),
) -> ExtractOcrService:
    return ExtractOcrService(guardrails, ekstraksi, waiter, settings)
