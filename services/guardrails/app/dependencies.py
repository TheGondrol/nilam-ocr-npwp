"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from fastapi import Depends

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.registry import Factory, build_backend

from app.clients.ekstraksi import EkstraksiJobClient, build_ekstraksi_client
from app.clients.stages import StageStatusClient, build_stage_status_clients
from app.config import Settings, get_settings
from app.ml.base import Classifier
from app.ml.efficientnet import EfficientNetPageClassifier
from app.ml.mock import MockPageClassifier
from app.ml.remote import RemoteGuardrailsModel
from app.services.guardrails_service import GuardrailsService
from app.services.job_service import GuardrailsJobService
from app.services.pipeline_waiter import PipelineWaiter


def _build_remote(settings: Settings) -> RemoteGuardrailsModel:
    if not settings.guardrails_model_url:
        raise RuntimeError("GUARDRAILS_MODEL_URL is required when GUARDRAILS_BACKEND=remote")
    headers = {"X-API-Key": settings.guardrails_model_api_key} if settings.guardrails_model_api_key else None
    client = RemoteModelClient(
        settings.guardrails_model_url,
        settings.guardrails_model_timeout_seconds,
        name="guardrails model",
        headers=headers,
    )
    return RemoteGuardrailsModel(client)


# GUARDRAILS_BACKEND -> how to build it. Add a backend here and, if it needs settings, in config.py.
CLASSIFIER_BACKENDS: dict[str, Factory[Classifier]] = {
    "mock": lambda settings: MockPageClassifier(),
    "efficientnet": lambda settings: EfficientNetPageClassifier(
        settings.guardrails_model_path, settings.guardrails_device, settings.guardrails_torch_threads
    ),
    "remote": _build_remote,
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_page_classifier() -> Classifier:
    settings: Settings = get_settings()
    return build_backend(CLASSIFIER_BACKENDS, settings.guardrails_backend, settings, "guardrails classifier")


# --- HTTP clients to the other stages (one per process, closed in main.lifespan) ---------------


@lru_cache
def get_ekstraksi_client() -> EkstraksiJobClient:
    return build_ekstraksi_client(get_settings())


@lru_cache
def get_stage_status_clients() -> tuple[StageStatusClient, ...]:
    return build_stage_status_clients(get_settings())


@lru_cache
def get_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(get_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds)


# The testing endpoint (TESTING_ENDPOINTS): the same check and wait, on the stages' `-test` endpoints
# (see ocr_common.testing_endpoints).


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


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_page_classifier(), get_settings())


def get_job_service(
    guardrails: GuardrailsService = Depends(get_guardrails_service),
    ekstraksi: EkstraksiJobClient = Depends(get_ekstraksi_client),
    waiter: PipelineWaiter = Depends(get_pipeline_waiter),
    settings: Settings = Depends(get_settings),
) -> GuardrailsJobService:
    return GuardrailsJobService(guardrails, ekstraksi, waiter, wait_seconds=settings.pipeline_wait_seconds)


def get_testing_job_service(
    guardrails: GuardrailsService = Depends(get_guardrails_service),
    ekstraksi: EkstraksiJobClient = Depends(get_testing_ekstraksi_client),
    waiter: PipelineWaiter = Depends(get_testing_pipeline_waiter),
    settings: Settings = Depends(get_settings),
) -> GuardrailsJobService:
    return GuardrailsJobService(guardrails, ekstraksi, waiter, wait_seconds=settings.pipeline_wait_seconds)
