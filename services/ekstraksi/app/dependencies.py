"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from fastapi import Depends

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.pipeline import (
    STAGE_OCR,
    NextStageClient,
    OutboxRelay,
    StagePipeline,
    StaleJobReaper,
    build_next_stage_client,
    build_outbox_relay,
    build_stage_pipeline,
    build_stale_job_reaper,
)
from ocr_common.registry import Factory, build_backend

from app.clients.stages import StageClients, build_stage_clients
from app.config import Settings, get_settings
from app.ml.base import OcrEngine
from app.ml.mock import MockOcrEngine
from app.ml.paddle import PaddleOcrEngine
from app.ml.remote import RemoteOcrEngine
from app.repositories.request_repository import RequestRepository, build_request_repository
from app.services.ekstraksi_service import EkstraksiService
from app.services.job_service import EkstraksiJobService
from app.services.ocr_service import OcrService

DB_TABLE_PREFIX = "ocr"


def _build_paddle(settings: Settings) -> PaddleOcrEngine:
    if not settings.ekstraksi_ocr_url:
        raise RuntimeError("EKSTRAKSI_OCR_URL is required when EKSTRAKSI_BACKEND=paddle")
    client = RemoteModelClient(
        settings.ekstraksi_ocr_url, settings.ekstraksi_ocr_timeout_seconds, name="ekstraksi OCR model"
    )
    return PaddleOcrEngine(client)


def _build_remote(settings: Settings) -> RemoteOcrEngine:
    if not settings.ekstraksi_ocr_url:
        raise RuntimeError("EKSTRAKSI_OCR_URL is required when EKSTRAKSI_BACKEND=remote")
    headers = {"X-API-Key": settings.ekstraksi_ocr_api_key} if settings.ekstraksi_ocr_api_key else None
    client = RemoteModelClient(
        settings.ekstraksi_ocr_url,
        settings.ekstraksi_ocr_timeout_seconds,
        name="ekstraksi OCR model",
        headers=headers,
    )
    return RemoteOcrEngine(client)


# EKSTRAKSI_BACKEND -> how to build it. Add a backend here and, if it needs settings, in config.py.
OCR_BACKENDS: dict[str, Factory[OcrEngine]] = {
    "mock": lambda settings: MockOcrEngine(),
    "paddle": _build_paddle,
    "remote": _build_remote,
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_ocr_engine() -> OcrEngine:
    settings: Settings = get_settings()
    return build_backend(OCR_BACKENDS, settings.ekstraksi_backend, settings, "ekstraksi OCR")


# --- storage and HTTP clients (one per process, closed in main.lifespan) -----------------------


@lru_cache
def get_request_repository() -> RequestRepository:
    return build_request_repository(get_settings().database_url)


@lru_cache
def get_stage_clients() -> StageClients:
    return build_stage_clients(get_settings())


# --- pipeline -----------------------------------------------------------------------


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


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(), stage=STAGE_OCR, table_prefix=DB_TABLE_PREFIX, next_stage=get_next_stage()
    )


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


# --- services (cheap to build: one per request) ------------------------------------------


def get_ekstraksi_service() -> EkstraksiService:
    return EkstraksiService(get_ocr_engine(), get_settings())


def get_job_service() -> EkstraksiJobService:
    settings = get_settings()
    return EkstraksiJobService(
        get_pipeline(),
        get_ekstraksi_service(),
        settings.max_upload_bytes,
        url_policy=settings.file_url_policy,
        simulate_delay=settings.is_local,
        handoff_by_reference=settings.pipeline_handoff_by_reference,
    )


def get_ocr_service(
    repository: RequestRepository = Depends(get_request_repository),
    ekstraksi: EkstraksiService = Depends(get_ekstraksi_service),
    stages: StageClients = Depends(get_stage_clients),
) -> OcrService:
    return OcrService(repository, ekstraksi, stages)


@lru_cache
def get_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_pipeline(), get_job_service().resume)
