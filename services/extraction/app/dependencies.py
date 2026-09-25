"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

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
from ocr_common.testing_endpoints import testing_path

from app.config import Settings, get_settings
from app.ml.base import OcrEngine
from app.ml.mock import MockOcrEngine
from app.ml.paddle import PaddleOcrEngine
from app.ml.remote import RemoteOcrEngine
from app.services.extraction_service import ExtractionService
from app.services.job_service import ExtractionJobService

DB_TABLE_PREFIX = "ocr"


def _build_paddle(settings: Settings) -> PaddleOcrEngine:
    if not settings.extraction_ocr_url:
        raise RuntimeError("EXTRACTION_OCR_URL is required when EXTRACTION_BACKEND=paddle")
    client = RemoteModelClient(
        settings.extraction_ocr_url, settings.extraction_ocr_timeout_seconds, name="extraction OCR model"
    )
    return PaddleOcrEngine(client)


def _build_remote(settings: Settings) -> RemoteOcrEngine:
    if not settings.extraction_ocr_url:
        raise RuntimeError("EXTRACTION_OCR_URL is required when EXTRACTION_BACKEND=remote")
    headers = {"X-API-Key": settings.extraction_ocr_api_key} if settings.extraction_ocr_api_key else None
    client = RemoteModelClient(
        settings.extraction_ocr_url,
        settings.extraction_ocr_timeout_seconds,
        name="extraction OCR model",
        headers=headers,
    )
    return RemoteOcrEngine(client, params=settings.extraction_ocr_params)


# EXTRACTION_BACKEND -> how to build it. Add a backend here and, if it needs settings, in config.py.
OCR_BACKENDS: dict[str, Factory[OcrEngine]] = {
    "mock": lambda settings: MockOcrEngine(),
    "paddle": _build_paddle,
    "remote": _build_remote,
}


# --- ML ---------------------------------------------------------------------------


@lru_cache
def get_ocr_engine() -> OcrEngine:
    settings: Settings = get_settings()
    return build_backend(OCR_BACKENDS, settings.extraction_backend, settings, "extraction OCR")


# --- pipeline -----------------------------------------------------------------------


STRUCTURING_JOBS_PATH = "/v1/structuring/jobs"


def _next_stage(path: str, name: str) -> NextStageClient:
    settings = get_settings()
    return build_next_stage_client(
        settings,
        base_url=settings.structuring_service_url,
        api_key=settings.structuring_api_key,
        timeout=settings.structuring_timeout_seconds,
        path=path,
        name=name,
    )


@lru_cache
def get_next_stage() -> NextStageClient:
    return _next_stage(STRUCTURING_JOBS_PATH, "structuring service")


@lru_cache
def get_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(), stage=STAGE_OCR, table_prefix=DB_TABLE_PREFIX, next_stage=get_next_stage()
    )


@lru_cache
def get_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_pipeline())


# The testing endpoints (TESTING_ENDPOINTS): the same pipeline on the testing_* tables, handing off to
# structuring's `-test` endpoint, without callbacks (see ocr_common.testing_endpoints).


@lru_cache
def get_testing_next_stage() -> NextStageClient:
    return _next_stage(testing_path(STRUCTURING_JOBS_PATH), "structuring service (testing)")


@lru_cache
def get_testing_pipeline() -> StagePipeline:
    return build_stage_pipeline(
        get_settings(),
        stage=STAGE_OCR,
        table_prefix=DB_TABLE_PREFIX,
        next_stage=get_testing_next_stage(),
        testing=True,
    )


@lru_cache
def get_testing_relay() -> OutboxRelay | None:
    return build_outbox_relay(get_settings(), get_testing_pipeline())


# --- services (cheap to build: one per request) ------------------------------------------


def get_extraction_service() -> ExtractionService:
    return ExtractionService(get_ocr_engine(), get_settings())


def _job_service(pipeline: StagePipeline) -> ExtractionJobService:
    settings = get_settings()
    return ExtractionJobService(
        pipeline,
        get_extraction_service(),
        settings.max_upload_bytes,
        url_policy=settings.file_url_policy,
        simulate_delay=settings.is_local,
        handoff_by_reference=settings.pipeline_handoff_by_reference,
    )


def get_job_service() -> ExtractionJobService:
    return _job_service(get_pipeline())


def get_testing_job_service() -> ExtractionJobService:
    return _job_service(get_testing_pipeline())


@lru_cache
def get_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_pipeline(), get_job_service().resume)


@lru_cache
def get_testing_reaper() -> StaleJobReaper | None:
    return build_stale_job_reaper(get_settings(), get_testing_pipeline(), get_testing_job_service().resume)
