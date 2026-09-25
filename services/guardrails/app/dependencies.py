"""Composition root: the one place that decides which implementation of each part runs.

Every `get_*` here is what the routes take through `Depends(...)` and what tests replace through
`app.dependency_overrides[...]`. Nothing else in the service builds these objects."""

from functools import lru_cache

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.registry import Factory, build_backend

from app.clients.reject_threshold import RejectThreshold, default_threshold
from app.config import Settings, get_settings
from app.ml.base import Classifier
from app.ml.efficientnet import EfficientNetPageClassifier
from app.ml.mock import MockPageClassifier
from app.ml.remote import RemoteGuardrailsModel
from app.services.guardrails_service import GuardrailsService


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


# --- clients ------------------------------------------------------------------------------


@lru_cache
def get_reject_threshold() -> RejectThreshold:
    """The reject threshold from the central orchestrator (one per process: it holds the cache)."""
    settings: Settings = get_settings()
    client = None
    if settings.guardrails_threshold_url:
        headers = (
            {"X-API-Key": settings.guardrails_threshold_api_key} if settings.guardrails_threshold_api_key else None
        )
        client = RemoteModelClient(
            settings.guardrails_threshold_url,
            settings.guardrails_threshold_timeout_seconds,
            name="orchestrator reject threshold",
            headers=headers,
        )
    return RejectThreshold(
        client,
        settings.guardrails_threshold_path,
        default_threshold(settings.guardrails_reject_threshold, get_page_classifier()),
        cache_seconds=settings.guardrails_threshold_cache_seconds,
    )


# --- services (cheap to build: one per request) ------------------------------------------


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_page_classifier(), get_settings(), get_reject_threshold())
