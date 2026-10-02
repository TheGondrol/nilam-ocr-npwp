from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.pipeline.schemas import ScoringStageCallback
from ocr_common.web.app import add_stage_callback_webhook, create_app, database_readiness

from app.api import direct, jobs, scoring, testing
from app.config import get_settings
from app.dependencies import (
    get_pipeline,
    get_reaper,
    get_relay,
    get_testing_pipeline,
    get_testing_reaper,
    get_testing_relay,
    get_trust_model,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_trust_model()
    if settings.database_url:
        from ocr_common.pipeline.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    relay = get_relay()
    if relay is not None:
        relay.start()
    reaper = get_reaper()
    if reaper is not None:
        reaper.start()
    testing_relay = testing_reaper = None
    if settings.testing_endpoints:
        testing_relay, testing_reaper = get_testing_relay(), get_testing_reaper()
        if testing_relay is not None:
            testing_relay.start()
        if testing_reaper is not None:
            testing_reaper.start()
    yield
    if reaper is not None:
        await reaper.stop()
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds, relay=relay)
    if settings.testing_endpoints:
        if testing_reaper is not None:
            await testing_reaper.stop()
        await get_testing_pipeline().aclose(settings.pipeline_drain_timeout_seconds, relay=testing_relay)
    if settings.database_url:
        from ocr_common.pipeline.database import dispose_engines

        await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP Scoring API",
    service_name="scoring",
    description=(
        "Pipeline step 4 (last) for Indonesian NPWP documents: the ML team's trust model gives each extracted "
        "field (nomor_npwp, nama) the probability that it is correct. "
        "**Async pipeline:** the structuring service POSTs /v1/scoring/jobs and gets 202; this service "
        "scores in the background, stores the result, and POSTs the stage callback carrying the final "
        "result to the orchestrator. /v1/scoring-direct scores a structuring result synchronously, for testing "
        "this stage alone; /v1/scoring/confidence runs the trust model alone on the ML team's payload. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Asynchronous pipeline stage: 202, background work, callback"},
        {"name": "Callbacks", "description": "Requests this service SENDS to the orchestrator (see Webhooks)"},
        {
            "name": "Direct",
            "description": "This stage alone on the previous stages' outputs, synchronous: nothing recorded (QC)",
        },
        {"name": "Scoring", "description": "The trust model on one payload, synchronous (debugging)"},
    ],
    routers=[jobs.router, direct.router, scoring.router, *([testing.router] if settings.testing_endpoints else [])],
    backends={
        "scoring": "trust_model",
        "storage": "postgres" if settings.database_url else "memory",
    },
    readiness=database_readiness(settings.database_url),
    health=database_readiness(settings.database_url),
    backends_example={"scoring": "trust_model", "storage": "postgres"},
    readiness_example={"database": "ok"},
    lifespan=lifespan,
)

add_stage_callback_webhook(
    app,
    body_model=ScoringStageCallback,
    sent=(
        "Once per job of `POST /v1/scoring/jobs`, the last callback of a request: `stage: SCORING` with "
        "`status: DONE` and `result` = the **final result** (fields, per-field confidences, and the guardrails "
        "result that was submitted), or `status: FAILED` with `error_message` and `result: null`."
    ),
)
