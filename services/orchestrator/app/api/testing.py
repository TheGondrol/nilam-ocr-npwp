from uuid import uuid4

from fastapi import APIRouter, Form

from ocr_common.web.testing_routes import build_testing_router

from app.api import extract_ocr
from app.dependencies import get_testing_extract_service

TESTING_REQUEST_ID_PREFIX = "TEST_"


def testing_request_id(
    run_id: str | None = Form(
        None,
        description=(
            "Optional name of the load-test run, put into the generated request_id "
            "(`TEST_<run_id>_<uuid>`) so one run's requests can be selected together"
        ),
        pattern=r"^[A-Za-z0-9_-]{1,40}$",
        examples=["burst5rps"],
    ),
) -> str:
    """The testing endpoint mints its own request_id instead of taking the central orchestrator's: a new one
    per request, so a load test never collides with an earlier run (the pipeline is idempotent per id)."""
    run = f"{run_id}_" if run_id else ""
    return f"{TESTING_REQUEST_ID_PREFIX}{run}{uuid4().hex}"


# Registered only with TESTING_ENDPOINTS (see app.main). Two builds, because the POST's request_id is
# minted here while the GET's is its path parameter and must stay one.
routers: list[APIRouter] = [
    build_testing_router(
        extract_ocr.router,
        {"/v1/extract-ocr": "/v1/extract-ocr-test"},
        note=(
            "**request_id is generated here**, not sent: `TEST_<uuid>`, or `TEST_<run_id>_<uuid>` with the "
            "optional `run_id` field. The response's `request_id` is the one to look the request up by, with "
            "`GET /v1/extract-ocr-test/{request_id}`. A `request_id` field in the form is ignored."
        ),
        service=get_testing_extract_service,
        request_id=testing_request_id,
    ),
    build_testing_router(
        extract_ocr.router,
        {"/v1/extract-ocr/{request_id}": "/v1/extract-ocr-test/{request_id}"},
        note="Reads the `-test` jobs of a request_id minted by `POST /v1/extract-ocr-test`.",
        service=get_testing_extract_service,
    ),
]
