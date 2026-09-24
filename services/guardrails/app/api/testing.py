from uuid import uuid4

from fastapi import Form

from ocr_common.web.testing_routes import build_testing_router

from app.api import guardrails
from app.dependencies import get_testing_job_service

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
    """The testing endpoint mints its own request_id instead of taking the orchestrator's: a new one per
    request, so a load test never collides with an earlier run (the pipeline is idempotent per id)."""
    run = f"{run_id}_" if run_id else ""
    return f"{TESTING_REQUEST_ID_PREFIX}{run}{uuid4().hex}"


# Registered only with TESTING_ENDPOINTS (see app.main).
router = build_testing_router(
    guardrails.router,
    {"/v1/extract-ocr": "/v1/extract-ocr-test"},
    note=(
        "**request_id is generated here**, not sent: `TEST_<uuid>`, or `TEST_<run_id>_<uuid>` with the optional "
        "`run_id` field. The response's `request_id` is the one to look the request up by. A `request_id` field "
        "in the form is ignored."
    ),
    service=get_testing_job_service,
    request_id=testing_request_id,
)
