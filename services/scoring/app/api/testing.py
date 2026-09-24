from ocr_common.web.testing_routes import build_testing_router

from app.api import jobs
from app.dependencies import get_testing_job_service

# Registered only with TESTING_ENDPOINTS (see app.main).
router = build_testing_router(
    jobs.router,
    {
        "/v1/scoring/jobs": "/v1/scoring/jobs-test",
        "/v1/scoring/jobs/{request_id}": "/v1/scoring/jobs-test/{request_id}",
    },
    service=get_testing_job_service,
)
