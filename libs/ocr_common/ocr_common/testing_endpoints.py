"""Names of the testing endpoints (`TESTING_ENDPOINTS`), shared by the four services.

Each endpoint of the pipeline has a `-test` twin that runs the same code on copies of the stage tables
(`testing_ocr_jobs`, ..., `testing_pipeline_outbox`), without callbacks and without writes to the
orchestrator's tables. The ML team uses them for load tests on dev. No dependency beyond the standard
library: guardrails imports it without SQLAlchemy.
"""

TESTING_TABLE_PREFIX = "testing_"
TESTING_PATH_SUFFIX = "-test"
TESTING_TAG = "Testing"


def testing_path(path: str) -> str:
    """The testing twin of an endpoint path: `/v1/structuring/jobs` -> `/v1/structuring/jobs-test`."""
    return f"{path}{TESTING_PATH_SUFFIX}"


def testing_metrics_stage(stage: str) -> str:
    """The `stage` label of the testing pipeline's metrics: `OCR` -> `TESTING_OCR`."""
    return f"TESTING_{stage}"
