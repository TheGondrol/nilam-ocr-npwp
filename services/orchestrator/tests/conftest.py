import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(AUTH_DISABLED="false")

from app.dependencies import (  # noqa: E402
    get_extraction_client,
    get_guardrails_client,
    get_guardrails_log,
    get_pipeline_waiter,
    get_testing_guardrails_log,
)
from app.main import app  # noqa: E402
from app.services.pipeline_waiter import WaitOutcome  # noqa: E402

# Only the content type and the size of a document are checked here; the bytes never reach a model.
JPEG = b"\xff\xd8fake-jpeg-bytes"

ACCEPTED_REPORT = {
    "passed": True,
    "reason": None,
    "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 1, "n_approve": 1, "n_reject": 0},
    "pages": [{"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}],
}
REJECTED_REPORT = {
    "passed": False,
    "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
    "document": {"verdict": "reject", "confidence": 0.8821, "n_pages": 1, "n_approve": 0, "n_reject": 1},
    "pages": [{"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}],
}

STRUCTURING_RESULT = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.99,
            "source": "NPWP : 12.345.678.9-012.345",
            "signals": {"has_homoglyph": False, "candidate_count": 1},
        },
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.97, "source": "NAMA : BUDI SANTOSO"},
        "nama_badan": {"value": None, "confidence": 0.0, "source": None},
    },
    "flag": False,
    "flag_reason": None,
}
SCORING_RESULT = {"npwp_confidence": 0.7296, "name_confidence": 0.9471, "payload": {"npwp": "123456789012345"}}
DONE = WaitOutcome(
    "SCORING",
    "DONE",
    results={"OCR": {"blocks": []}, "STRUCTURING": STRUCTURING_RESULT, "SCORING": SCORING_RESULT},
)


class StubGuardrails:
    """The guardrails service, answering like its mock model: a file name containing `blur`, `invalid` or
    `notnpwp` is rejected, anything else accepted. Set `error` to make it fail instead."""

    def __init__(self) -> None:
        self.checked: list[dict] = []
        self.error: Exception | None = None

    async def check(self, request_id, filename, content_type, content, threshold=None) -> dict:
        call = {"request_id": request_id, "filename": filename, "content_type": content_type}
        if threshold is not None:
            call["threshold"] = threshold
        self.checked.append(call)
        if self.error is not None:
            raise self.error
        rejected = any(word in filename for word in ("blur", "invalid", "notnpwp"))
        return REJECTED_REPORT if rejected else ACCEPTED_REPORT

    async def aclose(self) -> None:
        pass


class StubExtraction:
    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.error: Exception | None = None

    async def submit(
        self,
        request_id,
        document_type,
        guardrails,
        filename,
        content_type,
        content,
        *,
        file_url=None,
        sequence=None,
        column_thresholds=None,
    ) -> dict:
        if self.error is not None:
            raise self.error
        self.submitted.append(
            {
                "request_id": request_id,
                "document_type": document_type,
                "guardrails": guardrails,
                "file_url": file_url,
                "sequence": list(sequence) if sequence else None,
                "column_thresholds": column_thresholds,
            }
        )
        return {"request_id": request_id, "stage": "OCR", "status": "PROCESSING", "duplicate": False}

    async def aclose(self) -> None:
        pass


class RecordingGuardrailsLog:
    """Keeps the verdicts the service asks to record, instead of writing nilam_guardrails_results."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    async def record(self, request_id, report, *, threshold_from_request, sequence=None) -> None:
        self.records.append(
            {
                "request_id": request_id,
                "report": report,
                "threshold_from_request": threshold_from_request,
                "sequence": list(sequence) if sequence else None,
            }
        )

    async def latest(self, request_id):
        mine = [record for record in self.records if record["request_id"] == request_id]
        return {"report": mine[-1]["report"], "sequence": mine[-1]["sequence"]} if mine else None


class RecordingResponseLog:
    """Keeps the answers the response log middleware hands over, instead of writing nilam_ocr_results."""

    def __init__(self, table_prefix: str = "") -> None:
        self.table_prefix = table_prefix
        self.records: list[dict] = []

    async def record(self, request_id, status_code, body, *, guardrails) -> None:
        self.records.append(
            {"request_id": request_id, "status_code": status_code, "body": body, "guardrails": guardrails}
        )


class StubWaiter:
    def __init__(self) -> None:
        self.outcome = DONE
        self.snapshot_outcome: WaitOutcome | None = DONE
        self.snapshot_error: Exception | None = None
        self.calls: list[tuple[str, float]] = []
        self.last_stages: list[str | None] = []
        self.snapshots: list[str] = []

    async def wait(self, request_id: str, timeout: float, *, last_stage: str | None = None) -> WaitOutcome:
        self.calls.append((request_id, timeout))
        self.last_stages.append(last_stage)
        return self.outcome

    async def snapshot(self, request_id: str) -> WaitOutcome | None:
        self.snapshots.append(request_id)
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot_outcome


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()


@pytest.fixture(autouse=True)
def stub_guardrails():
    stub = StubGuardrails()
    app.dependency_overrides[get_guardrails_client] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_guardrails_client, None)


@pytest.fixture(autouse=True)
def stub_extraction():
    stub = StubExtraction()
    app.dependency_overrides[get_extraction_client] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_extraction_client, None)


@pytest.fixture(autouse=True)
def stub_waiter():
    stub = StubWaiter()
    app.dependency_overrides[get_pipeline_waiter] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_pipeline_waiter, None)


@pytest.fixture(autouse=True)
def guardrails_log():
    log = RecordingGuardrailsLog()
    app.dependency_overrides[get_guardrails_log] = lambda: log
    app.dependency_overrides[get_testing_guardrails_log] = lambda: log
    yield log
    app.dependency_overrides.pop(get_guardrails_log, None)
    app.dependency_overrides.pop(get_testing_guardrails_log, None)


@pytest.fixture(autouse=True)
def response_logs():
    """The live (key "") and testing ("testing_") response logs, recording instead of writing."""
    before = app.state.response_logs
    logs = {"": RecordingResponseLog(), "testing_": RecordingResponseLog("testing_")}
    app.state.response_logs = logs
    yield logs
    app.state.response_logs = before
