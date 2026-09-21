import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(GUARDRAILS_BACKEND="mock", AUTH_DISABLED="false")

from src.api.v1.guardrails import get_pipeline_waiter  # noqa: E402
from src.clients.ekstraksi import get_ekstraksi_client  # noqa: E402
from src.main import app  # noqa: E402
from src.services.pipeline_waiter import WaitOutcome  # noqa: E402

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
}
SCORING_RESULT = {"npwp_confidence": 0.7296, "name_confidence": 0.9471, "payload": {"npwp": "123456789012345"}}


class StubEkstraksi:
    def __init__(self) -> None:
        self.submitted: list[dict] = []

    async def submit(self, request_id, document_type, guardrails, filename, content_type, content) -> dict:
        self.submitted.append({"request_id": request_id, "document_type": document_type, "guardrails": guardrails})
        return {"request_id": request_id, "stage": "OCR", "status": "PROCESSING", "duplicate": False}

    async def aclose(self) -> None:
        pass


class StubWaiter:
    def __init__(self) -> None:
        self.outcome = WaitOutcome(
            "SCORING",
            "DONE",
            results={"OCR": {"blocks": []}, "STRUCTURING": STRUCTURING_RESULT, "SCORING": SCORING_RESULT},
        )
        self.calls: list[tuple[str, float]] = []

    async def wait(self, request_id: str, timeout: float) -> WaitOutcome:
        self.calls.append((request_id, timeout))
        return self.outcome


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()


@pytest.fixture(autouse=True)
def stub_ekstraksi():
    stub = StubEkstraksi()
    app.dependency_overrides[get_ekstraksi_client] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_ekstraksi_client, None)


@pytest.fixture(autouse=True)
def stub_waiter():
    stub = StubWaiter()
    app.dependency_overrides[get_pipeline_waiter] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_pipeline_waiter, None)
