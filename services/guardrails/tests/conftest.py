import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(GUARDRAILS_BACKEND="mock", AUTH_DISABLED="false")

from src.clients.ekstraksi import get_ekstraksi_client  # noqa: E402
from src.main import app  # noqa: E402


class StubEkstraksi:
    def __init__(self) -> None:
        self.submitted: list[dict] = []

    async def submit(self, request_id, document_type, guardrails, filename, content_type, content) -> dict:
        self.submitted.append({"request_id": request_id, "document_type": document_type, "guardrails": guardrails})
        return {"request_id": request_id, "stage": "OCR", "status": "PROCESSING", "duplicate": False}

    async def aclose(self) -> None:
        pass


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
