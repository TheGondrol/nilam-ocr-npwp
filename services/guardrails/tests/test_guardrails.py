import io

import pytest
from PIL import Image

from ocr_common.errors import ServiceError
from ocr_common.testing import image_upload
from src.core.config import Settings
from src.services.guardrails_service import GuardrailsService
from src.services.pages import render_pages


class StubClassifier:
    reject_threshold = 0.5

    def __init__(self, *predictions):
        self._predictions = list(predictions)
        self.seen_pages = 0

    def classify(self, filename, pages):
        self.seen_pages = len(pages)
        return self._predictions[: len(pages)]


def _settings(**overrides) -> Settings:
    return Settings(api_key="x", _env_file=None, **overrides)


def _jpeg(width=200, height=100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _pdf(n_pages: int) -> bytes:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for i in range(n_pages):
        page = document.new_page(width=300, height=200)
        page.insert_text((20, 40), f"halaman {i + 1}")
    return document.tobytes()


def test_image_is_one_page():
    pages = render_pages("image/jpeg", _jpeg(), dpi=150, max_pages=20)
    assert len(pages) == 1
    assert pages[0].mode == "RGB"


def test_pdf_is_rendered_per_page_up_to_max_pages():
    pages = render_pages("application/pdf", _pdf(3), dpi=72, max_pages=2)
    assert len(pages) == 2
    assert pages[0].size == (300, 200)


def test_unreadable_image_is_400():
    with pytest.raises(ServiceError) as exc:
        render_pages("image/jpeg", b"not really an image", dpi=150, max_pages=20)
    assert exc.value.status_code == 400
    assert exc.value.message == "Uploaded file is not a readable image"


def test_unreadable_pdf_is_400():
    with pytest.raises(ServiceError) as exc:
        render_pages("application/pdf", b"%PDF-1.4 broken", dpi=150, max_pages=20)
    assert exc.value.status_code == 400


async def test_all_pages_accepted_gives_accepted_with_weakest_page_confidence():
    classifier = StubClassifier((0.99, 0.01), (0.80, 0.20), (0.95, 0.05))
    report = await GuardrailsService(classifier, _settings()).check("a.pdf", "application/pdf", _pdf(3))
    assert classifier.seen_pages == 3
    assert report["document"] == {
        "verdict": "accepted",
        "confidence": 0.80,
        "n_pages": 3,
        "n_approve": 3,
        "n_reject": 0,
    }
    assert [p["verdict"] for p in report["pages"]] == ["accepted"] * 3
    assert report["pages"][1] == {"page_index": 1, "proba_approve": 0.80, "proba_reject": 0.20, "verdict": "accepted"}


async def test_policy_all_rejects_document_when_one_page_rejected():
    classifier = StubClassifier((0.99, 0.01), (0.30, 0.70))
    report = await GuardrailsService(classifier, _settings()).check("a.pdf", "application/pdf", _pdf(2))
    assert report["document"] == {"verdict": "reject", "confidence": 0.70, "n_pages": 2, "n_approve": 1, "n_reject": 1}


async def test_policy_majority_accepts_when_more_pages_accepted():
    classifier = StubClassifier((0.99, 0.01), (0.30, 0.70), (0.90, 0.10))
    settings = _settings(guardrails_document_policy="majority")
    report = await GuardrailsService(classifier, settings).check("a.pdf", "application/pdf", _pdf(3))
    assert report["document"]["verdict"] == "accepted"
    assert report["document"]["n_reject"] == 1


async def test_threshold_from_settings_overrides_classifier_threshold():
    classifier = StubClassifier((0.70, 0.30))
    strict = GuardrailsService(classifier, _settings(guardrails_reject_threshold=0.25))
    assert (await strict.check("a.jpg", "image/jpeg", _jpeg()))["document"]["verdict"] == "reject"
    default = GuardrailsService(classifier, _settings())
    assert (await default.check("a.jpg", "image/jpeg", _jpeg()))["document"]["verdict"] == "accepted"


async def test_unsupported_content_type_is_400():
    with pytest.raises(ServiceError) as exc:
        await GuardrailsService(StubClassifier(), _settings()).check("a.txt", "text/plain", b"x")
    assert exc.value.status_code == 400


def test_health_lists_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["backends"] == {"guardrails": "mock"}


def test_http_check_returns_contract_shape(client, auth):
    response = client.post(
        "/v1/extract-ocr", data={"request_id": "OCR_1"}, files=image_upload("npwp.jpg", _jpeg()), headers=auth
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status_code"] == 200
    assert body["message"] == "OK"
    assert body["request_id"] == "OCR_1"
    assert body["data"] == {
        "passed": True,
        "reason": None,
        "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 1, "n_approve": 1, "n_reject": 0},
        "pages": [{"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}],
        "job": {"request_id": "OCR_1", "stage": "OCR", "status": "PROCESSING", "duplicate": False},
    }


def test_http_check_pdf_reports_every_page(client, auth):
    response = client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_2"},
        files=image_upload("scan.pdf", _pdf(2), "application/pdf"),
        headers=auth,
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["n_pages"] == 2
    assert [p["page_index"] for p in data["pages"]] == [0, 1]


@pytest.mark.parametrize("filename", ["blur.jpg", "invalid.jpg", "notnpwp.jpg"])
def test_http_mock_scenarios_reject_with_200(client, auth, filename):
    response = client.post(
        "/v1/extract-ocr", data={"request_id": "OCR_3"}, files=image_upload(filename, _jpeg()), headers=auth
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document"]["verdict"] == "reject"
    assert data["document"]["n_reject"] == 1
    assert data["passed"] is False
    assert data["reason"] == "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)"


def test_http_request_id_is_required(client, auth):
    response = client.post("/v1/extract-ocr", files=image_upload("npwp.jpg", _jpeg()), headers=auth)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"] == "body.request_id: Field required"


def test_http_file_url_is_fetched_by_service(client, auth, monkeypatch):
    async def fake_fetch(url, *, limit, timeout=10.0):
        assert url == "http://minio.local/bucket/npwp.jpg"
        return _jpeg(), "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("ocr_common.intake.fetch", fake_fetch)
    response = client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_4", "file_url": "http://minio.local/bucket/npwp.jpg"},
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["data"]["document"]["verdict"] == "accepted"


def test_unsupported_content_type_returns_400(client, auth):
    response = client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_5"},
        files=image_upload(content_type="text/plain"),
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["message"].startswith("Unsupported content type")


def test_unreadable_image_returns_400(client, auth):
    response = client.post(
        "/v1/extract-ocr", data={"request_id": "OCR_6"}, files=image_upload("x.jpg", b"garbage"), headers=auth
    )
    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is not a readable image"


def test_missing_api_key_returns_401_envelope(client):
    response = client.post("/v1/extract-ocr", data={"request_id": "OCR_7"}, files=image_upload())
    assert response.status_code == 401
    assert response.json()["errors"] == "Invalid or missing API key"
