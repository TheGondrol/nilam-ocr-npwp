import pytest

from ocr_common.errors import ServiceError, UpstreamUnavailable
from ocr_common.testing import image_upload

from app.config import get_settings
from app.main import app
from app.services.pipeline_waiter import STATUS_REJECTED, WaitOutcome
from tests.conftest import JPEG

TOO_MANY_PAGES = "Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP"


def _submit(client, auth, filename="npwp.jpg", content=JPEG, content_type="image/jpeg", **data):
    return client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_1", **data},
        files=image_upload(filename, content, content_type),
        headers=auth,
    )


def test_health_has_no_backends(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["backends"] == {}


def test_extract_ocr_follows_the_central_orchestrators_contract(client, auth, stub_guardrails, stub_ekstraksi):
    response = _submit(client, auth)

    assert response.status_code == 200
    assert response.json() == {
        "status_code": 200,
        "status_desc": "OK",
        "message": "OCR extraction completed successfully",
        "data": {
            "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
            "nama": {"value": "BUDI SANTOSO", "confidence": 1},
        },
        "errors": None,
        "request_id": "OCR_1",
        "document_type": "npwp",
        "job_status": "completed",
        "guardrails": 0,
        "params": None,
    }
    assert stub_guardrails.checked == [{"request_id": "OCR_1", "filename": "npwp.jpg", "content_type": "image/jpeg"}]
    [handed] = stub_ekstraksi.submitted
    assert handed["guardrails"]["passed"] is True


def test_rejection_by_the_guardrails_model_is_400_with_guardrails_0(client, auth, stub_ekstraksi, stub_waiter):
    response = _submit(client, auth, filename="notnpwp.jpg")

    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "status_desc": "Bad Request",
        "message": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
        "data": None,
        "errors": "DOWNSTREAM_VALIDATION_ERROR",
        "request_id": "OCR_1",
        "document_type": "npwp",
        "job_status": "failed",
        "guardrails": 1,
        "params": None,
    }
    assert stub_ekstraksi.submitted == [] and stub_waiter.calls == []


def test_request_id_is_required(client, auth):
    response = client.post("/v1/extract-ocr", files=image_upload("npwp.jpg", JPEG), headers=auth)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"] == "body.request_id: Field required"


def test_file_url_is_fetched_here_and_forwarded_as_url(client, auth, monkeypatch, stub_ekstraksi):
    async def fake_fetch(url, *, limit, timeout=10.0, policy):
        assert url == "http://minio.local/bucket/npwp.jpg"
        return JPEG, "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("ocr_common.web.intake.fetch", fake_fetch)
    response = client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_4", "file_url": "http://minio.local/bucket/npwp.jpg"},
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["job_status"] == "completed"
    assert stub_ekstraksi.submitted[0]["file_url"] == "http://minio.local/bucket/npwp.jpg"


def test_unsupported_content_type_is_400_before_guardrails(client, auth, stub_guardrails):
    response = _submit(client, auth, content_type="text/plain")
    assert response.status_code == 400
    assert response.json()["message"].startswith("Unsupported content type")
    assert stub_guardrails.checked == []


def test_oversized_document_is_413_before_guardrails(client, auth, stub_guardrails):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"max_upload_bytes": 10})
    try:
        response = _submit(client, auth)
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 413
    body = response.json()
    assert body["status_desc"] == "Payload Too Large"
    assert body["message"].startswith("Ukuran dokumen melebihi batas")
    assert stub_guardrails.checked == []


def _pdf(n_pages: int) -> bytes:
    import fitz

    document = fitz.open()
    for i in range(n_pages):
        document.new_page(width=300, height=200).insert_text((20, 40), f"halaman {i + 1}")
    return document.tobytes()


def _submit_pdf(client, auth, content):
    return _submit(client, auth, filename="scan.pdf", content=content, content_type="application/pdf")


def test_more_than_two_pages_is_400_before_guardrails(client, auth, stub_guardrails, stub_ekstraksi):
    response = _submit_pdf(client, auth, _pdf(3))

    assert response.status_code == 400
    body = response.json()
    assert (body["message"], body["errors"]) == (TOO_MANY_PAGES, TOO_MANY_PAGES)
    assert "job_status" not in body
    assert stub_guardrails.checked == [] and stub_ekstraksi.submitted == []


def test_two_pages_are_within_the_limit(client, auth, stub_guardrails):
    response = _submit_pdf(client, auth, _pdf(2))

    assert response.status_code == 200
    assert len(stub_guardrails.checked) == 1


def test_the_page_limit_is_a_setting(client, auth, stub_guardrails):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"max_document_pages": 1})
    try:
        response = _submit_pdf(client, auth, _pdf(2))
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 400
    assert response.json()["message"] == TOO_MANY_PAGES
    assert stub_guardrails.checked == []


def test_unreadable_pdf_is_400_before_guardrails(client, auth, stub_guardrails):
    response = _submit_pdf(client, auth, b"%PDF-1.4 garbage")

    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is not a readable PDF"
    assert stub_guardrails.checked == []


def test_a_refusal_of_the_guardrails_service_is_answered_as_it_is(client, auth, stub_guardrails, stub_ekstraksi):
    stub_guardrails.error = ServiceError(400, "Uploaded file is not a readable image")

    response = _submit(client, auth)

    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is not a readable image"
    assert stub_ekstraksi.submitted == []


def test_guardrails_unreachable_is_503_and_nothing_starts(client, auth, stub_guardrails, stub_ekstraksi):
    stub_guardrails.error = UpstreamUnavailable("guardrails service is unavailable")

    response = _submit(client, auth)

    assert response.status_code == 503
    assert response.json()["message"] == "guardrails service is unavailable"
    assert stub_ekstraksi.submitted == []


@pytest.fixture
def settings_override():
    def install(**update):
        app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update=update)

    yield install
    app.dependency_overrides.pop(get_settings, None)


def test_skip_guardrails_is_refused_with_403_while_not_allowed(client, auth, stub_guardrails, stub_ekstraksi):
    response = _submit(client, auth, skip_guardrails="true")

    assert response.status_code == 403
    assert response.json() == {
        "status_code": 403,
        "status_desc": "Forbidden",
        "message": "skip_guardrails is not allowed here: GUARDRAILS_SKIP_ALLOWED is off",
        "data": None,
        "errors": "GUARDRAILS_SKIP_NOT_ALLOWED",
        "request_id": "OCR_1",
        "document_type": "npwp",
        "job_status": None,
        "guardrails": None,
        "params": None,
    }
    assert stub_guardrails.checked == [] and stub_ekstraksi.submitted == []


def test_skip_guardrails_false_is_never_refused(client, auth, stub_guardrails):
    response = _submit(client, auth, skip_guardrails="false")

    assert response.status_code == 200
    assert len(stub_guardrails.checked) == 1


def test_skip_guardrails_must_be_a_boolean(client, auth, stub_guardrails):
    response = _submit(client, auth, skip_guardrails="maybe")

    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"
    assert stub_guardrails.checked == []


def test_skipped_guardrails_hand_the_document_on_without_a_report(
    client, auth, settings_override, stub_guardrails, stub_ekstraksi
):
    settings_override(guardrails_skip_allowed=True)

    # A file name the guardrails model rejects: with the check skipped it never gets to judge it.
    response = _submit(client, auth, filename="notnpwp.jpg", skip_guardrails="true")

    assert response.status_code == 200
    body = response.json()
    assert (body["job_status"], body["guardrails"], body["errors"]) == ("completed", 0, None)
    assert stub_guardrails.checked == []
    [handed] = stub_ekstraksi.submitted
    assert handed["guardrails"] is None


def test_with_guardrails_skipped_the_structuring_rules_still_reject(client, auth, settings_override, stub_waiter):
    settings_override(guardrails_skip_allowed=True)
    stub_waiter.outcome = WaitOutcome("STRUCTURING", STATUS_REJECTED, "dokumen blur / blank")

    response = _submit(client, auth, skip_guardrails="true")

    assert response.status_code == 400
    body = response.json()
    assert (body["message"], body["errors"], body["guardrails"]) == (
        "dokumen blur / blank",
        "DOWNSTREAM_VALIDATION_ERROR",
        1,
    )


def test_with_guardrails_skipped_the_file_checks_still_run(client, auth, settings_override, stub_ekstraksi):
    settings_override(guardrails_skip_allowed=True)
    pages = _submit(
        client, auth, filename="scan.pdf", content=_pdf(3), content_type="application/pdf", skip_guardrails="true"
    )
    settings_override(guardrails_skip_allowed=True, max_upload_bytes=10)
    size = _submit(client, auth, skip_guardrails="true")

    assert (pages.status_code, pages.json()["message"]) == (400, TOO_MANY_PAGES)
    assert size.status_code == 413
    assert stub_ekstraksi.submitted == []


def test_missing_api_key_returns_401_envelope(client):
    response = client.post("/v1/extract-ocr", data={"request_id": "OCR_7"}, files=image_upload())
    assert response.status_code == 401
    assert response.json()["errors"] == "Invalid or missing API key"
