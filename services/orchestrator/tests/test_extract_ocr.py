import pytest

from ocr_common.errors import ServiceError, UpstreamUnavailable
from ocr_common.testing import image_upload

from app.config import get_settings
from app.main import app
from app.services.pipeline_waiter import STATUS_REJECTED, WaitOutcome
from tests.conftest import ACCEPTED_REPORT, JPEG, STRUCTURING_RESULT, WITH_GUARDRAILS_THRESHOLD

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


def test_extract_ocr_follows_the_central_orchestrators_contract(client, auth, stub_guardrails, stub_extraction):
    response = _submit(client, auth)

    # No thresholds sent: the trust model's probabilities as the confidences, the accepted probability as guardrails.
    assert response.status_code == 200
    assert response.json() == {
        "status_code": 200,
        "status_desc": "OK",
        "message": "OCR extraction completed successfully",
        "data": {
            "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.7296},
            "nama": {"value": "BUDI SANTOSO", "confidence": 0.9471},
        },
        "errors": None,
        "request_id": "OCR_1",
        "guardrails": 0.9821,
        "pipeline_last_stage": None,
    }
    assert stub_guardrails.checked == [{"request_id": "OCR_1", "filename": "npwp.jpg", "content_type": "image/jpeg"}]
    [handed] = stub_extraction.submitted
    assert (handed["guardrails"]["passed"], handed["guardrails"]["auto_accepted"]) == (True, True)


def test_with_both_thresholds_the_answer_has_0_1_flags(client, auth, stub_guardrails):
    response = _submit(client, auth, **WITH_GUARDRAILS_THRESHOLD, column_confidence_threshold='{"all_field": 0.8}')

    body = response.json()
    assert (response.status_code, body["guardrails"]) == (200, 0)
    assert body["data"] == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    }
    assert stub_guardrails.checked[0]["threshold"] == 0.5


def test_without_a_guardrails_threshold_a_document_the_model_rejects_is_accepted(client, auth, stub_extraction):
    response = _submit(client, auth, filename="notnpwp.jpg")

    # The model's accepted probability of that document (its only page) is the answer's guardrails.
    assert (response.status_code, response.json()["guardrails"]) == (200, 0.1179)
    [handed] = stub_extraction.submitted
    assert (handed["guardrails"]["document"]["verdict"], handed["guardrails"]["score"]) == ("reject", 0.1179)


def test_rejection_by_the_guardrails_model_is_400_with_guardrails_0(client, auth, stub_extraction, stub_waiter):
    response = _submit(client, auth, filename="notnpwp.jpg", **WITH_GUARDRAILS_THRESHOLD)

    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "status_desc": "Bad Request",
        "message": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
        "data": None,
        "errors": "DOWNSTREAM_VALIDATION_ERROR",
        "request_id": "OCR_1",
        "guardrails": 1,
        "pipeline_last_stage": "guardrails",
    }
    assert stub_extraction.submitted == [] and stub_waiter.calls == []


def test_request_id_is_required(client, auth):
    response = client.post("/v1/extract-ocr", files=image_upload("npwp.jpg", JPEG), headers=auth)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"] == "body.request_id: Field required"


def test_file_url_is_fetched_here_and_forwarded_as_url(client, auth, monkeypatch, stub_extraction):
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
    assert response.status_code == 200
    assert stub_extraction.submitted[0]["file_url"] == "http://minio.local/bucket/npwp.jpg"


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


def test_more_than_two_pages_is_400_before_guardrails(client, auth, stub_guardrails, stub_extraction):
    response = _submit_pdf(client, auth, _pdf(3))

    assert response.status_code == 400
    body = response.json()
    assert (body["message"], body["errors"]) == (TOO_MANY_PAGES, "TOO_MANY_PAGES")
    assert "guardrails" not in body, "a plain error envelope"
    assert stub_guardrails.checked == [] and stub_extraction.submitted == []


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


def test_a_refusal_of_the_guardrails_service_is_answered_as_it_is(client, auth, stub_guardrails, stub_extraction):
    stub_guardrails.error = ServiceError(400, "Uploaded file is not a readable image")

    response = _submit(client, auth)

    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is not a readable image"
    assert stub_extraction.submitted == []


def test_guardrails_unreachable_is_503_and_nothing_starts(client, auth, stub_guardrails, stub_extraction):
    stub_guardrails.error = UpstreamUnavailable("guardrails service is unavailable")

    response = _submit(client, auth)

    assert response.status_code == 503
    assert response.json()["message"] == "guardrails service is unavailable"
    assert stub_extraction.submitted == []


@pytest.fixture
def settings_override():
    def install(**update):
        app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update=update)

    yield install
    app.dependency_overrides.pop(get_settings, None)


NO_GUARDRAILS = ["extraction", "structuring", "scoring"]
FULL = ["guardrails", "extraction", "structuring", "scoring"]


def test_without_a_sequence_the_whole_pipeline_runs(client, auth, stub_guardrails, stub_extraction, stub_waiter):
    response = _submit(client, auth)

    assert response.status_code == 200
    assert len(stub_guardrails.checked) == 1
    assert stub_extraction.submitted[0]["sequence"] == FULL
    assert stub_waiter.last_stages == ["SCORING"]


def test_the_sequence_is_taken_as_repeated_fields_or_as_a_json_array(client, auth, stub_extraction, stub_waiter):
    repeated = _submit(client, auth, pipeline_name_sequence=["guardrails", "extraction"])
    as_json = _submit(client, auth, pipeline_name_sequence='["guardrails", "extraction"]')

    assert (repeated.status_code, as_json.status_code) == (200, 200)
    assert [handed["sequence"] for handed in stub_extraction.submitted] == [["guardrails", "extraction"]] * 2
    assert stub_waiter.last_stages == ["OCR", "OCR"]


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        (["extraction", "scoring"], "without skipping one in the middle"),
        (["guardrails", "structuring", "scoring"], "without skipping one in the middle"),
        (["structuring", "scoring"], "structuring cannot come first"),
        (["guardrails", "scoring", "structuring", "extraction"], "without skipping one in the middle"),
        (["guardrails", "guardrails"], "listed twice"),
        (["guardrails", "ekstraksi"], "unknown service 'ekstraksi'"),
        ('["guardrails", ', "JSON array of strings"),
    ],
)
def test_an_invalid_sequence_is_422_before_anything_runs(
    client, auth, stub_guardrails, stub_extraction, sequence, reason
):
    response = _submit(client, auth, pipeline_name_sequence=sequence)

    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "INVALID_PIPELINE_SEQUENCE"
    assert body["message"].startswith("Invalid pipeline_name_sequence: ") and reason in body["message"]
    assert stub_guardrails.checked == [] and stub_extraction.submitted == []


def test_guardrails_only_answers_with_the_report_as_it_is(client, auth, stub_extraction, stub_waiter):
    response = _submit(client, auth, pipeline_name_sequence=["guardrails"])

    assert response.status_code == 200
    body = response.json()
    assert (body["status_code"], body["guardrails"], body["errors"]) == (200, 0.9821, None)
    assert body["data"] == {**ACCEPTED_REPORT, "auto_accepted": True, "score": 0.9821}
    assert stub_extraction.submitted == [] and stub_waiter.calls == []


def test_guardrails_only_still_rejects_with_a_threshold(client, auth):
    response = _submit(
        client, auth, filename="notnpwp.jpg", pipeline_name_sequence=["guardrails"], **WITH_GUARDRAILS_THRESHOLD
    )

    assert response.status_code == 400
    assert (response.json()["errors"], response.json()["guardrails"]) == ("DOWNSTREAM_VALIDATION_ERROR", 1)


def test_a_sequence_ending_at_extraction_answers_with_the_ocr_result_as_it_is(client, auth, stub_waiter):
    ocr = {"text": "NPWP 12.345.678.9-012.345", "blocks": [{"text": "NPWP", "confidence": 0.99, "page": 0}]}
    stub_waiter.outcome = WaitOutcome("OCR", "DONE", results={"OCR": ocr})

    response = _submit(client, auth, pipeline_name_sequence=["guardrails", "extraction"])

    assert response.status_code == 200
    assert (response.status_code, response.json()["data"]) == (200, ocr)


def test_a_sequence_ending_at_structuring_answers_with_its_result_as_it_is(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome(
        "STRUCTURING", "DONE", results={"OCR": {"blocks": []}, "STRUCTURING": STRUCTURING_RESULT}
    )

    response = _submit(client, auth, pipeline_name_sequence=["guardrails", "extraction", "structuring"])

    assert response.status_code == 200
    assert response.json()["data"] == STRUCTURING_RESULT
    assert stub_waiter.last_stages == ["STRUCTURING"]


def test_without_guardrails_the_document_is_handed_on_without_a_report(client, auth, stub_guardrails, stub_extraction):
    """Leaving guardrails out is the central orchestrator's call: no setting here can refuse it."""
    # A file name the guardrails model rejects: left out of the sequence, it never gets to judge it.
    response = _submit(client, auth, filename="notnpwp.jpg", pipeline_name_sequence=NO_GUARDRAILS)

    assert response.status_code == 200
    body = response.json()
    assert (body["status_code"], body["guardrails"], body["errors"]) == (200, 0, None)
    assert stub_guardrails.checked == []
    [handed] = stub_extraction.submitted
    assert (handed["guardrails"], handed["sequence"]) == (None, NO_GUARDRAILS)


def test_without_guardrails_the_structuring_rules_still_reject(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome("STRUCTURING", STATUS_REJECTED, "dokumen blur / blank")

    response = _submit(client, auth, pipeline_name_sequence=NO_GUARDRAILS)

    assert response.status_code == 400
    body = response.json()
    assert (body["message"], body["errors"], body["guardrails"]) == (
        "dokumen blur / blank",
        "DOWNSTREAM_VALIDATION_ERROR",
        1,
    )


def test_without_guardrails_the_file_checks_still_run(client, auth, settings_override, stub_extraction):
    pages = _submit(
        client,
        auth,
        filename="scan.pdf",
        content=_pdf(3),
        content_type="application/pdf",
        pipeline_name_sequence=NO_GUARDRAILS,
    )
    settings_override(max_upload_bytes=10)
    size = _submit(client, auth, pipeline_name_sequence=NO_GUARDRAILS)

    assert (pages.status_code, pages.json()["message"]) == (400, TOO_MANY_PAGES)
    assert size.status_code == 413
    assert stub_extraction.submitted == []


def test_missing_api_key_returns_401_envelope(client):
    response = client.post("/v1/extract-ocr", data={"request_id": "OCR_7"}, files=image_upload())
    assert response.status_code == 401
    assert (response.json()["message"], response.json()["errors"]) == ("Invalid or missing API key", "UNAUTHORIZED")


@pytest.mark.parametrize(
    ("sequence", "outcome", "status", "stage"),
    [
        (None, None, 200, None),
        (["guardrails"], None, 200, None),
        (["guardrails", "extraction"], WaitOutcome("OCR", "DONE", results={"OCR": {"blocks": []}}), 200, None),
        (None, WaitOutcome("OCR", "FAILED", "OCR model is unavailable"), 422, "extraction"),
        (None, WaitOutcome("STRUCTURING", STATUS_REJECTED, "Kode provinsi pada NPWP tidak valid"), 400, "structuring"),
        (None, WaitOutcome("STRUCTURING", "PROCESSING"), 202, None),
    ],
)
def test_pipeline_last_stage_names_the_service_of_an_error_and_is_null_on_success(
    client, auth, stub_waiter, sequence, outcome, status, stage
):
    if outcome is not None:
        stub_waiter.outcome = outcome
    data = {"pipeline_name_sequence": sequence} if sequence else {}

    response = _submit(client, auth, **data)

    assert (response.status_code, response.json()["pipeline_last_stage"]) == (status, stage)


def test_a_guardrails_rejection_comes_from_guardrails(client, auth):
    response = _submit(client, auth, filename="notnpwp.jpg", **WITH_GUARDRAILS_THRESHOLD)

    assert (response.status_code, response.json()["pipeline_last_stage"]) == (400, "guardrails")


@pytest.mark.parametrize(
    ("failing", "stage"),
    [("guardrails", "guardrails"), ("extraction", "extraction")],
)
def test_an_unreachable_service_is_named_in_the_error(client, auth, stub_guardrails, stub_extraction, failing, stage):
    stub = stub_guardrails if failing == "guardrails" else stub_extraction
    stub.error = UpstreamUnavailable(f"{failing} service is unavailable")

    response = _submit(client, auth)

    assert response.status_code == 503
    body = response.json()
    assert (body["pipeline_last_stage"], body["message"], body["errors"]) == (
        stage,
        f"{failing} service is unavailable",
        "DOWNSTREAM_UNAVAILABLE",
    )
    assert body["request_id"] == "OCR_1"


def test_a_refusal_before_any_pipeline_service_names_the_orchestrator(client, auth):
    response = _submit(client, auth, pipeline_name_sequence=["extraction", "scoring"])

    assert (response.status_code, response.json()["pipeline_last_stage"]) == (422, "orchestrator")
