"""
Kontrak yang dipanggil ocr-orchestration (sama dengan mock ocr-npwp):
generate-request-id -> extract-ocr -> get-ocr-result, tiap field
{value, confidence}, skor dokumen `guardrails` di level envelope.
"""

from tests.conftest import image_upload


def _new_request_id(client, auth):
    response = client.post("/v1/generate-request-id", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["request_id"] == body["request_id"]
    assert body["request_id"].startswith("OCR_")
    return body["request_id"]


def _extract(client, auth, request_id, filename="npwp.jpg", content=b"\xff\xd8seed"):
    return client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": request_id}, files=image_upload(filename, content)
    )


def test_happy_path_returns_wrapped_fields_and_guardrails(client, auth):
    request_id = _new_request_id(client, auth)
    response = _extract(client, auth, request_id)
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == request_id
    assert set(body["data"]) == {"nomor_npwp", "nama", "nama_badan"}
    for field in body["data"].values():
        assert set(field) == {"value", "confidence"}
        assert field["value"]
        assert 0 <= field["confidence"] <= 1
    assert 0 <= body["guardrails"] <= 1

    poll = client.get(f"/v1/get-ocr-result/{request_id}", headers=auth)
    assert poll.status_code == 200
    poll_body = poll.json()
    assert poll_body["data"]["status"] == "completed"
    assert poll_body["data"]["result"] == body["data"]
    assert poll_body["guardrails"] == body["guardrails"]
    assert "guardrails" not in poll_body["data"]
    assert set(poll_body["data"]) == {"request_id", "status", "result", "error_message", "created_at", "updated_at"}


def test_generate_request_id_has_no_guardrails_key(client, auth):
    body = client.post("/v1/generate-request-id", headers=auth).json()
    assert "guardrails" not in body


def test_unknown_request_id_returns_400(client, auth):
    response = _extract(client, auth, "OCR_unknown")
    assert response.status_code == 400
    assert response.json()["message"].startswith("Unknown request_id")


def test_same_request_id_twice_returns_409(client, auth):
    request_id = _new_request_id(client, auth)
    assert _extract(client, auth, request_id).status_code == 200
    response = _extract(client, auth, request_id)
    assert response.status_code == 409
    assert response.json()["message"] == f"request_id {request_id} has already been processed"


def test_guardrails_failure_returns_400_and_marks_failed(client, auth):
    request_id = _new_request_id(client, auth)
    response = _extract(client, auth, request_id, filename="notnpwp.jpg")
    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded image is not recognized as an NPWP"

    poll = client.get(f"/v1/get-ocr-result/{request_id}", headers=auth).json()
    assert poll["data"]["status"] == "failed"
    assert poll["data"]["error_message"] == "Uploaded image is not recognized as an NPWP"
    assert poll["guardrails"] is None


def test_blur_returns_400(client, auth):
    request_id = _new_request_id(client, auth)
    response = _extract(client, auth, request_id, filename="blur.jpg")
    assert response.status_code == 400
    assert response.json()["message"] == "Image quality too low, NPWP could not be read"


def test_servererror_returns_500(client, auth):
    request_id = _new_request_id(client, auth)
    response = _extract(client, auth, request_id, filename="servererror.jpg")
    assert response.status_code == 500
    assert response.json()["message"] == "Internal server error while processing OCR"


def test_missing_form_returns_422_with_code(client, auth):
    response = client.post("/v1/extract-ocr", headers=auth, data={})
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"] == "body.request_id: Field required"


def test_get_result_unknown_returns_404_with_request_id(client, auth):
    response = client.get("/v1/get-ocr-result/OCR_missing", headers=auth)
    assert response.status_code == 404
    body = response.json()
    assert body["request_id"] == "OCR_missing"
    assert body["message"] == "No data found for request_id: OCR_missing"


def test_get_result_without_key_returns_401_with_request_id(client):
    response = client.get("/v1/get-ocr-result/OCR_missing")
    assert response.status_code == 401
    assert response.json()["request_id"] == "OCR_missing"
