from tests.conftest import image_upload


def test_health_is_public_and_lists_backends(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["backends"] == {
        "guardrails": "mock",
        "ekstraksi": "mock",
        "structuring": "rule_based",
        "scoring": "heuristic",
        "storage": "memory",
    }


def test_missing_api_key_returns_401_envelope(client):
    response = client.post("/v1/guardrails/check", files=image_upload())
    assert response.status_code == 401
    body = response.json()
    assert body["status_desc"] == "Unauthorized"
    assert body["errors"] == "Invalid or missing API key"
    assert body["data"] is None


def test_request_id_from_header_is_echoed(client, auth):
    response = client.post("/v1/guardrails/check", files=image_upload(), headers={**auth, "X-Request-ID": "OCR_abc"})
    assert response.status_code == 200
    assert response.json()["request_id"] == "OCR_abc"
    assert response.headers["X-Request-ID"] == "OCR_abc"


def test_request_id_is_generated_when_absent(client, auth):
    response = client.post("/v1/guardrails/check", files=image_upload(), headers=auth)
    assert response.json()["request_id"].startswith("REQ_")


def test_unsupported_content_type_returns_400(client, auth):
    response = client.post("/v1/guardrails/check", files=image_upload(content_type="application/pdf"), headers=auth)
    assert response.status_code == 400
    assert response.json()["message"].startswith("Unsupported content type")


def test_empty_file_returns_400(client, auth):
    response = client.post("/v1/ekstraksi/extract", files=image_upload(content=b""), headers=auth)
    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is empty"


def test_validation_error_uses_envelope_with_code(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": "not-a-list"}, headers=auth)
    assert response.status_code == 422
    body = response.json()
    assert body["status_desc"] == "Unprocessable Entity"
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"].startswith("body.lines:")


def test_file_and_file_url_both_missing_returns_400(client, auth):
    response = client.post("/v1/ekstraksi/extract", data={"file": ""}, headers=auth)
    assert response.status_code == 400
    assert response.json()["message"] == "Send exactly one of file or file_url"


def test_file_and_file_url_both_sent_returns_400(client, auth):
    response = client.post(
        "/v1/ekstraksi/extract", data={"file_url": "http://x/y.jpg"}, files=image_upload(), headers=auth
    )
    assert response.status_code == 400


def test_file_url_is_fetched_by_service(client, auth, monkeypatch):
    async def fake_fetch(url, *, limit, timeout=10.0):
        assert url == "http://minio.local/bucket/npwp.jpg"
        return b"\xff\xd8bytes-from-url", "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("src.api.v1.intake.fetch", fake_fetch)
    response = client.post(
        "/v1/ekstraksi/extract", data={"file_url": "http://minio.local/bucket/npwp.jpg"}, headers=auth
    )
    assert response.status_code == 200
    assert response.json()["data"]["engine"] == "mock"


def test_file_url_fetch_failure_returns_400(client, auth):
    response = client.post("/v1/ekstraksi/extract", data={"file_url": "file:///etc/passwd"}, headers=auth)
    assert response.status_code == 400
    assert "Unsupported URL scheme" in response.json()["message"]
