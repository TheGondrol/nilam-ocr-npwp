"""
Backend `remote`: service model guardrails milik ML engineer.

    POST {GUARDRAILS_MODEL_URL}/v1/predict/json   header X-API-Key, multipart `file`
    -> {"status_code": 200, "message": "OK", "data": {"document": {...}, "pages": [...]}}

Service model diganti httpx.MockTransport yang membalas contoh response dari
kontraknya (tanpa jaringan). Test ke service sungguhan: test_guardrails_live.py.
"""

from typing import Any

import httpx
import pytest

from ocr_common.errors import ServiceError
from ocr_common.remote import RemoteModelClient
from ocr_common.testing import image_upload
from src.core.config import Settings
from src.models.guardrails import CLASSIFIER_BACKENDS, RemoteGuardrailsModel
from src.services.guardrails_service import GuardrailsService

JPEG = b"\xff\xd8fake-jpeg-bytes"

ACCEPTED: dict[str, Any] = {
    "status_code": 200,
    "message": "OK",
    "data": {
        "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 2, "n_approve": 2, "n_reject": 0},
        "pages": [
            {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"},
            {"page_index": 1, "proba_approve": 0.9950, "proba_reject": 0.0050, "verdict": "accepted"},
        ],
    },
}
REJECTED: dict[str, Any] = {
    "status_code": 200,
    "message": "OK",
    "data": {
        "document": {"verdict": "reject", "confidence": 0.8821, "n_pages": 1, "n_approve": 0, "n_reject": 1},
        "pages": [{"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}],
    },
}


def _settings(**overrides) -> Settings:
    return Settings(api_key="x", _env_file=None, **overrides)


def _model(handler) -> RemoteGuardrailsModel:
    return RemoteGuardrailsModel(
        RemoteModelClient(
            "http://guardrails-model:8081",
            5.0,
            name="guardrails model",
            headers={"X-API-Key": "dummy-key"},
            transport=httpx.MockTransport(handler),
        )
    )


def _reply(body, status_code=200):
    return lambda request: httpx.Response(status_code, json=body)


async def test_request_follows_the_model_contract():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            method=request.method,
            path=request.url.path,
            api_key=request.headers.get("X-API-Key"),
            content_type=request.headers["Content-Type"],
            body=request.content,
        )
        return httpx.Response(200, json=ACCEPTED)

    await _model(handler).check_document("sample_npwp.jpg", JPEG, "image/jpeg")

    assert (seen["method"], seen["path"]) == ("POST", "/v1/predict/json")
    assert seen["api_key"] == "dummy-key"
    assert seen["content_type"].startswith("multipart/form-data")
    # Satu-satunya field adalah `file`, membawa nama, tipe, dan isi berkas apa adanya.
    assert b'name="file"; filename="sample_npwp.jpg"' in seen["body"]
    assert b"Content-Type: image/jpeg" in seen["body"]
    assert JPEG in seen["body"]
    assert b"request_id" not in seen["body"]


async def test_accepted_report_passes_through_with_passed_true():
    report = await GuardrailsService(_model(_reply(ACCEPTED)), _settings()).check("a.pdf", "application/pdf", b"%PDF")
    assert report == {"passed": True, "reason": None, **ACCEPTED["data"]}


async def test_rejected_report_gets_reason_for_the_orchestrator():
    report = await GuardrailsService(_model(_reply(REJECTED)), _settings()).check("a.jpg", "image/jpeg", JPEG)
    assert report["passed"] is False
    assert report["reason"] == "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)"
    assert report["document"] == REJECTED["data"]["document"]


async def test_model_verdict_is_not_overridden_by_local_settings():
    """Ambang dan kebijakan dokumen milik service model: setting lokal tidak boleh mengubah vonisnya."""
    settings = _settings(guardrails_reject_threshold=0.001, guardrails_document_policy="majority")
    report = await GuardrailsService(_model(_reply(ACCEPTED)), settings).check("a.jpg", "image/jpeg", JPEG)
    assert report["document"]["verdict"] == "accepted"
    assert [p["verdict"] for p in report["pages"]] == ["accepted", "accepted"]


async def test_file_is_validated_before_calling_the_model():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=ACCEPTED)

    service = GuardrailsService(_model(handler), _settings())
    with pytest.raises(ServiceError) as exc:
        await service.check("a.txt", "text/plain", b"x")
    assert exc.value.status_code == 400
    with pytest.raises(ServiceError):
        await service.check("a.jpg", "image/jpeg", b"")
    assert calls == 0


async def test_extra_fields_from_the_model_are_dropped():
    body = {
        "status_code": 200,
        "message": "OK",
        "data": {
            "document": {**REJECTED["data"]["document"], "model_version": "v7"},
            "pages": [{**REJECTED["data"]["pages"][0], "heatmap": "..."}],
            "debug": True,
        },
    }
    report = await _model(_reply(body)).check_document("a.jpg", JPEG, "image/jpeg")
    assert report == REJECTED["data"]


@pytest.mark.parametrize(
    "body",
    [
        {"status_code": 200, "message": "OK"},  # tanpa data
        {"data": {"document": {"verdict": "accepted"}, "pages": []}},  # document tidak lengkap
        {"data": {"document": {**ACCEPTED["data"]["document"], "verdict": "maybe"}, "pages": []}},  # vonis asing
        {"data": {"document": ACCEPTED["data"]["document"], "pages": [{"page_index": 0}]}},  # halaman tidak lengkap
        ["not", "an", "object"],
    ],
)
async def test_unexpected_response_shape_is_500_not_a_guess(body):
    with pytest.raises(ServiceError) as exc:
        await _model(_reply(body)).check_document("a.jpg", JPEG, "image/jpeg")
    assert exc.value.status_code == 500
    assert exc.value.message == "guardrails model returned an unexpected response"


async def test_model_error_status_becomes_500_with_detail():
    model = _model(_reply({"status_code": 401, "message": "Invalid API key"}, status_code=401))
    with pytest.raises(ServiceError) as exc:
        await model.check_document("a.jpg", JPEG, "image/jpeg")
    # 401 dari service MODEL adalah salah konfigurasi kita, bukan salah pemanggil kita: jangan teruskan 401.
    assert exc.value.status_code == 500
    assert exc.value.message == "guardrails model error (401): Invalid API key"


async def test_unreachable_model_is_503_and_slow_model_is_504():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ServiceError) as exc:
        await _model(refuse).check_document("a.jpg", JPEG, "image/jpeg")
    assert (exc.value.status_code, exc.value.message) == (503, "guardrails model is unavailable")

    with pytest.raises(ServiceError) as exc:
        await _model(stall).check_document("a.jpg", JPEG, "image/jpeg")
    assert (exc.value.status_code, exc.value.message) == (504, "guardrails model timed out after 5.0s")


def test_remote_backend_requires_model_url():
    with pytest.raises(RuntimeError, match="GUARDRAILS_MODEL_URL is required"):
        CLASSIFIER_BACKENDS["remote"](_settings(guardrails_backend="remote"))


async def test_remote_backend_is_built_from_settings():
    model = CLASSIFIER_BACKENDS["remote"](
        _settings(
            guardrails_backend="remote",
            guardrails_model_url="http://localhost:8081/",
            guardrails_model_api_key="dummy-key",
        )
    )
    try:
        assert isinstance(model, RemoteGuardrailsModel)
        assert model._client.base_url == "http://localhost:8081"
        assert model._client._client.headers["X-API-Key"] == "dummy-key"
    finally:
        await model.aclose()


def test_http_check_with_remote_backend(client, auth, monkeypatch):
    """Kontrak KITA ke orkestrator tidak berubah: request_id + file masuk, envelope + passed/reason keluar."""
    monkeypatch.setattr("src.api.v1.guardrails.get_page_classifier", lambda: _model(_reply(REJECTED)))
    response = client.post(
        "/v1/guardrails/check", data={"request_id": "OCR_R1"}, files=image_upload("npwp.jpg", JPEG), headers=auth
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "OCR_R1"
    assert body["data"]["passed"] is False
    assert body["data"]["document"] == REJECTED["data"]["document"]


def test_http_model_unreachable_returns_503_envelope(client, auth, monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("src.api.v1.guardrails.get_page_classifier", lambda: _model(refuse))
    response = client.post(
        "/v1/guardrails/check", data={"request_id": "OCR_R2"}, files=image_upload("npwp.jpg", JPEG), headers=auth
    )
    assert response.status_code == 503
    assert response.json()["message"] == "guardrails model is unavailable"
