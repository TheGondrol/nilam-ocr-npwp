import json

import httpx
import pytest
from pydantic import ValidationError

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.config import PipelineSettings
from ocr_common.npwp import final_result
from ocr_common.pipeline.callbacks import (
    OrchestrationCallback,
    ResultCallback,
    result_callback_body,
    stage_callback_body,
)
from ocr_common.pipeline.factory import build_callback

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"
GUARDRAILS = {"passed": True, "reason": None, "document": {"verdict": "accepted", "confidence": 0.98}}
PATH = "/v1/ocr-callback"


def _final(nama=None, nama_badan=None, npwp="09.254.294.3-407.000", npwp_confidence=0.98291, name_confidence=0.9512):
    structuring = {
        "fields": {
            "nomor_npwp": {"value": npwp, "confidence": 0.99},
            "nama": {"value": nama, "confidence": 0.97},
            "nama_badan": {"value": nama_badan, "confidence": 0.0},
        },
        "flag": False,
        "flag_reason": None,
    }
    scoring = {"npwp_confidence": npwp_confidence, "name_confidence": name_confidence}
    return dict(final_result("npwp", GUARDRAILS, structuring, scoring))


def test_scoring_done_becomes_the_completed_result_callback():
    body = result_callback_body(stage_callback_body(RID, "SCORING", "DONE", result=_final(nama="BUDI SANTOSO")))

    assert body == {
        "request_id": RID,
        "status": "completed",
        "result": {
            "nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 0.9829},
            "nama": {"value": "BUDI SANTOSO", "confidence": 0.9512},
            "nama_badan": {"value": "", "confidence": 0.0},
        },
        "guardrails": GUARDRAILS,
    }


def test_on_a_company_card_the_name_probability_goes_to_nama_badan():
    body = result_callback_body(
        stage_callback_body(RID, "SCORING", "DONE", result=_final(nama_badan="PT CONTOH INDONESIA"))
    )

    assert body is not None
    assert body["result"]["nama"] == {"value": "", "confidence": 0.0}
    assert body["result"]["nama_badan"] == {"value": "PT CONTOH INDONESIA", "confidence": 0.9512}


def test_a_field_that_was_not_found_is_empty_with_confidence_0():
    body = result_callback_body(
        stage_callback_body(RID, "SCORING", "DONE", result=_final(nama="BUDI", npwp=None, npwp_confidence=None))
    )

    assert body is not None
    assert body["result"]["nomor_npwp"] == {"value": "", "confidence": 0.0}


@pytest.mark.parametrize(
    ("stage", "error_message", "error_code", "expected_code"),
    [
        ("OCR", "ekstraksi OCR model is unavailable", None, "OCR_FAILED"),
        ("SCORING", "Internal error in SCORING stage", None, "SCORING_FAILED"),
        (
            "STRUCTURING",
            "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
            "DOWNSTREAM_VALIDATION_ERROR",
            "DOWNSTREAM_VALIDATION_ERROR",
        ),
    ],
)
def test_a_failed_stage_becomes_the_failed_result_callback(stage, error_message, error_code, expected_code):
    body = result_callback_body(
        stage_callback_body(RID, stage, "FAILED", error_message=error_message, error_code=error_code)
    )

    assert body == {
        "request_id": RID,
        "status": "failed",
        "result": None,
        "guardrails": {},
        "error_code": expected_code,
        "error_message": error_message,
    }


@pytest.mark.parametrize("stage", ["OCR", "STRUCTURING"])
def test_a_stage_that_does_not_end_the_request_sends_nothing(stage):
    assert result_callback_body(stage_callback_body(RID, stage, "DONE", result={"blocks": []})) is None


def _recording_client(seen: list[httpx.Request], status: int = 200) -> RemoteModelClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={})

    return RemoteModelClient(
        "http://ocr-orchestration.ocr-dev.svc.cluster.local",
        1.0,
        name="orchestration result callback",
        headers={"X-Callback-Key": "secret"},
        passthrough_client_errors=True,
        transport=httpx.MockTransport(handler),
    )


async def test_notify_posts_the_result_with_the_callback_key_once_the_request_ends():
    seen: list[httpx.Request] = []
    callback = ResultCallback(_recording_client(seen), PATH, attempts=1, delay=0)

    assert await callback.notify(RID, "OCR", "DONE", result={"blocks": []}) is False
    assert await callback.notify(RID, "SCORING", "DONE", result=_final(nama="BUDI SANTOSO")) is True

    [request] = seen
    assert request.url.path == PATH
    assert request.headers["X-Callback-Key"] == "secret"
    assert json.loads(request.content)["status"] == "completed"


async def test_outbox_send_turns_the_stored_stage_body_into_the_result():
    seen: list[httpx.Request] = []
    callback = ResultCallback(_recording_client(seen), PATH)

    await callback.send(stage_callback_body(RID, "STRUCTURING", "DONE", result={"fields": {}}))
    await callback.send(
        stage_callback_body(RID, "STRUCTURING", "FAILED", error_message="dokumen blur / blank", error_code="X")
    )

    [request] = seen
    assert json.loads(request.content) == {
        "request_id": RID,
        "status": "failed",
        "result": None,
        "guardrails": {},
        "error_code": "X",
        "error_message": "dokumen blur / blank",
    }


def _settings(**overrides) -> PipelineSettings:
    return PipelineSettings(api_key="k", environment="local", _env_file=None, **overrides)


def test_the_callback_format_picks_the_callback_class():
    assert isinstance(build_callback(_settings(orchestration_url="http://orch")), OrchestrationCallback)
    result = build_callback(
        _settings(
            orchestration_url="http://orch", orchestration_callback_format="result", orchestration_callback_key="s"
        )
    )
    assert isinstance(result, ResultCallback)


def _production(callback_key: str | None) -> PipelineSettings:
    return PipelineSettings(
        api_key="k",
        environment="production",
        database_url="postgresql+asyncpg://u:p@db/x",
        orchestration_url="http://ocr-orchestration.ocr-dev.svc.cluster.local",
        orchestration_callback_format="result",
        orchestration_callback_key=callback_key,
        _env_file=None,
    )


def test_the_result_format_needs_the_callback_key_outside_local():
    with pytest.raises(ValidationError, match="ORCHESTRATION_CALLBACK_KEY must be set"):
        _production(None)
    assert _production("secret").orchestration_callback_key == "secret"
