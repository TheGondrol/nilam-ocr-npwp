import time

import httpx

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

from app.clients.ekstraksi import EkstraksiJobClient
from app.clients.guardrails import GuardrailsClient
from app.clients.stages import StageStatusClient
from app.config import get_settings
from app.main import app
from app.services.extract_service import ExtractOcrService
from app.services.pipeline_waiter import STATUS_REJECTED, PipelineWaiter, WaitOutcome
from tests.conftest import ACCEPTED_REPORT, JPEG

RID = "REQ_wait"


def _submit(client, auth, filename="npwp.jpg"):
    return client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": RID}, files={"file": (filename, JPEG, "image/jpeg")}
    )


def _job(status, result=None, error_message=None):
    return {"status": status, "result": result, "error_message": error_message}


class FakeStage:
    def __init__(self, stage, *answers):
        self.stage = stage
        self._answers = list(answers)
        self.calls = 0

    async def get(self, request_id):
        self.calls += 1
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_finished_within_the_wait_is_200_with_the_final_result(client, auth, stub_waiter):
    response = _submit(client, auth)

    assert response.status_code == 200
    body = response.json()
    assert (body["job_status"], body["guardrails"], body["errors"]) == ("completed", 0, None)
    assert body["data"] == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    }
    [(request_id, timeout)] = stub_waiter.calls
    assert request_id == RID
    assert 10 < timeout <= 15


def test_still_running_when_the_wait_runs_out_is_202(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome("STRUCTURING", "PROCESSING", results={"OCR": {"blocks": []}})

    response = _submit(client, auth)

    assert response.status_code == 202
    body = response.json()
    assert (body["status_code"], body["status_desc"], body["message"]) == (
        202,
        "Accepted",
        "OCR job accepted; still processing",
    )
    assert (body["job_status"], body["data"], body["guardrails"], body["errors"]) == ("processing", None, None, None)


def test_failure_within_the_wait_is_422_with_the_failed_stage(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome("OCR", "FAILED", "ekstraksi OCR model is unavailable")

    response = _submit(client, auth)

    assert response.status_code == 422
    body = response.json()
    assert (body["errors"], body["message"]) == ("OCR_FAILED", "ekstraksi OCR model is unavailable")
    assert (body["job_status"], body["data"], body["guardrails"]) == ("failed", None, 0)


def test_rejection_by_the_structuring_rules_is_400_with_their_reason(client, auth, stub_waiter):
    reason = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
    stub_waiter.outcome = WaitOutcome("STRUCTURING", STATUS_REJECTED, reason)

    response = _submit(client, auth)

    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["message"]) == ("DOWNSTREAM_VALIDATION_ERROR", reason)
    assert (body["job_status"], body["data"], body["guardrails"]) == ("failed", None, 1)


def test_rejected_document_answers_at_once_without_waiting(client, auth, stub_waiter):
    response = _submit(client, auth, filename="notnpwp.jpg")

    assert response.status_code == 400
    assert response.json()["errors"] == "DOWNSTREAM_VALIDATION_ERROR"
    assert stub_waiter.calls == []


def test_waiting_disabled_answers_202_right_after_the_handoff(client, auth, stub_waiter):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"pipeline_wait_seconds": 0})
    try:
        response = _submit(client, auth)
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 202
    assert response.json()["job_status"] == "processing"
    assert stub_waiter.calls == []


async def test_the_wait_is_counted_from_the_arrival_of_the_request(stub_waiter):
    remote = RemoteModelClient(
        "http://ekstraksi:8030",
        5.0,
        name="ekstraksi service",
        passthrough_client_errors=True,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                202, json={"data": {"request_id": RID, "stage": "OCR", "status": "PROCESSING", "duplicate": False}}
            )
        ),
    )
    guardrails = RemoteModelClient(
        "http://guardrails:8031",
        5.0,
        name="guardrails service",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": ACCEPTED_REPORT})),
    )
    settings = get_settings().model_copy(update={"pipeline_wait_seconds": 15})
    service = ExtractOcrService(
        GuardrailsClient(guardrails), EkstraksiJobClient(remote, attempts=1, delay=0), stub_waiter, settings
    )

    await service.submit(RID, "npwp", "npwp.jpg", "image/jpeg", JPEG, received_at=time.monotonic() - 10)

    [(_, timeout)] = stub_waiter.calls
    assert 4 < timeout <= 5


async def test_waiter_follows_the_stages_in_order_and_collects_their_results():
    stages = [
        FakeStage("OCR", _job("PROCESSING"), _job("DONE", {"blocks": []})),
        FakeStage("STRUCTURING", None, _job("DONE", {"fields": {}})),
        FakeStage("SCORING", _job("DONE", {"npwp_confidence": 0.7, "name_confidence": 0.9})),
    ]

    outcome = await PipelineWaiter(stages, poll_interval=0.01).wait(RID, 5)

    assert (outcome.stage, outcome.status, outcome.error_message) == ("SCORING", "DONE", None)
    assert outcome.results == {
        "OCR": {"blocks": []},
        "STRUCTURING": {"fields": {}},
        "SCORING": {"npwp_confidence": 0.7, "name_confidence": 0.9},
    }
    assert [stage.calls for stage in stages] == [2, 2, 1]


async def test_waiter_stops_at_the_first_failed_stage():
    stages = [
        FakeStage("OCR", _job("DONE", {})),
        FakeStage("STRUCTURING", _job("FAILED", error_message="No text lines to structure")),
        FakeStage("SCORING", _job("DONE", {})),
    ]

    outcome = await PipelineWaiter(stages, poll_interval=0.01).wait(RID, 5)

    assert (outcome.stage, outcome.status, outcome.error_message) == (
        "STRUCTURING",
        "FAILED",
        "No text lines to structure",
    )
    assert stages[2].calls == 0


async def test_waiter_stops_at_a_rejection_of_the_structuring_rules():
    reason = "dokumen blur / blank"
    stages = [
        FakeStage("OCR", _job("DONE", {"blocks": []})),
        FakeStage("STRUCTURING", _job("DONE", {"fields": {}, "flag": True, "reject_reason": reason})),
        FakeStage("SCORING", _job("DONE", {})),
    ]

    outcome = await PipelineWaiter(stages, poll_interval=0.01).wait(RID, 5)

    assert (outcome.stage, outcome.status, outcome.error_message) == ("STRUCTURING", STATUS_REJECTED, reason)
    assert stages[2].calls == 0, "scoring never gets a rejected document, so it is not waited for"


async def test_waiter_goes_on_past_a_tolerated_flag():
    stages = [
        FakeStage("OCR", _job("DONE", {})),
        FakeStage("STRUCTURING", _job("DONE", {"fields": {}, "flag": True, "reject_reason": None})),
        FakeStage("SCORING", _job("DONE", {"npwp_confidence": 0.7, "name_confidence": 0.9})),
    ]

    outcome = await PipelineWaiter(stages, poll_interval=0.01).wait(RID, 5)

    assert (outcome.stage, outcome.status) == ("SCORING", "DONE")


async def test_waiter_gives_up_when_the_time_runs_out():
    stages = [FakeStage("OCR", _job("DONE", {})), FakeStage("STRUCTURING", _job("PROCESSING")), FakeStage("SCORING")]

    started = time.monotonic()
    outcome = await PipelineWaiter(stages, poll_interval=0.05).wait(RID, 0.3)

    assert (outcome.stage, outcome.status) == ("STRUCTURING", "PROCESSING")
    assert 0.25 < time.monotonic() - started < 1.0


async def test_waiter_keeps_polling_through_status_errors():
    stages = [
        FakeStage("OCR", ServiceError(503, "ekstraksi service is unavailable"), _job("DONE", {})),
        FakeStage("STRUCTURING", _job("DONE", {})),
        FakeStage("SCORING", _job("DONE", {"npwp_confidence": 0.7, "name_confidence": 0.9})),
    ]

    outcome = await PipelineWaiter(stages, poll_interval=0.01).wait(RID, 5)

    assert outcome.status == "DONE"
    assert stages[0].calls == 2


async def test_waiter_without_time_left_does_not_poll():
    stage = FakeStage("OCR", _job("DONE", {}))

    outcome = await PipelineWaiter([stage], poll_interval=0.01).wait(RID, 0)

    assert (outcome.stage, outcome.status) == ("OCR", "PROCESSING")
    assert stage.calls == 0


async def test_stage_status_client_quotes_the_request_id_and_maps_404_to_none():
    paths: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path)
        if request.url.raw_path.endswith(b"/missing"):
            return httpx.Response(404, json={"message": "No STRUCTURING job found"})
        return httpx.Response(200, json={"data": {"status": "DONE", "result": {"fields": {}}}})

    remote = RemoteModelClient(
        "http://structuring:8032",
        5.0,
        name="structuring service",
        passthrough_client_errors=True,
        transport=httpx.MockTransport(handler),
    )
    stage = StageStatusClient("STRUCTURING", remote, "/v1/structuring/jobs")

    assert await stage.get("a/b?c") == {"status": "DONE", "result": {"fields": {}}}
    assert paths[0] == b"/v1/structuring/jobs/a%2Fb%3Fc"
    assert await stage.get("missing") is None
