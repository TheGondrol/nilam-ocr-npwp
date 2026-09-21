import io
import time

import httpx
from PIL import Image

from ocr_common.errors import ServiceError
from ocr_common.npwp import final_result
from ocr_common.remote import RemoteModelClient
from src.api.v1.guardrails import get_guardrails_service
from src.clients.ekstraksi import EkstraksiJobClient
from src.clients.stages import StageStatusClient
from src.core.config import get_settings
from src.main import app
from src.services.job_service import GuardrailsJobService
from src.services.pipeline_waiter import PipelineWaiter, WaitOutcome

RID = "REQ_wait"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _submit(client, auth, filename="npwp.jpg"):
    return client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": RID}, files={"file": (filename, _jpeg(), "image/jpeg")}
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
    data = response.json()["data"]
    assert data["pipeline"] == {"stage": "SCORING", "status": "DONE", "error_message": None}
    report = {key: data[key] for key in ("passed", "reason", "document", "pages")}
    results = stub_waiter.outcome.results
    assert data["result"] == final_result("npwp", report, results["STRUCTURING"], results["SCORING"])
    assert data["result"]["fields"]["nama"] == {"value": "BUDI SANTOSO", "confidence": 0.97}
    assert data["result"]["scoring"] == {"npwp_confidence": 0.7296, "name_confidence": 0.9471}
    [(request_id, timeout)] = stub_waiter.calls
    assert request_id == RID
    assert 10 < timeout <= 15


def test_still_running_when_the_wait_runs_out_is_202(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome("STRUCTURING", "PROCESSING", results={"OCR": {"blocks": []}})

    response = _submit(client, auth)

    assert response.status_code == 202
    body = response.json()
    assert (body["status_code"], body["status_desc"]) == (202, "Accepted")
    assert body["data"]["job"]["stage"] == "OCR"
    assert body["data"]["pipeline"] == {"stage": "STRUCTURING", "status": "PROCESSING", "error_message": None}
    assert body["data"]["result"] is None


def test_failure_within_the_wait_is_200_with_the_failed_stage(client, auth, stub_waiter):
    stub_waiter.outcome = WaitOutcome("OCR", "FAILED", "ekstraksi OCR model is unavailable")

    response = _submit(client, auth)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["pipeline"] == {
        "stage": "OCR",
        "status": "FAILED",
        "error_message": "ekstraksi OCR model is unavailable",
    }
    assert data["result"] is None


def test_rejected_document_answers_at_once_without_waiting(client, auth, stub_waiter):
    response = _submit(client, auth, filename="notnpwp.jpg")

    assert response.status_code == 200
    data = response.json()["data"]
    assert (data["passed"], data["job"], data["pipeline"], data["result"]) == (False, None, None, None)
    assert stub_waiter.calls == []


def test_waiting_disabled_answers_202_right_after_the_handoff(client, auth, stub_waiter):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"pipeline_wait_seconds": 0})
    try:
        response = _submit(client, auth)
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 202
    assert response.json()["data"]["pipeline"] is None
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
    service = GuardrailsJobService(
        get_guardrails_service(), EkstraksiJobClient(remote, attempts=1, delay=0), stub_waiter, wait_seconds=15
    )

    await service.submit(RID, "npwp", "npwp.jpg", "image/jpeg", _jpeg(), received_at=time.monotonic() - 10)

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
