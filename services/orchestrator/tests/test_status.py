import httpx
import pytest

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError, UpstreamUnavailable

from app.clients.stages import StageStatusClient, build_stage_status_clients
from app.config import get_settings
from app.services.pipeline_waiter import STATUS_REJECTED, PipelineWaiter, WaitOutcome

RID = "REQ_status"


def _job(status, result=None, error_message=None):
    return {"status": status, "result": result, "error_message": error_message}


class FakeStage:
    def __init__(self, stage, answer):
        self.stage = stage
        self._answer = answer
        self.calls = 0

    async def get(self, request_id):
        self.calls += 1
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def _stages(ocr, structuring=None, scoring=None) -> list[FakeStage]:
    return [FakeStage("OCR", ocr), FakeStage("STRUCTURING", structuring), FakeStage("SCORING", scoring)]


def _get(client, auth, request_id=RID):
    return client.get(f"/v1/extract-ocr/{request_id}", headers=auth)


# --- the endpoint -----------------------------------------------------------------------------------


def test_finished_request_is_200_with_its_data_and_no_params(client, auth, stub_waiter, stub_guardrails):
    response = _get(client, auth)

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
        "request_id": RID,
        "document_type": "npwp",
        "job_status": "completed",
        "guardrails": 0,
        "pipeline_last_stage": "scoring",
        "params": None,
    }
    assert stub_waiter.snapshots == [RID]
    assert stub_waiter.calls == [], "the status is read, not waited for"
    assert stub_guardrails.checked == []


def test_running_request_is_202(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = WaitOutcome("STRUCTURING", "PROCESSING")

    response = _get(client, auth)

    assert response.status_code == 202
    assert (response.json()["job_status"], response.json()["guardrails"]) == ("processing", None)


def test_failed_stage_is_422(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = WaitOutcome("SCORING", "FAILED", "trust model is unavailable")

    body = _get(client, auth).json()

    assert (body["status_code"], body["errors"], body["message"]) == (
        422,
        "SCORING_FAILED",
        "trust model is unavailable",
    )


def test_rejection_by_the_structuring_rules_is_400(client, auth, stub_waiter):
    reason = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
    stub_waiter.snapshot_outcome = WaitOutcome("STRUCTURING", STATUS_REJECTED, reason)

    body = _get(client, auth).json()

    assert (body["status_code"], body["errors"], body["message"], body["guardrails"]) == (
        400,
        "DOWNSTREAM_VALIDATION_ERROR",
        reason,
        1,
    )


def test_unknown_request_id_is_404(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = None

    response = _get(client, auth, "REQ_unknown")

    assert response.status_code == 404
    body = response.json()
    assert (body["message"], body["request_id"]) == ("No request found for request_id REQ_unknown", "REQ_unknown")


def test_unreadable_stage_is_503_not_a_false_202(client, auth, stub_waiter):
    stub_waiter.snapshot_error = UpstreamUnavailable("structuring service is unavailable")

    response = _get(client, auth)

    assert response.status_code == 503
    assert response.json()["message"] == "structuring service is unavailable"


def test_status_needs_the_api_key(client):
    assert client.get(f"/v1/extract-ocr/{RID}").status_code == 401


# --- PipelineWaiter.snapshot --------------------------------------------------------------------------


async def test_snapshot_of_a_finished_pipeline_collects_every_result():
    stages = _stages(
        _job("DONE", {"blocks": []}),
        _job("DONE", {"fields": {}}),
        _job("DONE", {"npwp_confidence": 0.7, "name_confidence": 0.9}),
    )

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status) == ("SCORING", "DONE")
    assert outcome.results["SCORING"] == {"npwp_confidence": 0.7, "name_confidence": 0.9}
    assert [stage.calls for stage in stages] == [1, 1, 1]


async def test_snapshot_without_an_ocr_job_is_none():
    stages = _stages(None)

    assert await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID) is None
    assert [stage.calls for stage in stages] == [1, 0, 0]


async def test_snapshot_stops_at_the_first_running_stage():
    stages = _stages(_job("DONE", {}), _job("PROCESSING"))

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status) == ("STRUCTURING", "PROCESSING")
    assert stages[2].calls == 0


async def test_snapshot_reads_a_next_stage_without_a_job_yet_as_running():
    """The hand-off is still on its way (or, for good, lost: see the endpoint's limitation)."""
    stages = _stages(_job("DONE", {}), None)

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status) == ("STRUCTURING", "PROCESSING")


async def test_snapshot_stops_at_a_failed_stage():
    stages = _stages(_job("FAILED", error_message="extraction OCR model is unavailable"))

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status, outcome.error_message) == (
        "OCR",
        "FAILED",
        "extraction OCR model is unavailable",
    )
    assert [stage.calls for stage in stages] == [1, 0, 0]


async def test_snapshot_reports_a_rejection_even_though_scoring_has_no_job():
    reason = "dokumen blur / blank"
    stages = _stages(_job("DONE", {}), _job("DONE", {"fields": {}, "flag": True, "reject_reason": reason}), None)

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status, outcome.error_message) == ("STRUCTURING", STATUS_REJECTED, reason)
    assert stages[2].calls == 0


async def test_snapshot_raises_when_a_stage_cannot_be_read():
    stages = _stages(_job("DONE", {}), ServiceError(503, "structuring service is unavailable"))

    with pytest.raises(ServiceError) as exc:
        await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)
    assert exc.value.status_code == 503


async def test_snapshot_refuses_a_status_it_does_not_know():
    with pytest.raises(ServiceError) as exc:
        await PipelineWaiter(_stages(_job("QUEUED")), poll_interval=0.01).snapshot(RID)
    assert exc.value.status_code == 500


# --- the stage clients ---------------------------------------------------------------------------------


async def test_a_stage_answering_404_is_no_job_and_401_is_500_not_passed_on():
    """Configured like build_stage_status_clients: a 401 from a stage is our misconfiguration; passed on, the
    central orchestrator would read it as its own key being wrong."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"message": "No OCR job found"})
        return httpx.Response(401, json={"message": "Invalid API key"})

    remote = RemoteModelClient(
        "http://extraction:8030",
        5.0,
        name="extraction service",
        passthrough_statuses=(404,),
        transport=httpx.MockTransport(handler),
    )
    stage = StageStatusClient("OCR", remote, "/v1/extraction/jobs")

    assert await stage.get("missing") is None
    with pytest.raises(ServiceError) as exc:
        await stage.get(RID)
    assert (exc.value.status_code, exc.value.message) == (500, "extraction service error (401): Invalid API key")


def test_the_stage_clients_pass_only_404_through():
    for stage in build_stage_status_clients(get_settings()):
        assert stage._client._passthrough_statuses == frozenset({404})
        assert stage._client._passthrough is False


async def test_snapshot_stops_at_the_last_stage_of_the_stored_sequence():
    ocr = {**_job("DONE", {"blocks": []}), "pipeline_name_sequence": ["guardrails", "extraction"]}
    stages = _stages(ocr, _job("DONE", {}), _job("DONE", {}))

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status, outcome.results) == ("OCR", "DONE", {"OCR": {"blocks": []}})
    assert [stage.calls for stage in stages] == [1, 0, 0]


async def test_snapshot_of_a_job_without_a_sequence_reads_the_whole_pipeline():
    """Jobs submitted before pipeline_name_sequence existed, or with an unreadable one."""
    ocr = {**_job("DONE", {"blocks": []}), "pipeline_name_sequence": ["extraction", "scoring"]}
    stages = _stages(ocr, _job("DONE", {}), _job("PROCESSING"))

    outcome = await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)

    assert outcome is not None
    assert (outcome.stage, outcome.status) == ("SCORING", "PROCESSING")


def test_a_request_that_ended_before_scoring_is_answered_with_that_result(client, auth, stub_waiter):
    structuring = {"fields": {"nomor_npwp": {"value": "12.345.678.9-012.345"}}, "flag": False}
    stub_waiter.snapshot_outcome = WaitOutcome(
        "STRUCTURING", "DONE", results={"OCR": {"blocks": []}, "STRUCTURING": structuring}
    )

    response = _get(client, auth)

    assert response.status_code == 200
    assert (response.json()["job_status"], response.json()["data"]) == ("completed", structuring)


def test_an_unreachable_stage_is_named_in_the_error(client, auth, stub_waiter):
    from app.services.pipeline_waiter import StageError

    stub_waiter.snapshot_error = StageError("structuring", UpstreamUnavailable("structuring service is unavailable"))

    response = _get(client, auth)

    assert response.status_code == 503
    assert (response.json()["pipeline_last_stage"], response.json()["message"]) == (
        "structuring",
        "structuring service is unavailable",
    )


async def test_snapshot_names_the_stage_it_could_not_read():
    from app.services.pipeline_waiter import StageError

    stages = _stages(_job("DONE", {}), UpstreamUnavailable("structuring service is unavailable"))

    with pytest.raises(StageError) as exc:
        await PipelineWaiter(stages, poll_interval=0.01).snapshot(RID)
    assert (exc.value.service, exc.value.status_code) == ("structuring", 503)
