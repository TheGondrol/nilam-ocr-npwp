import pytest

from ocr_common.pipeline import STAGE_SCORING, InMemoryJobRepository, StagePipeline
from ocr_common.testing import RecordingCallback, make_client, wait_for_job

from app.dependencies import get_confidence_service, get_job_service, get_trust_model
from app.main import app
from app.services.confidence_service import ConfidenceService
from app.services.job_service import ScoringJobService

GUARDRAILS = {"passed": True, "reason": None}
STRUCTURING = {
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.96,
            "source": "NPWP : 12.345.678.9-012.345",
        },
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.95, "source": "NAMA : BUDI SANTOSO"},
        "nama_badan": {"value": None, "confidence": 0.0, "source": None},
    },
    "flag": True,
    "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
}


@pytest.fixture
def harness():
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    service = ScoringJobService(pipeline, get_confidence_service())
    app.dependency_overrides[get_job_service] = lambda: service
    with make_client(app) as client:
        yield client, callback
    app.dependency_overrides.pop(get_job_service, None)


def _payload(request_id):
    return {
        "request_id": request_id,
        "guardrails": GUARDRAILS,
        "ocr": {"blocks": [{"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "page": 0}]},
        "structuring": STRUCTURING,
    }


def test_submit_returns_202_then_scores_and_sends_final_result(harness, auth):
    client, callback = harness

    response = client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_1"))
    assert response.status_code == 202
    assert response.json()["data"] == {
        "request_id": "REQ_1",
        "stage": "SCORING",
        "status": "PROCESSING",
        "duplicate": False,
    }

    job = wait_for_job(client, "/v1/scoring/jobs/REQ_1")
    assert job["status"] == "DONE"
    result = job["result"]
    assert set(result) == {"npwp_confidence", "name_confidence", "fields", "payload"}
    assert result["fields"] == {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": int(result["npwp_confidence"] >= 0.5),
            "threshold": 0.5,
        },
        "nama": {"value": "BUDI SANTOSO", "confidence": int(result["name_confidence"] >= 0.5), "threshold": 0.5},
    }
    assert 0 <= result["npwp_confidence"] <= 1 and 0 <= result["name_confidence"] <= 1
    assert result["payload"]["npwp"] == "123456789012345"
    assert result["payload"]["name_base"] == "BUDI SANTOSO"
    assert result["payload"]["avg_doc_score"] == 0.96
    assert result["payload"]["flag"] is True
    assert result["payload"]["guardrail_probability"] is None

    assert len(callback.calls) == 1
    call = callback.calls[0]
    assert (call["stage"], call["status"]) == ("SCORING", "DONE")
    assert call["result"] == {
        "fields": {
            "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96},
            "nama": {"value": "BUDI SANTOSO", "confidence": 0.95},
            "nama_badan": {"value": None, "confidence": 0.0},
        },
        "scoring": {"npwp_confidence": result["npwp_confidence"], "name_confidence": result["name_confidence"]},
        "guardrails": GUARDRAILS,
        "flag": True,
        "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
    }


def test_same_request_id_is_not_processed_twice(harness, auth):
    client, callback = harness
    client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_2"))
    wait_for_job(client, "/v1/scoring/jobs/REQ_2")

    again = client.post("/v1/scoring/jobs", headers=auth, json=_payload("REQ_2"))
    assert again.status_code == 202
    assert again.json()["data"]["duplicate"] is True
    assert len(callback.calls) == 1


def test_missing_structuring_without_a_database_is_422(harness, auth):
    client, _ = harness
    response = client.post("/v1/scoring/jobs", headers=auth, json={"request_id": "REQ_4"})
    assert response.status_code == 422
    assert "structuring is missing" in response.json()["message"]


def test_get_unknown_job_is_404(harness, auth):
    client, _ = harness
    assert client.get("/v1/scoring/jobs/REQ_missing", headers=auth).status_code == 404


def test_requires_api_key(harness):
    client, _ = harness
    assert client.post("/v1/scoring/jobs", json=_payload("REQ_5")).status_code == 401


class FakeResults:
    """Stands in for the shared database: {stage_prefix: {request_id: result}}."""

    def __init__(self, **stored):
        self.stored = stored

    async def get(self, stage_prefix, request_id):
        return self.stored.get(stage_prefix, {}).get(request_id)


def test_by_reference_reads_structuring_and_ocr_from_the_database(auth):
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    ocr = {"blocks": [{"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "page": 0}]}
    results = FakeResults(structuring={"REQ_ref": STRUCTURING}, ocr={"REQ_ref": ocr})
    service = ScoringJobService(pipeline, get_confidence_service(), results=results)
    app.dependency_overrides[get_job_service] = lambda: service
    try:
        with make_client(app) as client:
            body = {"request_id": "REQ_ref", "guardrails": GUARDRAILS}
            assert client.post("/v1/scoring/jobs", headers=auth, json=body).status_code == 202
            job = wait_for_job(client, "/v1/scoring/jobs/REQ_ref")
    finally:
        app.dependency_overrides.pop(get_job_service, None)

    assert job["status"] == "DONE"
    assert job["result"]["payload"]["npwp"] == "123456789012345"
    assert job["result"]["payload"]["avg_doc_score"] == 0.96, "the OCR blocks were read from the database too"
    [call] = callback.calls
    assert call["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"
    assert call["result"]["guardrails"] == GUARDRAILS


async def test_a_stale_job_is_run_again_from_what_the_database_holds():
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=InMemoryJobRepository(), callback=callback)
    service = ScoringJobService(
        pipeline, get_confidence_service(), results=FakeResults(structuring={"REQ_stale": STRUCTURING})
    )
    await pipeline.repository.claim("REQ_stale", input={"guardrails": GUARDRAILS})

    await service.resume("REQ_stale", {"guardrails": GUARDRAILS})
    await pipeline.runner.drain(5)

    assert (await pipeline.get("REQ_stale"))["status"] == "DONE"
    [call] = callback.calls
    assert (call["stage"], call["status"]) == ("SCORING", "DONE")
    assert call["result"]["guardrails"] == GUARDRAILS
    assert call["result"]["fields"]["nomor_npwp"]["value"] == "12.345.678.9-012.345"


def test_scoring_ends_every_request_it_runs_for(harness, auth):
    client, callback = harness
    sequence = ["extraction", "structuring", "scoring"]

    client.post("/v1/scoring/jobs", headers=auth, json={**_payload("REQ_seq"), "pipeline_name_sequence": sequence})

    job = wait_for_job(client, "/v1/scoring/jobs/REQ_seq")
    assert (job["status"], job["pipeline_name_sequence"]) == ("DONE", sequence)
    [done] = callback.calls
    assert (done["stage"], done["status"], done["final"]) == ("SCORING", "DONE", True)


@pytest.mark.parametrize("sequence", [["guardrails", "extraction", "structuring"], ["extraction", "scoring"]])
def test_a_sequence_without_scoring_or_out_of_order_is_422(harness, auth, sequence):
    client, _ = harness

    response = client.post(
        "/v1/scoring/jobs", headers=auth, json={**_payload("REQ_seq_bad"), "pipeline_name_sequence": sequence}
    )

    assert response.status_code == 422


class RecordingRepository(InMemoryJobRepository):
    """Keeps the outcome row's `result_data` the SQL repository would write."""

    def __init__(self):
        super().__init__()
        self.outcomes: dict[str, dict | None] = {}

    async def complete(self, request_id, result, *, outcome_data=None, rejection=None, messages=()):
        self.outcomes[request_id] = outcome_data
        await super().complete(request_id, result, outcome_data=outcome_data, rejection=rejection, messages=messages)


class FixedConfidence(ConfidenceService):
    """The trust model's probabilities fixed at npwp 0.8, name 0.6."""

    def predict(self, payload):
        return {"npwp_confidence": 0.8, "name_confidence": 0.6}


@pytest.fixture
def outcome_harness():
    repository = RecordingRepository()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=repository, callback=RecordingCallback())
    service = ScoringJobService(pipeline, FixedConfidence(get_trust_model()), 0.5)
    app.dependency_overrides[get_job_service] = lambda: service
    with make_client(app) as client:
        yield client, repository, service
    app.dependency_overrides.pop(get_job_service, None)


def test_column_confidence_threshold_sets_each_fields_confidence(outcome_harness, auth):
    client, repository, _ = outcome_harness
    body = {**_payload("REQ_col"), "column_confidence_threshold": {"nomor_npwp": 0.9, "nama": 0.5}}

    assert client.post("/v1/scoring/jobs", headers=auth, json=body).status_code == 202
    wait_for_job(client, "/v1/scoring/jobs/REQ_col")

    assert repository.outcomes["REQ_col"] == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    }


def test_a_field_left_out_uses_field_confidence_threshold(outcome_harness, auth):
    client, repository, _ = outcome_harness

    client.post("/v1/scoring/jobs", headers=auth, json={**_payload("REQ_none")})
    client.post(
        "/v1/scoring/jobs",
        headers=auth,
        json={**_payload("REQ_nama"), "column_confidence_threshold": {"nama": 0.7}},
    )
    wait_for_job(client, "/v1/scoring/jobs/REQ_none")
    wait_for_job(client, "/v1/scoring/jobs/REQ_nama")

    assert [repository.outcomes["REQ_none"][name]["confidence"] for name in ("nomor_npwp", "nama")] == [1, 1]
    assert [repository.outcomes["REQ_nama"][name]["confidence"] for name in ("nomor_npwp", "nama")] == [1, 0]


async def test_a_stale_job_is_run_again_with_its_stored_column_thresholds(outcome_harness):
    _, repository, service = outcome_harness
    service._results = FakeResults(structuring={"REQ_stale_col": STRUCTURING})
    input = {"guardrails": GUARDRAILS, "column_confidence_threshold": {"nomor_npwp": 0.9}}
    await repository.claim("REQ_stale_col", input=input)

    await service.resume("REQ_stale_col", input)
    await service._pipeline.runner.drain(5)

    assert repository.outcomes["REQ_stale_col"]["nomor_npwp"]["confidence"] == 0


@pytest.mark.parametrize(
    "thresholds", [{"npwp": 0.9}, {"nomor_npwp": 1.5}, {"nama": -0.1}, {"nama": "tinggi"}, [0.9, 0.5]]
)
def test_an_invalid_column_confidence_threshold_is_422(harness, auth, thresholds):
    client, _ = harness

    response = client.post(
        "/v1/scoring/jobs", headers=auth, json={**_payload("REQ_col_bad"), "column_confidence_threshold": thresholds}
    )

    assert response.status_code == 422


def test_the_0_1_decision_is_stored_with_the_result_and_matches_the_outcome_row(outcome_harness, auth):
    client, repository, _ = outcome_harness
    body = {**_payload("REQ_store"), "column_confidence_threshold": {"nomor_npwp": 0.9}}

    client.post("/v1/scoring/jobs", headers=auth, json=body)
    job = wait_for_job(client, "/v1/scoring/jobs/REQ_store")

    # FixedConfidence: npwp 0.8 (< 0.9 given), name 0.6 (>= FIELD_CONFIDENCE_THRESHOLD 0.5)
    assert job["result"]["fields"] == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0, "threshold": 0.9},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1, "threshold": 0.5},
    }
    assert repository.outcomes["REQ_store"] == {
        name: {"value": field["value"], "confidence": field["confidence"]}
        for name, field in job["result"]["fields"].items()
    }
