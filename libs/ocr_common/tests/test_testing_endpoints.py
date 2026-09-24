import pytest
from fastapi import APIRouter, Depends
from sqlalchemy import MetaData, select

from ocr_common import testing_endpoints
from ocr_common.config import PipelineSettings
from ocr_common.pipeline import STAGE_OCR, STAGE_STRUCTURING, build_stage_pipeline, build_stage_results, database
from ocr_common.pipeline.outbox_sql import SqlOutbox
from ocr_common.pipeline.tables import outbox_table, pipeline_tables, repo_metadata
from ocr_common.testing import make_client
from ocr_common.web.app import create_app
from ocr_common.web.security import verify_api_key
from ocr_common.web.testing_routes import build_testing_router

KEY = {"X-API-Key": "k"}


def _settings(**values) -> PipelineSettings:
    return PipelineSettings(api_key="k", environment="local", _env_file=None, **values)


def _live_router() -> APIRouter:
    router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])

    def live_service() -> str:
        return "live"

    @router.get("/v1/demo/jobs/{request_id}", summary="Status of a job", operation_id="getDemoJob")
    async def get_job(request_id: str, service: str = Depends(live_service)):
        return {"request_id": request_id, "service": service}

    return router


def test_twin_runs_the_same_handler_with_the_swapped_dependency():
    live = _live_router()
    twin = build_testing_router(
        live, {"/v1/demo/jobs/{request_id}": "/v1/demo/jobs-test/{request_id}"}, service=lambda: "testing"
    )
    client = make_client(create_app(settings=_settings(), title="Demo", description="demo", routers=[live, twin]))

    assert client.get("/v1/demo/jobs/R1", headers=KEY).json() == {"request_id": "R1", "service": "live"}
    assert client.get("/v1/demo/jobs-test/R1", headers=KEY).json() == {"request_id": "R1", "service": "testing"}
    assert client.get("/v1/demo/jobs-test/R1").status_code == 401  # the live router's API-key check came along

    operation = client.app.openapi()["paths"]["/v1/demo/jobs-test/{request_id}"]["get"]
    assert operation["operationId"] == "getDemoJobTest"
    assert operation["summary"] == "[Testing] Status of a job"
    assert operation["tags"] == ["Testing"]


def test_twin_of_a_missing_route_is_refused():
    with pytest.raises(ValueError, match="/v1/demo/gone"):
        build_testing_router(_live_router(), {"/v1/demo/gone": "/v1/demo/gone-test"})


def test_testing_path():
    assert testing_endpoints.testing_path("/v1/structuring/jobs") == "/v1/structuring/jobs-test"


def test_testing_tables_are_migrated_with_the_live_ones():
    names = set(repo_metadata().tables)
    for stage in ("ocr", "structuring", "scoring"):
        assert {f"testing_{stage}_jobs", f"testing_{stage}_results"} <= names
    assert {"pipeline_outbox", "testing_pipeline_outbox"} <= names
    index_names = {index.name for index in outbox_table(MetaData(), "testing_").indexes}
    assert index_names == {
        "idx_testing_pipeline_outbox_due",
        "idx_testing_pipeline_outbox_dead",
        "idx_testing_pipeline_outbox_request_id",
    }


@pytest.fixture
async def database_url(tmp_path):
    await database.dispose_engines()
    url = f"sqlite+aiosqlite:///{tmp_path / 'testing.db'}"
    async with database.get_engine(url).begin() as conn:
        await conn.run_sync(repo_metadata().create_all)
    yield url
    await database.dispose_engines()


async def _table_rows(url: str, name: str) -> list[str]:
    async with database.get_engine(url).connect() as conn:
        table = pipeline_tables(name, MetaData())[0]
        return list((await conn.execute(select(table.c.request_id))).scalars())


async def test_testing_pipeline_writes_only_the_testing_tables_and_sends_no_callback(database_url):
    settings = _settings(
        database_url=database_url,
        orchestration_url="http://orchestrator.test",
        orchestration_outcome_table="orchestration_extract_ocr",
        pipeline_outbox=True,
    )
    pipeline = build_stage_pipeline(settings, stage=STAGE_OCR, table_prefix="ocr", testing=True)
    assert pipeline.callbacks is False
    assert pipeline.metrics_stage == "TESTING_OCR"
    assert isinstance(pipeline.outbox, SqlOutbox) and pipeline.outbox.table.name == "testing_pipeline_outbox"

    async def work():
        return {"full_text": "NPWP"}

    await pipeline.submit(
        "REQ_T1",
        work,
        handoff_payload=lambda result: {"request_id": "REQ_T1"},
        next_stage=STAGE_STRUCTURING,
        callback_result=lambda result: {"final": True},
    )
    await pipeline.runner.drain(5)

    # DONE also shows no outcome row was written: its table does not exist here, so the write would fail the job.
    assert (await pipeline.get("REQ_T1"))["status"] == "DONE"
    assert await _table_rows(database_url, "testing_ocr") == ["REQ_T1"]
    assert await _table_rows(database_url, "ocr") == []
    async with database.get_engine(database_url).connect() as conn:
        outbox = (await conn.execute(select(pipeline.outbox.table.c.kind))).scalars().all()
    assert outbox == ["handoff"]  # the hand-off only: no callback queued


async def test_live_pipeline_keeps_its_callbacks_and_metrics_label(database_url):
    settings = _settings(database_url=database_url, orchestration_url="http://orchestrator.test")
    pipeline = build_stage_pipeline(settings, stage=STAGE_OCR, table_prefix="ocr")
    assert pipeline.callbacks is True
    assert pipeline.metrics_stage == STAGE_OCR


async def test_testing_results_read_the_testing_tables(database_url):
    jobs, results = pipeline_tables("testing_ocr", MetaData())
    async with database.get_engine(database_url).begin() as conn:
        await conn.execute(jobs.insert().values(request_id="REQ_T2", status="DONE", ds="20260924"))
        await conn.execute(results.insert().values(request_id="REQ_T2", result={"full_text": "T"}, ds="20260924"))

    settings = _settings(database_url=database_url, orchestration_url="http://orchestrator.test")
    assert await build_stage_results(settings, testing=True).get("ocr", "REQ_T2") == {"full_text": "T"}
    assert await build_stage_results(settings).get("ocr", "REQ_T2") is None
