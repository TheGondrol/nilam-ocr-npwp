from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from ocr_common.pipeline import STAGE_OCR, InMemoryJobRepository, database
from ocr_common.pipeline.reaper import StaleJobReaper
from ocr_common.pipeline.repository_sql import SqlJobRepository

LEASE = 60.0
INPUT = {"document_type": "npwp", "guardrails": None, "file_url": "http://minio/npwp.jpg"}


class Resumed:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict | None]] = []

    async def __call__(self, request_id: str, input: dict | None) -> None:
        self.calls.append((request_id, input))


@pytest.fixture(params=["memory", "sql"])
async def repository(request, tmp_path):
    if request.param == "memory":
        yield InMemoryJobRepository(LEASE)
        return
    await database.dispose_engines()
    repo = SqlJobRepository(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}", "ocr", lease_seconds=LEASE, stage=STAGE_OCR)
    async with repo.engine.begin() as conn:
        await conn.run_sync(repo.metadata.create_all)
    yield repo
    await database.dispose_engines()


async def _age(repository, request_id: str, seconds: float) -> None:
    past = datetime.now(UTC) - timedelta(seconds=seconds)
    if isinstance(repository, InMemoryJobRepository):
        repository._jobs[request_id]["updated_at"] = past.isoformat()
        return
    jobs = repository._jobs
    async with repository.engine.begin() as conn:
        await conn.execute(update(jobs).where(jobs.c.request_id == request_id).values(updated_at=past))


async def test_only_jobs_past_their_lease_are_run_again_and_only_once(repository):
    await repository.claim("REQ_stale", input=INPUT)
    await repository.claim("REQ_fresh", input=INPUT)
    await repository.claim("REQ_done", input=INPUT)
    await repository.complete("REQ_done", {"blocks": []})
    await _age(repository, "REQ_stale", LEASE + 1)
    await _age(repository, "REQ_done", LEASE + 1)
    resumed = Resumed()
    reaper = StaleJobReaper(repository, resumed, stage=STAGE_OCR)

    assert await reaper.run_once() == 1
    assert resumed.calls == [("REQ_stale", INPUT)]
    assert (await repository.get("REQ_stale"))["status"] == "PROCESSING"

    assert await reaper.run_once() == 0, "reclaiming refreshed updated_at, so the job is not stale any more"


async def test_a_job_claimed_before_the_input_column_existed_is_still_run_again(repository):
    await repository.claim("REQ_old")
    await _age(repository, "REQ_old", LEASE + 1)
    resumed = Resumed()

    assert await StaleJobReaper(repository, resumed, stage=STAGE_OCR).run_once() == 1
    assert resumed.calls == [("REQ_old", None)]


async def test_the_loop_stops_cleanly(repository):
    reaper = StaleJobReaper(repository, Resumed(), stage=STAGE_OCR, interval_seconds=60)
    reaper.start()
    await reaper.stop()
    assert reaper._task is None
