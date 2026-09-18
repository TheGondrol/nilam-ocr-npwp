"""
Kedua implementasi repository diuji dengan kontrak yang sama. Yang SQL
dijalankan terhadap SQLite (aiosqlite) sungguhan, bukan fake session, supaya
model, query, dan konversi tipenya benar-benar teruji; di produksi dialeknya
PostgreSQL (JSONB), skema di db/schema.sql.
"""

import pytest
from src.core import database
from src.core.config import Settings, get_settings
from src.repositories.request_repository import (
    InMemoryRequestRepository,
    SqlRequestRepository,
    get_request_repository,
)


@pytest.fixture
async def sqlite_url(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    dummy = Settings(api_key="x", database_url=url, _env_file=None)
    monkeypatch.setattr("src.core.database.get_settings", lambda: dummy)
    monkeypatch.setattr("src.repositories.request_repository.get_settings", lambda: dummy)
    await database.dispose_engine()
    get_request_repository.cache_clear()
    async with database.get_engine().begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)
    yield url
    await database.dispose_engine()
    get_request_repository.cache_clear()


@pytest.fixture(params=["memory", "sql"])
async def repository(request, sqlite_url):
    if request.param == "memory":
        return InMemoryRequestRepository()
    return SqlRequestRepository()


async def test_create_then_get_is_pending(repository):
    await repository.create("OCR_1", file_name="npwp.jpg")
    record = await repository.get("OCR_1")
    assert record is not None
    assert record["status"] == "pending"
    assert record["result"] is None
    assert record["guardrails"] is None
    assert record["error_message"] is None
    assert record["created_at"].endswith("+00:00")
    assert record["created_at"] == record["updated_at"]


async def test_get_unknown_returns_none(repository):
    assert await repository.get("OCR_missing") is None


async def test_update_completed_stores_result_and_guardrails(repository):
    await repository.create("OCR_2")
    result = {"nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.9}}
    await repository.update("OCR_2", status="completed", result=result, guardrails=0.9)
    record = await repository.get("OCR_2")
    assert record["status"] == "completed"
    assert record["result"] == result
    assert record["guardrails"] == 0.9
    assert record["updated_at"] >= record["created_at"]


async def test_update_failed_stores_error_message(repository):
    await repository.create("OCR_3")
    await repository.update("OCR_3", status="failed", error_message="Uploaded image is not recognized as an NPWP")
    record = await repository.get("OCR_3")
    assert record["status"] == "failed"
    assert record["error_message"] == "Uploaded image is not recognized as an NPWP"
    assert record["result"] is None


async def test_get_returns_copy_not_internal_state(repository):
    await repository.create("OCR_4")
    record = await repository.get("OCR_4")
    record["status"] = "tampered"
    assert (await repository.get("OCR_4"))["status"] == "pending"


def test_repository_is_memory_without_database_url(monkeypatch):
    dummy = Settings(api_key="x", _env_file=None)
    monkeypatch.setattr("src.repositories.request_repository.get_settings", lambda: dummy)
    get_request_repository.cache_clear()
    try:
        assert isinstance(get_request_repository(), InMemoryRequestRepository)
    finally:
        get_request_repository.cache_clear()


def test_repository_is_sql_with_database_url(monkeypatch):
    dummy = Settings(api_key="x", database_url="postgresql+asyncpg://u:p@h/db", _env_file=None)
    monkeypatch.setattr("src.repositories.request_repository.get_settings", lambda: dummy)
    get_request_repository.cache_clear()
    try:
        assert isinstance(get_request_repository(), SqlRequestRepository)
    finally:
        get_request_repository.cache_clear()


def test_engine_requires_database_url(monkeypatch):
    dummy = Settings(api_key="x", _env_file=None)
    monkeypatch.setattr("src.core.database.get_settings", lambda: dummy)
    database.get_engine.cache_clear()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        database.get_engine()


async def test_ocr_contract_end_to_end_on_sql_repository(client, auth, sqlite_url, monkeypatch):
    """Alur HTTP lengkap dengan repository SQL di belakangnya."""
    from tests.conftest import image_upload

    monkeypatch.setattr("src.api.v1.ocr.get_request_repository", lambda: SqlRequestRepository())
    assert get_settings().database_url is None  # app tetap memakai settings test; hanya repo yang diganti

    request_id = client.post("/v1/generate-request-id", headers=auth).json()["request_id"]
    extract = client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": request_id}, files=image_upload("npwp.jpg")
    )
    assert extract.status_code == 200
    poll = client.get(f"/v1/get-ocr-result/{request_id}", headers=auth).json()
    assert poll["data"]["status"] == "completed"
    assert poll["data"]["result"] == extract.json()["data"]
    assert poll["guardrails"] == extract.json()["guardrails"]
    assert (
        client.post(
            "/v1/extract-ocr", headers=auth, data={"request_id": request_id}, files=image_upload("npwp.jpg")
        ).status_code
        == 409
    )
