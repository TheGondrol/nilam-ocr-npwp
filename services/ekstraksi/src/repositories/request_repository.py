"""
Penyimpanan status request_id: pending -> completed | failed.

Dua implementasi dengan method yang sama (create / get / update), dipilih
get_request_repository() berdasarkan DATABASE_URL:

- InMemoryRequestRepository: dict di proses ini, sama dengan mock ocr-*.
  Cukup untuk satu container; hilang saat restart.
- SqlRequestRepository: tabel ocr_npwp_requests di PostgreSQL lewat
  SQLAlchemy async (skema: db/schema.sql). Dipakai kalau service ini jalan
  lebih dari satu replika atau statusnya harus tahan restart.

Tidak ada file lain yang tahu implementasi mana yang aktif; OcrService hanya
memanggil ketiga method itu.
"""

from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, TypedDict

from sqlalchemy import DateTime, Float, Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column

from ocr_common.database import JSON_TYPE, Base, get_session_factory
from src.core.config import get_settings


class RequestRecord(TypedDict):
    status: str
    result: dict[str, Any] | None
    guardrails: float | None
    error_message: str | None
    created_at: str
    updated_at: str


def _now_with_ds() -> tuple[datetime, str]:
    """Satu `now` untuk kolom timestamp dan `ds` (YYYYMMDD UTC, partisi harian
    ala Hive/BigQuery untuk penarikan batch ke Big Data), supaya keduanya
    tidak pernah berbeda. Konvensi sama dengan ocr-nilam-db."""
    now = datetime.now(UTC)
    return now, now.strftime("%Y%m%d")


def _iso(value: datetime) -> str:
    # SQLite mengembalikan datetime naive; nilainya UTC karena kita yang menulis.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class InMemoryRequestRepository:
    name = "memory"

    def __init__(self) -> None:
        self._requests: dict[str, RequestRecord] = {}

    async def create(self, request_id: str, *, file_name: str | None = None) -> None:
        now = _iso(datetime.now(UTC))
        self._requests[request_id] = {
            "status": "pending",
            "result": None,
            "guardrails": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }

    async def get(self, request_id: str) -> RequestRecord | None:
        record = self._requests.get(request_id)
        return record.copy() if record else None

    async def update(self, request_id: str, **fields: Any) -> None:
        self._requests[request_id].update(**fields, updated_at=_iso(datetime.now(UTC)))


class RequestRow(Base):
    __tablename__ = "ocr_npwp_requests"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE, nullable=True)
    guardrails: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    # Metadata file yang disubmit; isi file tidak pernah disimpan.
    file_name: Mapped[str | None] = mapped_column(String, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ds: Mapped[str] = mapped_column(String, nullable=False)


class SqlRequestRepository:
    name = "postgres"

    def __init__(self, database_url: str):
        self._url = database_url

    async def create(self, request_id: str, *, file_name: str | None = None) -> None:
        now, ds = _now_with_ds()
        async with get_session_factory(self._url)() as session:
            session.add(
                RequestRow(
                    request_id=request_id,
                    status="pending",
                    file_name=file_name,
                    created_at=now,
                    updated_at=now,
                    ds=ds,
                )
            )
            await session.commit()

    async def get(self, request_id: str) -> RequestRecord | None:
        async with get_session_factory(self._url)() as session:
            row = (
                await session.execute(select(RequestRow).where(RequestRow.request_id == request_id))
            ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "status": row.status,
            "result": row.result,
            "guardrails": row.guardrails,
            "error_message": row.error_message,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    async def update(self, request_id: str, **fields: Any) -> None:
        async with get_session_factory(self._url)() as session:
            row = (await session.execute(select(RequestRow).where(RequestRow.request_id == request_id))).scalar_one()
            for name, value in fields.items():
                setattr(row, name, value)
            row.updated_at = datetime.now(UTC)
            await session.commit()


@lru_cache
def get_request_repository():
    """Satu instance per proses; engine di bawahnya sudah punya pool koneksi."""
    url = get_settings().database_url
    return SqlRequestRepository(url) if url else InMemoryRequestRepository()
