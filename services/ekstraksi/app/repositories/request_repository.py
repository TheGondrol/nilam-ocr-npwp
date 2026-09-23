from datetime import UTC, datetime
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Mapped

from ocr_common.pipeline.database import Base, get_session_factory
from ocr_common.pipeline.tables import OCR_NPWP_REQUESTS


class RequestRecord(TypedDict):
    status: str
    result: dict[str, Any] | None
    guardrails: float | None
    error_message: str | None
    created_at: str
    updated_at: str


def _now_with_ds() -> tuple[datetime, str]:
    now = datetime.now(UTC)
    return now, now.strftime("%Y%m%d")


def _iso(value: datetime) -> str:
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
    __table__ = OCR_NPWP_REQUESTS

    request_id: Mapped[str]
    status: Mapped[str]
    result: Mapped[dict[str, Any] | None]
    guardrails: Mapped[float | None]
    error_message: Mapped[str | None]
    file_name: Mapped[str | None]
    file_size_bytes: Mapped[int | None]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    ds: Mapped[str]


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


RequestRepository = InMemoryRequestRepository | SqlRequestRepository


def build_request_repository(database_url: str | None) -> RequestRepository:
    return SqlRequestRepository(database_url) if database_url else InMemoryRequestRepository()
