"""Threshold provider: periodically refreshes thresholds from the management DB.

The shared management service (`ms-bribrain-manage-thservice`) stores thresholds
for every prediction service in a Postgres table. Each service initializes a
`ThresholdProvider` once during FastAPI startup, passing in:

  - service_name (e.g. "dgc_dt")
  - in-memory defaults (used as fallback when DB is unreachable)

All other tunables (refresh interval, table/schema, pool sizing) are read from
the `threshold_provider:` section of the service's `config.yaml`. If the
section is missing, sensible defaults are used.

If the database is unreachable (or no connection URL is configured), the
provider falls back to the in-memory defaults so the service can still serve
traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL_SECONDS = 3600
DEFAULT_TABLE_NAME = "bribrain_model_thresholds"
DEFAULT_DB_SCHEMA = "public"
DEFAULT_POOL_SIZE = 2
DEFAULT_MAX_OVERFLOW = 2
DEFAULT_POOL_RECYCLE = 3600


def _load_settings() -> dict[str, Any]:
    """Pull the `threshold_provider:` block from the service's CONFIG_PATH yaml."""
    path = Path(os.environ.get("CONFIG_PATH", "config.yaml"))
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s for threshold_provider settings: %s", path, exc)
        return {}
    block = raw.get("threshold_provider") or {}
    if not isinstance(block, dict):
        return {}
    return block


class ThresholdProvider:
    """Caches thresholds for one service and refreshes from DB on a timer."""

    def __init__(self, service_name: str, defaults: dict[str, float]) -> None:
        self._service_name = service_name
        self._defaults: dict[str, float] = dict(defaults)
        self._values: dict[str, float] = dict(defaults)

        settings = _load_settings()
        self._refresh_interval = int(
            settings.get("refresh_interval_seconds", DEFAULT_REFRESH_INTERVAL_SECONDS)
        )
        self._table_name = str(settings.get("table_name", DEFAULT_TABLE_NAME))
        self._db_schema = str(settings.get("schema", DEFAULT_DB_SCHEMA))
        self._pool_size = int(settings.get("pool_size", DEFAULT_POOL_SIZE))
        self._max_overflow = int(settings.get("max_overflow", DEFAULT_MAX_OVERFLOW))
        self._pool_recycle = int(settings.get("pool_recycle", DEFAULT_POOL_RECYCLE))

        self._engine: Optional[AsyncEngine] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

    @property
    def _qualified_table(self) -> str:
        if self._db_schema:
            return f'"{self._db_schema}"."{self._table_name}"'
        return f'"{self._table_name}"'

    async def initialize(self) -> None:
        url = os.getenv("THRESHOLD_DB_URL") or os.getenv("DATABASE_URL")
        if not url:
            logger.warning(
                "Neither THRESHOLD_DB_URL nor DATABASE_URL is set; "
                "ThresholdProvider for '%s' will use defaults only",
                self._service_name,
            )
            return

        try:
            self._engine = create_async_engine(
                url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=True,
                pool_recycle=self._pool_recycle,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to create threshold DB engine for '%s': %s; "
                "falling back to defaults",
                self._service_name,
                exc,
            )
            self._engine = None
            return

        self._stop_event = asyncio.Event()
        await self._refresh_once()
        self._task = asyncio.create_task(
            self._refresh_loop(),
            name=f"threshold-refresh-{self._service_name}",
        )
        logger.info(
            "ThresholdProvider initialized for '%s' (refresh every %ds, table=%s)",
            self._service_name,
            self._refresh_interval,
            self._qualified_table,
        )

    async def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except BaseException:  # noqa: BLE001
                    pass
            self._task = None
        if self._engine is not None:
            try:
                await self._engine.dispose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error disposing threshold engine: %s", exc)
            self._engine = None
        self._stop_event = None
        logger.info("ThresholdProvider for '%s' shut down", self._service_name)

    async def _refresh_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._refresh_interval,
                )
                return
            except asyncio.TimeoutError:
                await self._refresh_once()

    async def _refresh_once(self) -> None:
        if self._engine is None:
            return
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    text(
                        f"SELECT threshold_key, threshold_value "
                        f"FROM {self._qualified_table} "
                        "WHERE service_name = :name"
                    ),
                    {"name": self._service_name},
                )
                rows = list(result.fetchall())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Threshold refresh failed for '%s': %s",
                self._service_name,
                exc,
            )
            return

        new_values = dict(self._defaults)
        for key, value in rows:
            try:
                new_values[key] = float(value)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping non-numeric threshold %s.%s=%r",
                    self._service_name,
                    key,
                    value,
                )

        old_values = self._values
        changed = {
            k: (old_values.get(k), new_values[k])
            for k in new_values
            if old_values.get(k) != new_values[k]
        }
        removed = [k for k in old_values if k not in new_values]
        self._values = new_values

        logger.info(
            "Threshold refresh: '%s' loaded %d row(s) from %s -> %s",
            self._service_name,
            len(rows),
            self._qualified_table,
            self._values,
        )
        if changed:
            for key, (old, new) in changed.items():
                logger.info(
                    "Threshold change: %s.%s %s -> %s",
                    self._service_name,
                    key,
                    old,
                    new,
                )
        else:
            logger.info(
                "Threshold change: %s no values changed since last refresh",
                self._service_name,
            )
        if removed:
            logger.info(
                "Threshold change: %s keys removed from DB (reverted to defaults): %s",
                self._service_name,
                removed,
            )

    def get(self, key: str) -> float:
        if key in self._values:
            return self._values[key]
        if key in self._defaults:
            return self._defaults[key]
        raise KeyError(
            f"Unknown threshold '{key}' for service '{self._service_name}'"
        )

    def get_all(self) -> dict[str, float]:
        return dict(self._values)

    @property
    def service_name(self) -> str:
        return self._service_name


_provider: Optional[ThresholdProvider] = None


def init_provider(service_name: str, defaults: dict[str, float]) -> ThresholdProvider:
    """Create the singleton provider for this process. Idempotent per process."""
    global _provider
    if _provider is None:
        _provider = ThresholdProvider(service_name, defaults)
    return _provider


def get_provider() -> ThresholdProvider:
    if _provider is None:
        raise RuntimeError(
            "ThresholdProvider not initialized; call init_provider() in lifespan startup"
        )
    return _provider
