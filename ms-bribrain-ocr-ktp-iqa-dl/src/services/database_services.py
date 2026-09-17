"""Database service for request logging."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.schemas.database_schema import Base, OcrKtpQualityLog

logger = logging.getLogger("quality")

# Global database engine and session factory
_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_engine() -> None:
    """
    Initialize async database engine and session factory.

    Uses DATABASE_URL environment variable for connection string.
    Format: postgresql+asyncpg://user:pass@host:port/dbname

    Raises:
        ValueError: If DATABASE_URL not set
    """
    global _async_engine, _async_session_factory

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set. Database logging will be disabled.")
        return

    logger.info("Initializing database engine")

    try:
        _async_engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=config.db_pool_pre_ping,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_recycle=config.db_pool_recycle,
            pool_timeout=config.db_pool_timeout,
            connect_args={
                "timeout": config.db_connect_timeout,
            },
        )

        _async_session_factory = async_sessionmaker(
            _async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Database engine initialized successfully")

        # Create tables if they don't exist
        async with _async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e!r}", exc_info=True)
        raise


async def dispose_engine() -> None:
    """Dispose of database engine and close connections."""
    global _async_engine

    if _async_engine:
        logger.info("Disposing database engine")
        await _async_engine.dispose()
        logger.info("Database engine disposed")


async def insert_log(
    request_id: str,
    response_code: int,
    payload: Optional[dict[str, Any]],
    error_message: Optional[str],
    result: Optional[dict[str, Any]],
    processing_time: Optional[float],
) -> None:
    """
    Insert request log into database.

    Args:
        request_id: Unique request identifier
        response_code: HTTP response code
        payload: Request payload (crops data)
        error_message: Error message if failed
        result: Classification result
        processing_time: Processing time in seconds
    """
    if not _async_session_factory:
        logger.debug("Database not configured, skipping log insert")
        return

    try:
        async with _async_session_factory() as session:
            log_entry = OcrKtpQualityLog(
                request_id=request_id,
                response_code=response_code,
                payload=payload,
                error_message=error_message,
                result=result,
                processing_time=processing_time,
            )

            session.add(log_entry)
            await session.commit()

            logger.info(f"Log inserted for request {request_id}")

    except Exception as e:
        logger.error(f"Failed to insert log: {e!r}", exc_info=True)
        # Don't raise - logging failure shouldn't break the API
