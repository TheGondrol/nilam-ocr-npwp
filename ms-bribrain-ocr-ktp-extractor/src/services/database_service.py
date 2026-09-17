"""Database service for async database operations."""

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine

from src.models.database import OcrKtpLog
from src.core.config import settings
from src.core.crypto import encrypt

load_dotenv()  # Auto-load .env from cwd

logger = logging.getLogger(__name__)

# ---------- Database Configuration ----------
DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_engine():
    """Initialize the async SQLAlchemy engine and session factory.

    Should be called during application startup.
    """
    global _async_engine, _async_session_factory
    if _async_engine is None and DATABASE_URL:
        _async_engine = create_async_engine(
            DATABASE_URL,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=settings.db_pool_pre_ping,
            pool_recycle=settings.db_pool_recycle,
            pool_timeout=settings.db_pool_timeout,
            connect_args={
                "timeout": settings.db_connect_timeout,
            },
        )
        _async_session_factory = async_sessionmaker(
            bind=_async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Async database engine initialized")


async def dispose_engine():
    """Dispose the async database engine and close all connections.

    Should be called during application shutdown for graceful cleanup.
    """
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("Async database engine disposed")


async def insert_log(
    request_id: str,
    response_code: int,
    payload: str,
    error_message: str,
    result: Any,
    processing_time: float
) -> None:
    """
    Insert a log entry for an OCR request asynchronously.

    Args:
        request_id: Unique request identifier
        response_code: HTTP response code
        payload: Request payload (filename)
        error_message: Error message if any
        result: OCR result
        processing_time: Processing time in seconds
    """
    if not settings.log_to_database:
        logger.debug(f"Database logging disabled, skipping for request_id: {request_id}")
        return

    if _async_session_factory is None:
        logger.warning(f"Database not initialized, skipping log for request_id: {request_id}")
        return

    try:
        async with _async_session_factory() as session:
            log = OcrKtpLog(
                request_id=request_id,
                response_code=response_code,
                payload=encrypt({"filename": payload} if isinstance(payload, str) else payload),
                error_message=error_message,
                result=encrypt(result if result else None),
                processing_time=processing_time,
            )
            session.add(log)
            await session.commit()

        logger.debug(f"Log inserted for request_id: {request_id}")

    except Exception as e:
        # Don't let database errors affect the main request
        logger.error(
            f"Failed to insert log for request_id: {request_id}, error: {e}",
            exc_info=True
        )
