"""Database service for async database operations."""

import logging
import os
from typing import Any, Optional
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from ..core.logging import logger
from src.schemas.database_schema import OcrKtpLog
from ..core.config import config

load_dotenv()  # otomatis cari .env di cwd

# ---------- Database ----------
DATABASE_URL = os.getenv("DATABASE_URL")

_async_engine: Optional[Any] = None
_async_session_factory: Optional[Any] = None


async def init_engine():
    """Initialize the async SQLAlchemy engine and session factory.
    
    Should be called during application startup.
    """
    global _async_engine, _async_session_factory
    if _async_engine is None and DATABASE_URL:
        _async_engine = create_async_engine(
            DATABASE_URL,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_pre_ping=config.db_pool_pre_ping,
            pool_recycle=config.db_pool_recycle,
            pool_timeout=config.db_pool_timeout,
            connect_args={
                "timeout": config.db_connect_timeout,
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


# ---------- Insert Log ----------
async def insert_log(
    request_id: str,
    response_code: int,
    payload: dict | None,
    error_message: str,
    result: dict | None,
    processing_time: float
) -> None:
    """
    Insert a log entry into the database asynchronously.
    
    Args:
        request_id: Request identifier
        response_code: HTTP response code
        payload: JSON-compatible dict for JSONB column (e.g., {"filename": "test.jpg"})
        error_message: Error message string
        result: JSON-compatible dict for JSONB column (e.g., {"status": "OK"})
        processing_time: Processing time in seconds
    """
    if not config.log_to_database:
        logger.info(f"Database logging disabled, skipping log for request_id: {request_id}")
        return
    
    if DATABASE_URL is None:
        logger.warning(f"DATABASE_URL not set, skipping log for request_id: {request_id}")
        return
    
    if _async_session_factory is None:
        logger.warning(f"Database not initialized, skipping log for request_id: {request_id}")
        return
    
    try:
        async with _async_session_factory() as session:
            log = OcrKtpLog(
                request_id=request_id,
                response_code=response_code,
                payload=payload,
                error_message=error_message,
                result=result,
                processing_time=processing_time,
            )
            session.add(log)
            await session.commit()
        logger.info(f"Log inserted for request_id: {request_id}")
    except Exception as e:
        logger.error(f"Failed to insert log for request_id: {request_id}, error: {e}")
        return
