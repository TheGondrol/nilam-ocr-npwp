"""Database service for async database operations."""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.schemas.database_schema import OcrKtpLog
from src.core.config import get_config
from src.core.crypto import encrypt
from src.core.logging import get_logger

load_dotenv()  # otomatis cari .env di cwd

logger = get_logger(__name__)

# ---------- Database ----------
DATABASE_URL = os.getenv("DATABASE_URL")

_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_engine():
    """Initialize the async SQLAlchemy engine and session factory.
    
    Should be called during application startup.
    """
    global _async_engine, _async_session_factory
    if _async_engine is None and DATABASE_URL:
        config = get_config()
        _async_engine = create_async_engine(
            DATABASE_URL,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=config.database.pool_pre_ping,
            pool_recycle=config.database.pool_recycle,
            pool_timeout=config.database.pool_timeout,
            connect_args={
                "timeout": config.database.connect_timeout,
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
    payload: Dict[str, Any],
    error_message: str,
    result: Dict[str, Any] | str,
    processing_time: float
) -> None:
    """
    Insert log entry into database asynchronously.
    
    Args:
        request_id: Unique request identifier
        response_code: HTTP response code
        payload: Request payload data
        error_message: Error message if any
        result: Processing result
        processing_time: Time taken to process request
    """
    config = get_config()
    if not config.logging.log_to_database:
        logger.info(f"Database logging is disabled, skipping log for request_id: {request_id}")
        return
    
    if _async_session_factory is None:
        logger.warning(f"Database not initialized, skipping log for request_id: {request_id}")
        return
    
    try:
        async with _async_session_factory() as session:
            log = OcrKtpLog(
                request_id=request_id,
                response_code=response_code,
                payload=encrypt(payload),
                error_message=error_message,
                result=encrypt(result),
                processing_time=processing_time,
            )
            session.add(log)
            await session.commit()
        logger.info(f"Log inserted for request_id: {request_id}")
    except Exception as e:
        logger.error(f"Failed to insert log for request_id: {request_id}, error: {e}")
        return
