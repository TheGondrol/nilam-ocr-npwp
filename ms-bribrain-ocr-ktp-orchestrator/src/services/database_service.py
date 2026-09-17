"""Database service for async database operations."""

import logging
import os
from typing import Optional
from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from src.schemas.database_schema import OcrKtpLog, OcrKtpResult, OcrStatus
from src.core.config import get_settings
from src.core.crypto import encrypt

settings = get_settings()
log_config = settings.logging

logger = logging.getLogger(__name__)

# ---------- Database ----------
DATABASE_URL = os.getenv("DATABASE_URL")

insert_to_database = log_config.log_to_database

_async_engine = None
_async_session_factory = None


async def init_engine():
    """Initialize the async SQLAlchemy engine and session factory.
    
    Should be called during application startup.
    """
    global _async_engine, _async_session_factory
    if _async_engine is None and DATABASE_URL:
        _async_engine = create_async_engine(
            DATABASE_URL,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=settings.database.pool_pre_ping,
            pool_recycle=settings.database.pool_recycle,
            pool_timeout=settings.database.pool_timeout,
            connect_args={
                "timeout": settings.database.connect_timeout,
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
    payload,
    error_message: Optional[str],
    result,
    processing_time: float
):
    """
    Insert a log entry into the database asynchronously.

    Args:
        request_id: Unique request identifier.
        response_code: HTTP response code.
        payload: Request payload data.
        error_message: Error message if any.
        result: Processing result.
        processing_time: Time taken to process the request.
    """
    if not insert_to_database:
        logger.info(f"Database is disabled, skipping log for request_id: {request_id}")
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


# ---------- OCR Result CRUD ----------
async def create_ocr_result(request_id: str) -> None:
    """Insert a new OCR result record with status 'pending'.

    Raises on failure (does NOT silently swallow like insert_log).
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized")

    async with _async_session_factory() as session:
        record = OcrKtpResult(request_id=request_id, status=OcrStatus.PENDING)
        session.add(record)
        await session.commit()
    logger.info(f"[{request_id}] OCR result record created with status 'pending'")


async def claim_ocr_result(request_id: str) -> bool:
    """Atomically claim a pending OCR result for processing.

    Returns True if claim succeeded (record was pending).
    Returns False if record not found or not in 'pending' status.
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized")

    async with _async_session_factory() as session:
        stmt = (
            update(OcrKtpResult)
            .where(
                OcrKtpResult.request_id == request_id,
                OcrKtpResult.status == OcrStatus.PENDING,
            )
            .values(status=OcrStatus.PROCESSING, updated_at=func.now())
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0


async def update_ocr_result(
    request_id: str,
    status: OcrStatus,
    result: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Update OCR result after processing. Only transitions FROM 'processing'.

    Best-effort: logs error but does not raise (client already has synchronous result).
    """
    if _async_session_factory is None:
        logger.warning(f"[{request_id}] Database not initialized, skipping result update")
        return

    try:
        async with _async_session_factory() as session:
            stmt = (
                update(OcrKtpResult)
                .where(
                    OcrKtpResult.request_id == request_id,
                    OcrKtpResult.status == OcrStatus.PROCESSING,
                )
                .values(
                    status=status,
                    result=encrypt(result),
                    error_message=error_message,
                    updated_at=func.now(),
                )
            )
            await session.execute(stmt)
            await session.commit()
        logger.info(f"[{request_id}] OCR result updated to '{status}'")
    except Exception as e:
        logger.error(f"[{request_id}] Failed to update OCR result: {e}")


async def get_ocr_result(request_id: str) -> OcrKtpResult | None:
    """Retrieve OCR result by request_id. Returns None if not found."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized")

    async with _async_session_factory() as session:
        stmt = select(OcrKtpResult).where(OcrKtpResult.request_id == request_id)
        result = await session.execute(stmt)
        return result.scalars().first()
