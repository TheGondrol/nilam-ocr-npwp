"""Database service for async database operations."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import os
import logging
from src.schemas.database_schema import OcrKtpLog
from src.schemas.api_schema import LogEntry
from dotenv import load_dotenv
from src.core.config import get_config
from src.core.crypto import encrypt

load_dotenv()  # otomatis cari .env di cwd
config = get_config()
log_config = config.get_logging_config()
db_config = config.get_database_config()

logger = logging.getLogger(__name__)

# ---------- Database ----------
DATABASE_URL = os.getenv("DATABASE_URL")

insert_to_database = log_config.get("insert_to_database", False)

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
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10),
            pool_pre_ping=db_config.get("pool_pre_ping", True),
            pool_recycle=db_config.get("pool_recycle", 3600),
            pool_timeout=db_config.get("pool_timeout", 10),
            connect_args={
                "timeout": db_config.get("connect_timeout", 5),
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
async def insert_log(log_entry: LogEntry):
    """
    Insert a log entry into the database asynchronously.

    Args:
        log_entry: LogEntry model containing all log fields.
    """
    if not insert_to_database:
        logger.info(f"Database is disabled, skipping log for request_id: {log_entry.request_id}")
        return
    
    if _async_session_factory is None:
        logger.warning(f"Database not initialized, skipping log for request_id: {log_entry.request_id}")
        return
    
    try:
        async with _async_session_factory() as session:
            log = OcrKtpLog(
                request_id=log_entry.request_id,
                response_code=log_entry.response_code,
                payload=encrypt(log_entry.payload),
                error_message=log_entry.error_message,
                result=encrypt(log_entry.result),
                processing_time=log_entry.processing_time,
            )
            session.add(log)
            await session.commit()
        logger.info(f"Log inserted for request_id: {log_entry.request_id}")
    except Exception as e:
        logger.error(f"Failed to insert log for request_id: {log_entry.request_id}, error: {e}")
        return
