"""Database schema for request logging."""

from __future__ import annotations

from sqlalchemy import TIMESTAMP, Column, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

from src.core.config import config
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class OcrKtpQualityLog(Base):
    """Log table for OCR quality classification requests."""

    __tablename__ = config.get("database.table_name", "bribrain_ocr_ktp_iqa_dl")
    __table_args__ = {"schema": config.get("database.schema", "public")}

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, comment="Unique request identifier")
    response_code = Column(Integer, nullable=False, comment="HTTP response code")
    payload = Column(JSONB, nullable=True, comment="Request payload (crops data)")
    error_message = Column(Text, nullable=True, comment="Error message if failed")
    result = Column(JSONB, nullable=True, comment="Classification result")
    processing_time = Column(Float, nullable=True, comment="Processing time in seconds")
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp of request",
    )

    def __repr__(self) -> str:
        """String representation of log entry."""
        return f"<OcrKtpQualityLog(id={self.id}, request_id={self.request_id}, response_code={self.response_code})>"
