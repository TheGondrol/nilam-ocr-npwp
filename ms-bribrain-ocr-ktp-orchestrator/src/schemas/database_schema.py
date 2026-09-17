import enum

from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    Integer,
    LargeBinary,
    TIMESTAMP,
    Float,
    CheckConstraint,
)
from sqlalchemy.orm import declarative_base, DeclarativeMeta
from sqlalchemy.sql import func
from src.core.config import config


class OcrStatus(str, enum.Enum):
    """Valid states for OCR result lifecycle."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # business rejection (HTTP 4xx), distinct from a server error

# ---------- Base ----------
Base: DeclarativeMeta = declarative_base()

# ---------- Model ----------
class OcrKtpLog(Base):
    __tablename__ = config.get("database.table_name")  # bribrain_ocr_ktp_orchestrator
    __table_args__ = {"schema": config.get("database.schema")}  # public

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(100), nullable=False, index=True)
    response_code = Column(Integer)
    payload = Column(LargeBinary)        # AES-256-GCM ciphertext (see core.crypto)
    error_message = Column(Text)
    result = Column(LargeBinary)         # AES-256-GCM ciphertext
    processing_time = Column(Float)
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class OcrKtpResult(Base):
    __tablename__ = config.get("database.result_table_name")  # bribrain_ocr_ktp_result
    __table_args__ = (
        CheckConstraint(
            f"status IN ({', '.join(repr(s.value) for s in OcrStatus)})",
            name="chk_ocr_ktp_result_status",
        ),
        {"schema": config.get("database.schema")},  # public
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False, server_default="pending")
    result = Column(LargeBinary)         # AES-256-GCM ciphertext
    error_message = Column(Text)
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )