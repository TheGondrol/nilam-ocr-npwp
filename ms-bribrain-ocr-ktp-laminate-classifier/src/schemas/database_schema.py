from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    TIMESTAMP
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, DeclarativeMeta
from sqlalchemy.sql import func

# ---------- Base ----------
Base: DeclarativeMeta = declarative_base()  # type: ignore[assignment]

# ---------- Model ----------
class OcrKtpLog(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "bribrain_ocr_ktp_laminate"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False)
    response_code = Column(Integer, nullable=False)
    payload = Column(JSONB)
    error_message = Column(Text)
    result = Column(JSONB)
    processing_time = Column(Float)  # in seconds
    created_at = Column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False
    )