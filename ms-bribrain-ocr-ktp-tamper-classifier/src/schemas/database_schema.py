from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    TIMESTAMP,
    Float
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

# ---------- Base ----------
Base = declarative_base()

# ---------- Model ----------
class OcrKtpLog(Base):
    __tablename__ = "bribrain_ocr_ktp_tamper"
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