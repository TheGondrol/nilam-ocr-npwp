from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    TIMESTAMP,
    Float,
    LargeBinary
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

# ---------- Base ----------
Base = declarative_base()

# ---------- Model ----------
class OcrKtpLog(Base):
    __tablename__ = "bribrain_ocr_ktp_extract"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False)
    response_code = Column(Integer, nullable=False)
    payload = Column(LargeBinary)        # AES-256-GCM ciphertext (see core.crypto)
    error_message = Column(Text)
    result = Column(LargeBinary)         # AES-256-GCM ciphertext
    processing_time = Column(Float)
    created_at = Column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False
    )