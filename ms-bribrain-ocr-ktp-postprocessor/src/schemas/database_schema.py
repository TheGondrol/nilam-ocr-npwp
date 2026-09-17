from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    Float,
    Integer,
    LargeBinary,
    TIMESTAMP
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from src.core.config import get_config

# ---------- Base ----------
Base = declarative_base()

# ---------- Config ----------
config = get_config()

# ---------- Model ----------
class OcrKtpLog(Base):
    __tablename__ = config.get("database.table_name")
    __table_args__ = {"schema": config.get("database.schema")}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(100), nullable=False)
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