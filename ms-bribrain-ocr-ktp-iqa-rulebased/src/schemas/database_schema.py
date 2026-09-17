from sqlalchemy import (
    Column,
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

# Get config for table name and schema
_config = get_config()

# ---------- Model ----------
class OcrKtpLog(Base):
    __tablename__ = _config.database.table_name
    __table_args__ = {"schema": _config.database.db_schema}

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False)
    response_code = Column(Integer, nullable=False)
    payload = Column(LargeBinary)        # AES-256-GCM ciphertext (see core.crypto)
    error_message = Column(Text)
    result = Column(LargeBinary)         # AES-256-GCM ciphertext
    processing_time = Column(Float)
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
