"""Unit tests for src.schemas.database_schema module."""

from src.schemas.database_schema import OcrKtpLog, Base


class TestOcrKtpLog:
    def test_tablename(self):
        assert OcrKtpLog.__tablename__ == "bribrain_ocr_ktp_laminate"

    def test_schema(self):
        assert OcrKtpLog.__table_args__["schema"] == "public"

    def test_columns_exist(self):
        cols = {c.name for c in OcrKtpLog.__table__.columns}
        expected = {"id", "request_id", "response_code", "payload", "error_message", "result", "processing_time", "created_at"}
        assert expected == cols

    def test_inherits_base(self):
        assert issubclass(OcrKtpLog, Base)
