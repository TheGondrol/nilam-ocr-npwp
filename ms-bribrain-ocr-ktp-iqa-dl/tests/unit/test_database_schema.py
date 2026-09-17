"""Unit tests for src.schemas.database_schema module."""

from datetime import datetime

from src.schemas.database_schema import Base, OcrKtpQualityLog


class TestOcrKtpQualityLog:
    def test_table_name(self):
        assert OcrKtpQualityLog.__tablename__ is not None

    def test_has_expected_columns(self):
        cols = {c.name for c in OcrKtpQualityLog.__table__.columns}
        expected = {
            "id", "request_id", "response_code", "payload",
            "error_message", "result", "processing_time", "created_at",
        }
        assert expected.issubset(cols)

    def test_repr(self):
        log = OcrKtpQualityLog()
        log.id = 1
        log.request_id = "test-id"
        log.response_code = 200
        r = repr(log)
        assert "test-id" in r
        assert "200" in r

    def test_base_metadata(self):
        assert "bribrain_ocr_ktp" in list(Base.metadata.tables.keys())[0]
