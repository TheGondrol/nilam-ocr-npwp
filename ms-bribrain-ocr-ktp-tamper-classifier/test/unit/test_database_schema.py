"""
Unit tests for database schema models
"""

import pytest
from sqlalchemy import inspect

from src.schemas.database_schema import OcrKtpLog, Base


@pytest.mark.unit
class TestOcrKtpLog:
    """Tests for OcrKtpLog model"""
    
    def test_table_name(self):
        """Test that table name is correct"""
        assert OcrKtpLog.__tablename__ == "bribrain_ocr_ktp_tamper"
    
    def test_table_schema(self):
        """Test that table schema is correct"""
        assert OcrKtpLog.__table_args__["schema"] == "public"
    
    def test_columns_exist(self):
        """Test that all required columns exist"""
        columns = [c.name for c in OcrKtpLog.__table__.columns]
        
        assert "id" in columns
        assert "request_id" in columns
        assert "response_code" in columns
        assert "payload" in columns
        assert "error_message" in columns
        assert "result" in columns
        assert "processing_time" in columns
        assert "created_at" in columns
    
    def test_id_is_primary_key(self):
        """Test that id is the primary key"""
        inspector = inspect(OcrKtpLog)
        pk = inspector.primary_key
        assert pk is not None, "Primary key should not be None"
        primary_keys = [key.name for key in pk]
        
        assert "id" in primary_keys
    
    def test_model_instantiation(self):
        """Test creating an instance of OcrKtpLog"""
        log = OcrKtpLog(
            request_id="test-123",
            response_code=200,
            payload={"file_name": "test.jpg"},
            error_message="",
            result={"prediction": "authentic"},
            processing_time=100.5
        )
        
        assert log.request_id == "test-123"
        assert log.response_code == 200
        assert log.payload == {"file_name": "test.jpg"}
        assert log.result == {"prediction": "authentic"}
        assert log.processing_time == 100.5
    
    def test_nullable_fields(self):
        """Test that nullable fields can be None"""
        log = OcrKtpLog(
            request_id="test-123",
            response_code=200,
            payload=None,
            error_message=None,
            result=None,
            processing_time=None
        )
        
        assert log.payload is None
        assert log.error_message is None
        assert log.result is None
        assert log.processing_time is None
    
    def test_jsonb_fields(self):
        """Test that JSONB fields accept complex data"""
        complex_payload = {
            "file_name": "test.jpg",
            "metadata": {
                "size": 1024,
                "format": "JPEG"
            }
        }
        
        complex_result = {
            "prediction": "tampered",
            "confidence": 0.87,
            "probabilities": {
                "authentic": 0.13,
                "tampered": 0.87
            }
        }
        
        log = OcrKtpLog(
            request_id="test-123",
            response_code=200,
            payload=complex_payload,
            error_message="",
            result=complex_result,
            processing_time=150.0
        )
        
        assert log.payload == complex_payload
        assert log.result == complex_result


@pytest.mark.unit
class TestBaseModel:
    """Tests for Base declarative base"""
    
    def test_base_is_declarative_base(self):
        """Test that Base is a declarative base"""
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')
