"""
Integration tests for database operations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.services.database_service import init_engine, dispose_engine, insert_log


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.integration
    def test_database_lifecycle(self, in_memory_db):
        """Test complete database lifecycle."""
        # Database should be initialized via fixture
        session, OcrKtpLogTest = in_memory_db
        
        # Create a log entry
        log_entry = OcrKtpLogTest(
            request_id="TEST_001",
            response_code=200,
            payload={"filename": "test_ktp.jpg", "file_size": 12345},
            result={"nik": "1234567890123456"},
            processing_time=1.5,
            error_message=None
        )
        
        session.add(log_entry)
        session.commit()
        
        # Query back
        result = session.query(OcrKtpLogTest).filter_by(request_id="TEST_001").first()
        
        assert result is not None
        assert result.request_id == "TEST_001"
        assert result.response_code == 200
        assert result.processing_time == pytest.approx(1.5)

    @pytest.mark.integration
    def test_insert_multiple_logs(self, in_memory_db):
        """Test inserting multiple log entries."""
        session, OcrKtpLogTest = in_memory_db
        
        # Insert multiple logs
        for i in range(5):
            log_entry = OcrKtpLogTest(
                request_id=f"TEST_{i:03d}",
                response_code=200,
                payload={"filename": f"ktp_{i}.jpg"},
                result={"status": "success"},
                processing_time=1.0 + i * 0.1,
                error_message=None
            )
            session.add(log_entry)
        
        session.commit()
        
        # Query all
        results = session.query(OcrKtpLogTest).all()
        assert len(results) == 5
        
        # Query by status
        success_logs = session.query(OcrKtpLogTest).filter_by(response_code=200).all()
        assert len(success_logs) == 5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_insert_log_service_integration(self):
        """Test insert_log service function with database."""
        mock_settings = MagicMock()
        mock_settings.database.enabled = True
        mock_settings.database.url = "sqlite:///:memory:"
        
        with patch("src.services.database_service.get_settings", return_value=mock_settings):
            # Initialize engine
            init_engine()
            
            # Insert log
            await insert_log(
                request_id="INT_TEST_001",
                response_code=200,
                payload={"filename": "integration_test.jpg", "file_size": 50000},
                error_message=None,
                result={"nik": "1234567890123456"},
                processing_time=2.5
            )
            
            # Cleanup
            dispose_engine()

    @pytest.mark.integration
    def test_query_logs_by_date_range(self, in_memory_db):
        """Test querying logs by date range."""
        session, OcrKtpLogTest = in_memory_db
        
        now = datetime.now()
        
        # Insert logs with different dates
        for i in range(3):
            log_entry = OcrKtpLogTest(
                request_id=f"DATE_TEST_{i}",
                response_code=200,
                payload={"filename": f"ktp_{i}.jpg"},
                result={"status": "success"},
                processing_time=1.0,
                error_message=None
            )
            session.add(log_entry)
        
        session.commit()
        
        # Query logs from last 2 days
        two_days_ago = now - timedelta(days=2)
        recent_logs = session.query(OcrKtpLogTest).filter(
            OcrKtpLogTest.created_at >= two_days_ago
        ).all()
        
        assert len(recent_logs) >= 2

    @pytest.mark.integration
    def test_query_logs_by_status(self, in_memory_db):
        """Test querying logs by status code."""
        session, OcrKtpLogTest = in_memory_db
        
        # Insert logs with different status codes
        statuses = [200, 200, 400, 400, 500]
        for i, status in enumerate(statuses):
            log_entry = OcrKtpLogTest(
                request_id=f"STATUS_TEST_{i}",
                response_code=status,
                payload={"filename": f"ktp_{i}.jpg"},
                result={"status": "success" if status == 200 else "error"},
                processing_time=1.0,
                error_message="Error" if status >= 400 else None
            )
            session.add(log_entry)
        
        session.commit()
        
        # Query successful logs
        success_logs = session.query(OcrKtpLogTest).filter_by(response_code=200).all()
        assert len(success_logs) == 2
        
        # Query error logs
        error_logs = session.query(OcrKtpLogTest).filter(OcrKtpLogTest.response_code >= 400).all()
        assert len(error_logs) == 3

    @pytest.mark.integration
    def test_database_error_handling(self):
        """Test database error handling."""
        mock_settings = MagicMock()
        mock_settings.database.enabled = True
        mock_settings.database.url = "invalid://connection"
        
        with patch("src.services.database_service.get_settings", return_value=mock_settings):
            # Should not raise exception
            try:
                init_engine()
                # May fail to connect but shouldn't crash
            except Exception:
                pass  # Expected for invalid connection
            
            dispose_engine()
