"""
Unit tests for GET /v1/result/{request_id} endpoint.
"""

import json
from datetime import datetime, timezone

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import HTTPException

from src.api.routes import get_result


VALID_REQUEST_ID = "OCR_12345678-1234-1234-1234-123456789abc"


class TestGetResultEndpoint:
    """Tests for the /v1/result/{request_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_result_completed(self):
        """Should return 200 with result data when status is completed."""
        mock_record = MagicMock()
        mock_record.request_id = VALID_REQUEST_ID
        mock_record.status = "completed"
        mock_record.result = {"nama": "JOHN DOE"}
        mock_record.error_message = None
        mock_record.created_at = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
        mock_record.updated_at = datetime(2026, 3, 4, 12, 0, 5, tzinfo=timezone.utc)

        # Stored results are encrypted; the route decrypts on read. Stub decrypt
        # to a passthrough so the test drives plain data through the endpoint.
        with patch("src.api.routes.get_ocr_result", new_callable=AsyncMock) as mock_get, \
             patch("src.api.routes.decrypt", side_effect=lambda v: v):
            mock_get.return_value = mock_record

            response = await get_result(VALID_REQUEST_ID)

            assert response.status_code == 200
            data = json.loads(response.body.decode())
            assert data["data"]["status"] == "completed"
            assert data["data"]["result"] == {"nama": "JOHN DOE"}

    @pytest.mark.asyncio
    async def test_get_result_pending(self):
        """Should return 200 with status 'pending' and null result."""
        mock_record = MagicMock()
        mock_record.request_id = VALID_REQUEST_ID
        mock_record.status = "pending"
        mock_record.result = None
        mock_record.error_message = None
        mock_record.created_at = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
        mock_record.updated_at = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)

        with patch("src.api.routes.get_ocr_result", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_record

            response = await get_result(VALID_REQUEST_ID)

            assert response.status_code == 200
            data = json.loads(response.body.decode())
            assert data["data"]["status"] == "pending"
            assert data["data"]["result"] is None

    @pytest.mark.asyncio
    async def test_get_result_not_found(self):
        """Should return 404 for unknown request_id."""
        with patch("src.api.routes.get_ocr_result", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = await get_result(VALID_REQUEST_ID)

            assert response.status_code == 404
            data = json.loads(response.body.decode())
            assert data["status_desc"] == "Not Found"

    @pytest.mark.asyncio
    async def test_get_result_invalid_format(self):
        """Should return 400 for malformed request_id."""
        with pytest.raises(HTTPException) as exc_info:
            await get_result("INVALID_ID")

        assert exc_info.value.status_code == 400
        assert "Invalid request ID format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_result_database_error(self):
        """Should return 500 with standard envelope (not unformatted FastAPI error)."""
        with patch("src.api.routes.get_ocr_result", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            response = await get_result(VALID_REQUEST_ID)

            assert response.status_code == 500
            data = json.loads(response.body.decode())
            assert data["status_code"] == 500
            assert data["status_desc"] == "Internal Server Error"
            # Should NOT leak exception details
            assert "Connection refused" not in json.dumps(data)

    @pytest.mark.asyncio
    async def test_get_result_returns_envelope_format(self):
        """Should return response in standard envelope format."""
        mock_record = MagicMock()
        mock_record.request_id = VALID_REQUEST_ID
        mock_record.status = "completed"
        mock_record.result = {"nama": "TEST"}
        mock_record.error_message = None
        mock_record.created_at = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
        mock_record.updated_at = datetime(2026, 3, 4, 12, 0, 5, tzinfo=timezone.utc)

        with patch("src.api.routes.get_ocr_result", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_record

            response = await get_result(VALID_REQUEST_ID)

            data = json.loads(response.body.decode())
            assert "status_code" in data
            assert "status_desc" in data
            assert "message" in data
            assert "data" in data
            assert "error_code" in data
            assert "errors" in data
            assert "request_id" in data
