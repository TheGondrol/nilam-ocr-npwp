"""
Unit tests for POST /v1/generate-id endpoint.
"""

import pytest
from unittest.mock import patch, AsyncMock

from src.api.routes import generate_request_id


class TestGenerateIdEndpoint:
    """Tests for the /v1/generate-id endpoint."""

    @pytest.mark.asyncio
    async def test_generate_id_success(self):
        """Should return 200 with a new request_id starting with OCR_."""
        with patch("src.api.routes.create_ocr_result", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None

            response = await generate_request_id()

            assert response.status_code == 200
            import json
            data = json.loads(response.body.decode())
            assert data["status_code"] == 200
            assert data["status_desc"] == "OK"
            assert "OCR_" in data["data"]["request_id"]
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_id_returns_valid_uuid_format(self):
        """Should return request_id matching OCR_{uuid4} pattern."""
        import re

        with patch("src.api.routes.create_ocr_result", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None

            response = await generate_request_id()

            content = response.body.decode()
            # Extract request_id from response
            import json
            data = json.loads(content)
            request_id = data["data"]["request_id"]

            pattern = r"^OCR_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            assert re.match(pattern, request_id), f"request_id {request_id} does not match expected format"

    @pytest.mark.asyncio
    async def test_generate_id_database_failure(self):
        """Should return 500 with static error message when DB insert fails."""
        with patch("src.api.routes.create_ocr_result", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = RuntimeError("Database not initialized")

            response = await generate_request_id()

            assert response.status_code == 500
            content = response.body.decode()
            assert "unexpected error" in content.lower()
            # Should NOT leak exception details
            assert "Database not initialized" not in content

    @pytest.mark.asyncio
    async def test_generate_id_returns_envelope_format(self):
        """Should return response in standard envelope format."""
        import json

        with patch("src.api.routes.create_ocr_result", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None

            response = await generate_request_id()

            data = json.loads(response.body.decode())
            assert "status_code" in data
            assert "status_desc" in data
            assert "message" in data
            assert "data" in data
            assert "error_code" in data
            assert "errors" in data
            assert "request_id" in data
