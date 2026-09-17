"""
Unit tests for postprocess service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import asyncio

from src.services.postprocess_service import request_postprocess


class TestPostprocessService:
    """Tests for postprocess service."""

    @pytest.mark.asyncio
    async def test_request_postprocess_success(
        self, mock_aiohttp_session, request_id
    ):
        """Test successful postprocess request."""
        ocr_raw = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], ["John Doe", 0.95]]
        ]
        
        postprocess_response = {
            "nik": "1234567890123456",
            "nama": "John Doe",
            "tempat_lahir": "Jakarta"
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": postprocess_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.postprocess_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.postprocess.url = "http://test:8000/postprocess"
            mock_settings.return_value.services.postprocess.timeout = 30

            result = await request_postprocess(
                mock_aiohttp_session,
                ocr_raw,
                request_id
            )

            assert result.status == "success"
            assert result.data is not None
            assert result.data.results == postprocess_response

    @pytest.mark.asyncio
    async def test_request_postprocess_http_error(
        self, mock_aiohttp_session, request_id
    ):
        """Test postprocess with HTTP error response."""
        ocr_raw = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], ["Text", 0.95]]
        ]
        
        mock_response = AsyncMock()
        mock_response.status = 500
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.postprocess_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.postprocess.url = "http://test:8000/postprocess"
            mock_settings.return_value.services.postprocess.timeout = 30

            result = await request_postprocess(
                mock_aiohttp_session,
                ocr_raw,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "http_error"
            assert "500" in result.error.message

    @pytest.mark.asyncio
    async def test_request_postprocess_timeout(
        self, mock_aiohttp_session, request_id
    ):
        """Test postprocess request timeout."""
        ocr_raw = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], ["Text", 0.95]]
        ]
        
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError()

        with patch("src.services.postprocess_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.postprocess.url = "http://test:8000/postprocess"
            mock_settings.return_value.services.postprocess.timeout = 30

            result = await request_postprocess(
                mock_aiohttp_session,
                ocr_raw,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_postprocess_connection_error(
        self, mock_aiohttp_session, request_id
    ):
        """Test postprocess connection error."""
        ocr_raw = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], ["Text", 0.95]]
        ]
        
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("src.services.postprocess_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.postprocess.url = "http://test:8000/postprocess"
            mock_settings.return_value.services.postprocess.timeout = 30

            result = await request_postprocess(
                mock_aiohttp_session,
                ocr_raw,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_postprocess_unexpected_error(
        self, mock_aiohttp_session, request_id
    ):
        """Test postprocess with unexpected error."""
        ocr_raw = [
            [[[10, 20], [100, 20], [100, 40], [10, 40]], ["Text", 0.95]]
        ]
        
        mock_aiohttp_session.post.side_effect = Exception("Unexpected error")

        with patch("src.services.postprocess_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.postprocess.url = "http://test:8000/postprocess"
            mock_settings.return_value.services.postprocess.timeout = 30

            result = await request_postprocess(
                mock_aiohttp_session,
                ocr_raw,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"
