"""
Unit tests for temper detection service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import asyncio

from src.services.temper_service import request_temper


class TestTemperService:
    """Tests for temper detection service."""

    @pytest.mark.asyncio
    async def test_request_temper_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper check that passes (no tampering detected)."""
        temper_response = {
            "prediction": "good",
            "confidence": 0.95
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": temper_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "success"
            assert result.data is not None
            assert result.data.passed is True

    @pytest.mark.asyncio
    async def test_request_temper_fail(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper check that fails (tampering detected)."""
        temper_response = {
            "prediction": "fake",
            "confidence": 0.85
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": temper_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "tampering" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_temper_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper with HTTP error response."""
        mock_response = AsyncMock()
        mock_response.status = 500
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "http_error"

    @pytest.mark.asyncio
    async def test_request_temper_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper request timeout."""
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError()

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_temper_connection_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_temper_unexpected_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test temper with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected error")

        with patch("src.services.temper_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.temper.url = "http://test:8000/temper"
            mock_settings.return_value.services.temper.timeout = 30

            result = await request_temper(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"
