"""
Unit tests for OCR service module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
import asyncio

from src.services.ocr_service import request_ocr
from src.api.models import OCRData


class TestOCRService:
    """Tests for OCR service."""

    @pytest.mark.asyncio
    async def test_request_ocr_success(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response
    ):
        """Test successful OCR request."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_ocr_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.ocr_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.ocr.url = "http://test:8000/ocr"
            mock_settings.return_value.services.ocr.timeout = 30

            # Call the service
            result = await request_ocr(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            # Assertions
            assert result.status == "success"
            assert result.data is not None
            assert isinstance(result.data, OCRData)

    @pytest.mark.asyncio
    async def test_request_ocr_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test OCR request with HTTP error."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.ocr_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.ocr.url = "http://test:8000/ocr"
            mock_settings.return_value.services.ocr.timeout = 30

            # Call the service
            result = await request_ocr(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            # Assertions
            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "http_error"
            assert result.error.status_code == 500

    @pytest.mark.asyncio
    async def test_request_ocr_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test OCR request timeout."""
        # Mock timeout
        mock_aiohttp_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("src.services.ocr_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.ocr.url = "http://test:8000/ocr"
            mock_settings.return_value.services.ocr.timeout = 30

            # Call the service
            result = await request_ocr(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            # Assertions
            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_ocr_connection_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test OCR request connection error."""
        # Mock connection error
        mock_aiohttp_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection failed")
        )

        with patch("src.services.ocr_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.ocr.url = "http://test:8000/ocr"
            mock_settings.return_value.services.ocr.timeout = 30

            # Call the service
            result = await request_ocr(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            # Assertions
            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_ocr_unexpected_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test OCR request with unexpected error."""
        # Mock unexpected error
        mock_aiohttp_session.post = MagicMock(
            side_effect=Exception("Unexpected error")
        )

        with patch("src.services.ocr_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.ocr.url = "http://test:8000/ocr"
            mock_settings.return_value.services.ocr.timeout = 30

            # Call the service
            result = await request_ocr(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            # Assertions
            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"
