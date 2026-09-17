"""
Unit tests for classifier service module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
import asyncio

from src.services.classifier_service import request_classifier
from src.api.models import ClassifierData


class TestClassifierService:
    """Tests for classifier service."""

    @pytest.mark.asyncio
    async def test_request_classifier_success(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_classifier_response_valid
    ):
        """Test successful classifier request."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_classifier_response_valid})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.classifier_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.classifier.url = "http://test:8000/classifier"
            mock_settings.return_value.services.classifier.timeout = 30

            result = await request_classifier(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "success"
            assert result.data is not None
            assert isinstance(result.data, ClassifierData)

    @pytest.mark.asyncio
    async def test_request_classifier_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test classifier request with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.classifier_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.classifier.url = "http://test:8000/classifier"
            mock_settings.return_value.services.classifier.timeout = 30

            result = await request_classifier(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "http_error"

    @pytest.mark.asyncio
    async def test_request_classifier_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test classifier request timeout."""
        mock_aiohttp_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("src.services.classifier_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.classifier.url = "http://test:8000/classifier"
            mock_settings.return_value.services.classifier.timeout = 30

            result = await request_classifier(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_classifier_connection_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test classifier request with connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection refused")

        with patch("src.services.classifier_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.classifier.url = "http://test:8000/classifier"
            mock_settings.return_value.services.classifier.timeout = 30
            mock_settings.return_value.services.classifier.api_key = ""

            result = await request_classifier(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_classifier_unexpected_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test classifier request with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected")

        with patch("src.services.classifier_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.classifier.url = "http://test:8000/classifier"
            mock_settings.return_value.services.classifier.timeout = 30
            mock_settings.return_value.services.classifier.api_key = ""

            result = await request_classifier(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"
