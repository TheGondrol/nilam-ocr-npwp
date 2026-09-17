"""
Unit tests for spoof detection service module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
import asyncio

from src.services.spoof_service import (
    request_lamination,
    request_recapture,
    request_graycopy
)


class TestSpoofService:
    """Tests for spoof detection services."""

    @pytest.mark.asyncio
    async def test_request_lamination_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_spoof_response_pass
    ):
        """Test lamination check that passes (no lamination detected)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_spoof_response_pass})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30

            result = await request_lamination(
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
    async def test_request_lamination_fail(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
    ):
        """Test lamination check that fails (lamination detected)."""
        lamination_fail_response = {"prediction": "UNLAMINATED", "confidence": 0.85}
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": lamination_fail_response})

        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30

            result = await request_lamination(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "lamination" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_lamination_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test lamination check with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 503

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30

            result = await request_lamination(
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
    async def test_request_lamination_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test lamination check timeout."""
        mock_aiohttp_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30

            result = await request_lamination(
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
    async def test_request_lamination_connection_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test lamination check with connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection refused")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30
            mock_settings.return_value.services.lamination.api_key = ""

            result = await request_lamination(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_lamination_unexpected_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test lamination check with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.lamination.url = "http://test:8000/lamination"
            mock_settings.return_value.services.lamination.timeout = 30
            mock_settings.return_value.services.lamination.api_key = ""

            result = await request_lamination(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"

    @pytest.mark.asyncio
    async def test_request_recapture_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_spoof_response_pass
    ):
        """Test recapture check that passes (no recapture detected)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_spoof_response_pass})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "success"
            assert result.data is not None

    @pytest.mark.asyncio
    async def test_request_recapture_fail(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
    ):
        """Test recapture check that fails (recapture detected)."""
        recapture_fail_response = {"prediction": "RECAPTURED", "confidence": 0.85}
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": recapture_fail_response})

        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "recapture" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_graycopy_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_spoof_response_pass
    ):
        """Test graycopy check that passes (no graycopy detected)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_spoof_response_pass})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "success"
            assert result.data is not None

    @pytest.mark.asyncio
    async def test_request_graycopy_fail(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
    ):
        """Test graycopy check that fails (graycopy detected)."""
        graycopy_fail_response = {"prediction": "GRAYCOPY", "confidence": 0.85}
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": graycopy_fail_response})

        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "graycopy" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_recapture_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test recapture with HTTP error response."""
        mock_response = AsyncMock()
        mock_response.status = 503
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
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
    async def test_request_recapture_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test recapture request timeout."""
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError()

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
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
    async def test_request_recapture_connection_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test recapture connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_recapture_unexpected_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test recapture with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected error")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.recapture.url = "http://test:8000/recapture"
            mock_settings.return_value.services.recapture.timeout = 30

            result = await request_recapture(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "unknown_error"

    @pytest.mark.asyncio
    async def test_request_graycopy_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test graycopy with HTTP error response."""
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
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
    async def test_request_graycopy_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test graycopy request timeout."""
        mock_aiohttp_session.post.side_effect = asyncio.TimeoutError()

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
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
    async def test_request_graycopy_connection_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test graycopy connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection failed")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_graycopy_unexpected_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test graycopy with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected error")

        with patch("src.services.spoof_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.graycopy.url = "http://test:8000/graycopy"
            mock_settings.return_value.services.graycopy.timeout = 30

            result = await request_graycopy(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "unknown_error"
