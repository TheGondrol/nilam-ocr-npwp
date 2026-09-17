"""
Unit tests for quality service module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
import asyncio

from src.services.quality_service import request_quality_rulebase, request_quality_dl, get_lowest_confidence


class TestQualityService:
    """Tests for quality service."""

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_quality_response_pass
    ):
        """Test quality rulebase check that passes."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_quality_response_pass})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "success"
            assert result.data is not None

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_blur(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test quality rulebase check that detects blur."""
        quality_response = {
            "is_blurry": True,
            "is_glare": False,
            "is_rotated": False
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": quality_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "blur" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_glare(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test quality rulebase check that detects glare."""
        quality_response = {
            "is_blurry": False,
            "is_glare": True,
            "is_rotated": False
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": quality_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert "glare" in result.rejection.message.lower()

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_multiple_issues(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_quality_response_fail
    ):
        """Test quality rulebase check with multiple issues."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": sample_quality_response_fail})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_http_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test quality rulebase check with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_aiohttp_session.post = AsyncMock(return_value=mock_response)
        mock_aiohttp_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_aiohttp_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "error"
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_timeout(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test quality rulebase check timeout."""
        mock_aiohttp_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30

            result = await request_quality_rulebase(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [],
                request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_quality_dl_success(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id
    ):
        """Test quality DL check that passes."""
        quality_dl_response = {
            "prediction": "good",
            "confidence": 0.95
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": quality_dl_response})
        
        # Create async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality_dl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.quality_dl.timeout = 30

            result = await request_quality_dl(
                mock_aiohttp_session,
                sample_jpeg_bytes,
                sample_filename,
                sample_content_type,
                [["test", ["test", 0.95]]],
                request_id
            )

            assert result.status == "success"
            assert result.data is not None

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_connection_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality rulebase check with connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection refused")

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30
            mock_settings.return_value.services.quality.api_key = ""

            result = await request_quality_rulebase(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_quality_rulebase_unexpected_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality rulebase check with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected")

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.quality.url = "http://test:8000/quality"
            mock_settings.return_value.services.quality.timeout = 30
            mock_settings.return_value.services.quality.api_key = ""

            result = await request_quality_rulebase(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"

    @pytest.mark.asyncio
    async def test_request_quality_dl_rejection(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality DL check that detects bad quality."""
        quality_dl_response = {"label": "bad", "prediction": "bad", "confidence": 0.92}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": quality_dl_response})

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.qualitydl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.qualitydl.timeout = 30
            mock_settings.return_value.services.qualitydl.api_key = ""

            result = await request_quality_dl(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [["test", ["test", 0.95]]], request_id
            )

            assert result.status == "rejection"
            assert result.rejection is not None
            assert result.rejection.rejection_type == "quality_dl"

    @pytest.mark.asyncio
    async def test_request_quality_dl_http_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality DL check with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_aiohttp_session.post = MagicMock(return_value=mock_context)

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.qualitydl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.qualitydl.timeout = 30
            mock_settings.return_value.services.qualitydl.api_key = ""

            result = await request_quality_dl(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [["test", ["test", 0.95]]], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "http_error"

    @pytest.mark.asyncio
    async def test_request_quality_dl_timeout(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality DL check timeout."""
        mock_aiohttp_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.qualitydl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.qualitydl.timeout = 30
            mock_settings.return_value.services.qualitydl.api_key = ""

            result = await request_quality_dl(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [["test", ["test", 0.95]]], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "timeout"

    @pytest.mark.asyncio
    async def test_request_quality_dl_connection_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality DL check connection error."""
        mock_aiohttp_session.post.side_effect = aiohttp.ClientError("Connection refused")

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.qualitydl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.qualitydl.timeout = 30
            mock_settings.return_value.services.qualitydl.api_key = ""

            result = await request_quality_dl(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [["test", ["test", 0.95]]], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "connection_error"

    @pytest.mark.asyncio
    async def test_request_quality_dl_unexpected_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id
    ):
        """Test quality DL check with unexpected error."""
        mock_aiohttp_session.post.side_effect = Exception("Unexpected")

        with patch("src.services.quality_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.services.qualitydl.url = "http://test:8000/quality-dl"
            mock_settings.return_value.services.qualitydl.timeout = 30
            mock_settings.return_value.services.qualitydl.api_key = ""

            result = await request_quality_dl(
                mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
                sample_content_type, [["test", ["test", 0.95]]], request_id
            )

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "unknown_error"


def _make_quality_mock(session, data):
    """Helper to set up aiohttp mock for quality service tests."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": data})
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=mock_context)


class TestQualityRulebaseEdgeCases:

    @pytest.mark.asyncio
    async def test_rotation_only(self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, request_id):
        _make_quality_mock(mock_aiohttp_session, {"is_blurry": False, "is_glare": False, "is_rotated": True})
        with patch("src.services.quality_service.get_settings") as m:
            m.return_value = MagicMock()
            m.return_value.services.quality.url = "http://test:8000"
            m.return_value.services.quality.timeout = 30
            result = await request_quality_rulebase(mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, [], request_id)
        assert result.status == "rejection"
        assert result.rejection is not None
        assert result.rejection.details is not None
        assert result.rejection.details["is_rotated"] is True
        assert result.rejection.details["is_blurry"] is False

    @pytest.mark.asyncio
    async def test_all_three_issues(self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, request_id):
        _make_quality_mock(mock_aiohttp_session, {"is_blurry": True, "is_glare": True, "is_rotated": True})
        with patch("src.services.quality_service.get_settings") as m:
            m.return_value = MagicMock()
            m.return_value.services.quality.url = "http://test:8000"
            m.return_value.services.quality.timeout = 30
            result = await request_quality_rulebase(mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, [], request_id)
        assert result.status == "rejection"
        assert result.rejection is not None
        assert result.rejection.details is not None
        assert result.rejection.details["is_blurry"] is True
        assert result.rejection.details["is_glare"] is True
        assert result.rejection.details["is_rotated"] is True

    @pytest.mark.asyncio
    async def test_blur_and_rotation_wildcard(self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, request_id):
        _make_quality_mock(mock_aiohttp_session, {"is_blurry": True, "is_glare": False, "is_rotated": True})
        with patch("src.services.quality_service.get_settings") as m:
            m.return_value = MagicMock()
            m.return_value.services.quality.url = "http://test:8000"
            m.return_value.services.quality.timeout = 30
            result = await request_quality_rulebase(mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, [], request_id)
        assert result.status == "rejection"
        assert result.rejection is not None
        assert "blur" in result.rejection.message
        assert "rotation" in result.rejection.message

    @pytest.mark.asyncio
    async def test_glare_and_rotation_wildcard(self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, request_id):
        _make_quality_mock(mock_aiohttp_session, {"is_blurry": False, "is_glare": True, "is_rotated": True})
        with patch("src.services.quality_service.get_settings") as m:
            m.return_value = MagicMock()
            m.return_value.services.quality.url = "http://test:8000"
            m.return_value.services.quality.timeout = 30
            result = await request_quality_rulebase(mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, [], request_id)
        assert result.status == "rejection"
        assert result.rejection is not None
        assert "glare" in result.rejection.message
        assert "rotation" in result.rejection.message

    @pytest.mark.asyncio
    async def test_non_dict_data_response(self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, request_id):
        """When data field is not a dict, all quality flags default to False."""
        _make_quality_mock(mock_aiohttp_session, "not_a_dict")
        with patch("src.services.quality_service.get_settings") as m:
            m.return_value = MagicMock()
            m.return_value.services.quality.url = "http://test:8000"
            m.return_value.services.quality.timeout = 30
            result = await request_quality_rulebase(mock_aiohttp_session, sample_jpeg_bytes, sample_filename, sample_content_type, [], request_id)
        assert result.status == "success"


class TestGetLowestConfidence:

    @pytest.mark.asyncio
    async def test_returns_lowest_k(self):
        ocr_raw = [
            [[0, 0, 10, 10], ["text1", 0.9]],
            [[0, 0, 10, 10], ["text2", 0.5]],
            [[0, 0, 10, 10], ["text3", 0.3]],
            [[0, 0, 10, 10], ["text4", 0.7]],
            [[0, 0, 10, 10], ["text5", 0.1]],
            [[0, 0, 10, 10], ["text6", 0.6]],
        ]
        result = await get_lowest_confidence(ocr_raw, k=3)
        assert len(result) == 3
        confidences = [r["confidence"] for r in result]
        assert sorted(confidences) == [0.1, 0.3, 0.5]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await get_lowest_confidence(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await get_lowest_confidence([])
        assert result == []

    @pytest.mark.asyncio
    async def test_fewer_items_than_k(self):
        ocr_raw = [
            [[0, 0, 10, 10], ["text1", 0.9]],
            [[0, 0, 10, 10], ["text2", 0.5]],
        ]
        result = await get_lowest_confidence(ocr_raw, k=5)
        assert len(result) == 2
