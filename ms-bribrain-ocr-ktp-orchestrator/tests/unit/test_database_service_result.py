"""
Unit tests for OCR result CRUD functions in database_service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.database_service import (
    create_ocr_result,
    claim_ocr_result,
    update_ocr_result,
    get_ocr_result,
)
from src.schemas.database_schema import OcrStatus


def _make_mock_session_factory(mock_session):
    """Helper to create a mock async session factory."""
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_factory


class TestCreateOcrResult:
    """Tests for create_ocr_result."""

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Should insert a new record with status pending."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            await create_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_raises_when_db_not_initialized(self):
        """Should raise RuntimeError when database not initialized."""
        with patch("src.services.database_service._async_session_factory", None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                await create_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_create_raises_on_db_error(self):
        """Should propagate exceptions (not swallow like insert_log)."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock(side_effect=Exception("Duplicate key"))
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            with pytest.raises(Exception, match="Duplicate key"):
                await create_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")


class TestClaimOcrResult:
    """Tests for claim_ocr_result."""

    @pytest.mark.asyncio
    async def test_claim_success(self):
        """Should return True when record is pending."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            result = await claim_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")
            assert result is True
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_claim_not_pending(self):
        """Should return False when record is not in pending status."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            result = await claim_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")
            assert result is False

    @pytest.mark.asyncio
    async def test_claim_not_found(self):
        """Should return False when request_id does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            result = await claim_ocr_result("OCR_00000000-0000-0000-0000-000000000000")
            assert result is False

    @pytest.mark.asyncio
    async def test_claim_raises_when_db_not_initialized(self):
        """Should raise RuntimeError when database not initialized."""
        with patch("src.services.database_service._async_session_factory", None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                await claim_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")


class TestUpdateOcrResult:
    """Tests for update_ocr_result."""

    @pytest.mark.asyncio
    async def test_update_to_completed(self):
        """Should update status and result when current status is processing."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            await update_ocr_result(
                "OCR_12345678-1234-1234-1234-123456789abc",
                status=OcrStatus.COMPLETED,
                result={"nama": "Test"},
            )

            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_to_failed(self):
        """Should update status and error_message on failure."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            await update_ocr_result(
                "OCR_12345678-1234-1234-1234-123456789abc",
                status=OcrStatus.FAILED,
                error_message="OCR service timeout",
            )

            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_swallows_exceptions(self):
        """Should log error but not raise when DB fails (best-effort)."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock(side_effect=Exception("DB error"))
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            # Should NOT raise
            await update_ocr_result(
                "OCR_12345678-1234-1234-1234-123456789abc",
                status=OcrStatus.COMPLETED,
                result={"nama": "Test"},
            )

    @pytest.mark.asyncio
    async def test_update_skips_when_db_not_initialized(self):
        """Should silently skip when database not initialized."""
        with patch("src.services.database_service._async_session_factory", None):
            # Should NOT raise
            await update_ocr_result(
                "OCR_12345678-1234-1234-1234-123456789abc",
                status=OcrStatus.COMPLETED,
            )


class TestGetOcrResult:
    """Tests for get_ocr_result."""

    @pytest.mark.asyncio
    async def test_get_found(self):
        """Should return the record by request_id."""
        mock_record = MagicMock()
        mock_record.request_id = "OCR_12345678-1234-1234-1234-123456789abc"

        mock_scalars = MagicMock()
        mock_scalars.first.return_value = mock_record

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            result = await get_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")
            assert result is not None
            assert result.request_id == "OCR_12345678-1234-1234-1234-123456789abc"

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        """Should return None for unknown request_id."""
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = None

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_factory = _make_mock_session_factory(mock_session)

        with patch("src.services.database_service._async_session_factory", mock_factory):
            result = await get_ocr_result("OCR_00000000-0000-0000-0000-000000000000")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_raises_when_db_not_initialized(self):
        """Should raise RuntimeError when database not initialized."""
        with patch("src.services.database_service._async_session_factory", None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                await get_ocr_result("OCR_12345678-1234-1234-1234-123456789abc")
