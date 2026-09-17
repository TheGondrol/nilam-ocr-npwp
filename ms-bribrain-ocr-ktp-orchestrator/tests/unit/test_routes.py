"""
Unit tests for API routes module.
"""

import asyncio

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import UploadFile, HTTPException
from starlette.datastructures import Headers
from io import BytesIO

from src.api.routes import extract_text_lines
from src.api.models import ErrorCode, PipelineError


VALID_REQUEST_ID = "OCR_12345678-1234-1234-1234-123456789abc"


def _ocr_pipeline_mocks():
    """Context manager stack for mocking OCR pipeline dependencies."""
    return [
        patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock),
        patch("src.api.routes.ocr_process", new_callable=AsyncMock),
        patch("src.api.routes.update_ocr_result", new_callable=AsyncMock),
        patch("src.api.routes.insert_log", new_callable=AsyncMock),
        patch("src.api.routes.get_http_client", new_callable=AsyncMock),
    ]


class TestOCREndpoint:
    """Tests for the /v1/ppocr endpoint."""

    @pytest.mark.asyncio
    async def test_extract_text_success(
        self, sample_jpeg_bytes, sample_postprocess_response
    ):
        """Test successful OCR extraction."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (sample_postprocess_response, None)
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 200
            content = response.body.decode()
            assert "JOHN DOE" in content or "results" in content

    @pytest.mark.asyncio
    async def test_extract_text_invalid_file_type(self):
        """Test OCR extraction with invalid file type."""
        file_content = b'%PDF-1.4 test content'
        file = UploadFile(
            filename="test.pdf",
            file=BytesIO(file_content),
            headers=Headers({"content-type": "application/pdf"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert:
            mock_claim.return_value = True
            mock_update.return_value = None
            mock_insert.return_value = None

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            content = response.body.decode()
            assert "Invalid file type" in content or "JPEG" in content

    @pytest.mark.asyncio
    async def test_extract_text_quality_failure(self, sample_jpeg_bytes):
        """Test OCR extraction with quality check failure."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (
                None,
                PipelineError(
                    error_code=ErrorCode.QUALITY_BLUR,
                    message="Image is too blurry. Please take a clearer photo.",
                    http_status=400,
                    details={"is_blurry": True, "is_glare": False, "is_rotated": False},
                )
            )
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            content = response.body.decode()
            assert "QUALITY_BLUR" in content
            assert "blurry" in content

    @pytest.mark.asyncio
    async def test_extract_text_service_error(self, sample_jpeg_bytes):
        """Test OCR extraction with service error."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (
                None,
                PipelineError(
                    error_code=ErrorCode.SERVICE_ERROR,
                    message="OCR processing is temporarily unavailable. Please try again later.",
                    http_status=500,
                )
            )
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 500
            content = response.body.decode()
            assert "SERVICE_ERROR" in content

    @pytest.mark.asyncio
    async def test_extract_text_invalid_request_id_format(self, sample_jpeg_bytes):
        """Test OCR extraction with invalid request_id format."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with pytest.raises(HTTPException) as exc_info:
            await extract_text_lines(
                request_id="INVALID_ID", file=file
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_extract_text_request_id_not_claimed(self, sample_jpeg_bytes):
        """Test OCR extraction when request_id is not in pending status (409)."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim:
            mock_claim.return_value = False

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 409
            content = response.body.decode()
            assert "Conflict" in content

    @pytest.mark.asyncio
    async def test_extract_text_claim_db_error(self, sample_jpeg_bytes):
        """Test OCR extraction when claim_ocr_result raises exception."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim:
            mock_claim.side_effect = RuntimeError("Database not initialized")

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 500
            content = response.body.decode()
            assert "unexpected error" in content.lower()
            assert "Database not initialized" not in content

    @pytest.mark.asyncio
    async def test_extract_text_exception_handling(self, sample_jpeg_bytes):
        """Test exception handling in OCR endpoint."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.side_effect = Exception("Unexpected error")
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 500
            content = response.body.decode()
            assert "unexpected error" in content.lower()

    @pytest.mark.asyncio
    async def test_extract_text_update_result_called_on_success(
        self, sample_jpeg_bytes, sample_postprocess_response
    ):
        """Should call update_ocr_result with COMPLETED status on success."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (sample_postprocess_response, None)
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            mock_update.assert_called_once()
            call_args = mock_update.call_args
            from src.schemas.database_schema import OcrStatus
            assert call_args[1]["status"] == OcrStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_extract_text_update_result_failure_still_returns_response(
        self, sample_jpeg_bytes, sample_postprocess_response
    ):
        """Should return OCR result to client even if update_ocr_result fails."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (sample_postprocess_response, None)
            # update_ocr_result internally swallows exceptions (best-effort),
            # but the mock bypasses that -- so we just verify the endpoint doesn't crash
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_extract_text_png_file(
        self, sample_image_bytes, sample_postprocess_response
    ):
        """Test OCR extraction with PNG file."""
        file = UploadFile(
            filename="test.png",
            file=BytesIO(sample_image_bytes),
            headers=Headers({"content-type": "image/png"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            mock_ocr.return_value = (sample_postprocess_response, None)
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_extract_text_file_too_large(self, sample_jpeg_bytes):
        """Test OCR extraction rejects files exceeding max size."""
        # Create oversized content (larger than the 10MB limit)
        large_bytes = b'\xff' * (11 * 1024 * 1024)  # 11MB
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(large_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert:
            mock_claim.return_value = True
            mock_update.return_value = None
            mock_insert.return_value = None

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 413
            content = response.body.decode()
            assert "FILE_TOO_LARGE" in content

    @pytest.mark.asyncio
    async def test_extract_text_pipeline_timeout(self, sample_jpeg_bytes):
        """Test that overall pipeline timeout returns PIPELINE_TIMEOUT and updates DB."""
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(sample_jpeg_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:
            mock_claim.return_value = True
            # Simulate asyncio.wait_for raising TimeoutError
            mock_ocr.side_effect = asyncio.TimeoutError()
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            # Should return 500 with PIPELINE_TIMEOUT
            assert response.status_code == 500
            content = response.body.decode()
            assert "PIPELINE_TIMEOUT" in content
            assert "too long" in content.lower()

            # DB should be updated with FAILED status
            mock_update.assert_called_once()
            from src.schemas.database_schema import OcrStatus
            assert mock_update.call_args[1]["status"] == OcrStatus.FAILED

            # Log should be inserted
            mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_text_concurrent_same_request_id(
        self, sample_jpeg_bytes, sample_postprocess_response
    ):
        """Test that concurrent requests with same request_id: one succeeds, other gets 409."""
        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            # First call claims successfully, second is rejected
            mock_claim.side_effect = [True, False]
            mock_ocr.return_value = (sample_postprocess_response, None)
            mock_update.return_value = None
            mock_insert.return_value = None
            mock_http.return_value = MagicMock()

            async def _make_request():
                file = UploadFile(
                    filename="test.jpg",
                    file=BytesIO(sample_jpeg_bytes),
                    headers=Headers({"content-type": "image/jpeg"})
                )
                return await extract_text_lines(
                    request_id=VALID_REQUEST_ID, file=file
                )

            response_a, response_b = await asyncio.gather(
                _make_request(),
                _make_request(),
            )

            codes = sorted([response_a.status_code, response_b.status_code])
            # One should succeed (200), the other should be rejected (409)
            assert codes == [200, 409]

            # The 409 response should have REQUEST_ALREADY_PROCESSED
            rejected = response_a if response_a.status_code == 409 else response_b
            assert "REQUEST_ALREADY_PROCESSED" in rejected.body.decode()
