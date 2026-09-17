"""
Integration tests for API endpoints with service orchestration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO
from fastapi import UploadFile
from starlette.datastructures import Headers

from src.api.routes import extract_text_lines
from src.api.models import ErrorCode, PipelineError

VALID_REQUEST_ID = "OCR_12345678-1234-1234-1234-123456789abc"


def _ocr_endpoint_mocks():
    """Return a dict of common patches needed for the OCR endpoint."""
    return {
        "claim": patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock),
        "update": patch("src.api.routes.update_ocr_result", new_callable=AsyncMock),
        "insert_log": patch("src.api.routes.insert_log", new_callable=AsyncMock),
        "http_client": patch("src.api.routes.get_http_client", new_callable=AsyncMock),
    }


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_ocr_workflow_success(
        self,
        sample_ktp_image_bytes,
        sample_ocr_response,
        sample_postprocess_response,
        mock_settings
    ):
        """Test complete OCR workflow from upload to final result."""
        file = UploadFile(
            filename="test_ktp.jpg",
            file=BytesIO(sample_ktp_image_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.ocr_process", new_callable=AsyncMock) as mock_ocr_process, \
             patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert_log, \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            mock_claim.return_value = True
            mock_ocr_process.return_value = (
                sample_postprocess_response,
                None,
            )
            mock_update.return_value = None
            mock_insert_log.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 200
            body = response.body.decode()
            assert "1234567890123456" in body
            assert "JOHN DOE" in body

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_workflow_with_quality_rejection(
        self,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR workflow with quality check rejection."""
        file = UploadFile(
            filename="low_quality.jpg",
            file=BytesIO(sample_ktp_image_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.ocr_process") as mock_ocr_process, \
             patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock), \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            mock_claim.return_value = True
            mock_ocr_process.return_value = (
                None,
                PipelineError(
                    error_code=ErrorCode.QUALITY_BLUR,
                    message="Image is too blurry. Please take a clearer photo.",
                )
            )
            mock_update.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            body = response.body.decode()
            assert "blurry" in body.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_workflow_with_spoof_rejection(
        self,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR workflow with spoof detection rejection."""
        file = UploadFile(
            filename="fake_ktp.jpg",
            file=BytesIO(sample_ktp_image_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.ocr_process") as mock_ocr_process, \
             patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock), \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            mock_claim.return_value = True
            mock_ocr_process.return_value = (
                None,
                PipelineError(
                    error_code=ErrorCode.IMAGE_REJECTED,
                    message="Image rejected: Document material does not meet standards",
                    details={
                        "reasons": [
                            {"error_code": ErrorCode.SPOOF_UNLAMINATED, "message": "Document material does not meet standards"}
                        ]
                    },
                )
            )
            mock_update.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            body = response.body.decode()
            assert "IMAGE_REJECTED" in body
            assert "SPOOF_UNLAMINATED" in body

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_workflow_with_invalid_file_type(self, mock_settings):
        """Test OCR workflow with invalid file type."""
        file = UploadFile(
            filename="document.pdf",
            file=BytesIO(b"%PDF-1.4 fake content"),
            headers=Headers({"content-type": "application/pdf"})
        )

        with patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock):

            mock_claim.return_value = True
            mock_update.return_value = None

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            body = response.body.decode()
            assert "invalid" in body.lower() or "type" in body.lower() or "jpeg" in body.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_ocr_requests(
        self,
        sample_ktp_image_bytes,
        sample_postprocess_response,
        mock_settings
    ):
        """Test handling multiple concurrent OCR requests."""
        import asyncio

        async def process_request():
            file = UploadFile(
                filename="test_ktp.jpg",
                file=BytesIO(sample_ktp_image_bytes),
                headers=Headers({"content-type": "image/jpeg"})
            )

            with patch("src.api.routes.ocr_process") as mock_ocr_process, \
                 patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
                 patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
                 patch("src.api.routes.insert_log", new_callable=AsyncMock), \
                 patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

                mock_claim.return_value = True
                mock_ocr_process.return_value = (
                    sample_postprocess_response,
                    None,
                )
                mock_update.return_value = None
                mock_http.return_value = MagicMock()

                return await extract_text_lines(
                    request_id=VALID_REQUEST_ID, file=file
                )

        # Execute 5 concurrent requests
        tasks = [process_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for response in responses:
            assert response.status_code in [200, 500]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_workflow_with_classifier_rejection(
        self,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR workflow with classifier rejection."""
        file = UploadFile(
            filename="not_ktp.jpg",
            file=BytesIO(sample_ktp_image_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.ocr_process") as mock_ocr_process, \
             patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock), \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            mock_claim.return_value = True
            mock_ocr_process.return_value = (
                None,
                PipelineError(
                    error_code=ErrorCode.IMAGE_REJECTED,
                    message="Image rejected: Image is not recognized as a valid KTP",
                    details={
                        "reasons": [
                            {"error_code": ErrorCode.NOT_KTP, "message": "Image is not recognized as a valid KTP"}
                        ]
                    },
                )
            )
            mock_update.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 400
            body = response.body.decode()
            assert "IMAGE_REJECTED" in body
            assert "NOT_KTP" in body

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_workflow_error_handling(
        self,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR workflow error handling."""
        file = UploadFile(
            filename="test_ktp.jpg",
            file=BytesIO(sample_ktp_image_bytes),
            headers=Headers({"content-type": "image/jpeg"})
        )

        with patch("src.api.routes.ocr_process") as mock_ocr_process, \
             patch("src.api.routes.claim_ocr_result", new_callable=AsyncMock) as mock_claim, \
             patch("src.api.routes.update_ocr_result", new_callable=AsyncMock) as mock_update, \
             patch("src.api.routes.insert_log", new_callable=AsyncMock), \
             patch("src.api.routes.get_http_client", new_callable=AsyncMock) as mock_http:

            mock_claim.return_value = True
            mock_ocr_process.side_effect = Exception("Service unavailable")
            mock_update.return_value = None
            mock_http.return_value = MagicMock()

            response = await extract_text_lines(
                request_id=VALID_REQUEST_ID, file=file
            )

            assert response.status_code == 500
            body = response.body.decode()
            assert "error" in body.lower()
