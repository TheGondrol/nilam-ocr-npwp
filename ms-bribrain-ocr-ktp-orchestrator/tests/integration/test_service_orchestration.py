"""
Integration tests for service orchestration and workflow.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.services.manage_service import ocr_process, run_quality_checks_parallel
from src.api.models import ErrorCode, PipelineError


class TestServiceOrchestration:
    """Integration tests for service orchestration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_ocr_process_flow(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        sample_ocr_response,
        sample_postprocess_response,
        mock_settings
    ):
        """Test complete OCR process flow with all services."""

        from src.api.models import ServiceResult, OCRData, ClassifierData, QualityData, SpoofData, PostprocessData

        # Create async mocks with return values
        mock_ocr = AsyncMock(return_value=ServiceResult.success(OCRData(text=sample_ocr_response)))
        mock_classifier = AsyncMock(return_value=ServiceResult.success(ClassifierData(raw_response={"class": "ktp", "confidence": 0.95, "detected": True})))
        mock_quality = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_postprocess = AsyncMock(return_value=ServiceResult.success(PostprocessData(results=sample_postprocess_response)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.save_image", new_callable=AsyncMock), \
             patch("src.services.manage_service.aiohttp.ClientSession") as mock_session_class, \
             patch("src.services.manage_service.request_ocr", mock_ocr), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy), \
             patch("src.services.manage_service.request_postprocess", mock_postprocess):

            # Mock aiohttp session
            mock_session_class.return_value.__aenter__.return_value = aiohttp_session
            mock_session_class.return_value.__aexit__.return_value = None

            # Execute OCR process (session, filename, file_bytes, content_type, request_id)
            result, pipeline_error = await ocr_process(
                aiohttp_session,
                "test_ktp.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_001"
            )

            # Verify success
            assert pipeline_error is None, f"Expected no error, got: {pipeline_error}"
            assert result is not None, "Result should not be None"
            assert result["nik"] == "1234567890123456"
            assert result["nama"] == "JOHN DOE"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_checks_parallel_all_pass(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test parallel quality checks when all pass."""

        from src.api.models import ServiceResult, ClassifierData, QualityData, SpoofData

        # Create async mocks with return values
        mock_classifier = AsyncMock(return_value=ServiceResult.success(ClassifierData(raw_response={"class": "ktp", "detected": True})))
        mock_quality = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_quality_dl = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_quality_dl", mock_quality_dl), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy):

            # Run parallel checks (session, filename, file_bytes, content_type, request_id)
            result = await run_quality_checks_parallel(
                aiohttp_session,
                "test_ktp.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_002"
            )

            # Result is Optional[PipelineError] - None means passed all checks
            assert result is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_checks_with_classifier_rejection(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test quality checks with classifier rejecting the image."""

        from src.api.models import ServiceResult, ClassifierData, QualityData, SpoofData

        # Create async mocks
        # Classifier returns success but with detected=False to indicate not a valid KTP
        mock_classifier = AsyncMock(return_value=ServiceResult.success(
            ClassifierData(raw_response={"class": "other", "detected": False})
        ))
        mock_quality = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_quality_dl = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_quality_dl", mock_quality_dl), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy):

            result = await run_quality_checks_parallel(
                aiohttp_session,
                "not_ktp.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_003"
            )

            # Result should be PipelineError indicating not a valid KTP
            assert result is not None
            assert isinstance(result, PipelineError)
            assert result.error_code == ErrorCode.IMAGE_REJECTED
            assert result.details is not None
            assert len(result.details["reasons"]) == 1
            assert result.details["reasons"][0]["error_code"] == ErrorCode.NOT_KTP
            assert "valid ktp" in result.message.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_checks_with_spoof_detection(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test quality checks with spoof detection."""

        from src.api.models import ServiceResult, ClassifierData, QualityData, SpoofData

        # Create async mocks
        mock_classifier = AsyncMock(return_value=ServiceResult.success(ClassifierData(raw_response={"class": "ktp", "detected": True})))
        mock_quality = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_quality_dl = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.reject(
            rejection_type="lamination",
            message="Lamination detected on document"
        ))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_quality_dl", mock_quality_dl), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy):

            result = await run_quality_checks_parallel(
                aiohttp_session,
                "laminated.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_004"
            )

            # Result should be PipelineError indicating lamination issue
            assert result is not None
            assert isinstance(result, PipelineError)
            assert result.error_code == ErrorCode.IMAGE_REJECTED
            assert result.details is not None
            assert len(result.details["reasons"]) == 1
            assert result.details["reasons"][0]["error_code"] == ErrorCode.SPOOF_UNLAMINATED
            assert "material" in result.message.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_process_with_service_timeout(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR process handling service timeout."""

        from src.api.models import ServiceResult, ClassifierData, QualityData, SpoofData

        # Create async mocks - Mock OCR timeout error
        mock_ocr = AsyncMock(return_value=ServiceResult.fail(
            error_type="timeout",
            message="OCR service timeout"
        ))
        mock_classifier = AsyncMock(return_value=ServiceResult.success(ClassifierData(raw_response={"class": "ktp", "detected": True})))
        mock_quality = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_quality_dl = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.save_image"), \
             patch("src.services.manage_service.request_ocr", mock_ocr), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_quality_dl", mock_quality_dl), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy):

            result, pipeline_error = await ocr_process(
                aiohttp_session,
                "test.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_005"
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ocr_process_with_poor_quality_image(
        self,
        aiohttp_session,
        sample_ktp_image_bytes,
        mock_settings
    ):
        """Test OCR process with poor quality image."""

        from src.api.models import ServiceResult, OCRData, ClassifierData, QualityData, SpoofData

        # Create async mocks
        mock_ocr = AsyncMock(return_value=ServiceResult.success(OCRData(text=[])))
        mock_classifier = AsyncMock(return_value=ServiceResult.success(ClassifierData(raw_response={"class": "ktp", "detected": True})))
        mock_quality = AsyncMock(return_value=ServiceResult.reject(
            rejection_type="quality",
            message="Poor image quality, please retake photo",
            details={"is_blurry": True, "is_glare": False, "is_rotated": False}
        ))
        mock_quality_dl = AsyncMock(return_value=ServiceResult.success(QualityData(passed=True)))
        mock_lamination = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_recapture = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))
        mock_graycopy = AsyncMock(return_value=ServiceResult.success(SpoofData(passed=True)))

        with patch("src.services.manage_service.get_settings", return_value=mock_settings), \
             patch("src.services.manage_service.save_image", new_callable=AsyncMock), \
             patch("src.services.manage_service.save_image_gcs", new_callable=AsyncMock), \
             patch("src.services.manage_service.request_ocr", mock_ocr), \
             patch("src.services.manage_service.request_classifier", mock_classifier), \
             patch("src.services.manage_service.request_quality_rulebase", mock_quality), \
             patch("src.services.manage_service.request_quality_dl", mock_quality_dl), \
             patch("src.services.manage_service.request_lamination", mock_lamination), \
             patch("src.services.manage_service.request_recapture", mock_recapture), \
             patch("src.services.manage_service.request_graycopy", mock_graycopy):

            result, pipeline_error = await ocr_process(
                aiohttp_session,
                "blurry.jpg",
                sample_ktp_image_bytes,
                "image/jpeg",
                "INT_TEST_006"
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.IMAGE_REJECTED
            assert pipeline_error.details is not None
            assert pipeline_error.details["reasons"][0]["error_code"] == ErrorCode.QUALITY_BLUR
