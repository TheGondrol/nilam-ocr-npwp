"""
Unit tests for manage_service module (orchestration logic).
"""

import pytest
from unittest.mock import patch, MagicMock

from src.services.manage_service import (
    run_quality_checks_parallel,
    ocr_process
)
from src.api.models import (
    ServiceResult,
    OCRData,
    ClassifierData,
    SpoofData,
    QualityData,
    PostprocessData,
    ErrorCode,
    PipelineError,
)


class TestManageService:
    """Tests for orchestration logic in manage_service."""

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_all_pass(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test parallel quality checks when all pass."""
        # Mock all services to return success
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": True})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is None  # No error

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_lamination_fail(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test parallel quality checks when lamination fails."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.reject(
                    rejection_type="spoof_lamination",
                    message="Lamination detected",
                    details={}
                )
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": True})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                assert len(result.details["reasons"]) == 1
                assert result.details["reasons"][0]["error_code"] == ErrorCode.SPOOF_UNLAMINATED
                assert "material" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_service_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test parallel quality checks with service error."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.fail(
                    error_type="timeout",
                    message="Service timeout",
                    details="Connection timeout"
                )
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": True})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                assert result.error_code == ErrorCode.SERVICE_ERROR
                assert result.http_status == 500

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_classifier_not_detected(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test parallel quality checks when classifier doesn't detect KTP."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": False})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                assert len(result.details["reasons"]) == 1
                assert result.details["reasons"][0]["error_code"] == ErrorCode.NOT_KTP
                assert "valid ktp" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_error_takes_priority_over_rejection(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test that a service error takes priority over rejections from other services."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                # Lamination errors out (timeout)
                mock_lam.return_value = ServiceResult.fail(
                    error_type="timeout",
                    message="Lamination service timed out",
                    details="Connection timeout"
                )
                # Recapture rejects (would be ignored because error takes priority)
                mock_recap.return_value = ServiceResult.reject(
                    rejection_type="spoof_recapture",
                    message="Recapture detected",
                    details={}
                )
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": True})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                # Error takes priority over rejection
                assert result.error_code == ErrorCode.SERVICE_ERROR
                assert result.http_status == 500
                assert "lamination" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_multiple_rejections(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test that multiple rejections are all collected into reasons list."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.reject(
                    rejection_type="spoof_lamination",
                    message="Lamination detected",
                    details={}
                )
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.reject(
                    rejection_type="spoof_graycopy",
                    message="Graycopy detected",
                    details={}
                )
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": True})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                # Both rejections should be in the reasons list
                reason_codes = [r["error_code"] for r in result.details["reasons"]]
                assert len(reason_codes) == 2
                assert ErrorCode.SPOOF_UNLAMINATED in reason_codes
                assert ErrorCode.SPOOF_GRAYCOPY in reason_codes

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_rejection_and_classifier_not_ktp(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test spoof rejection combined with classifier NOT_KTP in same response."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:

                mock_lam.return_value = ServiceResult.reject(
                    rejection_type="spoof_lamination",
                    message="Lamination detected",
                    details={}
                )
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"detected": False})
                )

                result = await run_quality_checks_parallel(
                    mock_aiohttp_session,
                    sample_filename,
                    sample_jpeg_bytes,
                    sample_content_type,
                    request_id
                )

                assert result is not None
                assert isinstance(result, PipelineError)
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                reason_codes = [r["error_code"] for r in result.details["reasons"]]
                assert len(reason_codes) == 2
                assert ErrorCode.SPOOF_UNLAMINATED in reason_codes
                assert ErrorCode.NOT_KTP in reason_codes

    @pytest.mark.asyncio
    async def test_ocr_process_success_flow(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        sample_postprocess_response,
        mock_settings
    ):
        """Test successful OCR process flow."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None  # No quality issues

            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )

            mock_quality_rb.return_value = ServiceResult.success(
                QualityData(passed=True)
            )

            mock_postprocess.return_value = ServiceResult.success(
                PostprocessData(results={"ocr_result": sample_postprocess_response})
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is not None
            assert pipeline_error is None

    @pytest.mark.asyncio
    async def test_ocr_process_quality_rejection(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test OCR process with quality rejection."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = PipelineError(
                error_code=ErrorCode.IMAGE_REJECTED,
                message="Image rejected: Document material does not meet standards",
                details={
                    "reasons": [
                        {"error_code": ErrorCode.SPOOF_UNLAMINATED, "message": "Document material does not meet standards"}
                    ]
                },
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.IMAGE_REJECTED
            assert pipeline_error.details is not None
            assert pipeline_error.details["reasons"][0]["error_code"] == ErrorCode.SPOOF_UNLAMINATED
            assert pipeline_error.http_status == 400

    @pytest.mark.asyncio
    async def test_ocr_process_ocr_service_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test OCR process with OCR service error."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None

            mock_ocr.return_value = ServiceResult.fail(
                error_type="timeout",
                message="OCR timeout",
                details="Service timeout"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.asyncio
    async def test_ocr_process_quality_rulebase_rejection(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        mock_settings
    ):
        """Test OCR process with quality rulebase rejection."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None

            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )

            mock_quality_rb.return_value = ServiceResult.reject(
                rejection_type="quality_rulebase",
                message="Poor image quality detected: blur",
                details={"is_blurry": True}
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.IMAGE_REJECTED
            assert pipeline_error.details is not None
            assert len(pipeline_error.details["reasons"]) == 1
            assert pipeline_error.details["reasons"][0]["error_code"] == ErrorCode.QUALITY_BLUR
            assert "blurry" in pipeline_error.details["reasons"][0]["message"].lower()

    @pytest.mark.asyncio
    async def test_ocr_process_postprocess_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        mock_settings
    ):
        """Test OCR process with postprocess service error."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None

            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )

            mock_quality_rb.return_value = ServiceResult.success(
                QualityData(passed=True)
            )

            mock_postprocess.return_value = ServiceResult.fail(
                error_type="http_error",
                message="Postprocess failed",
                details="Service unavailable"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.asyncio
    async def test_run_quality_checks_parallel_no_checks_enabled(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        mock_settings
    ):
        """Test parallel quality checks when no services are enabled — should pass through."""
        mock_settings.run_services.lamination = False
        mock_settings.run_services.recapture = False
        mock_settings.run_services.graycopy = False
        mock_settings.run_services.classifier = False

        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings

            result = await run_quality_checks_parallel(
                mock_aiohttp_session,
                sample_filename,
                sample_jpeg_bytes,
                sample_content_type,
                request_id
            )

            assert result is None  # No error, pass through

    @pytest.mark.asyncio
    async def test_ocr_process_quality_dl_rejection(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        mock_settings
    ):
        """Test OCR process with quality DL rejection (label=='bad')."""
        mock_settings.run_services.qualitydl = True

        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_quality_dl") as mock_quality_dl:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_quality_dl.return_value = ServiceResult.reject(
                rejection_type="quality_dl",
                message="Poor image quality detected by DL model",
                details={"quality": "bad"}
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.IMAGE_REJECTED
            assert pipeline_error.details is not None
            assert pipeline_error.details["reasons"][0]["error_code"] == ErrorCode.QUALITY_DL_POOR

    @pytest.mark.asyncio
    async def test_ocr_process_quality_dl_service_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        mock_settings
    ):
        """Test OCR process with quality DL service error."""
        mock_settings.run_services.qualitydl = True

        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_quality_dl") as mock_quality_dl:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_quality_dl.return_value = ServiceResult.fail(
                error_type="timeout", message="Quality DL service timed out"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.asyncio
    async def test_ocr_process_temper_rejection(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        sample_postprocess_response,
        mock_settings
    ):
        """Test OCR process with temper rejection (tampering detected)."""
        mock_settings.run_services.temper = True

        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess, \
             patch("src.services.manage_service.crop_image") as mock_crop, \
             patch("src.services.manage_service.request_temper") as mock_temper:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_postprocess.return_value = ServiceResult.success(
                PostprocessData(results={
                    "ocr_result": sample_postprocess_response,
                    "nik_image_box": [[0, 0], [100, 0], [100, 50], [0, 50]],
                })
            )
            mock_crop.return_value = sample_jpeg_bytes
            mock_temper.return_value = ServiceResult.reject(
                rejection_type="temper",
                message="Image tampering detected",
                details={"prediction": "fake"}
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.IMAGE_REJECTED
            assert pipeline_error.details is not None
            assert pipeline_error.details["reasons"][0]["error_code"] == ErrorCode.TAMPERED

    @pytest.mark.asyncio
    async def test_ocr_process_temper_service_error(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        sample_postprocess_response,
        mock_settings
    ):
        """Test OCR process with temper service error."""
        mock_settings.run_services.temper = True

        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess, \
             patch("src.services.manage_service.crop_image") as mock_crop, \
             patch("src.services.manage_service.request_temper") as mock_temper:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_postprocess.return_value = ServiceResult.success(
                PostprocessData(results={
                    "ocr_result": sample_postprocess_response,
                    "nik_image_box": [[0, 0], [100, 0], [100, 50], [0, 50]],
                })
            )
            mock_crop.return_value = sample_jpeg_bytes
            mock_temper.return_value = ServiceResult.fail(
                error_type="timeout", message="Temper service timed out"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.asyncio
    async def test_ocr_process_no_postprocess_result(
        self,
        mock_aiohttp_session,
        sample_jpeg_bytes,
        sample_filename,
        sample_content_type,
        request_id,
        sample_ocr_response,
        mock_settings
    ):
        """Test OCR process when postprocess returns no ocr_result — fallback error."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            # Postprocess returns data but without the ocr_result key
            mock_postprocess.return_value = ServiceResult.success(
                PostprocessData(results={})
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )

            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500
            assert "no postprocess result" in pipeline_error.message.lower()


class TestMapQualityRejections:
    """Tests for _map_quality_rejections edge cases."""

    def test_glare_only(self):
        from src.services.manage_service import _map_quality_rejections
        from src.api.models import ServiceRejection
        rejection = ServiceRejection(
            rejection_type="quality_rulebase",
            message="glare",
            details={"is_blurry": False, "is_glare": True, "is_rotated": False}
        )
        result = _map_quality_rejections(rejection)
        assert len(result) == 1
        assert result[0][0] == ErrorCode.QUALITY_GLARE

    def test_rotation_only(self):
        from src.services.manage_service import _map_quality_rejections
        from src.api.models import ServiceRejection
        rejection = ServiceRejection(
            rejection_type="quality_rulebase",
            message="rotated",
            details={"is_blurry": False, "is_glare": False, "is_rotated": True}
        )
        result = _map_quality_rejections(rejection)
        assert len(result) == 1
        assert result[0][0] == ErrorCode.QUALITY_ROTATION

    def test_all_three_issues(self):
        from src.services.manage_service import _map_quality_rejections
        from src.api.models import ServiceRejection
        rejection = ServiceRejection(
            rejection_type="quality_rulebase",
            message="all",
            details={"is_blurry": True, "is_glare": True, "is_rotated": True}
        )
        result = _map_quality_rejections(rejection)
        codes = [r[0] for r in result]
        assert ErrorCode.QUALITY_BLUR in codes
        assert ErrorCode.QUALITY_GLARE in codes
        assert ErrorCode.QUALITY_ROTATION in codes

    def test_fallback_no_issues(self):
        from src.services.manage_service import _map_quality_rejections
        from src.api.models import ServiceRejection
        rejection = ServiceRejection(
            rejection_type="quality_rulebase",
            message="unknown",
            details={"is_blurry": False, "is_glare": False, "is_rotated": False}
        )
        result = _map_quality_rejections(rejection)
        assert len(result) == 1
        assert result[0][0] == ErrorCode.QUALITY_MULTIPLE

    def test_none_details(self):
        from src.services.manage_service import _map_quality_rejections
        from src.api.models import ServiceRejection
        rejection = ServiceRejection(
            rejection_type="quality_rulebase",
            message="no details",
            details=None
        )
        result = _map_quality_rejections(rejection)
        assert len(result) == 1
        assert result[0][0] == ErrorCode.QUALITY_MULTIPLE


class TestManageServiceEdgeCases:

    @pytest.mark.asyncio
    async def test_run_quality_checks_recapture_rejection(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id, mock_settings
    ):
        """Test recapture rejection in parallel checks."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings
            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:
                mock_lam.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_recap.return_value = ServiceResult.reject(
                    rejection_type="recapture", message="Recapture detected"
                )
                mock_gray.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"class": "ktp", "detected": True})
                )
                result = await run_quality_checks_parallel(
                    mock_aiohttp_session, sample_filename, sample_jpeg_bytes,
                    sample_content_type, request_id
                )
                assert result is not None
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                assert result.details["reasons"][0]["error_code"] == ErrorCode.SPOOF_RECAPTURE

    @pytest.mark.asyncio
    async def test_run_quality_checks_graycopy_rejection(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id, mock_settings
    ):
        """Test graycopy rejection in parallel checks."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings:
            mock_get_settings.return_value = mock_settings
            with patch("src.services.manage_service.request_lamination") as mock_lam, \
                 patch("src.services.manage_service.request_recapture") as mock_recap, \
                 patch("src.services.manage_service.request_graycopy") as mock_gray, \
                 patch("src.services.manage_service.request_classifier") as mock_class:
                mock_lam.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_recap.return_value = ServiceResult.success(SpoofData(passed=True))
                mock_gray.return_value = ServiceResult.reject(
                    rejection_type="graycopy", message="Graycopy detected"
                )
                mock_class.return_value = ServiceResult.success(
                    ClassifierData(raw_response={"class": "ktp", "detected": True})
                )
                result = await run_quality_checks_parallel(
                    mock_aiohttp_session, sample_filename, sample_jpeg_bytes,
                    sample_content_type, request_id
                )
                assert result is not None
                assert result.error_code == ErrorCode.IMAGE_REJECTED
                assert result.details is not None
                assert result.details["reasons"][0]["error_code"] == ErrorCode.SPOOF_GRAYCOPY

    @pytest.mark.asyncio
    async def test_ocr_process_image_save_failure_does_not_block(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id, sample_ocr_response,
        sample_postprocess_response, mock_settings
    ):
        """Image save failure is logged but doesn't block pipeline."""
        mock_settings.logging.log_images = True
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image_gcs", side_effect=Exception("GCS down")), \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_postprocess") as mock_postprocess:

            mock_get_settings.return_value = mock_settings
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_postprocess.return_value = ServiceResult.success(
                PostprocessData(results={"ocr_result": sample_postprocess_response})
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )
            assert pipeline_error is None
            assert result is not None

    @pytest.mark.asyncio
    async def test_ocr_process_quality_rulebase_service_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id, sample_ocr_response, mock_settings
    ):
        """Quality rulebase service error returns SERVICE_ERROR."""
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.fail(
                error_type="timeout",
                message="Quality rulebase service timed out"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )
            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
            assert pipeline_error.http_status == 500

    @pytest.mark.asyncio
    async def test_ocr_process_quality_dl_error(
        self, mock_aiohttp_session, sample_jpeg_bytes, sample_filename,
        sample_content_type, request_id, sample_ocr_response, mock_settings
    ):
        """Quality DL service error returns SERVICE_ERROR."""
        mock_settings.run_services.qualitydl = True
        with patch("src.services.manage_service.get_settings") as mock_get_settings, \
             patch("src.services.manage_service.save_image") as mock_save, \
             patch("src.services.manage_service.run_quality_checks_parallel") as mock_quality, \
             patch("src.services.manage_service.request_ocr") as mock_ocr, \
             patch("src.services.manage_service.request_quality_rulebase") as mock_quality_rb, \
             patch("src.services.manage_service.request_quality_dl") as mock_quality_dl:

            mock_get_settings.return_value = mock_settings
            mock_save.return_value = None
            mock_quality.return_value = None
            mock_ocr.return_value = ServiceResult.success(
                OCRData(text=sample_ocr_response["ocr_result"])
            )
            mock_quality_rb.return_value = ServiceResult.success(QualityData(passed=True))
            mock_quality_dl.return_value = ServiceResult.fail(
                error_type="timeout",
                message="Quality DL service timed out"
            )

            result, pipeline_error = await ocr_process(
                mock_aiohttp_session, sample_filename,
                sample_jpeg_bytes, sample_content_type, request_id
            )
            assert result is None
            assert pipeline_error is not None
            assert pipeline_error.error_code == ErrorCode.SERVICE_ERROR
