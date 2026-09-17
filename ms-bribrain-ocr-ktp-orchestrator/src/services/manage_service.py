import os
import aiohttp
import asyncio

from typing import Optional, Tuple, Union, Any
from src.core.logging import get_logger
from src.core.config import get_settings
from src.services.ocr_service import request_ocr
from src.services.classifier_service import request_classifier
from src.services.postprocess_service import request_postprocess
from src.services.temper_service import request_temper
from src.services.spoof_service import (
    request_lamination,
    request_recapture,
    request_graycopy,
)
from src.services.quality_service import request_quality_rulebase, request_quality_dl
from src.api.models import ServiceResult, ServiceRejection, ErrorCode, PipelineError
from src.services.crop_helper import crop_image
from src.services.minio_service import save_image
from src.services.gcs_service import save_image_gcs

logger = get_logger(__name__)

APP_ENVIRO = os.getenv("APP_ENVIRO", "gcp")


def _map_quality_rejections(rejection: ServiceRejection) -> list[tuple[str, str]]:
    """Map a quality rulebase rejection to a list of (error_code, message) tuples."""
    details = rejection.details or {}
    is_blurry = details.get("is_blurry", False)
    is_glare = details.get("is_glare", False)
    is_rotated = details.get("is_rotated", False)

    rejections: list[tuple[str, str]] = []
    if is_blurry:
        rejections.append((ErrorCode.QUALITY_BLUR, "Image is too blurry. Please take a clearer photo."))
    if is_glare:
        rejections.append((ErrorCode.QUALITY_GLARE, "Image has glare. Please reduce reflections and retake the photo."))
    if is_rotated:
        rejections.append((ErrorCode.QUALITY_ROTATION, "Image is rotated or zoomed incorrectly. Please align the document and retake."))

    if not rejections:
        # Fallback (shouldn't happen - no issues detected but status was rejection)
        rejections.append((ErrorCode.QUALITY_MULTIPLE, "Image quality check failed. Please retake the photo."))

    return rejections


async def run_quality_checks_parallel(
    session: aiohttp.ClientSession,
    filename: str,
    file_bytes: bytes,
    content_type: str,
    request_id: str,
) -> Optional[PipelineError]:
    """
    Run lamination, recapture, graycopy, and classifier checks in parallel.

    Returns:
        A PipelineError if the image is rejected or a service error occurred,
        or None if all checks pass.
    """
    settings = get_settings()

    # Build list of tasks to run based on enabled services
    tasks: list[Any] = []
    task_names: list[str] = []

    if settings.run_services.lamination:
        tasks.append(
            request_lamination(session, file_bytes, filename, content_type, request_id)
        )
        task_names.append("lamination")

    if settings.run_services.recapture:
        tasks.append(
            request_recapture(session, file_bytes, filename, content_type, request_id)
        )
        task_names.append("recapture")

    if settings.run_services.graycopy:
        tasks.append(
            request_graycopy(session, file_bytes, filename, content_type, request_id)
        )
        task_names.append("graycopy")

    if settings.run_services.classifier:
        tasks.append(
            request_classifier(session, file_bytes, filename, content_type, request_id)
        )
        task_names.append("classifier")

    # If no checks are enabled, pass through
    if not tasks:
        return None

    # Run all enabled checks in parallel
    from src.api.models import ClassifierData, SpoofData
    results: list[Union[ServiceResult[SpoofData], ServiceResult[ClassifierData], ServiceResult[Any]]] = await asyncio.gather(*tasks)

    # Map results back to their service names
    result_map = dict(zip(task_names, results))

    # Check for service errors first
    for service_name, result in result_map.items():
        if result.status == "error" and result.error is not None:
            logger.error(
                f"[{request_id}] Service error in {service_name}: {result.error.message}"
            )
            return PipelineError(
                error_code=ErrorCode.SERVICE_ERROR,
                message=f"{service_name.capitalize()} check is temporarily unavailable. Please try again later.",
                http_status=500,
            )

    # Check for rejections (business logic)
    rejections: list[tuple[str, str]] = []

    # Lamination check
    lamination_result = result_map.get("lamination")
    if lamination_result and lamination_result.status == "rejection":
        rejections.append(
            (ErrorCode.SPOOF_UNLAMINATED, "Document material does not meet standards")
        )

    # Recapture check
    recapture_result = result_map.get("recapture")
    if recapture_result and recapture_result.status == "rejection":
        rejections.append(
            (ErrorCode.SPOOF_RECAPTURE, "Image appears to be a photo of a screen or printout")
        )

    # Graycopy check
    graycopy_result = result_map.get("graycopy")
    if graycopy_result and graycopy_result.status == "rejection":
        rejections.append(
            (ErrorCode.SPOOF_GRAYCOPY, "Image appears to be a grayscale photocopy")
        )

    # Classifier check - not detected as KTP
    from src.api.models import ClassifierData
    classifier_result = result_map.get("classifier")
    if classifier_result:
        if classifier_result.status == "success" and classifier_result.data is not None:
            classifier_data = classifier_result.data
            if hasattr(classifier_data, "raw_response"):
                raw_response = classifier_data.raw_response
                if not raw_response.get("detected", False):
                    rejections.append(
                        (ErrorCode.NOT_KTP, "Image is not recognized as a valid KTP")
                    )

    if not rejections:
        return None

    return PipelineError(
        error_code=ErrorCode.IMAGE_REJECTED,
        message="Image rejected: " + "; ".join(msg for _, msg in rejections),
        details={
            "reasons": [
                {"error_code": code, "message": msg} for code, msg in rejections
            ]
        },
    )


async def ocr_process(
    session: aiohttp.ClientSession,
    filename: str,
    file_bytes: bytes,
    content_type: str,
    request_id: str,
) -> Tuple[Optional[dict], Optional[PipelineError]]:
    """
    Execute the full OCR processing pipeline for a given image.

    Args:
        session: Shared aiohttp ClientSession for connection pooling.
        filename: The name of the image file.
        file_bytes: The image content in bytes.
        content_type: The MIME type of the image.
        request_id: The request ID.

    Returns:
        Tuple containing:
        - final OCR result dict (None on failure/rejection)
        - PipelineError with structured error info (None on success)

        Examples:
        - Success: (result_dict, None)
        - Business rejection: (None, PipelineError(error_code="IMAGE_REJECTED", ...))
        - Service error: (None, PipelineError(error_code="SERVICE_ERROR", http_status=500, ...))
    """
    settings = get_settings()

    # Run image saving and quality checks in parallel.
    # Image saving is fire-and-forget — failures are logged but don't block the pipeline.
    async def _save_image_task():
        if settings.logging.log_images:
            try:
                if APP_ENVIRO == "onprem":
                    await save_image(file_bytes, request_id)
                else:
                    await save_image_gcs(file_bytes, request_id)
            except Exception as e:
                logger.error(f"[{request_id}] Failed to save image to storage: {e}")

    _, pipeline_error = await asyncio.gather(
        _save_image_task(),
        run_quality_checks_parallel(
            session, filename, file_bytes, content_type, request_id
        ),
    )

    if pipeline_error is not None:
        return None, pipeline_error

    # Perform OCR
    ocr_text = None
    if settings.run_services.ocr:
        ocr_result = await request_ocr(
            session, file_bytes, filename, content_type, request_id
        )

        if ocr_result.status == "error" and ocr_result.error is not None:
            logger.error(
                f"[{request_id}] OCR service error: {ocr_result.error.message}"
            )
            return None, PipelineError(
                error_code=ErrorCode.SERVICE_ERROR,
                message="OCR processing is temporarily unavailable. Please try again later.",
                http_status=500,
            )

        if ocr_result.data is not None:
            ocr_text = ocr_result.data.text

    # Run rule-based quality checks (require OCR result)
    if settings.run_services.quality:
        quality_result = await request_quality_rulebase(
            session, file_bytes, filename, content_type, ocr_text, request_id
        )

        if quality_result.status == "error" and quality_result.error is not None:
            logger.error(
                f"[{request_id}] Quality rulebase service error: {quality_result.error.message}"
            )
            return None, PipelineError(
                error_code=ErrorCode.SERVICE_ERROR,
                message="Quality rulebase check is temporarily unavailable. Please try again later.",
                http_status=500,
            )

        if quality_result.status == "rejection" and quality_result.rejection is not None:
            rejections = _map_quality_rejections(quality_result.rejection)
            return None, PipelineError(
                error_code=ErrorCode.IMAGE_REJECTED,
                message="Image rejected: " + "; ".join(msg for _, msg in rejections),
                details={
                    "reasons": [
                        {"error_code": code, "message": msg} for code, msg in rejections
                    ]
                },
            )

    # Run DL-based quality check
    if settings.run_services.qualitydl:
        qualitydl_result = await request_quality_dl(
            session, file_bytes, filename, content_type, ocr_text, request_id
        )

        if qualitydl_result.status == "error" and qualitydl_result.error is not None:
            logger.error(
                f"[{request_id}] Quality DL service error: {qualitydl_result.error.message}"
            )
            return None, PipelineError(
                error_code=ErrorCode.SERVICE_ERROR,
                message="Quality DL check is temporarily unavailable. Please try again later.",
                http_status=500,
            )

        if qualitydl_result.status == "rejection" and qualitydl_result.rejection is not None:
            return None, PipelineError(
                error_code=ErrorCode.IMAGE_REJECTED,
                message="Image rejected: Image quality is too low. Please retake with better lighting and focus.",
                details={
                    "reasons": [
                        {"error_code": ErrorCode.QUALITY_DL_POOR, "message": "Image quality is too low. Please retake with better lighting and focus."}
                    ]
                },
            )

    # Postprocess OCR results
    result_text = None
    result_nik_bbox = None
    if settings.run_services.postprocess and ocr_text is not None:
        postprocess_result = await request_postprocess(session, ocr_text, request_id)

        if postprocess_result.status == "error" and postprocess_result.error is not None:
            logger.error(
                f"[{request_id}] Postprocess service error: {postprocess_result.error.message}"
            )
            return None, PipelineError(
                error_code=ErrorCode.SERVICE_ERROR,
                message="Post-processing is temporarily unavailable. Please try again later.",
                http_status=500,
            )

        if postprocess_result.data is not None:
            result_dict = postprocess_result.data.results
            result_text = result_dict.get("ocr_result")
            result_nik_bbox = result_dict.get("nik_image_box")

    # Temper check on cropped NIK region. A malformed/degenerate NIK bbox from
    # postprocess must not discard an already-successful OCR result, so guard the
    # crop and treat the optional tamper check as skipped on failure (BUG-03).
    if result_nik_bbox is not None and settings.run_services.temper:
        try:
            cropped_nik = crop_image(file_bytes, result_nik_bbox)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"[{request_id}] Invalid NIK bbox {result_nik_bbox!r}, skipping tamper check: {e}"
            )
            cropped_nik = None

        if cropped_nik is not None:
            temper_result = await request_temper(
                session, cropped_nik, filename, content_type, request_id
            )

            if temper_result.status == "error" and temper_result.error is not None:
                logger.error(
                    f"[{request_id}] Temper service error: {temper_result.error.message}"
                )
                return None, PipelineError(
                    error_code=ErrorCode.SERVICE_ERROR,
                    message="Tamper check is temporarily unavailable. Please try again later.",
                    http_status=500,
                )

            if temper_result.status == "rejection" and temper_result.rejection is not None:
                return None, PipelineError(
                    error_code=ErrorCode.IMAGE_REJECTED,
                    message="Image rejected: Document appears to have been digitally altered.",
                    details={
                        "reasons": [
                            {"error_code": ErrorCode.TAMPERED, "message": "Document appears to have been digitally altered."}
                        ]
                    },
                )

    # Return OCR result or error if postprocessing was skipped
    if result_text is not None:
        return result_text, None

    return None, PipelineError(
        error_code=ErrorCode.SERVICE_ERROR,
        message="OCR pipeline failed: no postprocess result available. Please try again later.",
        http_status=500,
    )
