"""API routes for quality classification."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.config import config
from src.core.logging import request_id_ctx
from src.schemas.api_schema import ClassificationResponse, Crop, HealthResponse
from src.services.database_services import insert_log
from src.services.quality_service import get_device, get_model, process_and_classify_sync

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound ML operations
_executor: Optional[ThreadPoolExecutor] = None

# API Router
router = APIRouter(tags=["KTP Quality Classification"])


def set_executor(executor: ThreadPoolExecutor) -> None:
    """Set global thread pool executor."""
    global _executor
    _executor = executor


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.get("api.title", "KTP Image Quality Classifier API"),
        "description": config.get(
            "api.description",
            "Classifies KTP image quality by analyzing OCR text crop quality",
        ),
        "version": config.get("api.version", "1.0.0"),
        "endpoints": {
            "health": "/health",
            "filter": "/filter (POST)",
        },
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model = get_model()
    device = get_device()

    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        version=config.get("api.version", "1.0.0"),
    )


@router.post("/filter", response_model=ClassificationResponse)
async def filter_quality(
    file: UploadFile = File(..., description="Image file to classify"),
    crops: str = Form(..., description="JSON string of crops list"),
):
    """
    Classify image quality based on OCR text crop quality.

    This endpoint:
    1. Receives an image and list of crops (bbox, text, confidence from OCR)
    2. Filters crops to focus on text (width > height)
    3. Extracts and classifies each crop as good/bad
    4. Returns overall image quality (bad if >= threshold bad crops)

    Args:
        file: Uploaded image file
        crops: JSON string containing list of crop objects with bbox, text, confidence

    Returns:
        ClassificationResponse with predictions and overall label

    Raises:
        HTTPException: If processing fails
    """
    request_id = request_id_ctx.get()
    start_time = time.time()
    response_code = 500
    error_message = ""
    result = None
    payload = None

    try:
        # Validate model loaded
        model = get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Parse crops from JSON string
        try:
            crops_data = json.loads(crops)
            if not isinstance(crops_data, list):
                raise ValueError("Crops must be a list")

            # Validate and convert to Crop objects
            crop_objects = [Crop(**crop_dict) for crop_dict in crops_data]

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in crops: {e}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid crop data: {e}")

        logger.info(f"Processing request with {len(crop_objects)} crops")

        # Store payload for logging
        payload = {
            "filename": file.filename,
            "content_type": file.content_type,
            "num_crops": len(crop_objects),
        }

        # Read file bytes
        contents = await file.read()

        # Run classification in thread pool (CPU-bound operation)
        loop = asyncio.get_event_loop()
        classification_result = await loop.run_in_executor(
            _executor,
            process_and_classify_sync,
            contents,
            crop_objects,
        )

        response_code = 200
        result = classification_result.model_dump()

        logger.info(
            f"Classification successful: {classification_result.label} "
            f"({classification_result.num_bad}/{classification_result.num_filtered} bad crops)"
        )

        return classification_result

    except HTTPException:
        raise
    except Exception as e:
        response_code = 500
        error_message = str(e)
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    finally:
        # Log to database (always executes)
        processing_time = time.time() - start_time

        if config.log_to_database:
            try:
                await insert_log(
                    request_id=request_id,
                    response_code=response_code,
                    payload=payload,
                    error_message=error_message,
                    result=result,
                    processing_time=processing_time,
                )
            except Exception as e:
                logger.error(f"Failed to log to database: {e}")

        logger.info(f"Request completed in {processing_time:.3f}s with code {response_code}")
