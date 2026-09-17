"""OCR service for text extraction from images"""

import atexit
import gc
import logging
import asyncio
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Any, List, Tuple, Optional
import cv2
import numpy as np

from src.core.config import settings
from src.core.device import get_use_gpu
from src.core.exceptions import OCRInitializationError, OCRProcessingError, ImageValidationError
from src.services.ocr_backends import create_ocr_backend
from src.services.threshold_provider import get_provider

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound image processing operations
_executor: Optional[ThreadPoolExecutor] = None

# Dedicated single-thread executor for OCR model calls. torch.compile(mode=
# "reduce-overhead") stores CUDA-graph state in thread-local storage, so
# invoking the compiled model from multiple threads trips an assert inside
# cudagraph_trees.get_obj. Pinning OCR to one worker keeps all calls on the
# thread that performed the warmup capture.
_ocr_executor: Optional[ThreadPoolExecutor] = None

# Lock for thread-safe OCR access (PaddlePaddle is not thread-safe)
_ocr_lock = threading.Lock()

# Global OCR instance
_ocr_instance: Optional[Any] = None
_ocr_initialized: bool = False


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr_worker")
    return _executor


def _get_ocr_executor() -> ThreadPoolExecutor:
    global _ocr_executor
    if _ocr_executor is None:
        _ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr_model")
    return _ocr_executor


def cleanup_ocr() -> None:
    """
    Clean up OCR resources.
    
    Should be called during application shutdown to properly release
    GPU memory and thread pool resources.
    """
    global _ocr_instance, _executor, _ocr_executor, _ocr_initialized

    logger.info("Cleaning up OCR resources...")

    # Shutdown thread pool
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=False)
        _executor = None
        logger.debug("Thread pool executor shut down")

    # Shutdown the dedicated OCR model executor too — its non-daemon thread is
    # otherwise leaked across in-process re-inits (reload/tests/atexit) (BUG-16).
    if _ocr_executor is not None:
        _ocr_executor.shutdown(wait=True, cancel_futures=False)
        _ocr_executor = None
        logger.debug("OCR model executor shut down")

    # Clear OCR instance — close the backend first so CUDA/Paddle/context
    # resources are released (gc.collect alone won't: no __del__ finalizer) (BUG-16).
    if _ocr_instance is not None:
        close = getattr(_ocr_instance, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning(f"Error closing OCR backend: {exc}")
        _ocr_instance = None
        _ocr_initialized = False
        gc.collect()  # Help release GPU memory
        logger.debug("OCR instance cleared")
    
    logger.info("OCR cleanup complete")


# Register cleanup on interpreter exit
atexit.register(cleanup_ocr)


def is_ocr_ready() -> bool:
    """
    Check if OCR engine is initialized and ready.
    
    Returns:
        bool: True if OCR is ready, False otherwise
    """
    return _ocr_initialized and _ocr_instance is not None


def _build_backend_settings():
    """Build a settings-like object that overlays env vars for backend selection.

    Env vars (when set) override config.yaml:
      OCR_BACKEND, OCR_AUTOKERNEL_ENABLED, OCR_SERVER_CONFIG_PATH,
      AUTOKERNEL_ROOT, AUTOKERNEL_PPOCR_ROOT, AUTOKERNEL_WORKSPACE,
      AUTOKERNEL_PPOCRV5_SERVER_REC_PTH, AUTOKERNEL_PPOCRV5_SERVER_DET_PTH,
      OCR_AUTOKERNEL_DTYPE, OCR_AUTOKERNEL_REC_BATCH_SIZE,
      OCR_AUTOKERNEL_REC_IMAGE_SHAPE,
      OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO,
      OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN, OCR_AUTOKERNEL_DET_LIMIT_TYPE,
      OCR_AUTOKERNEL_TORCH_COMPILE, OCR_AUTOKERNEL_TORCH_COMPILE_DET,
      OCR_AUTOKERNEL_TORCH_COMPILE_REC, OCR_AUTOKERNEL_TORCH_COMPILE_MODE,
      OCR_AUTOKERNEL_TORCH_COMPILE_DYNAMIC, OCR_AUTOKERNEL_WARMUP_IMAGE_PATH,
      OCR_AUTOKERNEL_EXCLUDE_KERNEL_TYPES.
    """
    def env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}

    server_cfg = os.getenv("OCR_SERVER_CONFIG_PATH", settings.ocr_server_config_path)
    exclude = os.getenv("OCR_AUTOKERNEL_EXCLUDE_KERNEL_TYPES")
    return SimpleNamespace(
        ocr_backend=os.getenv("OCR_BACKEND", "auto"),
        ocr_autokernel_enabled=env_bool("OCR_AUTOKERNEL_ENABLED"),
        ocr_server_config_path=server_cfg,
        ocr_mobile_config_path=settings.ocr_mobile_config_path,
        ocr_autokernel_root=os.getenv("AUTOKERNEL_ROOT", "autokernel"),
        ocr_autokernel_ppocr_root=os.getenv("AUTOKERNEL_PPOCR_ROOT", "PaddleOCR2Pytorch"),
        ocr_autokernel_workspace_path=os.getenv("AUTOKERNEL_WORKSPACE", "autokernel/workspace/graph_capture_eval"),
        ocr_autokernel_det_weights_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_DET_PTH", "autokernel/workspace/ppocrv5/server_det.pth"),
        ocr_autokernel_rec_weights_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_REC_PTH", "autokernel/workspace/ppocrv5/server_rec.pth"),
        ocr_autokernel_det_source_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_DET_SOURCE", ""),
        ocr_autokernel_rec_source_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_REC_SOURCE", ""),
        ocr_autokernel_auto_convert_weights=env_bool("OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS", False),
        ocr_autokernel_optimize_recognizer=env_bool("OCR_AUTOKERNEL_OPTIMIZE_RECOGNIZER", True),
        ocr_autokernel_optimize_detector=env_bool("OCR_AUTOKERNEL_OPTIMIZE_DETECTOR", False),
        ocr_autokernel_rec_batch_size=int(os.getenv("OCR_AUTOKERNEL_REC_BATCH_SIZE", "1")),
        ocr_autokernel_rec_image_shape=os.getenv("OCR_AUTOKERNEL_REC_IMAGE_SHAPE", "3,48,320"),
        ocr_autokernel_rec_bucket_max_width_ratio=float(os.getenv("OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO", "1.30")),
        ocr_autokernel_det_limit_side_len=int(os.getenv("OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN", "1280")),
        ocr_autokernel_det_limit_type=os.getenv("OCR_AUTOKERNEL_DET_LIMIT_TYPE", "max"),
        ocr_autokernel_torch_compile=env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ocr_autokernel_torch_compile_det=env_bool(
            "OCR_AUTOKERNEL_TORCH_COMPILE_DET",
            env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ),
        ocr_autokernel_torch_compile_rec=env_bool(
            "OCR_AUTOKERNEL_TORCH_COMPILE_REC",
            env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ),
        ocr_autokernel_torch_compile_mode=os.getenv("OCR_AUTOKERNEL_TORCH_COMPILE_MODE", "default"),
        ocr_autokernel_torch_compile_dynamic=env_bool("OCR_AUTOKERNEL_TORCH_COMPILE_DYNAMIC", True),
        ocr_autokernel_warmup_image_path=os.getenv("OCR_AUTOKERNEL_WARMUP_IMAGE_PATH", ""),
        ocr_autokernel_dtype=os.getenv("OCR_AUTOKERNEL_DTYPE", "float16"),
        ocr_autokernel_det_dtype=os.getenv("OCR_AUTOKERNEL_DET_DTYPE", "float32"),
        ocr_autokernel_exclude_kernel_types=[s.strip() for s in exclude.split(",") if s.strip()] if exclude else None,
    )


def get_ocr():
    """
    Get or create singleton OCR instance.

    Uses create_ocr_backend to select between PaddleOCR / AutoKernel / Hybrid
    backends based on env vars (OCR_BACKEND, OCR_AUTOKERNEL_ENABLED, etc.).

    Raises:
        OCRInitializationError: If OCR engine fails to initialize
    """
    global _ocr_instance, _ocr_initialized

    if _ocr_instance is None:
        try:
            use_gpu = get_use_gpu()
            backend_settings = _build_backend_settings()

            logger.info(
                f"Initializing OCR backend '{backend_settings.ocr_backend}' "
                f"with config: {backend_settings.ocr_server_config_path if use_gpu else backend_settings.ocr_mobile_config_path}"
            )
            logger.info(f"Using GPU: {use_gpu}")

            _ocr_instance = create_ocr_backend(
                backend_settings,
                use_gpu=use_gpu,
                paddle_ocr_cls=None,
            )
            _ocr_initialized = True

            logger.info(f"OCR backend '{_ocr_instance.name}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OCR backend: {str(e)}", exc_info=True)
            raise OCRInitializationError(
                message="OCR initialization failed",
                details=str(e)
            )

    return _ocr_instance


def warmup_ocr() -> None:
    """
    Pre-warm TensorRT engine by running a dummy inference.

    TensorRT compiles optimized GPU kernels on the first inference,
    which can take 10-30s. Running this at startup ensures the first
    real request gets normal latency (~500ms) instead of cold-start latency.
    """
    try:
        logger.info("Warming up OCR engine (TensorRT compilation)...")
        ocr = get_ocr()

        def _do_warmup() -> None:
            with _ocr_lock:
                warmup = getattr(ocr, "warmup", None)
                if callable(warmup):
                    warmup()
                else:
                    dummy_image = np.zeros((607, 1080, 3), dtype=np.uint8)
                    ocr.predict(dummy_image)

        # Run warmup on the dedicated OCR thread so torch.compile's CUDA
        # graph state (stored in TLS) is captured on the same thread that
        # will serve requests.
        _get_ocr_executor().submit(_do_warmup).result()
        logger.info("OCR engine warm-up complete")
    except Exception as e:
        logger.warning(f"OCR warm-up failed (non-fatal): {e}")


def transform_ocr_result(results: dict, width: int) -> List[Tuple]:
    """
    Transform OCR results into desired format and filter by width threshold.

    Args:
        results: Raw OCR results from PaddleOCR
        width: Image width for filtering

    Returns:
        List of tuples containing (coordinates, (text, score))
    """
    try:
        rec_texts = results["rec_texts"]
        rec_scores = results["rec_scores"]
        rec_polys = results["rec_polys"]

        # Combine into desired format
        output = []
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            clean_poly = np.nan_to_num(
                np.asarray(poly, dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).tolist()
            output.append((clean_poly, (text, _finite_float(score))))

        if not output or output[0] is None:
            logger.warning("No OCR results found")
            return []

        # Filter results by width threshold
        result = filter_result(output, width)
        logger.info(f"OCR results: {len(output)} total, {len(result)} after filtering")

        return result

    except Exception as e:
        logger.error(f"Error transforming OCR results: {str(e)}", exc_info=True)
        raise


def filter_result(result: List[Tuple], width: int) -> List[Tuple]:
    """
    Filter OCR results based on width threshold.

    Args:
        result: List of OCR results
        width: Image width

    Returns:
        Filtered list of results
    """
    threshold_ratio = get_provider().get("width_threshold_ratio")
    threshold = width * threshold_ratio

    filtered_line = []
    for line in result:
        coordinate = line[0]
        # Check if all x-coordinates are within threshold
        if all(float(c[0]) <= threshold for c in coordinate):
            filtered_line.append(line)

    logger.debug(
        f"Filtered {len(result) - len(filtered_line)} results based on width threshold"
    )
    return filtered_line


def _process_image_sync(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Synchronous image processing - runs in thread pool.

    Args:
        image_bytes: Image data in bytes

    Returns:
        Tuple of (numpy array, height, width)
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image bytes")
    image_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image_np.shape[:2]
    return image_np, height, width


def _run_ocr_sync(ocr: Any, image_np: np.ndarray) -> list:
    """
    Synchronous OCR prediction - runs in thread pool with lock.
    PaddlePaddle is not thread-safe, so we serialize access.

    Args:
        ocr: PaddleOCR instance
        image_np: Image as numpy array

    Returns:
        OCR prediction results
    """
    with _ocr_lock:
        return ocr.predict(image_np)


def perform_ocr(image_bytes: bytes) -> List[Tuple]:
    """
    Perform OCR on image synchronously (blocking).
    For async usage, use perform_ocr_async instead.

    Args:
        image_bytes: Image data in bytes

    Returns:
        List of tuples containing (coordinates, (text, confidence))
    """
    try:
        logger.debug("Starting OCR processing (sync)")

        ocr = get_ocr()

        # Process image
        try:
            image_np, height, width = _process_image_sync(image_bytes)
            logger.debug(f"Image loaded: shape={image_np.shape}")
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise ImageValidationError(
                message="Invalid image data",
                details=str(e)
            )

        # Perform OCR prediction
        try:
            results = _run_ocr_sync(ocr, image_np)

            if not results or len(results) == 0:
                logger.warning("OCR returned no results")
                return []

            results = results[0]

        except Exception as e:
            logger.error(f"OCR prediction failed: {str(e)}", exc_info=True)
            raise OCRProcessingError(
                message="OCR execution failed",
                details=str(e)
            )

        result = transform_ocr_result(results, width)
        logger.info(f"OCR processing completed: {len(result)} text regions detected")
        return result

    except Exception as e:
        logger.error(f"Error in perform_ocr: {str(e)}", exc_info=True)
        raise


async def perform_ocr_async(image_bytes: bytes) -> List[Tuple]:
    """
    Perform OCR on image asynchronously using thread pool.
    This prevents blocking the event loop during CPU-intensive operations.

    Args:
        image_bytes: Image data in bytes

    Returns:
        List of tuples containing (coordinates, (text, confidence))

    Raises:
        ValueError: If image processing fails
        RuntimeError: If OCR execution fails
    """
    loop = asyncio.get_running_loop()
    executor = _get_executor()

    try:
        logger.debug("Starting async OCR processing")

        # Get OCR instance (fast, can stay on main thread)
        ocr = get_ocr()

        # Run image processing in thread pool (blocking PIL operations)
        try:
            image_np, height, width = await loop.run_in_executor(
                executor, _process_image_sync, image_bytes
            )
            logger.debug(f"Image loaded: shape={image_np.shape}")
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise ImageValidationError(
                message="Invalid image data",
                details=str(e)
            )

        # Run OCR prediction in thread pool (CPU-intensive)
        try:
            logger.debug("Running OCR prediction in thread pool")
            results = await loop.run_in_executor(
                _get_ocr_executor(), partial(_run_ocr_sync, ocr, image_np)
            )

            if not results or len(results) == 0:
                logger.warning("OCR returned no results")
                return []

            results = results[0]

        except Exception as e:
            logger.error(f"OCR prediction failed: {str(e)}", exc_info=True)
            raise OCRProcessingError(
                message="OCR execution failed",
                details=str(e)
            )

        # Transform results (fast operation, can stay on main thread)
        result = transform_ocr_result(results, width)

        logger.info(f"OCR processing completed: {len(result)} text regions detected")
        return result

    except Exception as e:
        logger.error(f"Error in perform_ocr_async: {str(e)}", exc_info=True)
        raise
