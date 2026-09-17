"""
Rotation detection service.
Detects if image is rotated using face detection with InsightFace (RetinaFace).
"""

import threading
import numpy as np
from typing import Optional, Tuple

# insightface ships only source distributions and requires a C++ toolchain to
# build on Windows. Production targets Linux, so we import optionally and fail
# at first-use time on platforms without it.
try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:
    FaceAnalysis = None  # type: ignore

from src.core.config import get_config
from src.core.logging import get_logger
from src.services.threshold_provider import get_provider

logger = get_logger(__name__)

_config = get_config()
# det_size and the verification-box ratios are static tuning (not centrally
# managed): read from config once at import. The detection-confidence threshold
# is NOT here — it is owned by the threshold manager (dgc_mng) and read live from
# the ThresholdProvider via _sync_det_thresh(), so central updates take effect
# without a redeploy.
_det_size = tuple(_config.quality.rotation.det_size)
_box_width_ratio = _config.quality.rotation.verification_box_width_ratio
_box_height_ratio = _config.quality.rotation.verification_box_height_ratio
_box_margin_ratio = _config.quality.rotation.verification_box_margin_ratio
_box_vertical_center = _config.quality.rotation.verification_box_vertical_center

# Threshold key in the management table (bribrain_model_thresholds, dgc_irl).
_DET_THRESH_KEY = "rotation_face_detection_confidence"

# Singleton for InsightFace Face Analysis to prevent repeated model loading
_face_detector = None
_face_detector_lock = threading.Lock()
_face_detector_execution_lock = threading.Lock()


def _current_det_thresh() -> float:
    """Live face-detection confidence threshold from the threshold manager."""
    return float(get_provider().get(_DET_THRESH_KEY))


def _sync_det_thresh(detector) -> None:
    """Keep the detector's det_thresh aligned with the managed threshold.

    InsightFace bakes ``det_thresh`` into the detection model when ``prepare()``
    runs, so a change to the managed value only takes effect by re-preparing.
    We read the live value from the ThresholdProvider and re-prepare only when it
    actually changes; the steady-state path is a single float comparison.

    Must be called while holding ``_face_detector_execution_lock`` so the
    re-prepare cannot race a concurrent ``get()`` on the shared ONNX session.
    """
    desired = _current_det_thresh()
    applied = getattr(detector, "_applied_det_thresh", None)
    if not isinstance(applied, (int, float)) or abs(desired - applied) > 1e-9:
        detector.prepare(ctx_id=-1, det_size=_det_size, det_thresh=desired)
        detector._applied_det_thresh = desired
        logger.info(
            "Rotation face-detection threshold applied: %s -> %.4f",
            f"{applied:.4f}" if isinstance(applied, (int, float)) else "unset",
            desired,
        )


def get_face_detector():
    """
    Get or create singleton InsightFace FaceAnalysis app (detection module only).
    Thread-safe using double-checked locking pattern.

    The detector is prepared with the live detection-confidence threshold read
    from the ThresholdProvider; subsequent changes are applied by
    ``_sync_det_thresh`` before each detection.
    """
    global _face_detector
    if _face_detector is None:
        with _face_detector_lock:
            if _face_detector is None:
                if FaceAnalysis is None:
                    raise RuntimeError(
                        "insightface is not installed in this environment. "
                        "Rotation detection requires insightface (Linux/macOS)."
                    )
                app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"],
                    allowed_modules=["detection"],
                )
                det_thresh = _current_det_thresh()
                app.prepare(
                    ctx_id=-1,
                    det_size=_det_size,
                    det_thresh=det_thresh,
                )
                app._applied_det_thresh = det_thresh
                _face_detector = app
                logger.debug(
                    "InsightFace FaceAnalysis initialized "
                    "(singleton, det_thresh=%.4f)",
                    det_thresh,
                )
    return _face_detector


def is_landscape(image: np.ndarray) -> bool:
    """
    Check if image is in landscape orientation (width > height).

    Args:
        image: Image as numpy ndarray.
              Expected shape: (H, W) or (H, W, C).
              Color format agnostic - only dimensions are checked.

    Returns:
        bool: True if landscape (width > height), False otherwise
    """
    height, width = image.shape[:2]
    return width > height


def is_face_in_box(
    face_x: int,
    face_y: int,
    face_width: int,
    face_height: int,
    box: Tuple[int, int, int, int],
) -> bool:
    """
    Check if face is completely inside the specified box.

    Args:
        face_x: Face x coordinate
        face_y: Face y coordinate
        face_width: Face width
        face_height: Face height
        box: Box defined as (x, y, width, height)

    Returns:
        bool: True if face is completely inside the box, False otherwise
    """
    box_x, box_y, box_width, box_height = box

    if (
        face_x >= box_x
        and face_y >= box_y
        and face_x + face_width <= box_x + box_width
        and face_y + face_height <= box_y + box_height
    ):
        return True
    else:
        return False


def rotated_detection(
    image: Optional[np.ndarray], min_face_proportion: float = 0.2
) -> Tuple[bool, str]:
    """
    Process an image to detect if it's rotated based on face position.

    Args:
        image: Image as numpy ndarray in BGR color format (OpenCV-native).
              Expected shape: (H, W, 3) - 3-channel BGR image.
              Expected dtype: uint8.
              InsightFace expects BGR input (it uses cv2 internally).
        min_face_proportion: Minimum face to box proportion required.
                            Default: 0.2 (face must be at least 20% of box size).

    Returns:
        Tuple of (is_accepted: bool, reason: str)
            - is_accepted: True if face detected in correct position, False otherwise
            - reason: Description of why image was accepted/rejected
    """
    if image is None:
        logger.error("Received None image for rotation detection")
        return False, "Error: Could not read image"

    if not is_landscape(image):
        logger.info("Image rejected: not in landscape orientation")
        return False, "Rejected: Image is not in landscape orientation"

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    try:
        face_detector = get_face_detector()

        logger.debug("Processing image with singleton FaceAnalysis")

        # InsightFace's underlying ONNX session is not guaranteed to be safe
        # under concurrent .get() calls — keep the execution lock. Sync the
        # managed det_thresh under the same lock so a re-prepare cannot race a
        # concurrent .get().
        with _face_detector_execution_lock:
            _sync_det_thresh(face_detector)
            faces = face_detector.get(image)

        height, width = image.shape[:2]

        box_width = int(width * _box_width_ratio)
        box_height = int(height * _box_height_ratio)
        box_x = (
            width - box_width - int(width * _box_margin_ratio)
        )
        box_y = int(height * _box_vertical_center) - (box_height // 2)
        verification_box = (box_x, box_y, box_width, box_height)

        box_area = box_width * box_height

        if not faces:
            logger.info("No face detected in image")
            return False, "No face detected"

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_x = int(x1)
            face_y = int(y1)
            face_w = int(x2 - x1)
            face_h = int(y2 - y1)

            face_area = face_w * face_h
            face_box_proportion = face_area / box_area

            logger.debug(
                f"Face detected - area: {face_area}px, box_area: {box_area}px, "
                f"proportion: {face_box_proportion:.4f}"
            )

            if is_face_in_box(face_x, face_y, face_w, face_h, verification_box):
                if face_box_proportion < min_face_proportion:
                    reason = (
                        f"Face is positioned correctly but too small "
                        f"(proportion: {face_box_proportion:.4f}, min required: {min_face_proportion})"
                    )
                    logger.debug(reason)
                    return False, reason
                else:
                    reason = f"Face is properly positioned with good proportion ({face_box_proportion:.4f})"
                    logger.debug(reason)
                    return True, reason

        logger.info("Face not properly positioned in the verification area")
        return False, "Face not properly positioned in the verification area"

    except Exception as e:
        logger.error(f"Error in rotation detection: {e}", exc_info=True)

        # Reset the singleton on any detector failure — the ONNX session may
        # be in a bad state. The next call will re-initialize cleanly.
        global _face_detector
        with _face_detector_lock:
            _face_detector = None

        # Propagate the failure instead of masking an infrastructure error as a
        # business "rotated" verdict on a 200 response (BUG-08); the caller
        # surfaces it as a 5xx so "could not analyze" != "image is rotated".
        raise


def detect_image_rotation(image: np.ndarray) -> bool:
    """
    Detect if image is rotated.

    Args:
        image: Image as numpy array in BGR color format (OpenCV-native).

    Returns:
        bool: True if image is rotated (rejected), False if properly oriented (accepted)
    """
    try:
        is_accepted, reason = rotated_detection(
            image, min_face_proportion=get_provider().get("rotation_min_face_proportion")
        )

        result_status = "ACCEPTED" if is_accepted else "REJECTED"
        logger.info(f"Rotation detection result: {result_status} - {reason}")

        return not is_accepted

    except Exception as e:
        # Do not mask detector/infra failures as a verdict — let them propagate
        # so the API returns a 5xx rather than a misleading is_rotated result (BUG-08).
        logger.error(f"Error in detect_image_rotation: {e}", exc_info=True)
        raise
