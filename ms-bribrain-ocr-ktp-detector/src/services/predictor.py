"""
Prediction Service
Handles YOLO model loading and prediction logic.
Supports both PyTorch and OpenVINO model formats for optimized inference.
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path

from ..core.logging import logger
from ..core.config import config
from ..core.device import get_device, log_device_info
from .threshold_provider import get_provider


class PredictorService:
    """Service for loading YOLO model and making predictions"""

    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.device: Optional[str] = None
        self.device_info: Optional[dict] = None
        self.class_names: Dict[int, str] = config.class_names
        self.export_format: str = config.get("model.export_format", "pt")

    @property
    def confidence(self) -> float:
        return get_provider().get("confidence")

    @property
    def iou_threshold(self) -> float:
        return get_provider().get("iou_threshold")

    def load_model(self) -> None:
        """Load the YOLO model (supports .pt and OpenVINO formats)"""
        try:
            # Get device
            self.device, self.device_info = get_device()
            log_device_info(self.device, self.device_info)

            # Determine model path based on export format
            model_path = config.model_path
            export_format = config.get("model.export_format", "pt")

            # Check for OpenVINO model
            if export_format == "openvino":
                # First try the explicit openvino_path from config
                openvino_path_config = config.get("model.openvino_path", None)
                if openvino_path_config and Path(openvino_path_config).exists():
                    model_path = openvino_path_config
                    logger.info(f"Using OpenVINO model from config: {model_path}")
                else:
                    # Fallback to derived path from base model
                    openvino_path = self._get_openvino_path(model_path)
                    if openvino_path and openvino_path.exists():
                        model_path = str(openvino_path)
                        logger.info(f"Using OpenVINO model (derived): {model_path}")
                    else:
                        logger.warning(
                            f"OpenVINO model not found, falling back to: {model_path}"
                        )

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully from {model_path}")

            # Move model to device (only for PyTorch models)
            if self.device == "cuda" and model_path.endswith(".pt"):
                self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")

            # Log class names
            logger.info(f"Configured class names: {self.class_names}")
            logger.info(f"Confidence threshold: {self.confidence}")
            logger.info(f"IOU threshold: {self.iou_threshold}")
            logger.info(f"Export format: {export_format}")

            logger.info("Model initialization complete")
            
            # Warmup the model to trigger OpenVINO compilation before first request
            # This prevents lazy compilation during inference which can cause issues
            if export_format == "openvino":
                self._warmup_model()

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
    
    def _warmup_model(self) -> None:
        """
        Warmup the model by running a dummy inference.
        This triggers OpenVINO's lazy compilation during startup instead of during
        the first request, which can interfere with uvicorn's signal handlers.
        
        We save and restore signal handlers because OpenVINO can override them
        during model compilation, which breaks uvicorn's process management.
        """
        import signal
        
        # Save current signal handlers before OpenVINO touches them
        saved_handlers = {}
        signals_to_protect = [signal.SIGINT, signal.SIGTERM]
        
        # On Windows, SIGTERM might not be available
        if hasattr(signal, 'SIGTERM'):
            try:
                for sig in signals_to_protect:
                    saved_handlers[sig] = signal.getsignal(sig)
            except (ValueError, OSError) as e:
                logger.debug(f"Could not save signal handlers: {e}")
        
        try:
            logger.info("Warming up OpenVINO model...")
            # Create a small dummy image (640x640 is typical YOLO input size)
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Run inference to trigger compilation
            if self.model is not None:
                _ = self.model(
                    dummy_image,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )
            logger.info("OpenVINO model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {str(e)}")
        finally:
            # Restore original signal handlers
            for sig, handler in saved_handlers.items():
                try:
                    signal.signal(sig, handler)
                    logger.debug(f"Restored signal handler for {sig}")
                except (ValueError, OSError) as e:
                    logger.debug(f"Could not restore signal handler for {sig}: {e}")

    def _get_openvino_path(self, base_path: str) -> Optional[Path]:
        """
        Get the OpenVINO model path from base .pt path.

        Args:
            base_path: Path to the original .pt model

        Returns:
            Path to OpenVINO model directory, or None if not found
        """
        base = Path(base_path)

        # Check for _openvino_model directory (Ultralytics naming convention)
        openvino_dir = base.parent / f"{base.stem}_openvino_model"
        if openvino_dir.exists() and openvino_dir.is_dir():
            return openvino_dir

        # Also check without underscore
        openvino_dir_alt = base.parent / f"{base.stem}_openvino"
        if openvino_dir_alt.exists() and openvino_dir_alt.is_dir():
            return openvino_dir_alt

        return None

    def predict(self, image: Image.Image, filename: str) -> List[Dict[str, Any]]:
        """
        Make prediction on a single image

        Args:
            image: PIL Image
            filename: Original filename for logging

        Returns:
            List of detection dictionaries
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            logger.info(f"Processing image: {filename}, size: {image.size}")

            # Convert PIL Image to numpy array for YOLO.
            # Ultralytics treats numpy inputs as BGR (OpenCV convention), so flip
            # from PIL's RGB to BGR to match how the model was trained/exported.
            image_np = np.array(image)[:, :, ::-1]

            # Run inference
            # For OpenVINO models, additional options can improve throughput
            inference_kwargs: Dict[str, Any] = {
                "conf": self.confidence,
                "iou": self.iou_threshold,
                "device": self.device if self.device else "cpu",
                "verbose": False,
            }
            
            results = self.model(image_np, **inference_kwargs)

            # Parse results
            detections = []

            for result in results:
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get box coordinates (use .tolist() for efficient scalar extraction)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Get class and confidence (use .item() for scalar values - more efficient)
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        # Get class name from config or model
                        if class_id in self.class_names:
                            class_name = self.class_names[class_id]
                        elif hasattr(result, "names") and class_id in result.names:
                            class_name = result.names[class_id]
                        else:
                            if class_id == 0:
                                class_name = "ktp"
                            elif class_id == 1:
                                class_name = "non-ktp"
                            else:
                                class_name = f"class_{class_id}"

                        detection = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                        }
                        detections.append(detection)

            logger.info(f"Prediction complete: {len(detections)} detections found")

            return detections

        except Exception as e:
            logger.error(f"Prediction error for {filename}: {str(e)}", exc_info=True)
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_device_string(self) -> str:
        """Get device string"""
        return str(self.device) if self.device else "unknown"

    def get_device_info(self) -> dict:
        """Get device information"""
        return self.device_info if self.device_info else {}


# Global predictor instance
predictor = PredictorService()
