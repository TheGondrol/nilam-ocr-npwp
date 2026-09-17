"""
Graycopy Detection Service
Main service for detecting photocopied/graycopy ID documents
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.core.config import Config
from src.core.device import get_device
from src.core.logging import get_logger
from src.services.model_architecture import create_resnet50_graycopy_model

logger = get_logger()


class GraycopyDetectionService:
    """Service for graycopy/photocopy detection"""

    def __init__(self, config: Config):
        """
        Initialize graycopy detection service.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device  # Always set in _initialize_device()
        self._transform_cropped: Optional[transforms.Compose] = None
        self._transform_full: Optional[transforms.Compose] = None
        self._is_jit_model: bool = False  # Track if model is JIT-traced

        self._initialize_device()
        self._setup_transforms()
        self._load_model()

    def _initialize_device(self) -> None:
        """Initialize computing device (GPU or CPU)"""
        self.device = get_device(
            prefer_gpu=self.config.device.prefer_gpu,
            force_cpu=self.config.device.force_cpu,
        )
        if self.device.type == "cuda":
            logger.info(
                f"GPU Compilation: {'enabled' if self.config.device.compile_model else 'disabled'}, "
                f"mode: {self.config.device.compile_mode}"
            )
        else:
            logger.info(
                f"CPU JIT: {'enabled' if self.config.device.use_jit_cpu else 'disabled'}"
            )

    def _setup_transforms(self) -> None:
        """Pre-build transforms for efficiency"""
        image_size = self.config.model.image_size
        norm_mean = self.config.model.normalize_mean
        norm_std = self.config.model.normalize_std

        # Transform for already-cropped images (224x224)
        self._transform_cropped = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        # Transform for full images that need cropping
        self._transform_full = transforms.Compose(
            [
                transforms.Resize(self.config.model.normalize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

    def _load_model(self) -> None:
        """Load the trained ResNet-50 graycopy detection model"""
        try:
            # Initialize model
            self.model = create_resnet50_graycopy_model(
                num_classes=2, dropout_rate=self.config.model.dropout_rate
            )
            self.model = self.model.to(self.device)

            # Load weights
            model_path = Path(self.config.model.path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint.get("epoch", "unknown")
                logger.info(f"Loaded model from epoch {epoch}")
            else:
                self.model.load_state_dict(checkpoint)

            # Ensure model is on the correct device after loading weights
            self.model = self.model.to(self.device)
            self.model.eval()

            # Apply torch.compile() optimization for CUDA devices
            if self.device.type == "cuda" and self.config.device.compile_model:
                if not self.config.device.is_valid_compile_mode:
                    logger.warning(
                        f"Invalid compile mode '{self.config.device.compile_mode}', "
                        f"using 'default' instead"
                    )
                    compile_mode = "default"
                else:
                    compile_mode = self.config.device.compile_mode

                logger.info(f"Compiling model with mode='{compile_mode}'...")
                self.model = torch.compile(self.model, mode=compile_mode)  # type: ignore[assignment]
                logger.info(
                    "Model compilation enabled (first inference will trigger compilation)"
                )

            # Apply JIT (TorchScript) optimization for CPU devices
            elif self.device.type == "cpu" and self.config.device.use_jit_cpu:
                logger.info("Applying JIT optimization for CPU...")
                try:
                    # Use torch.jit.trace for better performance with fixed input shapes
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    self.model = torch.jit.trace(self.model, dummy_input)
                    self.model = torch.jit.optimize_for_inference(self.model)
                    self._is_jit_model = True
                    logger.info("JIT optimization applied successfully")
                except Exception as jit_error:
                    logger.warning(
                        f"JIT optimization failed, using original model: {str(jit_error)}"
                    )

            logger.info("Model loaded successfully")
            logger.info("Smart preprocessing enabled: auto-detect if cropping needed")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def warmup(self) -> None:
        """
        Perform warmup inference to pre-load model into GPU memory/cache.
        For compiled models, runs 3 iterations to ensure full compilation.
        This reduces latency for the first real request.
        """
        if self.model is None or self.device is None:
            logger.warning("Cannot warmup: model not loaded")
            return

        try:
            # Clear CUDA cache and synchronize before warmup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            # Determine number of warmup iterations
            # Compiled models need multiple runs for full compilation
            is_compiled = (
                self.device.type == "cuda" and self.config.device.compile_model
            )
            warmup_iterations = 3 if is_compiled else 1

            logger.info(
                f"Running {warmup_iterations} warmup iteration(s) "
                f"({'with compilation' if is_compiled else 'without compilation'})"
            )

            # Run warmup inference
            with torch.inference_mode():
                for i in range(warmup_iterations):
                    _ = self.model(dummy_input)
                    if is_compiled:
                        logger.debug(
                            f"Warmup iteration {i+1}/{warmup_iterations} completed"
                        )

            # Synchronize and clear cache after warmup
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info("Model warmup completed successfully")

        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {str(e)}")

    def cleanup(self) -> None:
        """
        Cleanup resources on shutdown.
        Releases GPU memory and model references.
        """
        if self.model is not None:
            del self.model
            self.model = None

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Detection service resources cleaned up")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image with smart cropping detection.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image tensor
        """
        original_size = image.size
        image_size = self.config.model.image_size

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            logger.debug(f"Converted image from {image.mode} to RGB")

        # Smart preprocessing based on image size
        if original_size == (image_size, image_size):
            # Image is already cropped to desired size
            logger.debug(f"Image already at {image_size}x{image_size} - skipping crop")
            transform = self._transform_cropped
        else:
            # Image needs cropping
            logger.debug(f"Image size {original_size} - applying crop preprocessing")
            transform = self._transform_full

        assert transform is not None, "Transform should be initialized"
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def predict(
        self,
        image: Image.Image,
        filename: str = "unknown",
        threshold: float = 0.5,
    ) -> dict:
        """
        Predict whether an image is graycopy or original.

        Args:
            image: PIL Image
            filename: Original filename for logging
            threshold: Minimum P(graycopy) required to label as GRAYCOPY.
                Defaults to 0.5 (equivalent to argmax on a 2-class softmax).

        Returns:
            Dictionary with prediction results:
                - prediction: "GRAYCOPY" or "ORIGINAL"
                - confidence: Probability of the predicted class (0-1)
                - probability_graycopy: Probability of graycopy class
                - probability_original: Probability of original class

        Raises:
            RuntimeError: If model not loaded
            Exception: If prediction fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            logger.info(
                f"Processing image: {filename}, size: {image.size}, threshold: {threshold:.4f}"
            )

            # Preprocess image
            image_tensor = self._preprocess_image(image)

            # Make prediction with inference mode and error handling
            # Class mapping (sorted alphabetically from directory names):
            # Class 0 = "graycopy", Class 1 = "original"
            try:
                with torch.inference_mode():
                    outputs = self.model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    prob_graycopy = probabilities[0][0].item()
                    prob_original = probabilities[0][1].item()

            except RuntimeError as e:
                # A GPU hiccup (cuDNN/CUDA error) must NOT trigger an in-place
                # device move of the *shared* self.model: predict() runs under a
                # ThreadPoolExecutor, so moving the shared model to CPU and back
                # races concurrent workers (a sibling GPU inference can see the
                # weights momentarily on CPU -> device-mismatch / corrupt output).
                # Fail this single request cleanly instead (BUG-04). For a
                # permanent CPU deployment, set device.force_cpu=true.
                if self.device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                logger.error(f"GPU inference failed, failing this request: {e}")
                raise

            # Guard against non-finite model output: NaN/inf slips past the
            # threshold compare (every NaN comparison is False) and then explodes
            # the response model (Field ge/le) as a ValidationError -> 500 (BUG-28).
            if not (math.isfinite(prob_graycopy) and math.isfinite(prob_original)):
                raise RuntimeError(
                    f"Non-finite model output: graycopy={prob_graycopy}, original={prob_original}"
                )

            # Apply threshold to gate the GRAYCOPY decision
            if prob_graycopy >= threshold:
                predicted_label = "GRAYCOPY"
                conf_value = prob_graycopy
            else:
                predicted_label = "ORIGINAL"
                conf_value = prob_original

            # Cleanup tensor
            del image_tensor

            logger.info(
                f"Prediction: {predicted_label}, "
                f"Confidence: {conf_value:.4f}, "
                f"P(graycopy): {prob_graycopy:.4f}, "
                f"P(original): {prob_original:.4f}, "
                f"Threshold: {threshold:.4f}"
            )

            return {
                "prediction": predicted_label,
                "confidence": conf_value,
                "probability_graycopy": prob_graycopy,
                "probability_original": prob_original,
            }

        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}", exc_info=True)
            raise

    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.model is not None and self.device is not None

    def get_device_info(self) -> str:
        """Get device information"""
        return str(self.device) if self.device else "unknown"
