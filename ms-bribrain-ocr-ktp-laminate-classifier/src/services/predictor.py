"""
Prediction Service
Handles model loading and prediction logic
"""

import gc
import torch
from torchvision import transforms
from PIL import Image
from typing import Any, Tuple, Optional
from pathlib import Path
import numpy as np

from ..core.logging import logger
from ..core.config import config
from ..core.device import get_device, log_device_info
from ..models.architecture import UnlaminatedCopyDetector
from .threshold_provider import get_provider


class PredictorService:
    """Service for loading model and making predictions"""

    def __init__(self):
        self.model: Optional[Any] = None
        self.device: Optional[str] = None
        self.device_info: Optional[dict] = None
        self.transform: Optional[Any] = None
        self.image_size: int = config.image_size

    @property
    def threshold(self) -> float:
        return get_provider().get("threshold")

    def load_model(self):
        """Load the trained model"""
        try:
            # Get device
            self.device, self.device_info = get_device()
            log_device_info(self.device, self.device_info)

            # Initialize model
            self.model = UnlaminatedCopyDetector().to(self.device)
            logger.info(f"Model initialized on {self.device}")

            # Load weights
            model_path = config.model_path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Load checkpoint. Prefer weights_only=True (no arbitrary-object
            # unpickling) and allowlist numpy's data-only array globals so a
            # trusted checkpoint that embeds numpy arrays still loads; if it holds
            # other non-tensor objects weights_only can't handle, fall back to a
            # full load of the (bucket-sourced, trusted) artifact (BUG-06).
            try:
                import numpy as _np
                import torch.serialization as _ts
                _allow = [_np.ndarray, _np.dtype]
                for _mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
                    try:
                        _ma = __import__(_mod_name, fromlist=["_reconstruct", "scalar"])
                        _allow += [g for g in (getattr(_ma, "_reconstruct", None),
                                               getattr(_ma, "scalar", None)) if g is not None]
                        break
                    except Exception:
                        continue
                _ts.add_safe_globals(_allow)
            except Exception:
                pass
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception as _wo_err:
                logger.warning(
                    "weights_only load rejected the checkpoint (%s); falling back to "
                    "a full load of the trusted artifact — keep the model bucket "
                    "access-controlled and downloads over TLS", _wo_err)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            
            # Apply device-specific model optimization
            if config.torch_compile:
                try:
                    if self.device == "cuda":
                        # Use torch.compile for GPU - better runtime optimization
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile (reduce-overhead mode) for GPU")
                    else:
                        # Use JIT tracing for CPU - better static graph optimization
                        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
                        self.model = torch.jit.trace(self.model, dummy_input)
                        self.model = torch.jit.optimize_for_inference(self.model)
                        logger.info("Model optimized with torch.jit.trace for CPU inference")
                except Exception as compile_error:
                    logger.warning(f"Model optimization failed, using eager mode: {compile_error}")

            # Initialize transform — RGB + ImageNet normalization to match
            # the MobileNetV3-small backbone trained with pretrained=True.
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            logger.info("Model initialization complete")
            
            # Warmup the model to trigger compilation before first request
            self.warmup()

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

    def warmup(self):
        """
        Warmup the model by running a dummy inference.
        This triggers any lazy compilation during startup instead of during
        the first request, improving response time consistency.
        """
        try:
            if self.transform is None or self.model is None:
                logger.warning("Model warmup skipped: transform or model not initialized")
                return
            logger.info("Warming up model...")
            # Create a dummy image
            dummy_image = Image.fromarray(
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )

            # Run inference
            tensor = self.transform(dummy_image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                _ = self.model(tensor)
            
            # Cleanup
            del tensor
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {str(e)}")

    def predict(self, image: Image.Image, filename: str) -> Tuple[str, float]:
        """
        Make prediction on a single image

        Args:
            image: PIL Image
            filename: Original filename for logging

        Returns:
            Tuple of (prediction, probability)
        """
        tensor = None
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            logger.info(f"Processing image: {filename}, size: {image.size}")

            # Transform and prepare tensor
            if self.transform is None:
                raise RuntimeError("Transform not initialized")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor_sum = tensor.sum().item()
            logger.info(f"Input tensor sum: {tensor_sum:.6f}")

            # Make prediction. Model emits 2-class logits; class 1 = UNLAMINATED.
            with torch.inference_mode():
                logits = self.model(tensor)
                prob_spoof = torch.softmax(logits, dim=1)[0, 1].item()

            # Determine prediction
            is_unlaminated = prob_spoof > self.threshold
            prediction = "UNLAMINATED" if is_unlaminated else "LAMINATED"

            logger.info(
                f"Prediction: {prediction}, " f"Prob(unlaminated): {prob_spoof:.4f}"
            )

            return prediction, prob_spoof

        except Exception as e:
            logger.error(f"Prediction error for {filename}: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup intermediate tensors to free memory
            if tensor is not None:
                del tensor
            if self.device == "cuda":
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                torch.cuda.empty_cache()

    def cleanup(self):
        """Cleanup resources on shutdown"""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model resources cleaned up")

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

