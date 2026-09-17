"""
Tamper detection service
Main service for document tamper detection using fine-tuned ViT model
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.services.image_preprocessing import load_image_from_bytes, preprocess_image
from src.services.model_loader import load_model

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound inference operations
_inference_executor: Optional[ThreadPoolExecutor] = None


def get_inference_executor() -> ThreadPoolExecutor:
    """Get or create the inference thread pool executor."""
    global _inference_executor
    if _inference_executor is None:
        # Use half of available cores for inference threads to avoid oversubscription
        max_workers = max(2, (os.cpu_count() or 4) // 2)
        _inference_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="inference"
        )
        logger.info(f"Created inference thread pool with {max_workers} workers")
    return _inference_executor


class TamperDetectionService:
    """Service for tamper detection operations"""

    def __init__(self, model_path: str, image_size: int = 320, force_cpu: bool = False):
        """
        Initialize tamper detection service

        Args:
            model_path: Path to the model directory
            image_size: Image size for preprocessing (default: 320)
            force_cpu: If True, force CPU usage even if GPU is available
        """
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.image_size = image_size
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device: Optional[str] = None

    def initialize(self) -> None:
        """Load the model and processor"""
        try:
            self.model, self.processor, self.device = load_model(
                self.model_path, force_cpu=self.force_cpu
            )
            # Initialize the thread pool executor
            get_inference_executor()
            logger.info("Tamper detection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tamper detection service: {str(e)}")
            raise

    def is_ready(self) -> bool:
        """Check if the service is ready"""
        return self.model is not None and self.processor is not None

    def _sync_predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Synchronous prediction for running in thread pool.

        Args:
            image: PIL Image

        Returns:
            dict: Prediction results
        """
        # Preprocess image
        pixel_values = preprocess_image(image, self.processor, self.image_size)
        pixel_values = pixel_values.to(self.device)

        # Run inference
        with torch.inference_mode():
            if self.model is None:
                raise RuntimeError("Model not initialized")
            outputs = self.model(pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(logits, dim=1).item()

        # Ensure model has config
        if not hasattr(self.model, 'config'):
            raise RuntimeError("Model missing config attribute")
        
        # Format results
        pred_class_idx: int = int(pred_class)
        result = {
            "predicted_class": self.model.config.id2label[pred_class_idx],
            "predicted_id": pred_class_idx,
            "confidence": probs[pred_class_idx].item(),
            "probabilities": {
                self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probs)
            },
        }

        return result

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether a document image is tampered or authentic.
        Runs inference in a thread pool to avoid blocking the event loop.

        Args:
            image: PIL Image

        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            # Run CPU-bound inference in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(get_inference_executor(), self._sync_predict, image)

            logger.info(
                f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.4f})"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    async def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict from image bytes

        Args:
            image_bytes: Image bytes

        Returns:
            dict: Prediction results
        """
        image = await load_image_from_bytes(image_bytes)
        return await self.predict(image)

    async def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Predict on multiple images concurrently.

        Args:
            images: List of PIL Images

        Returns:
            list: Prediction results for each image
        """

        async def safe_predict(idx: int, img: Image.Image) -> Dict[str, Any]:
            try:
                return await self.predict(img)
            except Exception as e:
                logger.error(f"Error processing image {idx}: {str(e)}")
                return {
                    "predicted_class": "ERROR",
                    "predicted_id": -1,
                    "confidence": 0.0,
                    "probabilities": {},
                    "error": str(e),
                }

        # Process all images concurrently using asyncio.gather
        tasks = [safe_predict(idx, img) for idx, img in enumerate(images)]
        results = await asyncio.gather(*tasks)

        return list(results)


# Global service instance
tamper_service: Optional[TamperDetectionService] = None


def get_tamper_service() -> TamperDetectionService:
    """Get the global tamper detection service instance"""
    if tamper_service is None:
        raise RuntimeError("Tamper detection service not initialized")
    return tamper_service


def initialize_tamper_service(
    model_path: str, image_size: int = 320, force_cpu: bool = False
) -> TamperDetectionService:
    """
    Initialize the global tamper detection service

    Args:
        model_path: Path to the model directory
        image_size: Image size for preprocessing
        force_cpu: If True, force CPU usage even if GPU is available

    Returns:
        TamperDetectionService instance
    """
    global tamper_service
    tamper_service = TamperDetectionService(model_path, image_size, force_cpu)
    tamper_service.initialize()
    return tamper_service
