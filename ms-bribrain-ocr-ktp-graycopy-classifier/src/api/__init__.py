"""
API module for OCR Graycopy service.
Contains FastAPI routes and endpoints.
"""

from src.api.routes import router, set_detection_service, shutdown_executor

__all__ = [
    "router",
    "set_detection_service",
    "shutdown_executor",
]
