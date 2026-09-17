"""
YOLO Detection Service - Entry Point

This is the main entry point for the YOLO Detection service.
Run with: uvicorn main:app --reload
"""

from src.main import app

__all__ = ["app"]
