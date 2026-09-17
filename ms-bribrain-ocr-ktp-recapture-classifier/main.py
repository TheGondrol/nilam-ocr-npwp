"""
OCR KTP Orchestrator - Entry Point

This is the main entry point for the OCR KTP Orchestrator service.
Run with: uvicorn main:app --reload
"""

from src.main import app

__all__ = ["app"]