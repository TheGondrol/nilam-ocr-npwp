"""Shared fixtures for unit tests."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset Config singleton between tests to avoid state leaks."""
    from src.core.config import Config
    Config._instance = None
    yield
    Config._instance = None
