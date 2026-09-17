"""
Unit tests for api.dependencies module
"""
import asyncio
from unittest.mock import patch
import pytest
from fastapi import HTTPException

from src.api.dependencies import verify_api_key


class TestVerifyApiKey:
    """Test verify_api_key dependency"""

    def test_valid_api_key(self):
        """Test that a valid API key passes without error"""
        with patch.dict("os.environ", {"API_KEY": "test-secret-key"}):
            result = asyncio.get_event_loop().run_until_complete(
                verify_api_key(api_key="test-secret-key")
            )
            assert result is None

    def test_invalid_api_key(self):
        """Test that an invalid API key raises 401"""
        with patch.dict("os.environ", {"API_KEY": "test-secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    verify_api_key(api_key="wrong-key")
                )

            assert exc_info.value.status_code == 401
            assert "Invalid or missing API key" in exc_info.value.detail

    def test_empty_api_key(self):
        """Test that an empty API key raises 401"""
        with patch.dict("os.environ", {"API_KEY": "test-secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    verify_api_key(api_key="")
                )

            assert exc_info.value.status_code == 401

    def test_api_key_env_not_set(self):
        """Test that missing API_KEY env var causes all keys to fail"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    verify_api_key(api_key="any-key")
                )

            assert exc_info.value.status_code == 401
