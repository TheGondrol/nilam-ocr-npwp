"""
Unit tests for api.dependencies module
"""
from unittest.mock import patch
import pytest
from fastapi import HTTPException

from src.api.dependencies import verify_api_key, validate_request_id


class TestVerifyApiKey:
    """Test verify_api_key dependency"""

    @pytest.mark.asyncio
    async def test_valid_api_key(self):
        """Test that a valid API key passes without error"""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "test-secret-key"}):
            result = await verify_api_key(api_key="test-secret-key")
            assert result is None

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test that an invalid API key raises 401"""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "test-secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="wrong-key")

            assert exc_info.value.status_code == 401
            detail = exc_info.value.detail
            assert isinstance(detail, dict)
            assert detail["message"] == "Invalid or missing API key"

    @pytest.mark.asyncio
    async def test_empty_api_key(self):
        """Test that an empty API key raises 401"""
        with patch.dict("os.environ", {"ORCHESTRATOR_SERVICE_API": "test-secret-key"}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_env_not_set(self):
        """Test that missing ORCHESTRATOR_SERVICE_API env var causes all keys to fail"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="any-key")

            assert exc_info.value.status_code == 401


class TestValidateRequestId:
    """Test validate_request_id dependency"""

    def test_valid_request_id(self):
        """Test that a valid OCR_{uuid4} request_id passes."""
        result = validate_request_id("OCR_12345678-1234-1234-1234-123456789abc")
        assert result == "OCR_12345678-1234-1234-1234-123456789abc"

    def test_invalid_prefix(self):
        """Test that a request_id without OCR_ prefix raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_request_id("INVALID_12345678-1234-1234-1234-123456789abc")
        assert exc_info.value.status_code == 400

    def test_invalid_uuid_format(self):
        """Test that a malformed UUID part raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_request_id("OCR_not-a-uuid")
        assert exc_info.value.status_code == 400

    def test_empty_string(self):
        """Test that an empty string raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_request_id("")
        assert exc_info.value.status_code == 400

    def test_prefix_only(self):
        """Test that OCR_ alone raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            validate_request_id("OCR_")
        assert exc_info.value.status_code == 400

    def test_uppercase_uuid_rejected(self):
        """Test that uppercase UUID hex chars are rejected (uuid4 is lowercase)."""
        with pytest.raises(HTTPException) as exc_info:
            validate_request_id("OCR_12345678-1234-1234-1234-123456789ABC")
        assert exc_info.value.status_code == 400
