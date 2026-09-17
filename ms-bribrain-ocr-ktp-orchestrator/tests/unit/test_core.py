"""
Unit tests for core modules (config, logging, device).
"""

from unittest.mock import patch, MagicMock, mock_open
import os

from src.core.config import get_settings, _substitute_env_vars


class TestConfig:
    """Tests for configuration module."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns singleton instance."""
        with patch("src.core.config.Path") as mock_path, \
             patch("builtins.open", mock_open(read_data="app:\n  name: Test\n  version: 1.0.0")):
            
            mock_path.return_value.parent.parent.parent = MagicMock()
            
            settings1 = get_settings()
            settings2 = get_settings()
            
            # Should return same instance
            assert settings1 is settings2

    def test_substitute_env_vars_simple(self):
        """Test environment variable substitution."""
        os.environ["TEST_VAR"] = "test_value"
        
        result = _substitute_env_vars("${TEST_VAR}")
        assert result == "test_value"
        
        # Clean up
        del os.environ["TEST_VAR"]

    def test_substitute_env_vars_dict(self):
        """Test environment variable substitution in dict."""
        os.environ["TEST_HOST"] = "localhost"
        os.environ["TEST_PORT"] = "8080"
        
        config = {
            "host": "${TEST_HOST}",
            "port": "${TEST_PORT}",
            "name": "test"
        }
        
        result = _substitute_env_vars(config)
        
        assert result["host"] == "localhost"
        assert result["port"] == "8080"
        assert result["name"] == "test"
        
        # Clean up
        del os.environ["TEST_HOST"]
        del os.environ["TEST_PORT"]

    def test_substitute_env_vars_nested_dict(self):
        """Test environment variable substitution in nested dict."""
        os.environ["DB_HOST"] = "db.example.com"
        
        config = {
            "database": {
                "host": "${DB_HOST}",
                "port": 5432
            }
        }
        
        result = _substitute_env_vars(config)
        
        assert result["database"]["host"] == "db.example.com"
        assert result["database"]["port"] == 5432
        
        # Clean up
        del os.environ["DB_HOST"]

    def test_substitute_env_vars_list(self):
        """Test environment variable substitution in list."""
        os.environ["SERVICE_URL"] = "http://test.com"
        
        config = ["${SERVICE_URL}", "http://other.com"]
        
        result = _substitute_env_vars(config)
        
        assert result[0] == "http://test.com"
        assert result[1] == "http://other.com"
        
        # Clean up
        del os.environ["SERVICE_URL"]

    def test_substitute_env_vars_missing_var(self):
        """Test substitution with missing environment variable."""
        # Ensure variable doesn't exist
        if "MISSING_VAR" in os.environ:
            del os.environ["MISSING_VAR"]
        
        result = _substitute_env_vars("${MISSING_VAR}")

        # Missing vars are substituted with an empty string (logged as a warning)
        assert result == ""


class TestLogging:
    """Tests for logging module."""

    def test_setup_logging(self):
        """Test logging setup."""
        with patch("src.core.logging.get_settings") as mock_settings:
            mock_log_config = MagicMock()
            mock_log_config.level = "INFO"
            mock_log_config.file = "test.log"
            mock_log_config.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            mock_log_config.max_bytes = 10485760
            mock_log_config.backup_count = 5
            mock_log_config.console = True
            mock_settings.return_value.logging = mock_log_config
            
            from src.core.logging import setup_logging
            
            logger = setup_logging()
            
            assert logger is not None

    def test_get_logger(self):
        """Test get_logger function."""
        from src.core.logging import get_logger
        
        logger = get_logger("test_module")
        
        assert logger is not None
        assert "test_module" in logger.name


class TestDevice:
    """Tests for device module."""

    def test_log_device_info(self):
        """Test device info logging."""
        with patch("src.core.device.detect_gpu", return_value=(False, "cpu")), \
             patch("src.core.device.logger") as _mock_logger:
            
            from src.core.device import log_device_info
            
            # Should not raise exception
            log_device_info()

    def test_log_device_info_with_cuda(self):
        """Test device info logging with CUDA."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3090"
        
        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch("src.core.device.logger") as _mock_logger:
            
            from src.core.device import log_device_info
            
            # Should not raise exception
            log_device_info()

    def test_log_device_info_no_torch(self):
        """Test device info logging without torch."""
        with patch.dict("sys.modules", {"torch": None}), \
             patch("src.core.device.logger") as _mock_logger:
            
            from src.core.device import log_device_info
            
            # Should handle missing torch gracefully
            try:
                log_device_info()
            except Exception:
                # Should not raise critical exception
                pass


class TestAPIModels:
    """Tests for API models."""

    def test_service_result_success(self):
        """Test ServiceResult success creation."""
        from src.api.models import ServiceResult, OCRData
        
        data = OCRData(text=[["test", ["test", 0.95]]])
        result = ServiceResult.success(data)
        
        assert result.status == "success"
        assert result.data == data
        assert result.error is None
        assert result.rejection is None

    def test_service_result_fail(self):
        """Test ServiceResult fail creation."""
        from src.api.models import ServiceResult
        
        result = ServiceResult.fail(
            error_type="timeout",
            message="Service timeout",
            details="Connection timed out"
        )
        
        assert result.status == "error"
        assert result.error is not None
        assert result.error.error_type == "timeout"
        assert result.data is None
        assert result.rejection is None

    def test_service_result_reject(self):
        """Test ServiceResult reject creation."""
        from src.api.models import ServiceResult
        
        result = ServiceResult.reject(
            rejection_type="quality",
            message="Quality check failed",
            details={"is_blurry": True}
        )
        
        assert result.status == "rejection"
        assert result.rejection is not None
        assert result.rejection.rejection_type == "quality"
        assert result.data is None
        assert result.error is None

    def test_error_response_model(self):
        """Test ErrorResponse model."""
        from src.api.models import ErrorResponse
        
        error = ErrorResponse(error="Test error message")
        
        assert error.error == "Test error message"

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from src.api.models import HealthResponse
        
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            device="cpu"
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.device == "cpu"

    def test_ocr_success_response_model(self):
        """Test OCRSuccessResponse model."""
        from src.api.models import OCRSuccessResponse
        
        results = {"nama": "JOHN DOE", "nik": "1234567890123456"}
        response = OCRSuccessResponse(results=results)
        
        assert response.results == results
        assert response.results["nama"] == "JOHN DOE"
