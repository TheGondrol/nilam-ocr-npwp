"""
Unit tests for src/core/validation module.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestValidateEnvironmentVariables:

    def test_all_env_vars_present(self):
        """All required env vars set returns valid."""
        from src.core.validation import validate_environment_variables

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "OCR_SERVICE_API": "k",
            "LAMINATION_SERVICE_API": "k",
            "RECAPTURE_SERVICE_API": "k",
            "GRAYCOPY_SERVICE_API": "k",
            "TEMPER_SERVICE_API": "k",
            "CLASSIFIER_SERVICE_API": "k",
            "QUALITY_SERVICE_API": "k",
            "QUALITYDL_SERVICE_API": "k",
            "POSTPROCESS_SERVICE_API": "k",
            "ORCHESTRATOR_SERVICE_API": "k",
        }
        with patch.dict("os.environ", env, clear=False):
            is_valid, missing = validate_environment_variables()
            assert is_valid is True
            assert missing == []

    def test_missing_env_vars(self):
        """Missing env vars are reported."""
        from src.core.validation import validate_environment_variables

        with patch.dict("os.environ", {}, clear=True):
            is_valid, missing = validate_environment_variables()
            assert is_valid is False
            assert len(missing) > 0
            assert any("DATABASE_URL" in m for m in missing)

    def test_empty_env_var_treated_as_missing(self):
        """Empty string env vars treated as missing."""
        from src.core.validation import validate_environment_variables

        env = {
            "MINIO_ENDPOINT": "",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "url",
            "OCR_SERVICE_API": "k",
            "LAMINATION_SERVICE_API": "k",
            "RECAPTURE_SERVICE_API": "k",
            "GRAYCOPY_SERVICE_API": "k",
            "TEMPER_SERVICE_API": "k",
            "CLASSIFIER_SERVICE_API": "k",
            "QUALITY_SERVICE_API": "k",
            "QUALITYDL_SERVICE_API": "k",
            "POSTPROCESS_SERVICE_API": "k",
            "ORCHESTRATOR_SERVICE_API": "k",
        }
        with patch.dict("os.environ", env, clear=True):
            is_valid, missing = validate_environment_variables()
            assert is_valid is False
            assert any("MINIO_ENDPOINT" in m for m in missing)


class TestValidateConfigValues:

    def test_valid_config(self, mock_settings):
        """Valid config returns no issues."""
        from src.core.validation import validate_config_values

        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert is_valid is True
            assert missing == []

    def test_missing_app_host(self, mock_settings):
        """Missing app.host reports error."""
        from src.core.validation import validate_config_values

        mock_settings.app.host = ""
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert is_valid is False
            assert any("app.host" in m for m in missing)

    def test_missing_app_port(self, mock_settings):
        """Missing app.port reports error."""
        from src.core.validation import validate_config_values

        mock_settings.app.port = 0
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert is_valid is False
            assert any("app.port" in m for m in missing)

    def test_missing_service_url_when_enabled(self, mock_settings):
        """Missing service URL when service is enabled reports error."""
        from src.core.validation import validate_config_values

        mock_settings.services.ocr.url = ""
        mock_settings.run_services.ocr = True
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert is_valid is False
            assert any("services.ocr.url" in m for m in missing)

    def test_missing_service_url_when_disabled_is_ok(self, mock_settings):
        """Missing service URL is OK when service is disabled."""
        from src.core.validation import validate_config_values

        mock_settings.services.ocr.url = ""
        mock_settings.run_services.ocr = False
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert not any("services.ocr.url" in m for m in missing)

    def test_all_service_urls_missing_when_enabled(self, mock_settings):
        """All service URLs missing when all enabled reports all errors."""
        from src.core.validation import validate_config_values

        for svc in ["ocr", "quality", "postprocess", "lamination", "recapture", "graycopy"]:
            getattr(mock_settings.services, svc).url = ""
        mock_settings.services.qualitydl = MagicMock()
        mock_settings.services.qualitydl.url = ""
        mock_settings.run_services.qualitydl = True
        with patch("src.core.validation.get_settings", return_value=mock_settings):
            is_valid, missing = validate_config_values()
            assert is_valid is False
            assert len(missing) >= 6


class TestValidateStartupConfiguration:

    def test_passes_when_all_valid(self, mock_settings):
        """Startup validation passes when everything is configured."""
        from src.core.validation import validate_startup_configuration

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "OCR_SERVICE_API": "k",
            "LAMINATION_SERVICE_API": "k",
            "RECAPTURE_SERVICE_API": "k",
            "GRAYCOPY_SERVICE_API": "k",
            "TEMPER_SERVICE_API": "k",
            "CLASSIFIER_SERVICE_API": "k",
            "QUALITY_SERVICE_API": "k",
            "QUALITYDL_SERVICE_API": "k",
            "POSTPROCESS_SERVICE_API": "k",
            "ORCHESTRATOR_SERVICE_API": "k",
        }
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch.dict("os.environ", env, clear=False), \
             patch("src.core.validation.get_settings", return_value=mock_settings):
            validate_startup_configuration()

    def test_exits_on_missing_env_vars(self, mock_settings):
        """Startup validation sys.exit(1) when env vars missing."""
        from src.core.validation import validate_startup_configuration

        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch.dict("os.environ", {}, clear=True), \
             patch("src.core.validation.get_settings", return_value=mock_settings), \
             pytest.raises(SystemExit) as exc_info:
            validate_startup_configuration()
        assert exc_info.value.code == 1

    def test_exits_on_missing_config(self, mock_settings):
        """Startup validation sys.exit(1) when config values missing."""
        from src.core.validation import validate_startup_configuration

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "DATABASE_URL": "postgresql://localhost/db",
            "OCR_SERVICE_API": "k",
            "LAMINATION_SERVICE_API": "k",
            "RECAPTURE_SERVICE_API": "k",
            "GRAYCOPY_SERVICE_API": "k",
            "TEMPER_SERVICE_API": "k",
            "CLASSIFIER_SERVICE_API": "k",
            "QUALITY_SERVICE_API": "k",
            "QUALITYDL_SERVICE_API": "k",
            "POSTPROCESS_SERVICE_API": "k",
            "ORCHESTRATOR_SERVICE_API": "k",
        }
        mock_settings.app.host = ""
        mock_settings.services.qualitydl = mock_settings.services.quality
        with patch.dict("os.environ", env, clear=False), \
             patch("src.core.validation.get_settings", return_value=mock_settings), \
             pytest.raises(SystemExit) as exc_info:
            validate_startup_configuration()
        assert exc_info.value.code == 1
