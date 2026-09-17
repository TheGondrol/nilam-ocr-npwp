"""Unit tests for src.services.minio_service module."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from src.services.minio_service import download_model_minio


class TestDownloadModelMinio:
    @patch("src.services.minio_service.config")
    def test_skips_when_model_exists(self, mock_config, tmp_path):
        model_file = tmp_path / "model.pth"
        model_file.write_text("fake model")
        mock_config.model_path = str(model_file)

        download_model_minio()  # should return early

    @patch("src.services.minio_service.config")
    def test_raises_when_no_bucket_config(self, mock_config, tmp_path):
        mock_config.model_path = str(tmp_path / "nonexistent.pth")
        mock_config.get.return_value = None

        with pytest.raises(ValueError, match="MinIO bucket and object"):
            download_model_minio()

    @patch("src.services.minio_service.config")
    def test_raises_when_no_credentials(self, mock_config, tmp_path):
        mock_config.model_path = str(tmp_path / "nonexistent.pth")
        mock_config.get.side_effect = lambda key: {
            "minio.bucket": "test-bucket",
            "minio.object": "model.pth",
        }.get(key)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MinIO credentials not found"):
                download_model_minio()

    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.config")
    def test_successful_download(self, mock_config, mock_minio_cls, tmp_path):
        model_path = tmp_path / "models" / "model.pth"
        mock_config.model_path = str(model_path)
        mock_config.get.side_effect = lambda key: {
            "minio.bucket": "test-bucket",
            "minio.object": "model.pth",
        }.get(key)

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "MINIO_SECURE": "false",
        }
        with patch.dict("os.environ", env, clear=True):
            mock_client = MagicMock()
            mock_minio_cls.return_value = mock_client
            download_model_minio()
            mock_client.fget_object.assert_called_once_with("test-bucket", "model.pth", str(model_path))

    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.config")
    def test_download_failure_raises(self, mock_config, mock_minio_cls, tmp_path):
        model_path = tmp_path / "model.pth"
        mock_config.model_path = str(model_path)
        mock_config.get.side_effect = lambda key: {
            "minio.bucket": "bucket",
            "minio.object": "obj",
        }.get(key)

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "a",
            "MINIO_SECRET_KEY": "s",
        }
        with patch.dict("os.environ", env, clear=True):
            mock_client = MagicMock()
            mock_client.fget_object.side_effect = Exception("Network error")
            mock_minio_cls.return_value = mock_client
            with pytest.raises(Exception, match="Network error"):
                download_model_minio()
