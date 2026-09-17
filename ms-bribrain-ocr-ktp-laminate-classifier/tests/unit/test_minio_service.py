"""Unit tests for src.services.minio_service module."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from src.services.minio_service import download_model_minio


class TestDownloadModelMinio:
    def test_skips_if_file_exists(self):
        with patch("src.services.minio_service.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            download_model_minio()  # Should return early

    def test_skips_if_no_endpoint(self):
        with patch("src.services.minio_service.Path") as mock_path, \
             patch.dict("os.environ", {}, clear=True):
            mock_path.return_value.exists.return_value = False
            download_model_minio()  # Should return early

    def test_downloads_from_minio(self):
        mock_client = MagicMock()
        with patch("src.services.minio_service.Path") as mock_path, \
             patch("src.services.minio_service.Minio", return_value=mock_client), \
             patch.dict("os.environ", {
                 "MINIO_ENDPOINT": "minio:9000",
                 "MINIO_ACCESS_KEY": "ak",
                 "MINIO_SECRET_KEY": "sk",
             }):
            mock_instance = MagicMock()
            mock_instance.exists.return_value = False
            mock_instance.parent.mkdir = MagicMock()
            mock_path.return_value = mock_instance
            download_model_minio()
            mock_client.fget_object.assert_called_once()

    def test_raises_on_s3_error(self):
        from minio.error import S3Error
        mock_client = MagicMock()
        mock_client.fget_object.side_effect = S3Error(
            MagicMock(), "NoSuchKey", "msg", "resource", "req", "host"
        )
        with patch("src.services.minio_service.Path") as mock_path, \
             patch("src.services.minio_service.Minio", return_value=mock_client), \
             patch.dict("os.environ", {
                 "MINIO_ENDPOINT": "minio:9000",
                 "MINIO_ACCESS_KEY": "ak",
                 "MINIO_SECRET_KEY": "sk",
             }):
            mock_instance = MagicMock()
            mock_instance.exists.return_value = False
            mock_instance.parent.mkdir = MagicMock()
            mock_path.return_value = mock_instance
            import pytest
            with pytest.raises(RuntimeError, match="MinIO download failed"):
                download_model_minio()
