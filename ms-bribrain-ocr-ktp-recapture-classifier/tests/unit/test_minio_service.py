"""Tests for MinIO service."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestDownloadModelMinio:
    """Tests for download_model_minio function."""

    @patch('src.services.minio_service.Path')
    def test_skip_if_file_exists(self, mock_path_cls):
        """Test that download is skipped if file already exists."""
        from src.services.minio_service import download_model_minio

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_cls.return_value = mock_path_instance

        download_model_minio()

        # Should not try to create Minio client

    @patch('src.services.minio_service.Minio')
    @patch('src.services.minio_service.Path')
    def test_download_success(self, mock_path_cls, mock_minio_cls):
        """Test successful model download from MinIO."""
        from src.services.minio_service import download_model_minio

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.parent = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "MINIO_SECURE": "false",
        }
        with patch.dict("os.environ", env):
            download_model_minio()

        mock_client.fget_object.assert_called_once()
        mock_path_instance.parent.mkdir.assert_called_once()

    @patch('src.services.minio_service.Minio')
    @patch('src.services.minio_service.Path')
    def test_download_s3_error(self, mock_path_cls, mock_minio_cls):
        """Test MinIO download raises RuntimeError on S3Error."""
        from src.services.minio_service import download_model_minio
        from minio.error import S3Error

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.parent = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_client.fget_object.side_effect = S3Error(
            MagicMock(), "NoSuchBucket", "Bucket not found", "resource", "req-id", "host-id"
        )
        mock_minio_cls.return_value = mock_client

        env = {
            "MINIO_ENDPOINT": "localhost:9000",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
        }
        with patch.dict("os.environ", env):
            with pytest.raises(RuntimeError, match="MinIO download failed"):
                download_model_minio()

    @patch('src.services.minio_service.Minio')
    @patch('src.services.minio_service.Path')
    def test_secure_mode(self, mock_path_cls, mock_minio_cls):
        """Test MinIO client with secure=true."""
        from src.services.minio_service import download_model_minio

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.parent = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        env = {
            "MINIO_ENDPOINT": "minio.example.com",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "MINIO_SECURE": "true",
        }
        with patch.dict("os.environ", env):
            download_model_minio()

        mock_minio_cls.assert_called_once_with(
            endpoint="minio.example.com",
            access_key="access",
            secret_key="secret",
            secure=True,
        )
