"""
Unit tests for minio_service
"""

import pytest
from unittest.mock import Mock, patch
from minio.error import S3Error

from src.services.minio_service import download_model_minio as download_model


@pytest.mark.unit
class TestDownloadModel:
    """Tests for download_model function"""
    
    @patch("src.services.minio_service.Path")
    @patch("src.services.minio_service.config")
    def test_skips_download_if_file_exists(self, mock_config, mock_path_class):
        """Test that download is skipped if file already exists"""
        mock_config.model.file_path = "/tmp/model.zip"
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path
        
        download_model()
        
        mock_path.exists.assert_called_once()
        # Should not attempt to create directory or download
    
    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.Path")
    @patch("src.services.minio_service.config")
    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "test_key",
        "MINIO_SECRET_KEY": "test_secret",
        "MINIO_SECURE": "false"
    })
    def test_downloads_model_when_not_exists(self, mock_config, mock_path_class, mock_minio_class):
        """Test that model is downloaded when file doesn't exist"""
        mock_config.model.file_path = "/tmp/model.zip"
        mock_config.minio.bucket = "test-bucket"
        mock_config.minio.object = "model.zip"
        
        # Mock path
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path.parent = Mock()
        mock_path_class.return_value = mock_path
        
        # Mock Minio client
        mock_client = Mock()
        mock_minio_class.return_value = mock_client
        
        download_model()
        
        # Verify directory creation
        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify Minio client creation
        mock_minio_class.assert_called_once_with(
            endpoint="localhost:9000",
            access_key="test_key",
            secret_key="test_secret",
            secure=False
        )
        
        # Verify download - path might be normalized
        mock_client.fget_object.assert_called_once()
        call_args = mock_client.fget_object.call_args
        assert call_args.kwargs["bucket_name"] == "test-bucket"
        assert call_args.kwargs["object_name"] == "model.zip"
    
    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.Path")
    @patch("src.services.minio_service.config")
    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "test_key",
        "MINIO_SECRET_KEY": "test_secret",
        "MINIO_SECURE": "true"
    })
    def test_uses_secure_connection_when_configured(self, mock_config, mock_path_class, mock_minio_class):
        """Test that secure connection is used when MINIO_SECURE is true"""
        mock_config.model.file_path = "/tmp/model.zip"
        mock_config.minio.bucket = "test-bucket"
        mock_config.minio.object = "model.zip"
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path.parent = Mock()
        mock_path_class.return_value = mock_path
        
        mock_client = Mock()
        mock_minio_class.return_value = mock_client
        
        download_model()
        
        # Verify secure=True
        mock_minio_class.assert_called_once_with(
            endpoint="localhost:9000",
            access_key="test_key",
            secret_key="test_secret",
            secure=True
        )
    
    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.Path")
    @patch("src.services.minio_service.config")
    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "test_key",
        "MINIO_SECRET_KEY": "test_secret"
    })
    def test_raises_runtime_error_on_s3_error(self, mock_config, mock_path_class, mock_minio_class):
        """Test that RuntimeError is raised when S3Error occurs"""
        mock_config.model.file_path = "/tmp/model.zip"
        mock_config.minio.bucket = "test-bucket"
        mock_config.minio.object = "model.zip"
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path.parent = Mock()
        mock_path_class.return_value = mock_path
        
        mock_client = Mock()
        mock_client.fget_object.side_effect = S3Error(
            code="NoSuchBucket",
            message="Bucket not found",
            resource="resource",
            request_id="req-id",
            host_id="host-id",
            response=Mock(),
        )
        mock_minio_class.return_value = mock_client
        
        with pytest.raises(RuntimeError, match="MinIO download failed"):
            download_model()
    
    @patch("src.services.minio_service.Minio")
    @patch("src.services.minio_service.Path")
    @patch("src.services.minio_service.config")
    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "localhost:9000",
        "MINIO_ACCESS_KEY": "test_key",
        "MINIO_SECRET_KEY": "test_secret"
    })
    def test_creates_parent_directories(self, mock_config, mock_path_class, mock_minio_class):
        """Test that parent directories are created"""
        mock_config.model.file_path = "/tmp/models/subfolder/model.zip"
        mock_config.minio.bucket = "test-bucket"
        mock_config.minio.object = "model.zip"
        
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_parent = Mock()
        mock_path.parent = mock_parent
        mock_path_class.return_value = mock_path
        
        mock_client = Mock()
        mock_minio_class.return_value = mock_client
        
        download_model()
        
        # Verify mkdir was called with parents=True and exist_ok=True
        mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
