"""
Unit tests for services.minio_service module
"""
import os
from unittest.mock import patch, MagicMock, mock_open
import pytest

from src.services.minio_service import download_model_minio as download_model


@pytest.mark.unit
class TestMinioService:
    """Test MinIO service functions"""
    
    def test_download_model_file_exists(self, clean_env, tmp_path):
        """Test that download is skipped when file exists"""
        # Create a mock file
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("mock model")
        
        with patch('src.core.config.config') as mock_config:
            mock_config.get.return_value = str(model_file)
            
            with patch('pathlib.Path.exists', return_value=True):
                # Should not attempt to download
                download_model()
    
    def test_download_model_success(self, clean_env, tmp_path, mock_minio_client):
        """Test successful model download"""
        model_path = tmp_path / "models" / "test_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'
        os.environ['MINIO_SECURE'] = 'false'
        
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"model data", b""]
        mock_minio_client.get_object.return_value = mock_response
        
        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt'
            }.get(key, default)
            
            # Create a mock Path that tracks state
            path_exists_calls = [0]  # Use list to allow mutation in nested function
            def mock_exists():
                # First call: file doesn't exist, subsequent: directory exists
                result = path_exists_calls[0] > 0
                path_exists_calls[0] += 1
                return result
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir') as mock_mkdir:
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('src.services.minio_service.Minio', return_value=mock_minio_client):
                            with patch('builtins.open', mock_open()) as mock_file:
                                with patch('os.path.isabs', return_value=True):
                                    download_model()
                                    
                                    # Should have created directory and written file
                                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                                    mock_file.assert_called_once()
    
    def test_download_model_minio_error(self, clean_env, tmp_path):
        """Test model download with MinIO error"""
        from minio.error import S3Error
        
        model_path = tmp_path / "models" / "test_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'
        
        mock_client = MagicMock()
        mock_client.get_object.side_effect = S3Error(
            response=MagicMock(),
            code="NoSuchKey",
            message="The specified key does not exist",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )
        
        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt'
            }.get(key, default)
            
            # Mock Path operations
            path_exists_calls = [0]
            def mock_exists():
                result = path_exists_calls[0] > 0
                path_exists_calls[0] += 1
                return result
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir'):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('src.services.minio_service.Minio', return_value=mock_client):
                            with patch('os.path.isabs', return_value=True):
                                with pytest.raises(RuntimeError, match="MinIO download failed"):
                                    download_model()
    
    def test_download_model_io_error(self, clean_env, tmp_path, mock_minio_client):
        """Test model download with IO error"""
        model_path = tmp_path / "models" / "test_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'
        
        mock_response = MagicMock()
        mock_response.read.return_value = b"model data"
        mock_minio_client.get_object.return_value = mock_response
        
        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt'
            }.get(key, default)
            
            # Mock Path operations
            path_exists_calls = [0]
            def mock_exists():
                result = path_exists_calls[0] > 0
                path_exists_calls[0] += 1
                return result
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir'):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('src.services.minio_service.Minio', return_value=mock_minio_client):
                            with patch('os.path.isabs', return_value=True):
                                with patch('builtins.open', side_effect=IOError("Permission denied")):
                                    with pytest.raises(RuntimeError, match="Failed to write file"):
                                        download_model()
    
    def test_download_model_absolute_path(self, clean_env, tmp_path, mock_minio_client):
        """Test model download with absolute path"""
        model_path = tmp_path / "models" / "test_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'
        
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"model data", b""]
        mock_minio_client.get_object.return_value = mock_response
        
        with patch('src.core.config.config') as mock_config:
            # Use absolute path
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path.absolute()),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt'
            }.get(key, default)
            
            # Mock Path operations
            path_exists_calls = [0]
            def mock_exists():
                result = path_exists_calls[0] > 0
                path_exists_calls[0] += 1
                return result
            
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir'):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('src.services.minio_service.Minio', return_value=mock_minio_client):
                            with patch('builtins.open', mock_open()):
                                # Should handle absolute path
                                download_model()

    @pytest.mark.unit
    def test_download_model_missing_endpoint(self, clean_env, tmp_path):
        """Test that missing MINIO_ENDPOINT raises RuntimeError"""
        model_path = tmp_path / "models" / "test_model.pt"

        # Ensure MINIO_ENDPOINT is NOT set (clean_env already removes it)
        os.environ.pop('MINIO_ENDPOINT', None)

        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
            }.get(key, default)

            # File does not exist -- so it proceeds past the exists() check
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(RuntimeError, match="MINIO_ENDPOINT environment variable is required"):
                    download_model()

    @pytest.mark.unit
    def test_download_model_directory_creation_failure(self, clean_env, tmp_path):
        """Test RuntimeError when mkdir succeeds but directory still doesn't exist"""
        model_path = tmp_path / "models" / "test_model.pt"

        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'

        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt',
            }.get(key, default)

            # First exists() call: file doesn't exist (triggers download path)
            # After mkdir, model_dir.exists() returns False (creation failure)
            exists_calls = [0]
            def mock_exists(self_path=None):
                exists_calls[0] += 1
                # Call 1: output_path.exists() -> False (file not found, proceed)
                # Call 2: model_dir.exists() -> False (directory creation failed)
                return False

            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir'):
                    with patch('src.services.minio_service.Minio', return_value=MagicMock()):
                        with pytest.raises(RuntimeError, match="Failed to create directory"):
                            download_model()

    @pytest.mark.unit
    def test_download_model_path_not_directory(self, clean_env, tmp_path):
        """Test RuntimeError when path exists but is not a directory"""
        model_path = tmp_path / "models" / "test_model.pt"

        os.environ['MINIO_ENDPOINT'] = 'minio:9000'
        os.environ['MINIO_ACCESS_KEY'] = 'testkey'
        os.environ['MINIO_SECRET_KEY'] = 'testsecret'

        with patch('src.core.config.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.path': str(model_path),
                'minio.bucket': 'models',
                'minio.object': 'test_model.pt',
            }.get(key, default)

            # First exists() call: file doesn't exist
            # Second exists() call: model_dir.exists() returns True (path exists)
            exists_calls = [0]
            def mock_exists(self_path=None):
                exists_calls[0] += 1
                if exists_calls[0] == 1:
                    return False  # output_path.exists() -> file not found
                return True  # model_dir.exists() -> path exists

            with patch('pathlib.Path.exists', side_effect=mock_exists):
                with patch('pathlib.Path.mkdir'):
                    with patch('pathlib.Path.is_dir', return_value=False):
                        with patch('src.services.minio_service.Minio', return_value=MagicMock()):
                            with pytest.raises(RuntimeError, match="Path exists but is not a directory"):
                                download_model()
