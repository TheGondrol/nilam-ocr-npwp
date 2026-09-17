"""Tests for GCS service."""

import pytest
from unittest.mock import patch, MagicMock


class TestBuildCredentialsInfo:
    """Tests for _build_credentials_info."""

    def test_builds_from_env(self):
        """Test building credentials info from environment variables."""
        from src.services.gcs_service import _build_credentials_info

        env = {
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "my-project",
            "SA_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\\nfake\\n-----END PRIVATE KEY-----",
            "SA_CLIENT_MAIL": "test@sa.iam.gserviceaccount.com",
            "SA_CLIENT_ID": "123456789",
            "SA_AUTH_URI": "https://accounts.google.com/o/oauth2/auth",
            "SA_TOKEN_URI": "https://oauth2.googleapis.com/token",
            "SA_AUTH_PROVIDER": "https://www.googleapis.com/oauth2/v1/certs",
            "SA_CERT_URL": "https://www.googleapis.com/robot/v1/metadata/x509/test",
        }
        with patch.dict("os.environ", env):
            info = _build_credentials_info()

        assert info["type"] == "service_account"
        assert info["project_id"] == "my-project"
        assert info["client_email"] == "test@sa.iam.gserviceaccount.com"
        assert "\\n" not in info["private_key"]  # should be real newlines

    def test_missing_env_raises(self):
        """Test that missing env var raises KeyError."""
        from src.services.gcs_service import _build_credentials_info

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(KeyError):
                _build_credentials_info()


class TestParseVersion:
    """Tests for _parse_version."""

    def test_standard_version(self):
        """Test parsing standard version string."""
        from src.services.gcs_service import _parse_version

        assert _parse_version("recapture_model_v1.2.pth") == (1, 2)

    def test_single_version(self):
        """Test parsing single version number."""
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v3.pth") == (3,)

    def test_triple_version(self):
        """Test parsing triple version number."""
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v2.1.3.pth") == (2, 1, 3)

    def test_no_version(self):
        """Test parsing filename with no version."""
        from src.services.gcs_service import _parse_version

        assert _parse_version("model.pth") == (0,)

    def test_version_ordering(self):
        """Test that parsed versions compare correctly."""
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v2.0.pth") > _parse_version("model_v1.1.pth")
        assert _parse_version("model_v1.2.pth") > _parse_version("model_v1.1.pth")


class TestGetCredentials:
    """Tests for _get_credentials."""

    @patch('src.services.gcs_service.service_account.Credentials.from_service_account_info')
    @patch('src.services.gcs_service._build_credentials_info')
    def test_creates_credentials(self, mock_build, mock_from_info):
        """Test creating new credentials."""
        import src.services.gcs_service as gcs_mod

        # Reset module state
        original_creds = gcs_mod._credentials
        gcs_mod._credentials = None

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_from_info.return_value = mock_creds
        mock_build.return_value = {"type": "service_account"}

        try:
            result = gcs_mod._get_credentials()
            assert result is mock_creds
            mock_from_info.assert_called_once()
        finally:
            gcs_mod._credentials = original_creds

    @patch('src.services.gcs_service._build_credentials_info')
    def test_create_credentials_failure(self, mock_build):
        """Test RuntimeError when credentials cannot be created."""
        import src.services.gcs_service as gcs_mod

        original_creds = gcs_mod._credentials
        gcs_mod._credentials = None

        mock_build.side_effect = KeyError("SA_TYPE")

        try:
            with pytest.raises(RuntimeError, match="Failed to create GCS credentials"):
                gcs_mod._get_credentials()
        finally:
            gcs_mod._credentials = original_creds

    @patch('src.services.gcs_service.google.auth.transport.requests.Request')
    @patch('src.services.gcs_service.requests.Session')
    def test_refreshes_expired_credentials(self, mock_session, mock_request):
        """Test refreshing expired credentials."""
        import src.services.gcs_service as gcs_mod

        original_creds = gcs_mod._credentials

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh = MagicMock()
        gcs_mod._credentials = mock_creds

        try:
            result = gcs_mod._get_credentials()
            mock_creds.refresh.assert_called_once()
            assert result is mock_creds
        finally:
            gcs_mod._credentials = original_creds

    @patch('src.services.gcs_service.google.auth.transport.requests.Request')
    @patch('src.services.gcs_service.requests.Session')
    def test_refresh_failure(self, mock_session, mock_request):
        """Test RuntimeError when refresh fails."""
        from google.auth.exceptions import RefreshError
        import src.services.gcs_service as gcs_mod

        original_creds = gcs_mod._credentials

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh.side_effect = RefreshError("token expired")
        gcs_mod._credentials = mock_creds

        try:
            with pytest.raises(RuntimeError, match="Failed to refresh GCS credentials"):
                gcs_mod._get_credentials()
        finally:
            gcs_mod._credentials = original_creds


class TestGetGcsClient:
    """Tests for get_gcs_client."""

    @patch('src.services.gcs_service._get_credentials')
    def test_creates_client(self, mock_get_creds):
        """Test creating GCS client singleton."""
        import src.services.gcs_service as gcs_mod

        original_client = gcs_mod._gcs_client
        gcs_mod._gcs_client = None

        mock_creds = MagicMock()
        mock_get_creds.return_value = mock_creds

        mock_storage = MagicMock()
        with patch.dict('sys.modules', {'google.cloud': MagicMock(), 'google.cloud.storage': mock_storage}):
            try:
                client = gcs_mod.get_gcs_client()
                assert client is not None
            finally:
                gcs_mod._gcs_client = original_client


class TestDownloadModelGcs:
    """Tests for download_model_gcs."""

    @patch('src.services.gcs_service.get_gcs_client')
    def test_no_matching_blobs_raises(self, mock_get_client):
        """Test RuntimeError when no matching blobs found."""
        from src.services.gcs_service import download_model_gcs

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="No blobs matching"):
            download_model_gcs()

    @patch('src.services.gcs_service.get_gcs_client')
    @patch('src.services.gcs_service.Path')
    def test_always_redownloads_newest_version(self, mock_path_cls, mock_get_client):
        """BUG-07: the skip-if-exists check was removed; the newest version is
        always (re)downloaded on startup, even when a local file is present."""
        from src.services.gcs_service import download_model_gcs

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.parent = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        mock_blob = MagicMock()
        mock_blob.name = "models/recapture_model_v1.0.pth"

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        download_model_gcs()

        mock_blob.download_to_filename.assert_called_once()

    @patch('src.services.gcs_service.get_gcs_client')
    @patch('src.services.gcs_service.Path')
    def test_downloads_latest_version(self, mock_path_cls, mock_get_client):
        """Test that the latest versioned model is downloaded."""
        from src.services.gcs_service import download_model_gcs

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.parent = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        blob_v1 = MagicMock()
        blob_v1.name = "models/recapture_model_v1.0.pth"
        blob_v2 = MagicMock()
        blob_v2.name = "models/recapture_model_v2.0.pth"

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [blob_v1, blob_v2]
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        download_model_gcs()

        blob_v2.download_to_filename.assert_called_once()
        blob_v1.download_to_filename.assert_not_called()
