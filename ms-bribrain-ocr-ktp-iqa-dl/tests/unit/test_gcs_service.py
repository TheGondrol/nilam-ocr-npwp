"""Unit tests for src.services.gcs_service module."""

from unittest.mock import patch, MagicMock

import pytest


class TestBuildCredentialsInfo:
    def test_builds_info_from_env(self):
        from src.services.gcs_service import _build_credentials_info

        env = {
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "proj",
            "SA_PRIVATE_KEY": "key\\nline2",
            "SA_CLIENT_MAIL": "mail@test.com",
            "SA_CLIENT_ID": "123",
            "SA_AUTH_URI": "https://auth",
            "SA_TOKEN_URI": "https://token",
            "SA_AUTH_PROVIDER": "https://provider",
            "SA_CERT_URL": "https://cert",
        }
        with patch.dict("os.environ", env, clear=True):
            info = _build_credentials_info()
            assert info["type"] == "service_account"
            assert info["project_id"] == "proj"
            assert "\n" in info["private_key"]  # \\n replaced with \n
            assert info["client_email"] == "mail@test.com"

    def test_missing_env_raises_keyerror(self):
        from src.services.gcs_service import _build_credentials_info

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(KeyError):
                _build_credentials_info()


class TestParseVersion:
    def test_single_version(self):
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v1.pth") == (1,)

    def test_multi_version(self):
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v2.1.pth") == (2, 1)

    def test_no_version(self):
        from src.services.gcs_service import _parse_version

        assert _parse_version("model.pth") == (0,)

    def test_complex_version(self):
        from src.services.gcs_service import _parse_version

        assert _parse_version("qualitydl_model_v1.2.3.pth") == (1, 2, 3)

    def test_version_ordering(self):
        from src.services.gcs_service import _parse_version

        assert _parse_version("model_v2.0.pth") > _parse_version("model_v1.1.pth")


class TestGetCredentials:
    @patch("src.services.gcs_service.service_account.Credentials")
    @patch("src.services.gcs_service._build_credentials_info")
    def test_creates_credentials(self, mock_build, mock_creds_cls):
        import src.services.gcs_service as gcs_mod
        gcs_mod._credentials = None

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds_cls.from_service_account_info.return_value = mock_creds
        mock_build.return_value = {"type": "service_account"}

        result = gcs_mod._get_credentials()
        assert result is mock_creds
        mock_creds_cls.from_service_account_info.assert_called_once()

        gcs_mod._credentials = None

    @patch("src.services.gcs_service._build_credentials_info")
    def test_creation_failure_raises_runtime_error(self, mock_build):
        import src.services.gcs_service as gcs_mod
        gcs_mod._credentials = None

        mock_build.side_effect = Exception("bad creds")
        with pytest.raises(RuntimeError, match="Failed to create GCS credentials"):
            gcs_mod._get_credentials()

        gcs_mod._credentials = None

    @patch("src.services.gcs_service.google.auth.transport.requests.Request")
    @patch("src.services.gcs_service.requests.Session")
    def test_refreshes_expired_credentials(self, mock_session, mock_request):
        import src.services.gcs_service as gcs_mod

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh = MagicMock()
        gcs_mod._credentials = mock_creds

        result = gcs_mod._get_credentials()
        mock_creds.refresh.assert_called_once()
        assert result is mock_creds

        gcs_mod._credentials = None


class TestGetGcsClient:
    @patch("src.services.gcs_service._get_credentials")
    def test_creates_client(self, mock_get_creds):
        import src.services.gcs_service as gcs_mod
        gcs_mod._gcs_client = None

        mock_creds = MagicMock()
        mock_get_creds.return_value = mock_creds

        with patch("google.cloud.storage.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            result = gcs_mod.get_gcs_client()
            assert result is mock_client

        gcs_mod._gcs_client = None

    @patch("src.services.gcs_service._get_credentials")
    def test_returns_singleton(self, mock_get_creds):
        import src.services.gcs_service as gcs_mod
        mock_client = MagicMock()
        gcs_mod._gcs_client = mock_client

        result = gcs_mod.get_gcs_client()
        assert result is mock_client

        gcs_mod._gcs_client = None


class TestDownloadModelGcs:
    @patch("src.services.gcs_service.get_gcs_client")
    @patch("src.services.gcs_service.config")
    def test_no_matching_blobs_raises(self, mock_config, mock_get_client, tmp_path):
        mock_config.gcs_bucket_name = "bucket"
        mock_config.gcs_prefix = "models/"
        mock_config.gcs_model_prefix = "qualitydl"
        mock_config.get.return_value = str(tmp_path / "model.pth")

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        from src.services.gcs_service import download_model_gcs
        with pytest.raises(RuntimeError, match="No blobs matching"):
            download_model_gcs()

    @patch("src.services.gcs_service.get_gcs_client")
    @patch("src.services.gcs_service.config")
    def test_skips_when_model_exists(self, mock_config, mock_get_client, tmp_path):
        model_file = tmp_path / "model.pth"
        model_file.write_text("fake")

        mock_config.gcs_bucket_name = "bucket"
        mock_config.gcs_prefix = "models/"
        mock_config.gcs_model_prefix = "qualitydl"
        mock_config.get.return_value = str(model_file)

        mock_blob = MagicMock()
        mock_blob.name = "models/qualitydl_model_v1.0.pth"
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        from src.services.gcs_service import download_model_gcs
        download_model_gcs()
        mock_blob.download_to_filename.assert_not_called()

    @patch("src.services.gcs_service.get_gcs_client")
    @patch("src.services.gcs_service.config")
    def test_downloads_latest_version(self, mock_config, mock_get_client, tmp_path):
        model_path = tmp_path / "nonexistent.pth"
        mock_config.gcs_bucket_name = "bucket"
        mock_config.gcs_prefix = "models/"
        mock_config.gcs_model_prefix = "qualitydl"
        mock_config.get.return_value = str(model_path)

        blob_v1 = MagicMock()
        blob_v1.name = "models/qualitydl_model_v1.0.pth"
        blob_v2 = MagicMock()
        blob_v2.name = "models/qualitydl_model_v2.0.pth"
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [blob_v1, blob_v2]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_get_client.return_value = mock_client

        from src.services.gcs_service import download_model_gcs
        download_model_gcs()
        blob_v2.download_to_filename.assert_called_once()
        blob_v1.download_to_filename.assert_not_called()
