"""Unit tests for src.services.gcs_service module."""

from unittest.mock import patch, MagicMock

from src.services.gcs_service import _build_credentials_info, _parse_version

import pytest


class TestBuildCredentialsInfo:
    def test_builds_info(self):
        env = {
            "SA_TYPE": "service_account",
            "SA_PROJECT_ID": "proj",
            "SA_PRIVATE_KEY": "pk\\nline2",
            "SA_CLIENT_MAIL": "mail@x.com",
            "SA_CLIENT_ID": "cid",
            "SA_AUTH_URI": "https://auth",
            "SA_TOKEN_URI": "https://token",
            "SA_AUTH_PROVIDER": "https://provider",
            "SA_CERT_URL": "https://cert",
        }
        with patch.dict("os.environ", env, clear=True):
            info = _build_credentials_info()
        assert info["type"] == "service_account"
        assert info["project_id"] == "proj"
        assert "\n" in info["private_key"]

    def test_missing_env_raises_key_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(KeyError):
                _build_credentials_info()


class TestParseVersion:
    def test_single_version(self):
        assert _parse_version("model_v1.pth") == (1,)

    def test_two_part_version(self):
        assert _parse_version("laminate_model_v2.3.pth") == (2, 3)

    def test_three_part_version(self):
        assert _parse_version("model_v1.2.3.pth") == (1, 2, 3)

    def test_no_version(self):
        assert _parse_version("model.pth") == (0,)

    def test_version_ordering(self):
        assert _parse_version("m_v2.0.pth") > _parse_version("m_v1.9.pth")


class TestGetCredentials:
    def test_creates_credentials(self):
        import src.services.gcs_service as gcs_mod
        gcs_mod._credentials = None

        mock_creds = MagicMock()
        mock_creds.valid = True

        with patch("src.services.gcs_service._build_credentials_info", return_value={"type": "sa"}), \
             patch("src.services.gcs_service.service_account.Credentials.from_service_account_info",
                   return_value=mock_creds):
            from src.services.gcs_service import _get_credentials
            creds = _get_credentials()
            assert creds is mock_creds

        gcs_mod._credentials = None

    def test_refreshes_expired_credentials(self):
        import src.services.gcs_service as gcs_mod

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh = MagicMock()
        gcs_mod._credentials = mock_creds

        from src.services.gcs_service import _get_credentials
        creds = _get_credentials()
        mock_creds.refresh.assert_called_once()

        gcs_mod._credentials = None


class TestDownloadModelGcs:
    def test_download_success(self):
        mock_blob = MagicMock()
        mock_blob.name = "ocr_models/laminate_models/laminate_model_v1.0.pth"

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client), \
             patch("src.services.gcs_service.Path") as mock_path:
            mock_path_inst = MagicMock()
            mock_path_inst.parent.mkdir = MagicMock()
            mock_path_inst.exists.return_value = False
            mock_path.return_value = mock_path_inst

            from src.services.gcs_service import download_model_gcs
            download_model_gcs()
            mock_blob.download_to_filename.assert_called_once()

    def test_no_blobs_raises(self):
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client), \
             patch("src.services.gcs_service.Path") as mock_path:
            mock_path_inst = MagicMock()
            mock_path_inst.parent.mkdir = MagicMock()
            mock_path.return_value = mock_path_inst

            from src.services.gcs_service import download_model_gcs
            with pytest.raises(RuntimeError, match="No blobs"):
                download_model_gcs()

    def test_skips_if_exists(self):
        mock_blob = MagicMock()
        mock_blob.name = "prefix/laminate_model_v1.0.pth"

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client), \
             patch("src.services.gcs_service.Path") as mock_path:
            mock_path_inst = MagicMock()
            mock_path_inst.parent.mkdir = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path.return_value = mock_path_inst

            from src.services.gcs_service import download_model_gcs
            download_model_gcs()
            mock_blob.download_to_filename.assert_not_called()
