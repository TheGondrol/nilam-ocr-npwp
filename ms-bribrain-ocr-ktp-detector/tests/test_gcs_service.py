"""
Unit tests for services.gcs_service module
"""
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

import src.services.gcs_service as gcs_mod
from src.services.gcs_service import (
    _build_credentials_info,
    _get_credentials,
    _parse_version,
    download_model_gcs,
    get_gcs_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SA_ENV_VARS = {
    "SA_TYPE": "service_account",
    "SA_PROJECT_ID": "my-project",
    "SA_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\\nMIIE...\\n-----END RSA PRIVATE KEY-----",
    "SA_CLIENT_MAIL": "sa@my-project.iam.gserviceaccount.com",
    "SA_CLIENT_ID": "123456789",
    "SA_AUTH_URI": "https://accounts.google.com/o/oauth2/auth",
    "SA_TOKEN_URI": "https://oauth2.googleapis.com/token",
    "SA_AUTH_PROVIDER": "https://www.googleapis.com/oauth2/v1/certs",
    "SA_CERT_URL": "https://www.googleapis.com/robot/v1/metadata/x509/sa",
}


@pytest.fixture(autouse=True)
def reset_gcs_singletons():
    """Reset module-level singletons before and after every test."""
    gcs_mod._credentials = None
    gcs_mod._gcs_client = None
    yield
    gcs_mod._credentials = None
    gcs_mod._gcs_client = None


# ---------------------------------------------------------------------------
# _build_credentials_info
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildCredentialsInfo:
    """Tests for _build_credentials_info."""

    def test_happy_path(self):
        """All env vars present produces correct dict."""
        with patch.dict(os.environ, _SA_ENV_VARS, clear=False):
            info = _build_credentials_info()

        assert info["type"] == "service_account"
        assert info["project_id"] == "my-project"
        assert info["client_email"] == "sa@my-project.iam.gserviceaccount.com"
        assert info["client_id"] == "123456789"
        assert info["auth_uri"] == "https://accounts.google.com/o/oauth2/auth"
        assert info["token_uri"] == "https://oauth2.googleapis.com/token"
        assert info["auth_provider_x509_cert_url"] == "https://www.googleapis.com/oauth2/v1/certs"
        assert info["client_x509_cert_url"] == "https://www.googleapis.com/robot/v1/metadata/x509/sa"

    def test_missing_env_var_raises_key_error(self):
        """Missing required env var raises KeyError."""
        incomplete = {k: v for k, v in _SA_ENV_VARS.items() if k != "SA_PROJECT_ID"}
        with patch.dict(os.environ, incomplete, clear=True):
            with pytest.raises(KeyError):
                _build_credentials_info()

    def test_private_key_newline_replacement(self):
        """Literal backslash-n sequences in SA_PRIVATE_KEY are replaced with real newlines."""
        env = {**_SA_ENV_VARS, "SA_PRIVATE_KEY": "line1\\nline2\\nline3"}
        with patch.dict(os.environ, env, clear=False):
            info = _build_credentials_info()

        assert "\\n" not in info["private_key"]
        assert info["private_key"] == "line1\nline2\nline3"


# ---------------------------------------------------------------------------
# _get_credentials
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCredentials:
    """Tests for _get_credentials."""

    def test_first_call_creates_credentials(self):
        """First invocation creates credentials via from_service_account_info."""
        mock_creds = MagicMock()
        mock_creds.valid = True

        with patch.dict(os.environ, _SA_ENV_VARS, clear=False):
            with patch(
                "src.services.gcs_service.service_account.Credentials.from_service_account_info",
                return_value=mock_creds,
            ) as mock_from_sa:
                result = _get_credentials()

        mock_from_sa.assert_called_once()
        assert result is mock_creds

    def test_creation_failure_raises_runtime_error(self):
        """Exception during credential creation is wrapped in RuntimeError."""
        with patch.dict(os.environ, _SA_ENV_VARS, clear=False):
            with patch(
                "src.services.gcs_service.service_account.Credentials.from_service_account_info",
                side_effect=ValueError("bad key"),
            ):
                with pytest.raises(RuntimeError, match="Failed to create GCS credentials"):
                    _get_credentials()

    def test_refresh_failure_raises_runtime_error(self):
        """RefreshError during credential refresh is wrapped in RuntimeError."""
        from google.auth.exceptions import RefreshError

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh.side_effect = RefreshError("token expired")

        with patch.dict(os.environ, _SA_ENV_VARS, clear=False):
            with patch(
                "src.services.gcs_service.service_account.Credentials.from_service_account_info",
                return_value=mock_creds,
            ):
                with pytest.raises(RuntimeError, match="Failed to refresh GCS credentials"):
                    _get_credentials()


# ---------------------------------------------------------------------------
# get_gcs_client
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetGcsClient:
    """Tests for get_gcs_client."""

    def test_creates_client_on_first_call(self):
        """First call creates a storage.Client with valid credentials."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_storage_client = MagicMock()

        with patch("src.services.gcs_service._get_credentials", return_value=mock_creds):
            with patch("google.cloud.storage.Client", return_value=mock_storage_client) as mock_cls:
                client = get_gcs_client()

        mock_cls.assert_called_once_with(credentials=mock_creds)
        assert client is mock_storage_client

    def test_returns_same_client_on_subsequent_calls(self):
        """Subsequent calls return the cached singleton client."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_storage_client = MagicMock()

        with patch("src.services.gcs_service._get_credentials", return_value=mock_creds):
            with patch("google.cloud.storage.Client", return_value=mock_storage_client) as mock_cls:
                first = get_gcs_client()
                second = get_gcs_client()

        # Constructor called only once
        mock_cls.assert_called_once()
        assert first is second


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseVersion:
    """Tests for _parse_version."""

    def test_single_dot_version(self):
        assert _parse_version("model_v1.1.pt") == (1, 1)

    def test_major_minor_version(self):
        assert _parse_version("model_v2.0.pt") == (2, 0)

    def test_three_part_version(self):
        assert _parse_version("model_v1.1.5.pt") == (1, 1, 5)

    def test_no_version_returns_zero(self):
        assert _parse_version("model.pt") == (0,)


# ---------------------------------------------------------------------------
# download_model_gcs
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDownloadModelGcs:
    """Tests for download_model_gcs."""

    def _make_blob(self, name: str) -> MagicMock:
        """Helper: create a mock blob with given .name."""
        blob = MagicMock()
        blob.name = name
        return blob

    def _patch_config(self, model_path: str):
        """Helper: return a patch context for config."""
        mock_cfg = MagicMock()
        mock_cfg.gcs_bucket_name = "my-bucket"
        mock_cfg.gcs_prefix = "models/"
        mock_cfg.gcs_model_prefix = "graycopy_model"
        mock_cfg.get.return_value = model_path
        return patch("src.services.gcs_service.config", mock_cfg)

    def test_successful_download(self, tmp_path):
        """Model is downloaded when it does not exist locally."""
        model_file = tmp_path / "models" / "ktp_detection_model.pt"

        blob = self._make_blob("models/graycopy_model_v1.0.pt")
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with self._patch_config(str(model_file)):
            with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client):
                download_model_gcs()

        blob.download_to_filename.assert_called_once_with(str(model_file))

    def test_no_matching_blobs_raises_runtime_error(self, tmp_path):
        """RuntimeError when no blobs match the model prefix."""
        model_file = tmp_path / "models" / "ktp_detection_model.pt"

        # Only non-matching blobs in the bucket
        other_blob = self._make_blob("models/other_file.pt")
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [other_blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with self._patch_config(str(model_file)):
            with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client):
                with pytest.raises(RuntimeError, match="No blobs matching"):
                    download_model_gcs()

    def test_model_already_exists_skips_download(self, tmp_path):
        """Download is skipped when the local model file already exists."""
        model_file = tmp_path / "models" / "ktp_detection_model.pt"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text("existing model")

        blob = self._make_blob("models/graycopy_model_v1.0.pt")
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with self._patch_config(str(model_file)):
            with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client):
                download_model_gcs()

        blob.download_to_filename.assert_not_called()

    def test_picks_highest_version_blob(self, tmp_path):
        """The blob with the highest version number is selected."""
        model_file = tmp_path / "models" / "ktp_detection_model.pt"

        blob_v1 = self._make_blob("models/graycopy_model_v1.0.pt")
        blob_v2 = self._make_blob("models/graycopy_model_v2.0.pt")
        blob_v1_5 = self._make_blob("models/graycopy_model_v1.5.pt")

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [blob_v1, blob_v2, blob_v1_5]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with self._patch_config(str(model_file)):
            with patch("src.services.gcs_service.get_gcs_client", return_value=mock_client):
                download_model_gcs()

        # Only v2.0 should be downloaded
        blob_v2.download_to_filename.assert_called_once_with(str(model_file))
        blob_v1.download_to_filename.assert_not_called()
        blob_v1_5.download_to_filename.assert_not_called()
