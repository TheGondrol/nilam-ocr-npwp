import base64
import hashlib
import json
from pathlib import Path

import pytest
from google.auth import exceptions

from ocr_common.clients import gcs, models
from ocr_common.clients.gcp import EntraTokenSupplier, WifConfig, wif_config
from ocr_common.config import BaseServiceSettings

URI = "gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/scoring/trust_model.joblib"
CONTENT = b"the trust model"
SHA256 = hashlib.sha256(CONTENT).hexdigest()
MD5 = base64.b64encode(hashlib.md5(CONTENT, usedforsecurity=False).digest()).decode()
WIF_ENV = {
    "AZURE_TENANT_ID": "tenant",
    "AZURE_CLIENT_ID": "client",
    "AZURE_CLIENT_SECRET": "s",
    "GCP_PROJECT_NUMBER": "849229018263",
    "GCP_POOL_ID": "pool",
    "GCP_PROVIDER_ID": "provider",
    "GCP_SERVICE_ACCOUNT_EMAIL": "sa@project.iam.gserviceaccount.com",
}


class Response:
    def __init__(self, status: int, body: object = None, content: bytes = b""):
        self.status_code = status
        self._body = body
        self._content = content
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        return self._body

    def iter_content(self, size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """GCS as a dict of objects; records the media downloads and uploads."""

    def __init__(self, objects: dict[str, tuple[bytes, dict]]):
        self.objects = objects
        self.media_reads = 0
        self.uploads: list[dict] = []

    def get(self, url, params=None, stream=False, timeout=None):
        name = url.rsplit("/o/", 1)[1].replace("%2F", "/")
        if name not in self.objects:
            return Response(404, {"error": {"message": "No such object"}})
        content, meta = self.objects[name]
        if params and params.get("alt") == "media":
            self.media_reads += 1
            return Response(200, None, content)
        return Response(200, meta)

    def post(self, url, params=None, data: bytes = b"", headers=None, timeout=None):
        self.uploads.append({"params": params, "data": data, "headers": headers})
        head = json.loads(data.split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0])
        return Response(200, {"name": head["name"], "metadata": head["metadata"], "md5Hash": MD5, "size": "15"})


@pytest.fixture
def session(monkeypatch):
    name = gcs.GcsObject.parse(URI).name
    fake = FakeSession({name: (CONTENT, {"md5Hash": MD5, "metadata": {"sha256": SHA256}})})
    monkeypatch.setattr(gcs, "_session", lambda credentials: fake)
    return fake


def test_a_gcs_uri_is_bucket_and_object():
    obj = gcs.GcsObject.parse(URI)
    assert obj.bucket == "gc-bribrain-dev-gcs-ocr-nilam-01"
    assert obj.name.endswith("/trust_model.joblib")
    assert obj.path.startswith("/b/gc-bribrain-dev-gcs-ocr-nilam-01/o/nilam-ocr-npwp%2Fscoring")
    for bad in ("gs://bucket", "gs://bucket/", "gs://bucket/dir/", "s3://bucket/x", "bucket/x"):
        with pytest.raises(ValueError):
            gcs.GcsObject.parse(bad)


def test_a_download_is_verified_against_the_sha256_recorded_at_upload(session, tmp_path):
    target = gcs.download(URI, tmp_path / "m.joblib", credentials=None)

    assert target.read_bytes() == CONTENT


def test_a_download_that_does_not_match_the_pinned_sha256_leaves_nothing(session, tmp_path):
    with pytest.raises(gcs.GcsError, match="SHA-256"):
        gcs.download(URI, tmp_path / "m.joblib", credentials=None, sha256="0" * 64)

    assert list(tmp_path.iterdir()) == []


def test_a_download_whose_md5_differs_from_the_object_is_refused(session, tmp_path):
    name = gcs.GcsObject.parse(URI).name
    session.objects[name] = (b"tampered in transit", session.objects[name][1])

    with pytest.raises(gcs.GcsError, match="MD5"):
        gcs.download(URI, tmp_path / "m.joblib", credentials=None)
    assert list(tmp_path.iterdir()) == []


def test_a_file_already_there_with_the_expected_sha256_is_not_downloaded_again(session, tmp_path):
    (tmp_path / "m.joblib").write_bytes(CONTENT)

    gcs.download(URI, tmp_path / "m.joblib", credentials=None, sha256=SHA256)

    assert session.media_reads == 0


def test_a_missing_object_says_which(session, tmp_path):
    with pytest.raises(gcs.GcsError, match="404"):
        gcs.download(URI.replace("trust_model", "no_such_model"), tmp_path / "m", credentials=None)


def test_an_upload_overwrites_the_object_and_records_the_sha256(session, tmp_path):
    model = tmp_path / "trust_model.joblib"
    model.write_bytes(CONTENT)

    stored = gcs.upload(model, URI, credentials=None)  # the object already exists: replaced

    assert stored["metadata"] == {"sha256": SHA256}
    assert session.uploads[-1]["params"] == {"uploadType": "multipart"}


def test_wif_config_is_all_or_nothing():
    assert wif_config({}) is None
    with pytest.raises(ValueError, match="AZURE_CLIENT_SECRET"):
        wif_config({**WIF_ENV, "AZURE_CLIENT_SECRET": ""})
    config = wif_config(WIF_ENV)
    assert config is not None
    assert config.audience == (
        "//iam.googleapis.com/projects/849229018263/locations/global/workloadIdentityPools/pool/providers/provider"
    )


class EntraAnswer:
    def __init__(self, status: int, body: dict):
        self.status = status
        self.data = json.dumps(body).encode()


def _config() -> WifConfig:
    config = wif_config(WIF_ENV)
    assert config is not None
    return config


def test_the_entra_token_falls_back_to_the_plain_client_scope():
    scopes = []

    def request(url, method, body, headers):
        scopes.append(body.split("scope=")[1])
        if "api" in scopes[-1]:
            return EntraAnswer(400, {"error": "invalid_resource", "error_description": "AADSTS500011: not found"})
        return EntraAnswer(200, {"access_token": "entra-token"})

    assert EntraTokenSupplier(_config()).get_subject_token(None, request) == "entra-token"
    assert scopes == ["api%3A%2F%2Fclient%2F.default", "client%2F.default"]


def test_an_entra_failure_never_repeats_the_secret():
    def request(url, method, body, headers):
        return EntraAnswer(401, {"error": "invalid_client", "error_description": "AADSTS7000215: bad secret\nmore"})

    with pytest.raises(exceptions.RefreshError) as raised:
        EntraTokenSupplier(_config()).get_subject_token(None, request)
    assert "invalid_client: AADSTS7000215: bad secret" in str(raised.value)
    assert "client_secret" not in str(raised.value)


def _settings(**values) -> BaseServiceSettings:
    return BaseServiceSettings(api_key="k", environment="local", _env_file=None, **values)


def test_without_a_gcs_uri_the_local_path_is_used():
    assert models.model_file("weights/m.joblib", None, None, _settings()) == "weights/m.joblib"


def test_a_gcs_uri_without_credentials_stops_the_start():
    with pytest.raises(RuntimeError, match="needs GCP credentials"):
        models.model_file("weights/m.joblib", URI, SHA256, _settings())


def test_a_gcs_uri_is_downloaded_into_the_models_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        gcs, "download", lambda uri, target, credentials, sha256: calls.append((uri, target, sha256)) or target
    )
    settings = _settings(models_dir=str(tmp_path), **{name.lower(): value for name, value in WIF_ENV.items()})

    path = models.model_file("weights/m.joblib", URI, SHA256, settings)

    assert path == str(tmp_path / "trust_model.joblib")
    assert calls == [(URI, Path(path), SHA256)]


def test_a_model_uri_must_name_a_gcs_object():
    BaseServiceSettings.check_model_uri("scoring_model_gcs_uri", URI)
    BaseServiceSettings.check_model_uri("scoring_model_gcs_uri", None)
    for bad in ("https://x/m.joblib", "gs://bucket-only", "gs://bucket/dir/"):
        with pytest.raises(ValueError, match="SCORING_MODEL_GCS_URI must be gs://bucket"):
            BaseServiceSettings.check_model_uri("scoring_model_gcs_uri", bad)
