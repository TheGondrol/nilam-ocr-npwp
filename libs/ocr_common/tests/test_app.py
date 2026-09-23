import pytest
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.testclient import TestClient

from ocr_common.config import BaseServiceSettings
from ocr_common.web.app import create_app
from ocr_common.web.envelope import envelope
from ocr_common.web.intake import FileField, FileUrlField, read_image
from ocr_common.web.request_id import get_request_id
from ocr_common.web.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/v1/echo")
async def echo(request: Request, file: UploadFile | str | None = FileField, file_url: str | None = FileUrlField):
    content, filename, content_type = await read_image(request, file, file_url)
    return envelope(200, "Success", {"size": len(content), "filename": filename}, get_request_id(request))


@router.post("/v1/json")
async def json_endpoint(request: Request, body: dict):
    return envelope(200, "Success", body, get_request_id(request))


settings = BaseServiceSettings(api_key="k", _env_file=None)
app = create_app(
    settings=settings,
    title="Demo",
    description="demo",
    routers=[router],
    backends={"demo": "mock"},
)
client = TestClient(app, raise_server_exceptions=False)
AUTH = {"X-API-Key": "k"}


def test_health_is_public():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["backends"] == {"demo": "mock"}


def test_missing_api_key_is_401_envelope():
    response = client.post("/v1/echo", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert response.status_code == 401
    body = response.json()
    assert body["status_desc"] == "Unauthorized"
    assert body["errors"] == "Invalid or missing API key"


@pytest.mark.parametrize("key", ["salah", "kk", "ké"])
def test_wrong_api_key_is_401_including_non_ascii(key):
    response = client.post("/v1/json", json={}, headers={"X-API-Key": key.encode("latin-1")})
    assert response.status_code == 401


def test_auth_disabled_skips_the_api_key_check():
    open_app = create_app(
        settings=BaseServiceSettings(api_key="k", auth_disabled=True, environment="local", _env_file=None),
        title="Demo",
        description="demo",
        routers=[router],
    )
    open_client = TestClient(open_app, raise_server_exceptions=False)
    assert open_client.post("/v1/echo", files={"file": ("a.jpg", b"x", "image/jpeg")}).status_code == 200
    assert (
        open_client.post(
            "/v1/echo", files={"file": ("a.jpg", b"x", "image/jpeg")}, headers={"X-API-Key": "salah"}
        ).status_code
        == 200
    )


def _app_with_readiness(readiness):
    return TestClient(
        create_app(settings=settings, title="Demo", description="demo", readiness=readiness),
        raise_server_exceptions=False,
    )


def test_ready_without_dependencies_is_ready_and_public():
    response = _app_with_readiness(None).get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {}}


def test_ready_is_503_when_a_required_dependency_fails_but_health_stays_up():
    async def ok():
        return None

    async def down():
        raise ConnectionError("postgresql://user:secret@db:5432 unreachable")

    probe = _app_with_readiness({"database": down, "cache": ok})
    response = probe.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "checks": {"database": "failed", "cache": "ok"}}
    assert "secret" not in response.text
    assert probe.get("/health").status_code == 200


def test_request_id_header_is_echoed_or_generated():
    response = client.post(
        "/v1/echo", files={"file": ("a.jpg", b"x", "image/jpeg")}, headers={**AUTH, "X-Request-ID": "OCR_1"}
    )
    assert response.json()["request_id"] == "OCR_1"
    assert response.headers["X-Request-ID"] == "OCR_1"
    generated = client.post("/v1/echo", files={"file": ("a.jpg", b"x", "image/jpeg")}, headers=AUTH)
    assert generated.json()["request_id"].startswith("REQ_")


def test_intake_requires_exactly_one_of_file_or_file_url():
    neither = client.post("/v1/echo", data={"file": ""}, headers=AUTH)
    assert neither.status_code == 400
    assert neither.json()["message"] == "Send exactly one of file or file_url"
    both = client.post(
        "/v1/echo", data={"file_url": "http://x/y.jpg"}, files={"file": ("a.jpg", b"x", "image/jpeg")}, headers=AUTH
    )
    assert both.status_code == 400


def test_intake_rejects_bad_url_scheme_as_400():
    response = client.post("/v1/echo", data={"file_url": "file:///etc/passwd"}, headers=AUTH)
    assert response.status_code == 400
    assert "Unsupported URL scheme" in response.json()["message"]


def test_intake_refuses_internal_file_url_as_400():
    response = client.post("/v1/echo", data={"file_url": "http://169.254.169.254/latest/meta-data/"}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["message"] == "file_url host is not allowed: 169.254.169.254"


def test_validation_error_uses_envelope_with_code():
    response = client.post("/v1/json", json=[1, 2], headers=AUTH)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"].startswith("body:")


def test_every_accepted_key_opens_the_door_during_a_rotation():
    rotating = create_app(
        settings=BaseServiceSettings(api_key="baru", api_keys="lama, baru ,", _env_file=None),
        title="Demo",
        description="demo",
        routers=[router],
    )
    rotating_client = TestClient(rotating, raise_server_exceptions=False)
    assert rotating.state.settings.accepted_api_keys == ("baru", "lama")
    for key in ("lama", "baru"):
        assert rotating_client.post("/v1/json", json={}, headers={"X-API-Key": key}).status_code == 200
    assert rotating_client.post("/v1/json", json={}, headers={"X-API-Key": "k"}).status_code == 401
