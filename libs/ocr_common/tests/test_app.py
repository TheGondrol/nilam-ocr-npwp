"""create_app: perilaku yang harus sama di semua service (health, auth, envelope, request_id, intake)."""

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.testclient import TestClient

from ocr_common.app import create_app
from ocr_common.config import BaseServiceSettings
from ocr_common.envelope import envelope
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.request_id import get_request_id
from ocr_common.security import verify_api_key

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


def test_validation_error_uses_envelope_with_code():
    response = client.post("/v1/json", json=[1, 2], headers=AUTH)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"] == "VALIDATION_ERROR"
    assert body["message"].startswith("body:")
