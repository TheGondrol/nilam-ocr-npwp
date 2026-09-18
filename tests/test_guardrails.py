from src.core.config import Settings
from src.services.guardrails_service import GuardrailsService

from tests.conftest import image_upload


class StubQuality:
    def __init__(self, score):
        self._score = score

    def assess(self, filename, content):
        return {"score": self._score, "notes": []}


class StubClassifier:
    def __init__(self, label, confidence):
        self._result = {"label": label, "confidence": confidence}

    def classify(self, filename, content):
        return self._result


SETTINGS = Settings(api_key="x", _env_file=None)
IMAGE = ("x.jpg", "image/jpeg", b"abc")


def test_service_passes_when_all_checks_pass():
    report = GuardrailsService(StubQuality(0.9), StubClassifier("npwp", 0.95), SETTINGS).check(*IMAGE)
    assert report["passed"] is True
    assert report["document_type"] == "npwp"
    assert [c["name"] for c in report["checks"]] == ["image_quality", "document_type"]


def test_service_fails_on_low_quality_but_still_reports_classification():
    report = GuardrailsService(StubQuality(0.2), StubClassifier("npwp", 0.95), SETTINGS).check(*IMAGE)
    assert report["passed"] is False
    assert [c["name"] for c in report["checks"] if not c["passed"]] == ["image_quality"]
    assert report["document_type"] == "npwp"


def test_service_fails_on_low_classification_confidence():
    report = GuardrailsService(StubQuality(0.9), StubClassifier("npwp", 0.4), SETTINGS).check(*IMAGE)
    assert report["passed"] is False
    assert [c["name"] for c in report["checks"] if not c["passed"]] == ["document_type"]


def test_http_check_ok(client, auth):
    response = client.post("/v1/guardrails/check", files=image_upload("npwp.jpg"), headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["passed"] is True
    assert data["document_type"] == "npwp"


def test_http_check_blur_reports_failure_with_200(client, auth):
    response = client.post("/v1/guardrails/check", files=image_upload("blur.jpg"), headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["passed"] is False
    assert [c["name"] for c in data["checks"] if not c["passed"]] == ["image_quality"]


def test_http_check_notnpwp(client, auth):
    data = client.post("/v1/guardrails/check", files=image_upload("notnpwp.jpg"), headers=auth).json()["data"]
    assert data["passed"] is False
    assert data["document_type"] == "unknown"
