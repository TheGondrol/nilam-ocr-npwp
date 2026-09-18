from src.core.config import Settings
from src.services.ekstraksi_service import EkstraksiService

from tests.conftest import image_upload


class StubEngine:
    name = "stub"

    def extract(self, filename, content):
        return [{"text": "A", "confidence": 0.9, "bbox": None}, {"text": "B", "confidence": 0.8, "bbox": None}]


def test_service_wraps_engine_output_with_metadata():
    result = EkstraksiService(StubEngine(), Settings(api_key="x", _env_file=None)).extract(
        "x.jpg", "image/jpeg", b"abc"
    )
    assert result["engine"] == "stub"
    assert result["full_text"] == "A\nB"
    assert result["elapsed_ms"] >= 0


def test_http_extract_is_deterministic_per_content(client, auth):
    first = client.post("/v1/ekstraksi/extract", files=image_upload(content=b"same"), headers=auth).json()["data"]
    second = client.post("/v1/ekstraksi/extract", files=image_upload(content=b"same"), headers=auth).json()["data"]
    assert first["blocks"] == second["blocks"]
    assert first["engine"] == "mock"
    assert any(block["text"].startswith("NPWP :") for block in first["blocks"])
    assert all(0 <= block["confidence"] <= 1 for block in first["blocks"])
    assert first["blocks"][0]["bbox"] is not None
