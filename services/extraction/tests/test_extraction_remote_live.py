import os

import pytest

from ocr_common.clients.remote import RemoteModelClient

from app.config import Settings
from app.ml.remote import RemoteOcrEngine
from app.services.extraction_service import ExtractionService
from tests.test_extraction_live import synthetic_npwp_jpeg

MODEL_URL = os.environ.get("EXTRACTION_REMOTE_URL")
MODEL_API_KEY = os.environ.get("EXTRACTION_REMOTE_API_KEY")
pytestmark = pytest.mark.skipif(not MODEL_URL, reason="EXTRACTION_REMOTE_URL tidak di-set; test live dilewati")


@pytest.fixture
async def engine():
    assert MODEL_URL
    headers = {"X-API-Key": MODEL_API_KEY} if MODEL_API_KEY else None
    instance = RemoteOcrEngine(RemoteModelClient(MODEL_URL, 60.0, name="extraction OCR model", headers=headers))
    yield instance
    await instance.aclose()


async def test_live_engine_reads_synthetic_npwp(engine):
    result = await ExtractionService(engine, Settings(api_key="x", _env_file=None)).extract(
        "npwp.jpg", "image/jpeg", synthetic_npwp_jpeg()
    )
    assert result["engine"] == "remote"
    texts = [block["text"].replace(" ", "") for block in result["blocks"]]
    assert any("12.345.678.9-012.345" in t for t in texts), texts
    assert all(0 <= block["confidence"] <= 1 and block["bbox"] and block["page"] == 0 for block in result["blocks"])
