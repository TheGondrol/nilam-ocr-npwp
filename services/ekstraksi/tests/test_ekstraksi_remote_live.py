"""
Test integrasi ke service model ekstraksi sungguhan (backend `remote`,
POST /v1/predict/json). Dilewati kecuali env EKSTRAKSI_REMOTE_URL di-set, mis.:

    cd services/ekstraksi
    EKSTRAKSI_REMOTE_URL=http://localhost:8082 EKSTRAKSI_REMOTE_API_KEY=dummy-key \
        python -m pytest tests/test_ekstraksi_remote_live.py -q

Env-nya sengaja bukan EKSTRAKSI_OCR_URL: variabel itu sudah dipakai
test_ekstraksi_live.py untuk API lama (`paddle`, POST /ocr), dan kedua test
tidak boleh saling menyalakan ke server yang salah.
"""

import os

import pytest

from ocr_common.remote import RemoteModelClient
from src.core.config import Settings
from src.models.ekstraksi import RemoteOcrEngine
from src.services.ekstraksi_service import EkstraksiService
from tests.test_ekstraksi_live import synthetic_npwp_jpeg

MODEL_URL = os.environ.get("EKSTRAKSI_REMOTE_URL")
MODEL_API_KEY = os.environ.get("EKSTRAKSI_REMOTE_API_KEY")
pytestmark = pytest.mark.skipif(not MODEL_URL, reason="EKSTRAKSI_REMOTE_URL tidak di-set; test live dilewati")


@pytest.fixture
async def engine():
    assert MODEL_URL  # dijamin oleh skipif di atas
    headers = {"X-API-Key": MODEL_API_KEY} if MODEL_API_KEY else None
    instance = RemoteOcrEngine(RemoteModelClient(MODEL_URL, 60.0, name="ekstraksi OCR model", headers=headers))
    yield instance
    await instance.aclose()


async def test_live_engine_reads_synthetic_npwp(engine):
    result = await EkstraksiService(engine, Settings(api_key="x", _env_file=None)).extract(
        "npwp.jpg", "image/jpeg", synthetic_npwp_jpeg()
    )
    assert result["engine"] == "remote"
    texts = [block["text"].replace(" ", "") for block in result["blocks"]]
    assert any("12.345.678.9-012.345" in t for t in texts), texts
    assert all(0 <= block["confidence"] <= 1 and block["bbox"] and block["page"] == 0 for block in result["blocks"])
