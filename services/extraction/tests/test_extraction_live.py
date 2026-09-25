import io
import os

import pytest

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.testing import image_upload

from app.config import Settings
from app.ml.paddle import PaddleOcrEngine
from app.services.extraction_service import ExtractionService

OCR_URL = os.environ.get("EXTRACTION_OCR_URL")
pytestmark = pytest.mark.skipif(not OCR_URL, reason="EXTRACTION_OCR_URL tidak di-set; test live dilewati")


def synthetic_npwp_jpeg() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1000, 460), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, text in enumerate(
        [
            "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA",
            "NPWP : 12.345.678.9-012.345",
            "NAMA : BUDI SANTOSO",
            "NAMA BADAN : PT CIPTA KARYA MANDIRI",
        ]
    ):
        draw.text((40, 40 + i * 90), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


@pytest.fixture
def engine() -> PaddleOcrEngine:
    assert OCR_URL
    return PaddleOcrEngine(RemoteModelClient(OCR_URL, 60.0, name="extraction OCR model"))


async def test_live_engine_reads_synthetic_npwp(engine):
    result = await ExtractionService(engine, Settings(api_key="x", _env_file=None)).extract(
        "npwp.jpg", "image/jpeg", synthetic_npwp_jpeg()
    )
    assert result["engine"] == "paddle"
    assert result["model"]
    texts = [block["text"].replace(" ", "") for block in result["blocks"]]
    assert any(t.startswith("NPWP:12.345.678.9-012.345") for t in texts), texts
    assert all(0 <= block["confidence"] <= 1 and block["bbox"] for block in result["blocks"])


def test_live_http_extract_returns_model_and_blocks(client, auth, engine, use_engine):
    use_engine(engine)
    response = client.post(
        "/v1/extraction/extract", headers=auth, files=image_upload("npwp.jpg", synthetic_npwp_jpeg())
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["engine"] == "paddle"
    assert data["model"].startswith("PP-OCRv6")
    assert any("NPWP" in block["text"] for block in data["blocks"])
