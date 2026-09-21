import io
import os

import pytest
from PIL import Image, ImageDraw

from ocr_common.remote import RemoteModelClient
from src.core.config import Settings
from src.models.guardrails import RemoteGuardrailsModel
from src.services.guardrails_service import GuardrailsService

MODEL_URL = os.environ.get("GUARDRAILS_MODEL_URL")
MODEL_API_KEY = os.environ.get("GUARDRAILS_MODEL_API_KEY")
pytestmark = pytest.mark.skipif(not MODEL_URL, reason="GUARDRAILS_MODEL_URL tidak di-set; test live dilewati")


def synthetic_npwp_jpeg() -> bytes:
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(["KEMENTERIAN KEUANGAN REPUBLIK INDONESIA", "NPWP : 12.345.678.9-012.345"]):
        draw.text((40, 40 + i * 90), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


@pytest.fixture
async def model():
    assert MODEL_URL
    headers = {"X-API-Key": MODEL_API_KEY} if MODEL_API_KEY else None
    instance = RemoteGuardrailsModel(RemoteModelClient(MODEL_URL, 60.0, name="guardrails model", headers=headers))
    yield instance
    await instance.aclose()


async def test_live_model_answers_in_contract_shape(model):
    report = await GuardrailsService(model, Settings(api_key="x", _env_file=None)).check(
        "sample_npwp.jpg", "image/jpeg", synthetic_npwp_jpeg()
    )
    document = report["document"]
    assert document["verdict"] in {"accepted", "reject"}
    assert report["passed"] is (document["verdict"] == "accepted")
    assert document["n_pages"] == len(report["pages"]) == 1
    assert document["n_approve"] + document["n_reject"] == document["n_pages"]
    page = report["pages"][0]
    assert page["page_index"] == 0
    assert 0 <= page["proba_approve"] <= 1 and 0 <= page["proba_reject"] <= 1
