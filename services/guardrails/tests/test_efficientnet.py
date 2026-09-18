"""
Backend efficientnet dengan checkpoint sungguhan. Dilewati kalau torch tidak
terpasang atau weights/best_model.pt tidak ada (bobot tidak ikut git).
"""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best_model.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.is_file(), reason="weights/best_model.pt tidak ada")


@pytest.fixture(scope="module")
def classifier():
    pytest.importorskip("torch")
    from src.models.guardrails import EfficientNetPageClassifier

    return EfficientNetPageClassifier(str(WEIGHTS))


def _page(text: str) -> Image.Image:
    image = Image.new("RGB", (800, 500), "white")
    ImageDraw.Draw(image).text((40, 40), text, fill="black")
    return image


def test_checkpoint_metadata_is_read(classifier):
    assert classifier.class_names == ["accepted", "reject"]
    assert classifier.image_size == 224
    assert classifier.reject_threshold == 0.5
    assert classifier.metadata["architecture"] == "efficientnet_b0"
    assert 0 < classifier.metadata["val_macro_f1"] <= 1


def test_probabilities_are_valid_per_page(classifier):
    predictions = classifier.classify("x.jpg", [_page("NPWP : 12.345.678.9-012.345"), _page("halaman dua")])
    assert len(predictions) == 2
    for proba_approve, proba_reject in predictions:
        assert 0 <= proba_approve <= 1 and 0 <= proba_reject <= 1
        assert abs(proba_approve + proba_reject - 1) < 1e-3


def test_empty_page_list_returns_no_predictions(classifier):
    assert classifier.classify("x.jpg", []) == []


def test_http_with_real_model(client, auth, classifier, monkeypatch):
    from ocr_common.testing import image_upload

    monkeypatch.setattr("src.api.v1.guardrails.get_page_classifier", lambda: classifier)
    buffer = io.BytesIO()
    _page("NPWP : 12.345.678.9-012.345").save(buffer, format="JPEG")
    response = client.post(
        "/v1/guardrails/check",
        data={"request_id": "OCR_real"},
        files=image_upload("npwp.jpg", buffer.getvalue()),
        headers=auth,
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["document"]["verdict"] in ("accepted", "reject")
    assert data["document"]["n_pages"] == 1
    assert set(data["pages"][0]) == {"page_index", "proba_approve", "proba_reject", "verdict"}
