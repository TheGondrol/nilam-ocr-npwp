import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best_model.pt"
needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason="weights/best_model.pt tidak ada")


@pytest.fixture(scope="module")
def classifier():
    pytest.importorskip("torch")
    from app.ml.efficientnet import EfficientNetPageClassifier

    return EfficientNetPageClassifier(str(WEIGHTS))


def _page(text: str) -> Image.Image:
    image = Image.new("RGB", (800, 500), "white")
    ImageDraw.Draw(image).text((40, 40), text, fill="black")
    return image


def _checkpoint(tmp_path: Path, class_names: list[str]) -> str:
    """A checkpoint with random weights and the given class order, in the ML team's file layout."""
    import torch
    from torchvision.models import efficientnet_b0

    torch.manual_seed(0)
    path = tmp_path / f"{'_'.join(class_names)}.pt"
    torch.save(
        {
            "model_state_dict": efficientnet_b0(weights=None, num_classes=2).state_dict(),
            "class_names": class_names,
            "image_size": 64,
            "epoch": 1,
            "val_macro_f1": 0.5,
            "reject_threshold": 0.5,
        },
        path,
    )
    return str(path)


def test_class_order_is_read_from_the_checkpoint_not_assumed(tmp_path):
    pytest.importorskip("torch")
    from app.ml.efficientnet import EfficientNetPageClassifier

    # Same weights, opposite class order: the same output index must be reported under the opposite name.
    accepted_first = EfficientNetPageClassifier(_checkpoint(tmp_path, ["accepted", "reject"]))
    reject_first = EfficientNetPageClassifier(_checkpoint(tmp_path, ["reject", "accepted"]))
    page = _page("NPWP")
    [(approve_a, reject_a)] = accepted_first.classify("x.jpg", [page])
    [(approve_b, reject_b)] = reject_first.classify("x.jpg", [page])
    assert (approve_a, reject_a) == (reject_b, approve_b)
    assert reject_first.class_names == ["reject", "accepted"]


def test_a_checkpoint_with_other_classes_is_refused(tmp_path):
    pytest.importorskip("torch")
    from app.ml.efficientnet import EfficientNetPageClassifier

    with pytest.raises(RuntimeError, match="Unexpected class_names"):
        EfficientNetPageClassifier(_checkpoint(tmp_path, ["ktp", "npwp"]))


@needs_weights
def test_checkpoint_metadata_is_read(classifier):
    assert sorted(classifier.class_names) == ["accepted", "reject"]
    assert classifier.image_size == 224
    assert classifier.reject_threshold == 0.5
    assert classifier.metadata["architecture"] == "efficientnet_b0"
    assert 0 < classifier.metadata["val_macro_f1"] <= 1


@needs_weights
def test_probabilities_are_valid_per_page(classifier):
    predictions = classifier.classify("x.jpg", [_page("NPWP : 12.345.678.9-012.345"), _page("halaman dua")])
    assert len(predictions) == 2
    for proba_approve, proba_reject in predictions:
        assert 0 <= proba_approve <= 1 and 0 <= proba_reject <= 1
        assert abs(proba_approve + proba_reject - 1) < 1e-3


@needs_weights
def test_a_blank_page_is_rejected(classifier):
    """Guards the class order against the real weights: a white page is not an NPWP card. With the
    23 Sep 2026 checkpoint read in the wrong order it would be accepted at 0.98."""
    [(proba_approve, proba_reject)] = classifier.classify("x.jpg", [Image.new("RGB", (800, 500), "white")])
    assert proba_reject > 0.9, (proba_approve, proba_reject)


@needs_weights
def test_empty_page_list_returns_no_predictions(classifier):
    assert classifier.classify("x.jpg", []) == []


@needs_weights
def test_http_with_real_model(client, auth, classifier, use_classifier):
    from ocr_common.testing import image_upload

    use_classifier(classifier)
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
