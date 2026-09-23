import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError
from ocr_common.testing import image_upload

from app.config import Settings
from app.dependencies import OCR_BACKENDS
from app.ml.remote import RemoteOcrEngine
from app.services.ekstraksi_service import EkstraksiService

JPEG = b"\xff\xd8fake-jpeg-bytes"
SAMPLE: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "remote_npwp_response.json").read_text(encoding="utf-8")
)


def _settings(**overrides) -> Settings:
    return Settings(api_key="x", _env_file=None, **overrides)


def _engine(handler) -> RemoteOcrEngine:
    return RemoteOcrEngine(
        RemoteModelClient(
            "http://ocr-model:8082",
            5.0,
            name="ekstraksi OCR model",
            headers={"X-API-Key": "dummy-key"},
            transport=httpx.MockTransport(handler),
        )
    )


def _reply(body, status_code=200):
    return lambda request: httpx.Response(status_code, json=body)


def _page(**overrides) -> dict[str, Any]:
    page = {
        "page_index": 0,
        "rec_texts": ["NPWP"],
        "rec_scores": [0.9],
        "rec_polys": [[[0, 0], [9, 0], [9, 9], [0, 9]]],
    }
    return {**page, **overrides}


async def test_request_follows_the_model_contract():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            method=request.method,
            path=request.url.path,
            api_key=request.headers.get("X-API-Key"),
            content_type=request.headers["Content-Type"],
            body=request.content,
        )
        return httpx.Response(200, json=SAMPLE)

    await _engine(handler).extract("sample_npwp.jpg", JPEG, "image/jpeg")

    assert (seen["method"], seen["path"]) == ("POST", "/v1/predict/json")
    assert seen["api_key"] == "dummy-key"
    assert seen["content_type"].startswith("multipart/form-data")
    assert b'name="file"; filename="sample_npwp.jpg"' in seen["body"]
    assert b"Content-Type: image/jpeg" in seen["body"]
    assert JPEG in seen["body"]


async def test_real_npwp_sample_maps_parallel_arrays_to_blocks():
    result = await _engine(_reply(SAMPLE)).extract("sample_npwp.jpg", JPEG, "image/jpeg")

    assert result["model"] is None
    blocks = result["blocks"]
    assert [b["text"] for b in blocks] == SAMPLE[0]["rec_texts"]
    assert all(b["page"] == 0 for b in blocks)

    npwp = blocks[2]
    assert npwp["text"] == "95.844.800.1-805.000"
    assert npwp["confidence"] == 1.0
    assert npwp["bbox"] == {"x1": 271, "y1": 358, "x2": 1347, "y2": 511}
    assert blocks[3] == {
        "text": "RAHMAT HIDAYAT",
        "confidence": 0.9931,
        "bbox": {"x1": 285, "y1": 557, "x2": 725, "y2": 601},
        "page": 0,
    }


async def test_service_output_for_real_sample():
    data = await EkstraksiService(_engine(_reply(SAMPLE)), _settings()).extract("sample_npwp.jpg", "image/jpeg", JPEG)
    assert data["engine"] == "remote"
    assert data["model"] is None
    assert data["full_text"].splitlines()[2:4] == ["95.844.800.1-805.000", "RAHMAT HIDAYAT"]
    assert len(data["blocks"]) == 10


async def test_multi_page_pdf_keeps_page_index():
    body = [_page(page_index=0, rec_texts=["HALAMAN SATU"]), _page(page_index=1, rec_texts=["HALAMAN DUA"])]
    blocks = (await _engine(_reply(body)).extract("a.pdf", b"%PDF", "application/pdf"))["blocks"]
    assert [(b["text"], b["page"]) for b in blocks] == [("HALAMAN SATU", 0), ("HALAMAN DUA", 1)]


async def test_page_index_defaults_to_position():
    body = [{k: v for k, v in _page().items() if k != "page_index"}] * 2
    blocks = (await _engine(_reply(body)).extract("a.pdf", b"%PDF", "application/pdf"))["blocks"]
    assert [b["page"] for b in blocks] == [0, 1]


async def test_blank_texts_are_skipped_without_shifting_scores():
    body = [_page(rec_texts=["  ", "NAMA"], rec_scores=[0.1, 0.8], rec_polys=[None, None])]
    blocks = (await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg"))["blocks"]
    assert blocks == [{"text": "NAMA", "confidence": 0.8, "bbox": None, "page": 0}]


async def test_missing_rec_polys_gives_blocks_without_bbox():
    body = [{"page_index": 0, "rec_texts": ["NPWP"], "rec_scores": [0.9]}]
    blocks = (await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg"))["blocks"]
    assert blocks == [{"text": "NPWP", "confidence": 0.9, "bbox": None, "page": 0}]


async def test_no_pages_is_empty_blocks_not_an_error():
    assert (await _engine(_reply([])).extract("a.jpg", JPEG, "image/jpeg"))["blocks"] == []


async def test_envelope_with_data_list_is_accepted():
    body = {"status_code": 200, "message": "OK", "data": SAMPLE}
    blocks = (await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg"))["blocks"]
    assert len(blocks) == 10


CURRENT_CONTRACT = {
    "models": {"detection": "PP-OCRv6_medium_det", "recognition": "PP-OCRv6_medium_rec"},
    "num_pages": 1,
    "pages": SAMPLE,
    "n_boxes": 10,
    "avg_doc_score": 0.9813,
    "min_doc_score": 0.9438,
}


async def test_current_contract_with_pages_and_models_is_accepted():
    """The ML team's OCR service of 23 Sep 2026 answers {models, num_pages, pages, n_boxes, avg/min_doc_score}."""
    result = await _engine(_reply(CURRENT_CONTRACT)).extract("a.jpg", JPEG, "image/jpeg")
    assert result["model"] == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"
    assert len(result["blocks"]) == 10
    assert result["blocks"][2]["text"] == "95.844.800.1-805.000"


async def test_no_pages_in_the_current_contract_is_empty_blocks():
    body = {**CURRENT_CONTRACT, "num_pages": 0, "pages": [], "n_boxes": 0, "avg_doc_score": None}
    assert (await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg"))["blocks"] == []


async def test_ocr_params_are_sent_as_form_fields():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json=CURRENT_CONTRACT)

    engine = RemoteOcrEngine(
        RemoteModelClient(
            "http://ocr-model:8082", 5.0, name="ekstraksi OCR model", transport=httpx.MockTransport(handler)
        ),
        params={"document_type": "npwp", "scale": 3.5},
    )
    await engine.extract("a.jpg", JPEG, "image/jpeg")
    assert b'name="document_type"\r\n\r\nnpwp' in seen["body"]
    assert b'name="scale"\r\n\r\n3.5' in seen["body"]
    assert b'name="file"; filename="a.jpg"' in seen["body"]


async def test_without_params_only_the_file_is_sent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json=SAMPLE)

    await _engine(handler).extract("a.jpg", JPEG, "image/jpeg")
    assert seen["body"].count(b"Content-Disposition") == 1


@pytest.mark.parametrize(
    "body",
    [
        {"status_code": 200, "message": "OK"},
        ["not-a-page"],
        [{"page_index": 0, "rec_scores": [0.9]}],
        [_page(rec_texts=["A", "B"], rec_scores=[0.9])],
        [_page(rec_texts=["A", "B"], rec_scores=[0.9, 0.8], rec_polys=[[[0, 0]]])],
        [_page(page_index="dua")],
    ],
)
async def test_misaligned_or_unexpected_response_is_500_not_a_guess(body):
    with pytest.raises(ServiceError) as exc:
        await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg")
    assert exc.value.status_code == 500
    assert exc.value.message == "ekstraksi OCR model returned an unexpected response"


async def test_model_error_status_becomes_500_with_detail():
    engine = _engine(_reply({"status_code": 401, "message": "Invalid API key"}, status_code=401))
    with pytest.raises(ServiceError) as exc:
        await engine.extract("a.jpg", JPEG, "image/jpeg")
    assert (exc.value.status_code, exc.value.message) == (500, "ekstraksi OCR model error (401): Invalid API key")


async def test_unreachable_model_is_503_and_slow_model_is_504():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ServiceError) as exc:
        await _engine(refuse).extract("a.jpg", JPEG, "image/jpeg")
    assert (exc.value.status_code, exc.value.message) == (503, "ekstraksi OCR model is unavailable")

    with pytest.raises(ServiceError) as exc:
        await _engine(stall).extract("a.jpg", JPEG, "image/jpeg")
    assert (exc.value.status_code, exc.value.message) == (504, "ekstraksi OCR model timed out after 5.0s")


def test_remote_backend_requires_url():
    with pytest.raises(RuntimeError, match="EKSTRAKSI_OCR_URL is required when EKSTRAKSI_BACKEND=remote"):
        OCR_BACKENDS["remote"](_settings(ekstraksi_backend="remote"))


async def test_remote_backend_is_built_from_settings():
    engine = OCR_BACKENDS["remote"](
        _settings(
            ekstraksi_backend="remote",
            ekstraksi_ocr_url="http://localhost:8082/",
            ekstraksi_ocr_api_key="dummy-key",
            ekstraksi_ocr_params={"document_type": "npwp"},
        )
    )
    try:
        assert isinstance(engine, RemoteOcrEngine)
        assert engine._client.base_url == "http://localhost:8082"
        assert engine._client._client.headers["X-API-Key"] == "dummy-key"
        assert engine._params == {"document_type": "npwp"}
    finally:
        await engine.aclose()


def test_http_extract_with_remote_backend(client, auth, use_engine):
    use_engine(_engine(_reply(SAMPLE)))
    response = client.post("/v1/ekstraksi/extract", headers=auth, files=image_upload("sample_npwp.jpg", JPEG))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["engine"] == "remote"
    assert data["blocks"][2]["text"] == "95.844.800.1-805.000"
