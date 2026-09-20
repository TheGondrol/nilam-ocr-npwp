"""
Backend `remote`: service model ekstraksi milik ML engineer.

    POST {EKSTRAKSI_OCR_URL}/v1/predict/json   header X-API-Key, multipart `file`
    -> [{"page_index", "rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}, ...]

Service model diganti httpx.MockTransport yang membalas response ASLI dari
kontraknya (tests/fixtures/remote_npwp_response.json: kartu NPWP sungguhan),
tanpa jaringan. Test ke service sungguhan: test_ekstraksi_remote_live.py.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ocr_common.errors import ServiceError
from ocr_common.remote import RemoteModelClient
from ocr_common.testing import image_upload
from src.core.config import Settings
from src.models.ekstraksi import OCR_BACKENDS, RemoteOcrEngine
from src.services.ekstraksi_service import EkstraksiService

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

    assert result["model"] is None  # response tidak membawa identitas model
    blocks = result["blocks"]
    assert [b["text"] for b in blocks] == SAMPLE[0]["rec_texts"]  # urutan baca dipertahankan
    assert all(b["page"] == 0 for b in blocks)

    # Indeks i dari ketiga array adalah satu baris: teks, skor, dan poly-nya tetap berpasangan.
    npwp = blocks[2]
    assert npwp["text"] == "95.844.800.1-805.000"
    assert npwp["confidence"] == 1.0  # 0.99996 dibulatkan 4 desimal
    assert npwp["bbox"] == {"x1": 271, "y1": 358, "x2": 1347, "y2": 511}  # kotak tegak dari poly miring
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
    """Berkas tanpa teks: structuring yang akan menjawab "No text lines to structure"."""
    assert (await _engine(_reply([])).extract("a.jpg", JPEG, "image/jpeg"))["blocks"] == []


async def test_envelope_with_data_list_is_accepted():
    body = {"status_code": 200, "message": "OK", "data": SAMPLE}
    blocks = (await _engine(_reply(body)).extract("a.jpg", JPEG, "image/jpeg"))["blocks"]
    assert len(blocks) == 10


@pytest.mark.parametrize(
    "body",
    [
        {"status_code": 200, "message": "OK"},  # objek tanpa data
        ["not-a-page"],
        [{"page_index": 0, "rec_scores": [0.9]}],  # tanpa rec_texts
        [_page(rec_texts=["A", "B"], rec_scores=[0.9])],  # skor tidak sejajar dengan teks
        [_page(rec_texts=["A", "B"], rec_scores=[0.9, 0.8], rec_polys=[[[0, 0]]])],  # poly tidak sejajar
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
    # 401 dari service MODEL adalah salah konfigurasi kita, bukan salah pemanggil kita.
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
            ekstraksi_backend="remote", ekstraksi_ocr_url="http://localhost:8082/", ekstraksi_ocr_api_key="dummy-key"
        )
    )
    try:
        assert isinstance(engine, RemoteOcrEngine)
        assert engine._client.base_url == "http://localhost:8082"
        assert engine._client._client.headers["X-API-Key"] == "dummy-key"
    finally:
        await engine.aclose()


def test_http_extract_with_remote_backend(client, auth, monkeypatch):
    """Kontrak KITA tidak berubah: /v1/ekstraksi/extract tetap mengembalikan blocks + envelope."""
    monkeypatch.setattr("src.api.v1.ekstraksi.get_ocr_engine", lambda: _engine(_reply(SAMPLE)))
    response = client.post("/v1/ekstraksi/extract", headers=auth, files=image_upload("sample_npwp.jpg", JPEG))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["engine"] == "remote"
    assert data["blocks"][2]["text"] == "95.844.800.1-805.000"
