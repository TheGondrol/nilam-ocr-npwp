import httpx

from ocr_common.remote import RemoteModelClient
from ocr_common.testing import image_upload
from src.core.config import Settings
from src.models.ekstraksi import MockOcrEngine, PaddleOcrEngine
from src.services.ekstraksi_service import EkstraksiService

SETTINGS = Settings(api_key="x", _env_file=None)

PADDLE_RESPONSE = {
    "models": {
        "detection": "PP-OCRv6_medium_det",
        "recognition": "PP-OCRv6_medium_rec",
        "pipeline": "PaddleOCR",
        "device": "gpu",
    },
    "num_pages": 1,
    "pages": [
        {
            "page_index": 0,
            "width": 1000,
            "height": 620,
            "texts": [
                {
                    "text": "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA",
                    "score": 0.9998849034309387,
                    "poly": [[0, 4], [929, 9], [928, 67], [0, 62]],
                },
                {
                    "text": "NPWP:12.345.678.9-012.345",
                    "score": 0.9967303276062012,
                    "poly": [[0, 162], [444, 166], [443, 214], [0, 210]],
                },
                {
                    "text": "NAMA : BUDI SANTOSO",
                    "score": 0.9885978698730469,
                    "poly": [[0, 240], [363, 242], [363, 291], [0, 288]],
                },
                {"text": "   ", "score": 0.5, "poly": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            ],
        }
    ],
    "filename": "npwp_synth.jpg",
}


class StubEngine:
    name = "stub"

    async def extract(self, filename, content, content_type=None):
        return {
            "blocks": [
                {"text": "A", "confidence": 0.9, "bbox": None, "page": 0},
                {"text": "B", "confidence": 0.8, "bbox": None, "page": 0},
            ],
            "model": "stub-v1",
        }


def _paddle_engine(handler) -> PaddleOcrEngine:
    client = RemoteModelClient(
        "http://ocr.test", 5.0, name="ekstraksi OCR model", transport=httpx.MockTransport(handler)
    )
    return PaddleOcrEngine(client)


async def test_service_wraps_engine_output_with_metadata():
    result = await EkstraksiService(StubEngine(), SETTINGS).extract("x.jpg", "image/jpeg", b"abc")
    assert result["engine"] == "stub"
    assert result["model"] == "stub-v1"
    assert result["full_text"] == "A\nB"
    assert result["elapsed_ms"] >= 0


async def test_paddle_engine_posts_multipart_and_maps_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(200, json=PADDLE_RESPONSE)

    result = await _paddle_engine(handler).extract("npwp.jpg", b"\xff\xd8jpeg", "image/jpeg")

    assert seen["url"] == "http://ocr.test/ocr"
    assert seen["content_type"].startswith("multipart/form-data")
    assert b'name="file"; filename="npwp.jpg"' in seen["body"]
    assert b"Content-Type: image/jpeg" in seen["body"]
    assert b"\xff\xd8jpeg" in seen["body"]

    assert result["model"] == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"
    assert [b["text"] for b in result["blocks"]] == [
        "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA",
        "NPWP:12.345.678.9-012.345",
        "NAMA : BUDI SANTOSO",
    ]
    npwp = result["blocks"][1]
    assert npwp["confidence"] == 0.9967
    assert npwp["bbox"] == {"x1": 0, "y1": 162, "x2": 444, "y2": 214}
    assert npwp["page"] == 0


async def test_paddle_engine_accepts_split_list_response():
    engine = _paddle_engine(lambda request: httpx.Response(200, json=[PADDLE_RESPONSE]))
    result = await engine.extract("npwp.jpg", b"x", "image/jpeg")
    assert len(result["blocks"]) == 3
    assert result["model"] == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"


async def test_paddle_engine_returns_no_blocks_for_undecodable_file():
    empty = {"models": PADDLE_RESPONSE["models"], "num_pages": 0, "pages": [], "filename": "t.txt"}
    engine = _paddle_engine(lambda request: httpx.Response(200, json=empty))
    result = await engine.extract("t.txt", b"hello", "image/jpeg")
    assert result["blocks"] == []
    assert result["model"] == "PP-OCRv6_medium_det+PP-OCRv6_medium_rec"


async def test_paddle_engine_defaults_filename_and_content_type():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json=PADDLE_RESPONSE)

    await _paddle_engine(handler).extract("", b"x", None)
    assert b'filename="upload"' in seen["body"]
    assert b"Content-Type: image/jpeg" in seen["body"]


async def test_mock_engine_shape_matches_contract():
    result = await MockOcrEngine().extract("npwp.jpg", b"seed", "image/jpeg")
    assert result["model"] is None
    assert all(set(block) == {"text", "confidence", "bbox", "page"} for block in result["blocks"])


def test_http_extract_is_deterministic_per_content(client, auth):
    first = client.post("/v1/ekstraksi/extract", files=image_upload(content=b"same"), headers=auth).json()["data"]
    second = client.post("/v1/ekstraksi/extract", files=image_upload(content=b"same"), headers=auth).json()["data"]
    assert first["blocks"] == second["blocks"]
    assert first["engine"] == "mock"
    assert first["model"] is None
    assert any(block["text"].startswith("NPWP :") for block in first["blocks"])
    assert all(0 <= block["confidence"] <= 1 for block in first["blocks"])
    assert first["blocks"][0]["bbox"] is not None
    assert first["blocks"][0]["page"] == 0


def test_http_extract_accepts_pdf(client, auth):
    response = client.post(
        "/v1/ekstraksi/extract", files=image_upload("npwp.pdf", b"%PDF-1.4 fake", "application/pdf"), headers=auth
    )
    assert response.status_code == 200


def test_http_extract_empty_file_returns_400(client, auth):
    response = client.post("/v1/ekstraksi/extract", files=image_upload(content=b""), headers=auth)
    assert response.status_code == 400
    assert response.json()["message"] == "Uploaded file is empty"


def test_http_file_url_is_fetched_by_service(client, auth, monkeypatch):
    async def fake_fetch(url, *, limit, timeout=10.0):
        assert url == "http://minio.local/bucket/npwp.jpg"
        return b"\xff\xd8bytes-from-url", "npwp.jpg", "image/jpeg"

    monkeypatch.setattr("ocr_common.intake.fetch", fake_fetch)
    response = client.post(
        "/v1/ekstraksi/extract", data={"file_url": "http://minio.local/bucket/npwp.jpg"}, headers=auth
    )
    assert response.status_code == 200
    assert response.json()["data"]["engine"] == "mock"


def test_http_extract_surfaces_model_unavailable_as_503(client, auth, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr("src.api.v1.ekstraksi.get_ocr_engine", lambda: _paddle_engine(handler))
    response = client.post("/v1/ekstraksi/extract", files=image_upload(), headers=auth)
    assert response.status_code == 503
    body = response.json()
    assert body["status_desc"] == "Service Unavailable"
    assert body["message"] == "ekstraksi OCR model is unavailable"
