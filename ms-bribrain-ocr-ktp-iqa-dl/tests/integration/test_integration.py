"""Integration tests for the IQA-DL API pipeline."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, AsyncMock, call

import httpx
import pytest
import torch
from PIL import Image

from src.api.routes import register_exception_handlers, router, set_executor
from src.schemas.api_schema import Crop
from src.services.quality_service import set_model_globals

from fastapi import FastAPI
from src.middleware.add_requestid import RequestIdMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_bytes(width=200, height=200):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_crop_dict(x1, y1, x2, y2, text="t"):
    return {
        "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "text": text,
        "confidence": 0.9,
    }


def _mock_transform(img):
    """Mock transform that returns a 3x64x320 tensor without using torchvision."""
    return torch.randn(3, 64, 320)


def _create_app_with_model(model_output):
    """Create a test app with a mock model that returns specified output."""
    import src.services.quality_service as qs

    orig_qs = (qs._model, qs._device, qs._transform)

    mock_model = MagicMock()
    mock_model.side_effect = lambda x: torch.tensor([model_output] * x.shape[0])

    device = torch.device("cpu")
    set_model_globals(mock_model, device, _mock_transform)

    executor = ThreadPoolExecutor(max_workers=2)
    set_executor(executor)

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
    app.include_router(router)

    return app, (orig_qs, executor)


def _restore_globals(orig):
    import src.services.quality_service as qs
    orig_qs, executor = orig
    qs._model, qs._device, qs._transform = orig_qs
    executor.shutdown(wait=False)
    set_executor(None)  # type: ignore[arg-type]


async def _post_filter(app, crops_data, image_bytes=None, headers=None):
    """Helper to POST /filter with given crops and image."""
    if image_bytes is None:
        image_bytes = _image_bytes()
    if headers is None:
        headers = {"X-API-Key": "test"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v1/ocr_qualitydl",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"crops": json.dumps(crops_data)},
            headers=headers,
        )


@pytest.fixture(autouse=True)
def _set_api_key():
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


@pytest.fixture(autouse=True)
def _mock_threshold_provider():
    """Mirror the patched quality_service config through a stub ThresholdProvider."""
    import src.services.quality_service as qs

    def _get(key):
        if key == "bad_crop_threshold":
            return qs.config.bad_crop_threshold
        if key == "confidence_threshold":
            return qs.config.get("prediction.confidence_threshold", 0.67)
        raise KeyError(key)

    provider = MagicMock()
    provider.get.side_effect = _get
    with patch("src.services.quality_service.get_provider", return_value=provider):
        yield


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------

class TestClassificationPipeline:
    """End-to-end tests for the /filter classification pipeline."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_good_classification(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            crops = [_make_crop_dict(0, 0, 150, 30), _make_crop_dict(0, 50, 120, 70)]
            resp = await _post_filter(app, crops)

            content = resp.json()
            assert content["status_code"] == 200
            assert content["data"]["label"] == "good"
            assert content["data"]["num_bad"] == 0
            assert len(content["data"]["predictions"]) == 2

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_bad_classification(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.9, 0.1])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 1
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            resp = await _post_filter(app, [_make_crop_dict(0, 0, 150, 30)])

            content = resp.json()
            assert content["status_code"] == 200
            assert content["data"]["label"] == "bad"
            assert content["data"]["num_bad"] >= 1

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_empty_crops_returns_good(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.5, 0.5])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4

            resp = await _post_filter(app, [])

            content = resp.json()
            assert content["status_code"] == 200
            assert content["data"]["label"] == "good"
            assert content["data"]["total_crops"] == 0

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_mixed_crops_threshold_good(self, mock_config, _log):
        """3 bad + 1 good with threshold=4 -> overall good (below threshold)."""
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1

        import src.services.quality_service as qs
        orig = (qs._model, qs._device, qs._transform)

        # Model returns bad for first 3 crops, good for 4th
        call_count = [0]
        def _model_fn(x):
            batch_size = x.shape[0]
            results = []
            for _ in range(batch_size):
                if call_count[0] < 3:
                    results.append([0.9, 0.1])  # bad
                else:
                    results.append([0.1, 0.9])  # good
                call_count[0] += 1
            return torch.tensor(results)

        mock_model = MagicMock(side_effect=_model_fn)
        qs._model = mock_model
        qs._device = torch.device("cpu")
        qs._transform = _mock_transform

        executor = ThreadPoolExecutor(max_workers=2)
        set_executor(executor)

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
        app.include_router(router)

        try:
            with patch("src.services.quality_service.config") as qs_config:
                qs_config.min_width_ratio = 1.0
                qs_config.min_width = 0
                qs_config.bad_crop_threshold = 4  # need >= 4 bad to be "bad"
                qs_config.debug_mode = False
                qs_config.get.return_value = 0.67

                crops = [_make_crop_dict(0, 0, 150, 30 + i * 5) for i in range(4)]
                resp = await _post_filter(app, crops)

                content = resp.json()
                assert content["status_code"] == 200
                assert content["data"]["label"] == "good"
                assert content["data"]["num_bad"] == 3
        finally:
            qs._model, qs._device, qs._transform = orig
            executor.shutdown(wait=False)
            set_executor(None)  # type: ignore[arg-type]

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_tall_crops_filtered_out(self, mock_config, _log):
        """Crops taller than wide should be filtered, leaving 0 processed."""
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.9, 0.1])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 2.0  # strict: need width >= 2x height
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 1
            qs_config.debug_mode = False

            # All crops are square (ratio=1.0), below min_width_ratio=2.0
            crops = [_make_crop_dict(0, 0, 50, 50), _make_crop_dict(60, 0, 110, 50)]
            resp = await _post_filter(app, crops)

            content = resp.json()
            assert content["status_code"] == 200
            assert content["data"]["label"] == "good"  # no crops to judge = good
            assert content["data"]["num_filtered"] == 0
            assert content["data"]["total_crops"] == 2

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_large_batch_of_crops(self, mock_config, _log):
        """Verify pipeline handles a batch of 20 crops."""
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 20
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            crops = [_make_crop_dict(0, i * 8, 150, i * 8 + 7) for i in range(20)]
            resp = await _post_filter(app, crops, image_bytes=_image_bytes(300, 300))

            content = resp.json()
            assert content["status_code"] == 200
            assert content["data"]["total_crops"] == 20
            assert len(content["data"]["predictions"]) <= 20

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_debug_mode_includes_text_and_confidence(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = True
            qs_config.get.return_value = 0.67

            crops = [_make_crop_dict(0, 0, 150, 30, text="NAMA")]
            resp = await _post_filter(app, crops)

            content = resp.json()
            assert content["status_code"] == 200
            preds = content["data"]["predictions"]
            assert preds[0]["text"] == "NAMA"
            assert preds[0]["confidence"] == 0.9

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_debug_mode_off_hides_text(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            crops = [_make_crop_dict(0, 0, 150, 30, text="NAMA")]
            resp = await _post_filter(app, crops)

            content = resp.json()
            preds = content["data"]["predictions"]
            assert preds[0]["text"] is None
            assert preds[0]["confidence"] is None

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_corrupted_image_returns_400(self, mock_config, _log):
        """An undecodable image is a client error → 400 INVALID_INPUT (not 500)."""
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4

            crops = [_make_crop_dict(0, 0, 150, 30)]
            resp = await _post_filter(app, crops, image_bytes=b"not-a-real-image")

            content = resp.json()
            assert content["status_code"] == 400
            assert content["error_code"] == "INVALID_INPUT"

        _restore_globals(orig)


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class TestResponseEnvelope:
    """Verify every response has the standard envelope fields."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_success_envelope_structure(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            resp = await _post_filter(app, [_make_crop_dict(0, 0, 150, 30)])

        content = resp.json()
        expected_keys = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected_keys == set(content.keys())
        assert content["status_desc"] == "OK"
        assert content["error_code"] is None
        assert content["errors"] is None
        assert len(content["request_id"]) == 36

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=None)
    async def test_error_envelope_structure(self, _model, _log):
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
        app.include_router(router)

        resp = await _post_filter(app, [])
        content = resp.json()
        expected_keys = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected_keys == set(content.keys())
        assert content["status_code"] == 503
        assert content["error_code"] == "MODEL_NOT_LOADED"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Integration tests for error paths through the full stack."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=None)
    async def test_model_not_loaded_503(self, _model, _log):
        app = FastAPI()
        app.include_router(router)

        resp = await _post_filter(app, [])
        content = resp.json()
        assert content["status_code"] == 503
        assert content["error_code"] == "MODEL_NOT_LOADED"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_file_type_400(self, _model, _log):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.txt", b"hello", "text/plain")},
                data={"crops": "[]"},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_FILE_TYPE"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_crops_json_400(self, _model, _log):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "{not valid json"},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_crops_not_a_list_400(self, _model, _log):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": json.dumps({"not": "a list"})},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_invalid_crop_schema_400(self, _model, _log):
        """Crop with missing bbox should return 400."""
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": json.dumps([{"text": "no bbox"}])},
                headers={"X-API-Key": "test"},
            )
        content = resp.json()
        assert content["status_code"] == 400
        assert content["error_code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    """Integration tests for API key authentication."""

    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_missing_api_key_rejected(self, _model):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "[]"},
                # No X-API-Key header
            )
        assert resp.status_code in (401, 403)

    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_wrong_api_key_returns_401(self, _model):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "[]"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    async def test_health_does_not_require_api_key(self):
        """Health and root endpoints should not require auth."""
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            root = await client.get("/")
        assert health.status_code == 200
        assert root.status_code == 200


# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------

class TestDatabaseLogging:
    """Verify insert_log is called correctly during requests."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_insert_log_called_on_success(self, mock_config, mock_log):
        mock_config.log_to_database = True
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            await _post_filter(app, [_make_crop_dict(0, 0, 150, 30)])

        mock_log.assert_awaited_once()
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["response_code"] == 200
        assert call_kwargs["error_message"] == ""
        assert call_kwargs["result"] is not None
        assert call_kwargs["processing_time"] > 0

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_insert_log_called_on_error(self, mock_config, mock_log):
        mock_config.log_to_database = True
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4

            await _post_filter(app, [_make_crop_dict(0, 0, 150, 30)], image_bytes=b"broken")

        mock_log.assert_awaited_once()
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["response_code"] == 400
        assert call_kwargs["error_message"] != ""

        _restore_globals(orig)

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_insert_log_not_called_when_disabled(self, mock_config, mock_log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1
        app, orig = _create_app_with_model([0.1, 0.9])

        with patch("src.services.quality_service.config") as qs_config:
            qs_config.min_width_ratio = 1.0
            qs_config.min_width = 0
            qs_config.bad_crop_threshold = 4
            qs_config.debug_mode = False
            qs_config.get.return_value = 0.67

            await _post_filter(app, [_make_crop_dict(0, 0, 150, 30)])

        mock_log.assert_not_awaited()
        _restore_globals(orig)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TestMiddlewareIntegration:
    """Verify middleware works correctly through the full stack."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    async def test_request_id_in_response_header(self, _log):
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 36

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    async def test_different_requests_get_different_ids(self, _log):
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get("/health")
            r2 = await client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    async def test_request_id_in_envelope(self, mock_config, _log):
        mock_config.log_to_database = False
        mock_config.get.return_value = "1.0.0"
        mock_config.max_image_size_mb = 1

        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
        app.include_router(router)

        with patch("src.api.routes.get_model", return_value=None):
            resp = await _post_filter(app, [])

        content = resp.json()
        assert content["request_id"] == resp.headers["x-request-id"]


# ---------------------------------------------------------------------------
# Health & root endpoints
# ---------------------------------------------------------------------------

class TestInfoEndpoints:
    @patch("src.api.routes._check_database", return_value={"status": "up"})
    async def test_health_when_model_loaded(self, _db):
        import src.services.quality_service as qs
        orig = qs._model
        qs._model = MagicMock()

        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

        qs._model = orig

    async def test_health_when_model_not_loaded(self):
        import src.services.quality_service as qs
        orig = qs._model
        qs._model = None

        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False

        qs._model = orig

    async def test_root_returns_api_info(self):
        app = FastAPI()
        app.include_router(router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        data = resp.json()
        assert "name" in data
        assert "version" in data
        assert data["endpoints"]["health"] == "/health"
        assert "filter" in data["endpoints"]["filter"]


# ---------------------------------------------------------------------------
# Unified error envelope for early rejections (auth / validation)
# ---------------------------------------------------------------------------

_ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}


def _app_with_handlers():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)  # ty: ignore[invalid-argument-type]
    app.include_router(router)
    register_exception_handlers(app)
    return app


class TestEarlyRejectionEnvelope:
    """401/403/422 should use the same envelope and be logged to OcrKtpQualityLog."""

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_wrong_api_key_uses_envelope_and_logs(self, _model, mock_config, mock_log):
        mock_config.log_to_database = True
        app = _app_with_handlers()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("t.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                data={"crops": "[]"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401
        content = resp.json()
        assert _ENVELOPE_KEYS == set(content.keys())
        assert content["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args[1]["response_code"] == 401

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    @patch("src.api.routes.config")
    @patch("src.api.routes.get_model", return_value=MagicMock())
    async def test_missing_field_uses_envelope_and_logs(self, _model, mock_config, mock_log):
        mock_config.log_to_database = True
        app = _app_with_handlers()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/ocr_qualitydl",
                files={"file": ("t.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                headers={"X-API-Key": "test"},  # crops field omitted
            )
        assert resp.status_code == 422
        content = resp.json()
        assert _ENVELOPE_KEYS == set(content.keys())
        assert content["error_code"] == "VALIDATION_ERROR"
        assert content["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args[1]["response_code"] == 422
