"""Unit tests for src.api.routes module (_envelope helper)."""

import json

from fastapi.responses import JSONResponse

from src.api.routes import _envelope


class TestEnvelope:
    def test_success_envelope(self):
        resp = _envelope(200, "req-1", data={"key": "val"})
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body["status_code"] == 200
        assert body["status_desc"] == "OK"
        assert body["data"]["key"] == "val"
        assert body["request_id"] == "req-1"
        assert body["error_code"] is None

    def test_error_400(self):
        resp = _envelope(400, "req-2", message="Bad", error_code="BAD")
        body = json.loads(resp.body)
        assert body["status_code"] == 400
        assert body["status_desc"] == "Bad Request"
        assert body["error_code"] == "BAD"

    def test_error_413(self):
        resp = _envelope(413, "req-3", message="Too large", error_code="BIG")
        body = json.loads(resp.body)
        assert body["status_desc"] == "Payload Too Large"

    def test_error_500(self):
        resp = _envelope(500, "req-4", message="fail", error_code="ERR")
        body = json.loads(resp.body)
        assert body["status_desc"] == "Internal Server Error"

    def test_error_503(self):
        resp = _envelope(503, "req-5")
        body = json.loads(resp.body)
        assert body["status_desc"] == "Service Unavailable"

    def test_unknown_status(self):
        resp = _envelope(418, "req-6")
        body = json.loads(resp.body)
        assert body["status_desc"] == "Unknown"

    def test_empty_data(self):
        resp = _envelope(200, "req-7")
        body = json.loads(resp.body)
        assert body["data"] == ""

    def test_errors_field(self):
        resp = _envelope(400, "req-8", errors=["e1", "e2"])
        body = json.loads(resp.body)
        assert body["errors"] == ["e1", "e2"]

    def test_all_keys_present(self):
        resp = _envelope(200, "req-9", data={"x": 1})
        body = json.loads(resp.body)
        expected = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected == set(body.keys())


class TestPredictRounding:
    """Verify the predict endpoint rounds probability to 4 decimal places."""

    @staticmethod
    async def _call_predict(monkeypatch, prob_value: float):
        import io
        import json
        import numpy as np
        from PIL import Image
        from unittest.mock import MagicMock
        from fastapi import UploadFile

        from src.api import routes as routes_mod
        from src.services import predictor as predictor_mod

        # Stub predictor.
        monkeypatch.setattr(predictor_mod.predictor, "is_loaded", lambda: True)
        monkeypatch.setattr(
            predictor_mod.predictor, "predict", lambda img, fn: ("LAMINATED", prob_value)
        )
        # threshold is a read-only property backed by the ThresholdProvider; the
        # route reads predictor.threshold, so stub the provider rather than the attr.
        _stub_provider = MagicMock()
        _stub_provider.get.return_value = 0.5
        monkeypatch.setattr(predictor_mod, "get_provider", lambda: _stub_provider)

        # Bypass DB logging.
        async def _noop_insert(**kwargs):
            return None
        monkeypatch.setattr(routes_mod, "insert_log", _noop_insert)

        # Use a fresh executor — the module-level one may be shut down by other tests.
        from concurrent.futures import ThreadPoolExecutor
        monkeypatch.setattr(routes_mod, "_executor", ThreadPoolExecutor(max_workers=1))

        # Build an in-memory PNG UploadFile.
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        # UploadFile derives content_type from headers; pass it explicitly.
        from starlette.datastructures import Headers
        upload = UploadFile(
            filename="img.png",
            file=buf,
            headers=Headers({"content-type": "image/png"}),
        )

        request = MagicMock()
        response = await routes_mod.predict(request=request, file=upload, _="test-key")
        body = json.loads(response.body)
        return body

    async def test_probability_rounded_to_four_decimals(self, monkeypatch):
        body = await self._call_predict(monkeypatch, prob_value=0.123456789)
        assert body["status_code"] == 200
        prob = body["data"]["prob"]
        # round(0.123456789, 4) == 0.1235
        assert prob == 0.1235

    async def test_probability_rounding_near_one(self, monkeypatch):
        body = await self._call_predict(monkeypatch, prob_value=0.99995)
        assert body["status_code"] == 200
        prob = body["data"]["prob"]
        # Must be bounded and have at most 4 decimal places.
        assert 0.0 <= prob <= 1.0
        assert round(prob, 4) == prob
