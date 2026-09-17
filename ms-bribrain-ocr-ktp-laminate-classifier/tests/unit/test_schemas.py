"""Unit tests for src.api.schemas module."""

from src.api.schemas import (
    PredictionResponse,
    HealthResponse,
    RootResponse,
    BatchResultItem,
    BatchPredictionResponse,
)


class TestPredictionResponse:
    def test_create(self):
        r = PredictionResponse(
            filename="test.jpg",
            prediction="LAMINATED",
            prob=0.3,
            threshold=0.5,
            timestamp="2024-01-01T00:00:00",
        )
        assert r.filename == "test.jpg"
        assert r.prediction == "LAMINATED"
        assert r.prob == 0.3

    def test_dict(self):
        r = PredictionResponse(
            filename="f.jpg",
            prediction="UNLAMINATED",
            prob=0.8,
            threshold=0.5,
            timestamp="t",
        )
        d = r.model_dump()
        assert "filename" in d
        assert "prediction" in d


class TestHealthResponse:
    def test_create(self):
        r = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cpu",
            version="1.0.0",
        )
        assert r.status == "healthy"
        assert r.model_loaded is True


class TestRootResponse:
    def test_create(self):
        r = RootResponse(
            message="API",
            description="desc",
            version="1.0.0",
            endpoints={"health": "/health"},
        )
        assert r.version == "1.0.0"


class TestBatchResultItem:
    def test_with_result(self):
        pred = PredictionResponse(
            filename="f.jpg",
            prediction="LAMINATED",
            prob=0.2,
            threshold=0.5,
            timestamp="t",
        )
        item = BatchResultItem(filename="f.jpg", result=pred)
        assert item.result is not None
        assert item.error is None

    def test_with_error(self):
        item = BatchResultItem(filename="f.jpg", error="bad image")
        assert item.result is None
        assert item.error == "bad image"


class TestBatchPredictionResponse:
    def test_create(self):
        items = [BatchResultItem(filename="a.jpg"), BatchResultItem(filename="b.jpg")]
        r = BatchPredictionResponse(results=items)
        assert len(r.results) == 2
