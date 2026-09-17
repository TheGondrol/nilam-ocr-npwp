"""Unit tests for src.schemas.api_schema module."""

import pytest
from pydantic import ValidationError

from src.schemas.api_schema import (
    ClassificationResponse,
    Crop,
    CropPrediction,
    HealthResponse,
)


class TestCrop:
    def test_valid_crop(self):
        c = Crop(
            bbox=[[0, 0], [100, 0], [100, 50], [0, 50]],
            text="hello",
            confidence=0.95,
        )
        assert c.text == "hello"
        assert c.confidence == 0.95

    def test_crop_optional_fields(self):
        c = Crop(bbox=[[0, 0], [10, 0], [10, 10], [0, 10]])
        assert c.text is None
        assert c.confidence is None

    def test_crop_invalid_bbox_point_count(self):
        with pytest.raises(ValidationError):
            Crop(bbox=[[0, 0], [10, 0], [10, 10]])

    def test_crop_invalid_bbox_coord_count(self):
        with pytest.raises(ValidationError):
            Crop(bbox=[[0], [10, 0], [10, 10], [0, 10]])

    def test_crop_five_points_rejected(self):
        with pytest.raises(ValidationError):
            Crop(bbox=[[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])

    def test_crop_negative_coordinates_allowed(self):
        """Schema allows negative ints — clamping is the image service's responsibility."""
        c = Crop(bbox=[[-10, -5], [50, -5], [50, 30], [-10, 30]])
        assert c.bbox[0] == [-10, -5]


class TestCropPrediction:
    def test_valid_prediction(self):
        cp = CropPrediction(
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
            prediction=1,
            label="good",
            score=0.95,
        )
        assert cp.label == "good"
        assert cp.prediction == 1

    def test_bad_prediction(self):
        cp = CropPrediction(
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
            prediction=0,
            label="bad",
            score=0.8,
        )
        assert cp.label == "bad"

    def test_optional_debug_fields(self):
        cp = CropPrediction(
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
            prediction=1,
            label="good",
        )
        assert cp.text is None
        assert cp.confidence is None
        assert cp.score is None

    def test_invalid_prediction_value(self):
        with pytest.raises(ValidationError):
            CropPrediction(
                bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
                prediction=2,
                label="good",
            )

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            CropPrediction(
                bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
                prediction=1,
                label="unknown",  # ty: ignore[invalid-argument-type]
            )


class TestClassificationResponse:
    def test_valid_response(self):
        r = ClassificationResponse(
            label="good",
            num_bad=0,
            num_filtered=5,
            total_crops=10,
            num_failed_crops=0,
            bad_crop_threshold=4,
            predictions=[],
        )
        assert r.label == "good"
        assert r.total_crops == 10

    def test_bad_response(self):
        r = ClassificationResponse(
            label="bad",
            num_bad=5,
            num_filtered=8,
            total_crops=10,
            num_failed_crops=2,
            bad_crop_threshold=4,
            predictions=[],
        )
        assert r.label == "bad"

    def test_with_predictions(self):
        pred = CropPrediction(
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
            prediction=0,
            label="bad",
            score=0.9,
        )
        r = ClassificationResponse(
            label="bad",
            num_bad=1,
            num_filtered=1,
            total_crops=1,
            num_failed_crops=0,
            bad_crop_threshold=1,
            predictions=[pred],
        )
        assert len(r.predictions) == 1

    def test_model_dump(self):
        r = ClassificationResponse(
            label="good",
            num_bad=0,
            num_filtered=0,
            total_crops=0,
            num_failed_crops=0,
            bad_crop_threshold=4,
            predictions=[],
        )
        d = r.model_dump()
        assert "label" in d
        assert "predictions" in d

    def test_invalid_label_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationResponse(
                label="maybe",  # ty: ignore[invalid-argument-type]
                num_bad=0,
                num_filtered=0,
                total_crops=0,
                num_failed_crops=0,
                bad_crop_threshold=4,
                predictions=[],
            )


class TestHealthResponse:
    def test_healthy(self):
        h = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda",
            version="1.0.0",
        )
        assert h.status == "healthy"
        assert h.model_loaded is True

    def test_unhealthy(self):
        h = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            version="1.0.0",
        )
        assert h.model_loaded is False
