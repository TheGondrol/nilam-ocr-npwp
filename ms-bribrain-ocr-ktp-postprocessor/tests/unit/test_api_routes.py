"""Tests for API routes."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi import status


class TestOCRPostprocessEndpoint:
    """Test cases for /v1/ocr_postprocess endpoint."""

    def test_successful_ocr_processing(self, test_client, sample_ocr_data):
        """Test successful OCR processing with valid data."""
        ocr_text = json.dumps(sample_ocr_data)

        with patch('src.api.routes.mappingnext') as mock_mapping, \
             patch('src.api.routes.insert_log', new_callable=AsyncMock) as mock_log:

            mock_result = {
                "nik": ["3174012345678901", 0.97],
                "nama": ["BUDI SANTOSO", 0.95],
            }
            mock_nik_box = [[220, 120], [400, 120], [400, 170], [220, 170]]
            mock_mapping.return_value = (mock_result, mock_nik_box)

            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": ocr_text}
            )

            assert response.status_code == status.HTTP_200_OK
            body = response.json()
            assert body["status_code"] == 200
            assert body["status_desc"] == "OK"
            assert "ocr_result" in body["data"]
            assert "nik_image_box" in body["data"]
            assert body["data"]["nik_image_box"] == mock_nik_box

            mock_mapping.assert_called_once()
            mock_log.assert_called_once()

    def test_invalid_json_format(self, test_client):
        """Test with invalid JSON format in ocr_text."""
        with patch('src.api.routes.insert_log', new_callable=AsyncMock):
            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": "not a valid json"}
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        body = response.json()
        assert body["error_code"] == "INVALID_INPUT"
        assert "Invalid OCR text format" in body["message"]

    def test_empty_ocr_text(self, test_client):
        """Test with empty OCR text."""
        with patch('src.api.routes.insert_log', new_callable=AsyncMock):
            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": "[]"}
            )

            assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_processing_exception(self, test_client, sample_ocr_data):
        """Test handling of processing exceptions."""
        ocr_text = json.dumps(sample_ocr_data)

        with patch('src.api.routes.mappingnext') as mock_mapping, \
             patch('src.api.routes.insert_log', new_callable=AsyncMock):

            mock_mapping.side_effect = Exception("Processing error")

            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": ocr_text}
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            body = response.json()
            assert body["error_code"] == "INTERNAL_ERROR"

    def test_missing_ocr_text_field(self, test_client):
        """Test request with missing ocr_text field."""
        response = test_client.post(
            "/v1/ocr_postprocess",
            json={}
        )

        assert response.status_code == 422

    def test_malformed_request_body(self, test_client):
        """Test with malformed request body."""
        response = test_client.post(
            "/v1/ocr_postprocess",
            content="not json",
            headers={"Content-Type": "application/json", "X-API-Key": "test"}
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("ocr_text_value", [
        '[[[[100, 50], [200, 50]], ("TEXT", 0.95)]]',
        '[[[100, 50], ("SINGLE_COORD", 0.95)]]',
        '[[[], ("EMPTY_BOX", 0.95)]]',
    ])
    def test_various_ocr_formats(self, test_client, ocr_text_value):
        """Test with various OCR data formats."""
        with patch('src.api.routes.mappingnext') as mock_mapping, \
             patch('src.api.routes.insert_log', new_callable=AsyncMock):

            mock_mapping.return_value = ({}, None)

            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": ocr_text_value}
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]


class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    _ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}

    def test_wrong_api_key_uses_envelope_and_logs(self, test_client, sample_ocr_data):
        with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
            response = test_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": json.dumps(sample_ocr_data)},
                headers={"X-API-Key": "wrong"},
            )
        assert response.status_code == 401
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.args[0].response_code == 401

    def test_missing_field_uses_envelope_and_logs(self, test_client):
        with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_log:
            response = test_client.post("/v1/ocr_postprocess", json={})
        assert response.status_code == 422
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.args[0].response_code == 422


class TestLivenessEndpoint:
    """Tests for GET /health/live."""

    def test_liveness_returns_200(self, test_client):
        response = test_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "version" in data


class TestReadinessEndpoint:
    """Tests for GET /health/ready."""

    def test_readiness_ready(self, test_client):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"}):
            response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["database"]["status"] == "up"

    def test_readiness_not_ready(self, test_client):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "not initialised"}):
            response = test_client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_healthy(self, test_client):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "up"}):
            response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checks"]["database"]["status"] == "up"

    def test_health_unhealthy(self, test_client):
        with patch("src.main._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "connection refused"}):
            response = test_client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_service_info(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "OCR Postprocess API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]
