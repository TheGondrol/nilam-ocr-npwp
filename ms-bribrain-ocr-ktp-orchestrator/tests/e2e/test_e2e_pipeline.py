"""
End-to-end tests for the OCR KTP orchestrator pipeline.

Runs every image in tests/data/ through the full extract-ocr endpoint
with REAL calls to all downstream microservices. Only the database is mocked.

Prerequisites:
    All downstream services must be running and reachable
    (OCR, classifier, quality, lamination, recapture, graycopy, postprocess, temper).

Usage:
    pytest tests/e2e/ -m e2e -v
"""

import io
import uuid
from pathlib import Path

import pytest


# Expected outcomes per test image
EXPECTED_SUCCESS_IMAGES = {"good_data.png"}

EXPECTED_REJECTION_CODES = {
    "unlaminated.jpeg": "SPOOF_UNLAMINATED",
    "recapture.jpeg": "SPOOF_RECAPTURE",
    "graycopy.jpeg": "SPOOF_GRAYCOPY",
    "tamper.jpeg": "TAMPERED",
    "classifier.jpeg": "NOT_KTP",
}


@pytest.mark.e2e
class TestExtractOcrPipeline:
    """E2E: POST /v1/extract-ocr with real images and real microservices."""

    async def test_extract_ocr_with_image(
        self, e2e_client, image_path: Path, image_bytes: bytes, image_content_type: str
    ):
        """
        Send each test image through the full OCR pipeline.

        - good_data.png → 200 with extracted KTP fields
        - rejection images → 400 with the expected error_code
        """
        request_id = f"OCR_{uuid.uuid4()}"
        image_name = image_path.name

        response = await e2e_client.post(
            "/v1/extract-ocr",
            data={"request_id": request_id},
            files={"file": (image_name, io.BytesIO(image_bytes), image_content_type)},
        )

        body = response.json()

        if image_name in EXPECTED_SUCCESS_IMAGES:
            assert response.status_code == 200, (
                f"[{image_name}] expected 200, got {response.status_code}: {body}"
            )
            assert body["status_code"] == 200
            assert body["error_code"] is None
            assert isinstance(body["data"], dict), (
                f"[{image_name}] expected data dict, got {type(body['data'])}"
            )
            # Verify KTP fields are present
            data = body["data"]
            for field in ("nik", "nama"):
                assert field in data, f"[{image_name}] missing '{field}' in result"

        elif image_name in EXPECTED_REJECTION_CODES:
            expected_code = EXPECTED_REJECTION_CODES[image_name]
            assert response.status_code == 400, (
                f"[{image_name}] expected 400, got {response.status_code}: {body}"
            )
            assert body["status_code"] == 400

            actual_code = body["error_code"]
            if actual_code == "IMAGE_REJECTED":
                # Spoof/classifier rejections are wrapped in IMAGE_REJECTED
                reasons = body.get("errors", {}).get("reasons", [])
                reason_codes = [r["error_code"] for r in reasons]
                assert expected_code in reason_codes, (
                    f"[{image_name}] expected {expected_code} in reasons, got {reason_codes}"
                )
            else:
                assert actual_code == expected_code, (
                    f"[{image_name}] expected {expected_code}, got {actual_code}"
                )

        else:
            # Unknown image — just ensure a valid HTTP response
            assert response.status_code in (200, 400, 500), (
                f"[{image_name}] unexpected status {response.status_code}"
            )

    async def test_response_envelope_format(
        self, e2e_client, image_path: Path, image_bytes: bytes, image_content_type: str
    ):
        """Verify the response always follows the standard envelope format."""
        request_id = f"OCR_{uuid.uuid4()}"

        response = await e2e_client.post(
            "/v1/extract-ocr",
            data={"request_id": request_id},
            files={"file": (image_path.name, io.BytesIO(image_bytes), image_content_type)},
        )

        body = response.json()

        # All responses must have these envelope fields
        for field in ("status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"):
            assert field in body, f"[{image_path.name}] missing envelope field '{field}'"

        assert body["status_code"] == response.status_code
        assert isinstance(body["message"], str)
        assert body["request_id"] == request_id


@pytest.mark.e2e
class TestExtractOcrValidation:
    """E2E: input validation with real images."""

    async def test_invalid_request_id_format(
        self, e2e_client, image_path: Path, image_bytes: bytes, image_content_type: str
    ):
        """Reject requests with invalid request_id format."""
        response = await e2e_client.post(
            "/v1/extract-ocr",
            data={"request_id": "INVALID_FORMAT"},
            files={"file": (image_path.name, io.BytesIO(image_bytes), image_content_type)},
        )

        assert response.status_code == 400

    async def test_missing_file(self, e2e_client, image_path: Path):
        """Reject requests with no file uploaded."""
        request_id = f"OCR_{uuid.uuid4()}"

        response = await e2e_client.post(
            "/v1/extract-ocr",
            data={"request_id": request_id},
        )

        assert response.status_code == 422


@pytest.mark.e2e
class TestGenerateRequestId:
    """E2E: POST /v1/generate-request-id."""

    async def test_generate_request_id(self, e2e_client, image_path: Path):
        """Generate a unique request ID."""
        response = await e2e_client.post("/v1/generate-request-id")

        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        assert body["data"]["request_id"].startswith("OCR_")


@pytest.mark.e2e
class TestFullFlow:
    """E2E: full flow generate → extract → get-result (per image)."""

    async def test_full_flow(
        self, e2e_client, image_path: Path, image_bytes: bytes, image_content_type: str
    ):
        """
        Complete flow for each image:
        1. Generate request ID
        2. Submit image for OCR extraction
        3. Retrieve the stored result
        """
        image_name = image_path.name

        # Step 1: Generate request ID
        gen_response = await e2e_client.post("/v1/generate-request-id")
        assert gen_response.status_code == 200
        request_id = gen_response.json()["data"]["request_id"]

        # Step 2: Extract OCR
        extract_response = await e2e_client.post(
            "/v1/extract-ocr",
            data={"request_id": request_id},
            files={"file": (image_name, io.BytesIO(image_bytes), image_content_type)},
        )

        extract_body = extract_response.json()

        if image_name in EXPECTED_SUCCESS_IMAGES:
            assert extract_response.status_code == 200, (
                f"[{image_name}] full flow extract failed: {extract_body}"
            )
            assert isinstance(extract_body["data"], dict)

        elif image_name in EXPECTED_REJECTION_CODES:
            assert extract_response.status_code == 400, (
                f"[{image_name}] full flow expected rejection: {extract_body}"
            )

        # Step 3: Retrieve result
        result_response = await e2e_client.get(f"/v1/get-ocr-result/{request_id}")
        assert result_response.status_code == 200
        result_body = result_response.json()
        assert result_body["status_code"] == 200
        assert result_body["data"]["request_id"] == request_id
