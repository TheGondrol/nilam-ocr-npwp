import io
import json
from typing import cast

import pytest
from PIL import Image

from ocr_common.npwp import contract_fields
from ocr_common.types import FinalResult

from app.config import get_settings
from app.main import app

RID = "REQ_contract"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _submit(client, auth, **form):
    return client.post(
        "/v1/extract-ocr",
        headers=auth,
        data={"request_id": RID, **form},
        files={"file": ("npwp.jpg", _jpeg(), "image/jpeg")},
    )


def _result(nomor, nama, nama_badan, npwp_confidence, name_confidence) -> FinalResult:
    # Only the keys contract_fields reads.
    return cast(
        FinalResult,
        {
            "fields": {
                "nomor_npwp": {"value": nomor, "confidence": 0.99},
                "nama": {"value": nama, "confidence": 0.97},
                "nama_badan": {"value": nama_badan, "confidence": 0.95},
            },
            "scoring": {"npwp_confidence": npwp_confidence, "name_confidence": name_confidence},
        },
    )


def test_confidence_is_1_from_the_threshold_up():
    fields = contract_fields(_result("12.345.678.9-012.345", "BUDI SANTOSO", None, 0.5, 0.49), 0.5)
    assert fields == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0},
    }


def test_missing_value_or_score_gives_confidence_0():
    fields = contract_fields(_result(None, "BUDI SANTOSO", None, None, None), 0.5)
    assert fields == {
        "nomor_npwp": {"value": None, "confidence": 0},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0},
    }


def test_company_card_reports_the_registered_name_as_nama():
    fields = contract_fields(_result("01.234.567.8-901.000", None, "PT CIPTA KARYA MANDIRI", 0.9, 0.8), 0.5)
    assert fields["nama"] == {"value": "PT CIPTA KARYA MANDIRI", "confidence": 1}


@pytest.mark.parametrize(
    "params",
    ['{"nik": "3123456711950001", "refno": "PK19039Y8U"}', '"halo"'],
)
def test_params_are_returned_unchanged(client, auth, params):
    response = _submit(client, auth, params=params)
    assert response.status_code == 200
    assert response.json()["params"] == json.loads(params)


@pytest.mark.parametrize("params", ["{not json", "[1, 2]", "42"])
def test_invalid_params_are_422_before_anything_runs(client, auth, stub_ekstraksi, params):
    response = _submit(client, auth, params=params)
    assert response.status_code == 422
    body = response.json()
    assert (body["errors"], body["message"]) == (
        "INVALID_PARAMS",
        "params must be valid JSON: an object, or a quoted string",
    )
    assert body["job_status"] is None
    assert stub_ekstraksi.submitted == []


def test_unsupported_document_type_is_400_before_anything_runs(client, auth, stub_ekstraksi):
    response = _submit(client, auth, document_type="ktp")
    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["message"]) == (
        "UNSUPPORTED_DOCUMENT_TYPE",
        "Unsupported document_type: ktp. Supported: npwp",
    )
    assert body["document_type"] == "ktp"
    assert stub_ekstraksi.submitted == []


def test_confidence_threshold_can_be_changed(client, auth):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(
        update={"field_confidence_threshold": 0.8}
    )
    try:
        response = _submit(client, auth)
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.json()["data"] == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    }
