from typing import cast

from ocr_common.npwp import contract_fields
from ocr_common.types import FinalResult

from tests.conftest import JPEG

RID = "REQ_contract"


def _submit(client, auth, **form):
    return client.post(
        "/v1/extract-ocr",
        headers=auth,
        data={"request_id": RID, **form},
        files={"file": ("npwp.jpg", JPEG, "image/jpeg")},
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


def test_without_a_threshold_the_confidence_is_the_trust_probability():
    fields = contract_fields(_result("12.345.678.9-012.345", "BUDI SANTOSO", None, 0.729649, 0.49))
    assert fields == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.7296},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.49},
    }


def test_a_threshold_for_one_field_leaves_the_other_its_probability():
    fields = contract_fields(_result("12.345.678.9-012.345", "BUDI SANTOSO", None, 0.7296, 0.49), None, {"nama": 0.4})
    assert fields == {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.7296},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    }


def test_company_card_reports_the_registered_name_as_nama():
    fields = contract_fields(_result("01.234.567.8-901.000", None, "PT CIPTA KARYA MANDIRI", 0.9, 0.8), 0.5)
    assert fields["nama"] == {"value": "PT CIPTA KARYA MANDIRI", "confidence": 1}


def test_the_answer_carries_no_job_status_document_type_or_params(client, auth):
    response = _submit(client, auth)

    assert response.status_code == 200
    assert not {"job_status", "document_type", "params"} & set(response.json())


def test_params_sent_by_an_old_caller_are_ignored(client, auth):
    response = _submit(client, auth, params="{not json")

    assert response.status_code == 200
    assert "params" not in response.json()


def test_unsupported_document_type_is_400_before_anything_runs(client, auth, stub_guardrails, stub_extraction):
    response = _submit(client, auth, document_type="ktp")
    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["message"]) == (
        "UNSUPPORTED_DOCUMENT_TYPE",
        "Unsupported document_type: ktp. Supported: npwp",
    )
    assert "document_type" not in body
    assert stub_guardrails.checked == [] and stub_extraction.submitted == []


