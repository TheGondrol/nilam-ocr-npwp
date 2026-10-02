"""`POST /v1/structuring-direct`: the structuring rules on the extraction service's output, synchronous, for
testing this stage on its own. Nothing is recorded."""

import pytest

OCR = {
    "engine": "mock",
    "model": None,
    "full_text": "NPWP : 12.345.678.9-012.345\nNAMA : BUDI SANTOSO",
    "blocks": [
        {"text": "NPWP : 12.345.678.9-012.345", "confidence": 0.96, "bbox": None, "page": 0},
        {"text": "NAMA : BUDI SANTOSO", "confidence": 0.95, "bbox": None, "page": 0},
    ],
}


def _post(client, auth, **body):
    return client.post("/v1/structuring-direct", headers=auth, json=body)


def test_the_extraction_output_is_structured_now_and_nothing_is_recorded(client, auth):
    response = _post(client, auth, request_id="QC_1", ocr=OCR)

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "QC_1"
    data = body["data"]
    assert set(data) == {"document_type", "fields", "flag", "flag_reason", "reject_reason"}
    assert data["document_type"] == "npwp"
    assert data["fields"]["nomor_npwp"]["value"] == "123456789012345"
    assert data["fields"]["nama"]["value"] == "BUDI SANTOSO"
    assert data["reject_reason"] is None

    assert client.get("/v1/structuring/jobs/QC_1", headers=auth).status_code == 404


def test_the_answer_is_what_the_scoring_stage_receives(client, auth):
    """The same OCR output structured twice gives the same document: the output can be pasted on as the
    `structuring` of `/v1/scoring-direct`."""
    first = _post(client, auth, ocr=OCR).json()["data"]
    second = _post(client, auth, ocr=OCR).json()["data"]

    assert first == second
    assert set(first["fields"]) == {"nomor_npwp", "nama", "nama_badan"}


def test_a_rejected_document_still_answers_200_with_the_reason(client, auth):
    blocks = [{"text": "KARTU TANDA PENDUDUK", "confidence": 0.9}, {"text": "NIK 3201234567890001", "confidence": 0.9}]

    response = _post(client, auth, ocr={"blocks": blocks})

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["flag"] is True
    assert data["reject_reason"]


def test_no_text_lines_is_400(client, auth):
    response = _post(client, auth, ocr={"blocks": [{"text": "   ", "confidence": 0.5}]})

    assert response.status_code == 400
    assert response.json()["message"] == "No text lines to structure"


def test_another_document_type_is_400(client, auth):
    response = _post(client, auth, document_type="ktp", ocr=OCR)

    assert response.status_code == 400
    assert "Unsupported document_type: ktp" in response.json()["message"]


@pytest.mark.parametrize("body", [{}, {"ocr": {}}, {"ocr": {"blocks": "NPWP"}}])
def test_a_body_without_ocr_blocks_is_422(client, auth, body):
    response = client.post("/v1/structuring-direct", headers=auth, json=body)

    assert response.status_code == 422
    assert response.json()["errors"] == "VALIDATION_ERROR"


def test_requires_the_api_key(client):
    assert client.post("/v1/structuring-direct", json={"ocr": OCR}).status_code == 401
