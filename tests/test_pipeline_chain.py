"""
Test integrasi: hasil satu app bisa langsung dimakan app berikutnya lewat
HTTP, dan rangkaiannya menghasilkan data yang sama dengan /v1/extract-ocr
(yang menjalankan pipeline yang sama in-process).
"""

from tests.conftest import image_upload

CONTENT = b"\xff\xd8chain-seed"


def test_four_apps_chain_end_to_end(client, auth):
    guard = client.post("/v1/guardrails/check", files=image_upload("npwp.jpg", CONTENT), headers=auth).json()["data"]
    assert guard["passed"] is True

    ocr = client.post("/v1/ekstraksi/extract", files=image_upload("npwp.jpg", CONTENT), headers=auth).json()["data"]
    lines = [{"text": b["text"], "confidence": b["confidence"]} for b in ocr["blocks"]]

    structured = client.post("/v1/structuring/structure", json={"lines": lines}, headers=auth).json()["data"]
    assert set(structured["fields"]) == {"nomor_npwp", "nama", "nama_badan"}

    score_body = {
        "document_type": structured["document_type"],
        "fields": {k: {"value": v["value"], "confidence": v["confidence"]} for k, v in structured["fields"].items()},
    }
    score = client.post("/v1/scoring/score", json=score_body, headers=auth).json()["data"]
    assert score["decision"] == "approve"
    assert score["reasons"] == []

    # Jalur kontrak menghasilkan hal yang sama dari pipeline yang sama.
    request_id = client.post("/v1/generate-request-id", headers=auth).json()["request_id"]
    contract = client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": request_id}, files=image_upload("npwp.jpg", CONTENT)
    ).json()
    assert contract["data"] == {
        k: {"value": v["value"], "confidence": v["confidence"]} for k, v in structured["fields"].items()
    }
    assert contract["guardrails"] == score["score"]
