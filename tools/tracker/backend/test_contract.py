"""Pemeriksa kontrak callback. Body yang benar dibuat oleh kode pipeline sendiri (`result_callback_body`), jadi
pemeriksa dan pengirim tidak bisa diam-diam berbeda pendapat. Jalankan: python -m pytest tools/tracker/backend"""

import pytest
from contract import FAIL, GUARDRAILS_COMPLETED_RULE, PASS, WARN, check_result_callback, check_stage_callback

from ocr_common.pipeline.callbacks import REJECTED_CODE, result_callback_body, stage_callback_body

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"
DATA = {"nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 1}, "nama": {"value": "BUDI", "confidence": 0}}
REASON = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"


def _completed() -> dict:
    return result_callback_body(stage_callback_body(RID, "SCORING", "DONE", final=True, answer=DATA))


def _rejected() -> dict:
    return result_callback_body(
        stage_callback_body(RID, "STRUCTURING", "FAILED", error_message=REASON, error_code=REJECTED_CODE)
    )


def _failed() -> dict:
    return result_callback_body(stage_callback_body(RID, "SCORING", "FAILED", error_message="trust model is down"))


def _answer(status: int, **body) -> dict:
    return {"http_status": status, "body": {"status_code": status, **body}}


def _failing(report: dict) -> list[str]:
    return [check["rule"] for check in report["checks"] if check["level"] == FAIL]


def _warning(report: dict) -> list[str]:
    return [check["rule"] for check in report["checks"] if check["level"] == WARN]


@pytest.mark.parametrize(
    ("body", "answer", "kind"),
    [
        (_completed, _answer(200, data=DATA, guardrails=0, errors=None), "completed"),
        (_rejected, _answer(400, data=None, guardrails=1, errors=REJECTED_CODE, message=REASON), "rejected"),
        (_failed, _answer(422, data=None, guardrails=0, errors="SCORING_FAILED"), "failed"),
    ],
)
def test_what_the_pipeline_sends_passes(body, answer, kind):
    report = check_result_callback(body(), answer=answer)

    assert (report["kind"], report["verdict"]) == (kind, PASS), report["checks"]


def test_probabilities_without_thresholds_pass():
    """Tanpa threshold dari pusat (8 Okt 2026): confidence dan guardrails berupa probabilitas."""
    result = {"nomor_npwp": {"value": "x", "confidence": 0.9926}, "nama": DATA["nama"]}
    body = {**_completed(), "result": result, "guardrails": 0.9821}

    assert _failing(check_result_callback(body, answer=_answer(200, data=result, guardrails=0.9821))) == []


def test_a_confidence_outside_0_1_fails():
    body = {**_completed(), "result": {"nomor_npwp": {"value": "x", "confidence": 1.5}, "nama": DATA["nama"]}}

    assert _failing(check_result_callback(body, answer=None)) == [
        "result.nomor_npwp = {value: string | null, confidence: 0 | 1 | probabilitas}"
    ]


def test_a_boolean_is_not_the_integer_the_contract_asks_for():
    body = {**_completed(), "guardrails": False}

    assert GUARDRAILS_COMPLETED_RULE in _failing(check_result_callback(body))


def test_the_old_shape_with_a_guardrails_report_and_nama_badan_fails():
    body = {
        **_completed(),
        "guardrails": {"passed": True},
        "result": {**DATA, "nama_badan": {"value": "", "confidence": 0.0}},
    }

    assert _failing(check_result_callback(body)) == [
        GUARDRAILS_COMPLETED_RULE,
        "result berisi tepat nomor_npwp dan nama",
    ]


def test_a_rejection_must_say_why_and_carry_guardrails_1():
    body = {**_rejected(), "guardrails": 0, "message": ""}

    assert _failing(check_result_callback(body)) == ["guardrails = 1 (integer)", "message berisi alasan penolakan"]


def test_a_rejection_sent_as_failed_is_not_recognised_as_one():
    """Pusat hanya mengenali penolakan dari result null + guardrails 1."""
    body = {"request_id": RID, "status": "failed", "error_code": REJECTED_CODE, "message": REASON}

    report = check_result_callback(body, answer=_answer(400, guardrails=1, errors=REJECTED_CODE, message=REASON))

    assert report["kind"] == "failed"
    assert "error_code <STAGE>_FAILED" in _warning(report)
    assert any(rule.startswith("Sama dengan jawaban GET") for rule in _failing(report))


def test_internal_fields_must_not_leave():
    body = {**_completed(), "traceparent": "00-ab-cd-01", "answer": DATA}

    assert _failing(check_result_callback(body)) == ["Tidak membawa data internal pipeline"]


def test_a_result_that_differs_from_the_200_answer_fails():
    other = {**DATA, "nama": {"value": "BUDI", "confidence": 1}}

    report = check_result_callback(_completed(), answer=_answer(200, data=other, guardrails=0))

    [rule] = _failing(report)
    assert rule.endswith("result = data")
    assert "nama" in next(check["detail"] for check in report["checks"] if check["rule"] == rule)


def test_a_pipeline_that_ends_before_scoring_sends_the_last_result_as_it_is():
    ocr = {"full_text": "NPWP", "blocks": []}
    body = result_callback_body(stage_callback_body(RID, "OCR", "DONE", result=ocr, final=True, answer=ocr))

    report = check_result_callback(body, ends_at_scoring=False, answer=_answer(200, data=ocr, guardrails=0))

    assert report["verdict"] == PASS, report["checks"]


def test_a_wrong_key_extra_fields_and_no_answer_to_compare():
    report = check_result_callback({**_completed(), "extra": 1}, key_ok=False, answer=None)

    assert _failing(report) == ["X-Callback-Key benar"]
    assert _warning(report) == [
        "Tidak ada field di luar kontrak",
        "Sama dengan jawaban GET /v1/extract-ocr/{request_id}",
    ]


def test_a_failed_callback_for_a_dead_handoff_is_only_a_warning_while_get_still_says_202():
    report = check_result_callback(_failed(), answer=_answer(202, data=None))

    assert report["verdict"] == WARN


@pytest.mark.parametrize("body", [None, [], "completed", {"status": "done"}])
def test_a_body_that_is_not_a_callback_fails(body):
    assert check_result_callback(body)["verdict"] == FAIL


def test_the_stage_format_passes_what_the_pipeline_sends_and_never_the_answer():
    done = stage_callback_body(RID, "SCORING", "DONE", result={"x": 1}, final=True)
    failed = stage_callback_body(RID, "STRUCTURING", "FAILED", error_message=REASON, error_code=REJECTED_CODE)

    assert check_stage_callback(done)["verdict"] == PASS
    assert check_stage_callback(failed)["verdict"] == PASS
    assert _failing(check_stage_callback({**done, "answer": DATA})) == ["Tidak membawa data internal pipeline"]
