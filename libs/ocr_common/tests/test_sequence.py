import pytest

from ocr_common.pipeline import (
    DEFAULT_SEQUENCE,
    InMemoryJobRepository,
    InvalidSequence,
    chain,
    checked_sequence,
    next_service,
    validate_sequence,
)
from ocr_common.pipeline.callbacks import result_callback_body, stage_callback_body


@pytest.mark.parametrize(
    "sequence",
    [
        ["guardrails", "extraction", "structuring", "scoring"],
        ["guardrails", "extraction", "structuring"],
        ["guardrails", "extraction"],
        ["guardrails"],
        ["extraction", "structuring", "scoring"],
        ["extraction", "structuring"],
        ["extraction"],
    ],
)
def test_guardrails_may_be_left_out_and_the_end_cut_off(sequence):
    assert validate_sequence(sequence) == tuple(sequence)


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        (["extraction", "scoring"], "without skipping one in the middle"),
        (["guardrails", "extraction", "scoring"], "without skipping one in the middle"),
        (["guardrails", "structuring"], "without skipping one in the middle"),
        (["structuring", "scoring"], "structuring cannot come first"),
        (["scoring"], "scoring cannot come first"),
        (["extraction", "guardrails"], "without skipping one in the middle"),
        (["guardrails", "guardrails", "extraction"], "listed twice"),
        (["guardrails", "ekstraksi"], "unknown service 'ekstraksi'"),
    ],
)
def test_a_skipped_middle_a_wrong_order_or_an_unknown_name_is_refused(sequence, reason):
    with pytest.raises(InvalidSequence, match=reason):
        validate_sequence(sequence)


def test_no_sequence_is_the_full_pipeline():
    assert validate_sequence(None) == validate_sequence([]) == DEFAULT_SEQUENCE
    assert next_service(None, "structuring") == "scoring"


def test_next_service_is_none_for_the_last_one():
    assert next_service(["guardrails", "extraction", "structuring"], "extraction") == "structuring"
    assert next_service(["guardrails", "extraction", "structuring"], "structuring") is None
    with pytest.raises(InvalidSequence, match="does not include scoring"):
        next_service(["extraction"], "scoring")


def test_a_stage_checks_that_it_is_part_of_the_sequence():
    assert checked_sequence(None, "scoring") is None
    assert checked_sequence(("extraction", "structuring"), "structuring") == ["extraction", "structuring"]
    with pytest.raises(InvalidSequence):
        checked_sequence(["guardrails", "extraction"], "structuring")


def test_chain_hands_on_or_ends_the_request_with_the_result_as_it_is():
    def handoff(result):
        return {"handed": result}

    assert chain(None, "extraction", handoff) == {"handoff_payload": handoff, "next_stage": "STRUCTURING"}
    assert chain(["extraction", "structuring", "scoring"], "structuring", handoff)["next_stage"] == "SCORING"
    last = chain(["guardrails", "extraction"], "extraction", handoff)
    assert "next_stage" not in last and "handoff_payload" not in last
    assert last["callback_result"]({"text": "x"}) == last["outcome_data"]({"text": "x"}) == {"text": "x"}


def test_the_result_callback_completes_on_the_final_stage_with_its_result_as_it_is():
    ocr = {"text": "NPWP", "blocks": []}
    final = stage_callback_body("REQ", "OCR", "DONE", result=ocr, final=True)
    not_final = stage_callback_body("REQ", "OCR", "DONE", result=ocr)

    assert final["final"] is True and "final" not in not_final
    assert result_callback_body(final) == {"request_id": "REQ", "status": "completed", "result": ocr, "guardrails": {}}
    assert result_callback_body(not_final) is None


def test_a_scoring_callback_queued_before_the_final_flag_still_completes():
    body = stage_callback_body(
        "REQ",
        "SCORING",
        "DONE",
        result={"fields": {"nomor_npwp": {"value": "1"}}, "scoring": {"npwp_confidence": 0.9}, "guardrails": {}},
    )

    completed = result_callback_body(body)

    assert completed is not None and completed["status"] == "completed"
    assert completed["result"]["nomor_npwp"] == {"value": "1", "confidence": 0.9}


async def test_a_job_record_carries_the_sequence_it_was_submitted_with():
    repository = InMemoryJobRepository()
    await repository.claim("REQ_1", input={"pipeline_name_sequence": ["extraction"]})
    await repository.claim("REQ_2", input={"document_type": "npwp"})

    first, second = await repository.get("REQ_1"), await repository.get("REQ_2")

    assert first is not None and first["pipeline_name_sequence"] == ["extraction"]
    assert second is not None and second["pipeline_name_sequence"] is None
