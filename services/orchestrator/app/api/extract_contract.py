from collections.abc import Mapping
from typing import Any

from ocr_common.npwp import COMPLETED_MESSAGE, GUARDRAILS_REJECTED, REJECTED_CODE, contract_fields, guardrails_value
from ocr_common.pipeline import EXTRACTION, GUARDRAILS, SERVICE_OF_STAGE, STAGE_SCORING, STATUS_DONE, STATUS_FAILED
from ocr_common.web.envelope import envelope

from app.services.pipeline_waiter import STATUS_REJECTED

PROCESSING_MESSAGE = "OCR job accepted; still processing"


def extract_body(
    status_code: int,
    message: str,
    *,
    request_id: str | None,
    data: Mapping[str, Any] | None = None,
    errors: str | None = None,
    guardrails: int | float | None = None,
    pipeline_last_stage: str | None = None,
) -> dict[str, Any]:
    """The extract-ocr answer: the standard envelope plus `pipeline_last_stage` and `guardrails`. The HTTP status
    (`status_code`) says where the request is: 200 finished, 202 still running, 4xx / 5xx failed or refused."""
    return {
        **envelope(status_code, message, dict(data) if data is not None else None, request_id, errors=errors),
        "pipeline_last_stage": pipeline_last_stage,
        "guardrails": guardrails,
    }


def last_stage(outcome: dict[str, Any]) -> str | None:
    """The pipeline service an error answer comes from: guardrails when it rejected, else the stage the
    pipeline reached (the one that failed or rejected). A success answer (200, 202) does not name one."""
    if not outcome["passed"]:
        return GUARDRAILS
    pipeline = outcome.get("pipeline")
    if pipeline:
        return SERVICE_OF_STAGE.get(pipeline["stage"])
    return EXTRACTION if outcome.get("job") else None


def answer_guardrails(outcome: dict[str, Any]) -> int | float:
    """`guardrails` of a request that passed: the one kept with its first stage's job (a GET), else the one of the
    guardrails report the outcome carries (a POST); 0 when guardrails did not run."""
    kept = outcome.get("guardrails")
    if kept is not None:
        return kept
    value = guardrails_value(outcome)
    return 0 if value is None else value


def extract_response(
    outcome: dict[str, Any],
    *,
    request_id: str,
    column_thresholds: Mapping[str, float] | None = None,
) -> tuple[int, dict[str, Any]]:
    """`column_thresholds` is the central orchestrator's column_confidence_threshold: a field it leaves out gets
    the trust model's probability as its confidence. A request without a guardrails threshold passed whatever
    the model said, and its `guardrails` is the accepted probability (`answer_guardrails`)."""
    stage_name = last_stage(outcome)
    if not outcome["passed"]:
        body = extract_body(
            400,
            outcome["reason"],
            errors=REJECTED_CODE,
            guardrails=GUARDRAILS_REJECTED,
            request_id=request_id,
            pipeline_last_stage=stage_name,
        )
        return 400, body
    pipeline = outcome["pipeline"] or {}
    if pipeline.get("status") == STATUS_REJECTED:
        # A rejecting check of the structuring rules: answered like a guardrails rejection, with the
        # rules' own Indonesian reason as the message. `guardrails` stays 1 also without a guardrails threshold:
        # the central orchestrator recognises a rejection by it.
        return 400, extract_body(
            400,
            pipeline["error_message"],
            errors=REJECTED_CODE,
            guardrails=GUARDRAILS_REJECTED,
            request_id=request_id,
            pipeline_last_stage=stage_name,
        )
    if pipeline.get("status") == STATUS_DONE:
        # Scoring ended the request: the contract's fields. An earlier last service of the
        # pipeline_name_sequence: its result as it is.
        result = outcome["result"]
        data = contract_fields(result, None, column_thresholds) if pipeline.get("stage") == STAGE_SCORING else result
        return 200, extract_body(
            200, COMPLETED_MESSAGE, data=data, guardrails=answer_guardrails(outcome), request_id=request_id
        )
    if pipeline.get("status") == STATUS_FAILED:
        stage = pipeline["stage"]
        message = pipeline.get("error_message") or f"{stage} stage failed"
        return 422, extract_body(
            422,
            message,
            errors=f"{stage}_FAILED",
            guardrails=answer_guardrails(outcome),
            request_id=request_id,
            pipeline_last_stage=stage_name,
        )
    return 202, extract_body(202, PROCESSING_MESSAGE, request_id=request_id)
