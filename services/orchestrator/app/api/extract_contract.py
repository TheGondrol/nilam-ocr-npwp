from collections.abc import Mapping
from typing import Any

from ocr_common.npwp import REJECTED_CODE, contract_fields
from ocr_common.pipeline import STATUS_DONE, STATUS_FAILED
from ocr_common.web.envelope import envelope

from app.services.pipeline_waiter import STATUS_REJECTED

COMPLETED_MESSAGE = "OCR extraction completed successfully"
PROCESSING_MESSAGE = "OCR job accepted; still processing"


def extract_body(
    status_code: int,
    message: str,
    *,
    request_id: str | None,
    document_type: str | None,
    data: Mapping[str, Any] | None = None,
    errors: str | None = None,
    job_status: str | None = None,
    guardrails: int | None = None,
    params: Any = None,
) -> dict[str, Any]:
    return {
        **envelope(status_code, message, dict(data) if data is not None else None, request_id, errors=errors),
        "document_type": document_type,
        "job_status": job_status,
        "guardrails": guardrails,
        "params": params,
    }


def extract_response(
    outcome: dict[str, Any], *, request_id: str, document_type: str, params: Any, threshold: float
) -> tuple[int, dict[str, Any]]:
    if not outcome["passed"]:
        body = extract_body(
            400,
            outcome["reason"],
            errors=REJECTED_CODE,
            job_status="failed",
            guardrails=1,
            request_id=request_id,
            document_type=document_type,
            params=params,
        )
        return 400, body
    pipeline = outcome["pipeline"] or {}
    if pipeline.get("status") == STATUS_REJECTED:
        # A rejecting check of the structuring rules: answered like a guardrails rejection, with the
        # rules' own Indonesian reason as the message.
        return 400, extract_body(
            400,
            pipeline["error_message"],
            errors=REJECTED_CODE,
            job_status="failed",
            guardrails=1,
            request_id=request_id,
            document_type=document_type,
            params=params,
        )
    if pipeline.get("status") == STATUS_DONE:
        data = contract_fields(outcome["result"], threshold)
        return 200, extract_body(
            200,
            COMPLETED_MESSAGE,
            data=data,
            job_status="completed",
            guardrails=0,
            request_id=request_id,
            document_type=document_type,
            params=params,
        )
    if pipeline.get("status") == STATUS_FAILED:
        stage = pipeline["stage"]
        message = pipeline.get("error_message") or f"{stage} stage failed"
        return 422, extract_body(
            422,
            message,
            errors=f"{stage}_FAILED",
            job_status="failed",
            guardrails=0,
            request_id=request_id,
            document_type=document_type,
            params=params,
        )
    return 202, extract_body(
        202,
        PROCESSING_MESSAGE,
        job_status="processing",
        request_id=request_id,
        document_type=document_type,
        params=params,
    )
