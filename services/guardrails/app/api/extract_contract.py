from typing import Any

from ocr_common.npwp import contract_fields
from ocr_common.pipeline import STATUS_DONE, STATUS_FAILED
from ocr_common.web.envelope import envelope

COMPLETED_MESSAGE = "OCR extraction completed successfully"
PROCESSING_MESSAGE = "OCR job accepted; still processing"
REJECTED_CODE = "DOWNSTREAM_VALIDATION_ERROR"


def extract_body(
    status_code: int,
    message: str,
    *,
    request_id: str | None,
    document_type: str | None,
    data: dict[str, Any] | None = None,
    errors: str | None = None,
    job_status: str | None = None,
    guardrails: int | None = None,
) -> dict[str, Any]:
    return {
        **envelope(status_code, message, data, request_id, errors=errors),
        "document_type": document_type,
        "job_status": job_status,
        "guardrails": guardrails,
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
            guardrails=0,
            request_id=request_id,
            document_type=document_type,
        )
        return 400, body
    pipeline = outcome["pipeline"] or {}
    if pipeline.get("status") == STATUS_DONE:
        data = contract_fields(outcome["result"], threshold)
        return 200, extract_body(
            200,
            COMPLETED_MESSAGE,
            data=data,
            job_status="completed",
            guardrails=1,
            request_id=request_id,
            document_type=document_type,
        )
    if pipeline.get("status") == STATUS_FAILED:
        stage = pipeline["stage"]
        message = pipeline.get("error_message") or f"{stage} stage failed"
        return 422, extract_body(
            422,
            message,
            errors=f"{stage}_FAILED",
            job_status="failed",
            guardrails=1,
            request_id=request_id,
            document_type=document_type,
        )
    return 202, extract_body(
        202,
        PROCESSING_MESSAGE,
        job_status="processing",
        request_id=request_id,
        document_type=document_type,
    )
