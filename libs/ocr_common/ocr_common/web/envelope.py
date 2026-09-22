from typing import Any

STATUS_DESC = {
    200: "OK",
    202: "Accepted",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def envelope(
    status_code: int,
    message: str,
    data: Any,
    request_id: str | None,
    errors: str | None = None,
    guardrails: float | None = None,
) -> dict:
    body = {
        "status_code": status_code,
        "status_desc": STATUS_DESC.get(status_code, "Error"),
        "message": message,
        "data": data,
        "errors": errors,
        "request_id": request_id,
    }
    if guardrails is not None:
        body["guardrails"] = guardrails
    return body
