from typing import Any

STATUS_DESC = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


def envelope(
    status_code: int,
    message: str,
    data: Any,
    request_id: str | None,
    errors: str | None = None,
    guardrails: float | None = None,
) -> dict:
    # guardrails saudara `data`, bukan isinya: skor itu milik dokumennya, bukan
    # salah satu fieldnya. Hanya muncul kalau ada, jadi respons error dan
    # generate-request-id tidak ikut membawa kunci kosong.
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
