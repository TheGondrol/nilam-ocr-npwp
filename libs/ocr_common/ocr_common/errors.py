"""Domain errors of the services. Each carries the HTTP status it maps to, and `create_app` turns
any of them into the standard error envelope in one exception handler, so routes and services raise
and never translate.

Use the named subclasses when raising; `ServiceError(status_code, message)` itself is for the one
case where the status is data (a remote service's 4xx passed through unchanged).

`code` is the stable, machine-readable value of the envelope's `errors` (the API spec's Error Codes): the
client branches on it, never on the wording of `message`. Without one, `error_code` falls back to the code of
the status."""

# The code of an error raised without a more precise one.
DEFAULT_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "REQUEST_ID_NOT_FOUND",
    409: "CONFLICT",
    413: "FILE_TOO_LARGE",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_SERVER_ERROR",
    503: "DOWNSTREAM_UNAVAILABLE",
    504: "DOWNSTREAM_TIMEOUT",
}

# The codes of the document checks (400, 413).
EMPTY_FILE = "EMPTY_FILE"
UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
UNREADABLE_FILE = "UNREADABLE_FILE"
TOO_MANY_PAGES = "TOO_MANY_PAGES"
INVALID_FILE_SOURCE = "INVALID_FILE_SOURCE"
FILE_URL_REJECTED = "FILE_URL_REJECTED"
FILE_TOO_LARGE = "FILE_TOO_LARGE"


def error_code(status_code: int, code: str | None = None) -> str:
    """`code`, else the default code of `status_code` (`ERROR` for a status without one)."""
    return code or DEFAULT_CODES.get(status_code, "ERROR")


class ServiceError(Exception):
    """A failure with a known HTTP status. `message` is safe to show to the caller; `code` is the stable
    `errors` value (None: the default code of the status)."""

    status_code: int = 500

    def __init__(self, status_code: int, message: str, code: str | None = None):
        self.status_code = status_code
        self.message = message
        self.code = code
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """5xx: the caller may try again unchanged; 4xx: it must change the request first."""
        return self.status_code >= 500


class _StatusError(ServiceError):
    def __init__(self, message: str, code: str | None = None):
        super().__init__(type(self).status_code, message, code)


class BadRequest(_StatusError):
    """400: the request or the document is unusable as sent."""

    status_code = 400


class NotFound(_StatusError):
    """404: no job or request for this request_id."""

    status_code = 404


class Conflict(_StatusError):
    """409: the request clashes with the current state (a request_id already processed, an outbox that is off)."""

    status_code = 409


class PayloadTooLarge(_StatusError):
    """413: the document exceeds `MAX_UPLOAD_BYTES`; the caller must send a smaller file."""

    status_code = 413


class UnprocessableEntity(_StatusError):
    """422: the request is well-formed but cannot be acted on."""

    status_code = 422


class InternalError(_StatusError):
    """500: this service or its model failed, or a dependency answered in an unexpected shape."""

    status_code = 500


class UpstreamUnavailable(_StatusError):
    """503: a dependency (model service, next stage, database) could not be reached."""

    status_code = 503


class UpstreamTimeout(_StatusError):
    """504: a dependency did not answer within its timeout."""

    status_code = 504
