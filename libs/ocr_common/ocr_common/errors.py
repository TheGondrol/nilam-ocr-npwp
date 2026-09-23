"""Domain errors of the services. Each carries the HTTP status it maps to, and `create_app` turns
any of them into the standard error envelope in one exception handler, so routes and services raise
and never translate.

Use the named subclasses when raising; `ServiceError(status_code, message)` itself is for the one
case where the status is data (a remote service's 4xx passed through unchanged)."""


class ServiceError(Exception):
    """A failure with a known HTTP status. `message` is safe to show to the caller."""

    status_code: int = 500

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """5xx: the caller may try again unchanged; 4xx: it must change the request first."""
        return self.status_code >= 500


class _StatusError(ServiceError):
    def __init__(self, message: str):
        super().__init__(type(self).status_code, message)


class BadRequest(_StatusError):
    """400: the request or the document is unusable as sent."""

    status_code = 400


class NotFound(_StatusError):
    """404: no job or request for this request_id."""

    status_code = 404


class Conflict(_StatusError):
    """409: the request clashes with the current state (a request_id already processed, an outbox that is off)."""

    status_code = 409


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
