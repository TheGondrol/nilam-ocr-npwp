"""Elastic APM: a trace of every HTTP request and every background job, with the calls they make (httpx to the
other services, asyncpg / SQLAlchemy to the database), sent to the APM server at `ELASTIC_APM_SERVER_URL`.

Without `ELASTIC_APM_SERVER_URL` the agent is not loaded at all and every function here does nothing, so a
service without an APM server (local, CI, a cluster where it is not set up yet) runs exactly as before.

Request and response bodies and headers are never captured: the documents and their fields (NPWP numbers, names)
must not leave for the APM server, and neither must the API keys. Every transaction carries the `request_id`
label, the same id as in the logs, and JSON log lines carry `trace.id` / `transaction.id` while one is active.

One request is one trace: a background job continues the trace of the HTTP request that queued it, and an outbox
message (callback, hand-off) carries the `traceparent` of the job that wrote it, so its delivery by the relay, and
the next stage's request and job after it, join the same trace.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ocr_common.config import BaseServiceSettings

logger = logging.getLogger(__name__)

# The agent's client once `start` made one; None while APM is off.
_client: Any = None


def start(settings: BaseServiceSettings, service_name: str) -> Any | None:
    """The APM client for this process, made from the settings, or None (and nothing loaded) when
    `ELASTIC_APM_SERVER_URL` is not set. `service_name` is the default of `ELASTIC_APM_SERVICE_NAME`."""
    global _client
    if not settings.elastic_apm_server_url:
        logger.info("Elastic APM off: ELASTIC_APM_SERVER_URL is not set")
        return None
    from elasticapm.contrib.starlette import make_apm_client

    config: dict[str, Any] = {
        "SERVICE_NAME": settings.elastic_apm_service_name or service_name,
        "SERVER_URL": settings.elastic_apm_server_url,
        "ENVIRONMENT": settings.elastic_apm_environment or settings.environment,
        "TRANSACTION_SAMPLE_RATE": settings.elastic_apm_transaction_sample_rate,
        "VERIFY_SERVER_CERT": settings.elastic_apm_verify_server_cert,
        "CAPTURE_BODY": "off",
        "CAPTURE_HEADERS": False,
    }
    if settings.elastic_apm_secret_token:
        config["SECRET_TOKEN"] = settings.elastic_apm_secret_token
    if settings.elastic_apm_api_key:
        config["API_KEY"] = settings.elastic_apm_api_key
    if settings.elastic_apm_sanitize_field_names:
        config["SANITIZE_FIELD_NAMES"] = settings.elastic_apm_sanitize_field_names
    _client = make_apm_client(config)
    logger.info("Elastic APM on: service %s, server %s", config["SERVICE_NAME"], config["SERVER_URL"])
    return _client


def stop() -> None:
    """Flushes and closes the client, if any (shutdown, tests)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def label_request_id(request_id: str | None) -> None:
    """Puts `request_id` on the current transaction, if APM is on and one is active."""
    if _client is None or not request_id:
        return
    import elasticapm

    elasticapm.label(request_id=request_id)


def trace_parent() -> str | None:
    """The W3C `traceparent` of the active transaction (its current span, if one is open), for work that continues
    its trace later: a background job, an outbox message. None while APM is off or nothing is active.

    Not `elasticapm.get_trace_parent_header()`: in a transaction that continues an incoming trace it still names
    the caller's span, so what follows would hang off the caller instead of this transaction."""
    if _client is None:
        return None
    from elasticapm.traces import execution_context

    transaction = execution_context.get_transaction()
    if transaction is None:
        return None
    span = execution_context.get_span()
    return transaction.trace_parent.copy_from(span_id=span.id if span is not None else transaction.id).to_string()


def _begin(client: Any, transaction_type: str, parent: str | None) -> None:
    from elasticapm.utils.disttracing import TraceParent

    client.begin_transaction(transaction_type, trace_parent=TraceParent.from_string(parent) if parent else None)


@contextmanager
def job_transaction(stage: str, request_id: str) -> Iterator[None]:
    """A transaction `<stage> job` around a background job (not part of the HTTP request that answered 202), so
    its OCR, model and database calls show up in APM, in the trace of the request that queued it (the task
    inherits that request's context). Its result is `done` unless `job_failed` marked it."""
    if _client is None:
        yield
        return
    import elasticapm

    client = _client
    _begin(client, "job", trace_parent())
    elasticapm.label(request_id=request_id, stage=stage)
    elasticapm.set_transaction_result("done")
    elasticapm.set_transaction_outcome("success")
    try:
        yield
    except BaseException:
        elasticapm.set_transaction_result("interrupted")
        elasticapm.set_transaction_outcome("failure")
        raise
    finally:
        client.end_transaction(f"{stage} job")


@contextmanager
def delivery_transaction(
    stage: str, kind: str, request_id: str, parent: str | None, *, attempt: int, target: str | None = None
) -> Iterator[None]:
    """A transaction `<stage> <kind>` (`callback` / `handoff`) around one delivery by the outbox relay, in the
    trace of the job that queued the message (`parent`, its `traceparent`; a new trace without one), so a
    callback to the central orchestrator shows up with the HTTP status it got. Its result is set by
    `delivery_result`; an exception marks it `error`."""
    if _client is None:
        yield
        return
    import elasticapm

    client = _client
    _begin(client, "outbox", parent)
    labels: dict[str, str | int] = {"request_id": request_id, "stage": stage, "kind": kind, "attempt": attempt}
    if target:
        labels["target"] = target
    elasticapm.label(**labels)
    try:
        yield
    except BaseException:
        elasticapm.set_transaction_result("error")
        elasticapm.set_transaction_outcome("failure")
        raise
    finally:
        client.end_transaction(f"{stage} {kind}")


# Results of a delivery: sent and accepted; nothing to send (an event that does not end the request, or no
# ORCHESTRATION_URL); to be sent again later; given up (dead letter).
DELIVERED = "delivered"
SKIPPED = "skipped"
RETRY = "retry"
DEAD = "dead"


def delivery_result(result: str) -> None:
    """The result of the current delivery transaction: `retry` and `dead` are failures."""
    if _client is None:
        return
    import elasticapm

    elasticapm.set_transaction_result(result)
    elasticapm.set_transaction_outcome("failure" if result in (RETRY, DEAD) else "success")


def job_failed(*, crashed: bool = False) -> None:
    """Marks the current job transaction failed; `crashed` also sends the exception being handled."""
    if _client is None:
        return
    import elasticapm

    elasticapm.set_transaction_result("failed")
    elasticapm.set_transaction_outcome("failure")
    if crashed:
        _client.capture_exception(handled=True)


def trace_fields() -> dict[str, str]:
    """`trace.id` and `transaction.id` of the active transaction for a log line (ECS names); empty when off."""
    if _client is None:
        return {}
    import elasticapm

    fields = {}
    if trace_id := elasticapm.get_trace_id():
        fields["trace.id"] = trace_id
    if transaction_id := elasticapm.get_transaction_id():
        fields["transaction.id"] = transaction_id
    return fields
