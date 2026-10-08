"""The two HTTP calls a stage makes when a job finishes: the callback to the orchestrator and the
hand-off to the next stage. `with_retry` is the retry policy of the direct (non-outbox) mode.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

from ocr_common.clients.remote import RemoteClientError, RemoteModelClient
from ocr_common.errors import ServiceError
from ocr_common.npwp import GUARDRAILS_PASSED, GUARDRAILS_REJECTED, REJECTED_CODE, contract_fields
from ocr_common.types import FinalResult

logger = logging.getLogger(__name__)

# The central orchestrator answers a result callback that arrives before it recorded its 202 for the
# request with 409 RESULT_NOT_READY; its contract (2 Oct 2026) says to send it again after 1-2 s, at
# most 5 times. Every other 4xx is final.
RESULT_NOT_READY = "RESULT_NOT_READY"
NOT_READY_RETRIES = 5
NOT_READY_DELAY_SECONDS = 1.5
# Set by the outbox on a stored message (the job's W3C traceparent, ocr_common/web/apm.py) so its delivery joins
# the job's trace; never sent.
TRACE_PARENT_KEY = "traceparent"


def not_ready(exc: ServiceError) -> bool:
    """The central orchestrator has not recorded the request's 202 yet: send the callback again shortly."""
    return isinstance(exc, RemoteClientError) and exc.status_code == 409 and exc.remote_code == RESULT_NOT_READY


async def with_retry(call: Callable[[], Awaitable[Any]], attempts: int, delay: float) -> Any:
    """Calls `call` up to `attempts` times, doubling `delay` between tries, on 5xx-class `ServiceError`s only;
    a 4xx is raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except ServiceError as exc:
            if exc.status_code < 500 or attempt >= attempts:
                raise
            await asyncio.sleep(delay * 2 ** (attempt - 1))


class StageCallback(Protocol):
    """What the pipeline needs from the callback to the orchestrator."""

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        error_code: str | None = None,
        final: bool = False,
        answer: dict[str, Any] | None = None,
        guardrails: int | float | None = None,
    ) -> bool:
        """Direct mode: build and send the callback with retries; returns False when it was skipped or gave up."""
        ...

    async def send(self, body: dict[str, Any]) -> bool | None:
        """Outbox mode: send an already-built callback body once; False when there was nothing to send.
        Raises `ServiceError` on failure."""
        ...

    async def aclose(self) -> None:
        """Close the HTTP client."""
        ...


class NextStage(Protocol):
    """What the pipeline needs from the client of the next stage."""

    async def submit(self, payload: dict[str, Any]) -> None:
        """Direct mode: POST the hand-off with retries."""
        ...

    async def send(self, payload: dict[str, Any]) -> None:
        """Outbox mode: POST the hand-off once; raises `ServiceError` on failure."""
        ...

    async def aclose(self) -> None:
        """Close the HTTP client."""
        ...


def stage_callback_body(
    request_id: str,
    stage: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    error_code: str | None = None,
    final: bool = False,
    answer: dict[str, Any] | None = None,
    guardrails: int | float | None = None,
) -> dict[str, Any]:
    """The per-stage callback body. `error_code` is only present when set: `DOWNSTREAM_VALIDATION_ERROR`
    on a rejection, so a FAILED callback tells a rejected document from a stage that broke. `final: true`
    is only present on the DONE of the stage that ends the request (the last of its
    pipeline_name_sequence), whose `result` is then the request's answer.

    `answer`, only on that final DONE, is the `data` the orchestrator's `extract-ocr` 200 answers with for
    this request (scoring: the confidences decided with the request's thresholds), and `guardrails` its
    `guardrails` (0, or the accepted probability when the request sent no guardrails threshold). Both are kept
    for the result callback, which must carry exactly that; the per-stage callback leaves them out."""
    body: dict[str, Any] = {
        "request_id": request_id,
        "stage": stage,
        "status": status,
        "result": result,
        "error_message": error_message,
    }
    if error_code is not None:
        body["error_code"] = error_code
    if final:
        body["final"] = True
    if answer is not None:
        body["answer"] = answer
    if guardrails is not None:
        body["guardrails"] = guardrails
    return body


def _sendable(body: dict[str, Any]) -> dict[str, Any]:
    """A stored stage body without what is kept for us only: the result callback's `answer` and `guardrails`, the
    trace parent."""
    return {key: value for key, value in body.items() if key not in ("answer", "guardrails", TRACE_PARENT_KEY)}


class OrchestrationCallback:
    """The callback to `ORCHESTRATION_URL`; with no client (URL unset) every call is skipped and logged."""

    def __init__(self, client: RemoteModelClient | None, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        error_code: str | None = None,
        final: bool = False,
        answer: dict[str, Any] | None = None,
        guardrails: int | float | None = None,
    ) -> bool:
        """Send `{request_id, stage, status, result, error_message[, error_code]}` with retries; False when
        skipped or failed. `answer` and `guardrails` are only for the result callback and are not sent."""
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s %s %s", request_id, stage, status)
            return False
        client = self._client
        payload = stage_callback_body(
            request_id, stage, status, result=result, error_message=error_message, error_code=error_code, final=final
        )
        try:
            await with_retry(lambda: client.post_json(self._path, payload), self._attempts, self._delay)
        except ServiceError as exc:
            logger.error("callback failed: %s %s %s: %s", request_id, stage, status, exc.message)
            return False
        return True

    async def send(self, body: dict[str, Any]) -> bool:
        """Send one callback body without retries (the outbox relay retries); False without ORCHESTRATION_URL."""
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return False
        await self._client.post_json(self._path, _sendable(body))
        return True

    async def aclose(self) -> None:
        """Close the HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()


RESULT_COMPLETED = "completed"
RESULT_FAILED = "failed"
_FINAL_STAGE = "SCORING"
# The former FIELD_CONFIDENCE_THRESHOLD default: only for a SCORING body queued before `answer` existed.
_LEGACY_THRESHOLD = 0.5


def result_callback_body(stage_body: dict[str, Any]) -> dict[str, Any] | None:
    """The central orchestrator's result callback (`ORCHESTRATION_CALLBACK_FORMAT=result`, its contract
    "Callback Hasil OCR" of 2 Oct 2026) for a per-stage callback body, or None when that stage event
    is not the end of the request.

    Completed (DONE of the stage that ends the request: scoring, or the last of a shorter
    pipeline_name_sequence). `result` and `guardrails` are exactly the `data` and `guardrails` the extract-ocr
    200 answers with for the same request (scoring: `nomor_npwp` and `nama` with 0/1 confidences, or the trust
    model's probabilities for the fields the request sent no threshold for; an earlier stage: its result as it
    is; `guardrails` 0, or the accepted probability when the request sent no guardrails threshold)::

        {"request_id", "status": "completed", "result": {...}, "guardrails": 0 | 0.9821}

    Rejected by the structuring rules (a FAILED with `DOWNSTREAM_VALIDATION_ERROR`): the orchestrator
    recognises a rejection by `result: null` with `guardrails: 1`, and passes `message` to its client::

        {"request_id", "status": "completed", "result": null, "guardrails": 1,
         "message": "<the rules' reason>", "error_code": "DOWNSTREAM_VALIDATION_ERROR"}

    Failed (any other FAILED)::

        {"request_id", "status": "failed", "error_code": "<STAGE>_FAILED", "message": str}

    `error_code` is outside the contract: the orchestrator ignores it for now and plans to pass it on
    like the synchronous answer does."""
    request_id, stage, status = stage_body["request_id"], stage_body["stage"], stage_body["status"]
    error_code = stage_body.get("error_code")
    message = stage_body.get("error_message")
    if status == "FAILED" and error_code == REJECTED_CODE:
        return {
            "request_id": request_id,
            "status": RESULT_COMPLETED,
            "result": None,
            "guardrails": GUARDRAILS_REJECTED,
            "message": message,
            "error_code": REJECTED_CODE,
        }
    if status == "FAILED":
        return {
            "request_id": request_id,
            "status": RESULT_FAILED,
            "error_code": error_code or f"{stage}_FAILED",
            "message": message,
        }
    # A body without `final` was queued before pipeline_name_sequence existed: only SCORING ended a request.
    ends_request = stage_body.get("final", stage == _FINAL_STAGE)
    if status != "DONE" or not ends_request:
        return None
    answer = stage_body.get("answer")
    if answer is None:
        answer = _legacy_answer(stage, stage_body.get("result"))
    if answer is None:
        return None
    # A body queued before `guardrails` was kept with it: the 0 every completed callback had then.
    guardrails = stage_body.get("guardrails", GUARDRAILS_PASSED)
    return {"request_id": request_id, "status": RESULT_COMPLETED, "result": answer, "guardrails": guardrails}


def _legacy_answer(stage: str, final: dict[str, Any] | None) -> dict[str, Any] | None:
    """The answer of a DONE body queued before `answer` existed: an earlier stage's result as it is; for
    scoring, the 0/1 confidences decided with the former FIELD_CONFIDENCE_THRESHOLD default (the request's own
    thresholds were not kept with the body)."""
    if not final:
        return None
    if stage != _FINAL_STAGE:
        return final
    return dict(contract_fields(cast(FinalResult, final), _LEGACY_THRESHOLD))


class ResultCallback:
    """The orchestrator's single result callback (`ORCHESTRATION_CALLBACK_FORMAT=result`): one POST per
    request when it ends, completed by the last stage of its pipeline_name_sequence, rejected, or failed
    at any stage. It takes the same per-stage events as `OrchestrationCallback` (so the pipeline and the
    outbox are unchanged) and turns them into that body; the events that do not end a request (a `DONE`
    without `final`) are skipped."""

    def __init__(self, client: RemoteModelClient | None, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        error_code: str | None = None,
        final: bool = False,
        answer: dict[str, Any] | None = None,
        guardrails: int | float | None = None,
    ) -> bool:
        """Send the result callback with retries (5xx, and a 409 RESULT_NOT_READY a few times); False when
        this event is not final, or when it failed."""
        body = result_callback_body(
            stage_callback_body(
                request_id,
                stage,
                status,
                result=result,
                error_message=error_message,
                error_code=error_code,
                final=final,
                answer=answer,
                guardrails=guardrails,
            )
        )
        if body is None:
            return False
        if self._client is None:
            logger.info("result callback skipped (ORCHESTRATION_URL not set): %s %s", request_id, body["status"])
            return False
        client = self._client
        try:
            for retry in range(NOT_READY_RETRIES + 1):
                try:
                    await with_retry(lambda: client.post_json(self._path, body), self._attempts, self._delay)
                    break
                except ServiceError as exc:
                    if not not_ready(exc) or retry == NOT_READY_RETRIES:
                        raise
                    await asyncio.sleep(NOT_READY_DELAY_SECONDS)
        except ServiceError as exc:
            logger.error("result callback failed: %s %s: %s", request_id, body["status"], exc.message)
            return False
        return True

    async def send(self, body: dict[str, Any]) -> bool:
        """Outbox mode: turn a stored per-stage body into the result callback and send it once; nothing (False)
        for an event that does not end the request, or without ORCHESTRATION_URL."""
        result_body = result_callback_body(body)
        if result_body is None:
            return False
        if self._client is None:
            logger.info("result callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return False
        await self._client.post_json(self._path, result_body)
        return True

    async def aclose(self) -> None:
        """Close the HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()


Delivered = Callable[[dict[str, Any]], Awaitable[None]]
"""Called with the per-stage body of a callback that ended the request, once it reached the orchestrator."""


class LoggedCallback:
    """Wraps the callback to the orchestrator: every callback that ends a request (`result_callback_body` is not
    None: completed, rejected or failed) and was actually sent (the orchestrator answered 2xx) is passed to
    `delivered`, once, after the send; a skipped one (no ORCHESTRATION_URL) or a failed try is not. The same for
    the direct mode (`notify`) and the outbox relay (`send`)."""

    def __init__(self, inner: OrchestrationCallback | ResultCallback, delivered: Delivered):
        self.inner = inner
        self._delivered = delivered

    async def notify(
        self,
        request_id: str,
        stage: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        error_code: str | None = None,
        final: bool = False,
        answer: dict[str, Any] | None = None,
        guardrails: int | float | None = None,
    ) -> bool:
        """`inner.notify`, then `delivered` when it sent a callback that ends the request."""
        sent = await self.inner.notify(
            request_id,
            stage,
            status,
            result=result,
            error_message=error_message,
            error_code=error_code,
            final=final,
            answer=answer,
            guardrails=guardrails,
        )
        body = stage_callback_body(
            request_id,
            stage,
            status,
            result=result,
            error_message=error_message,
            error_code=error_code,
            final=final,
            answer=answer,
            guardrails=guardrails,
        )
        if sent and result_callback_body(body) is not None:
            await self._delivered(body)
        return sent

    async def send(self, body: dict[str, Any]) -> bool:
        """`inner.send` (raises on failure, so the relay retries), then `delivered` when it sent a callback that
        ends the request."""
        sent = await self.inner.send(body)
        if sent and result_callback_body(body) is not None:
            await self._delivered(body)
        return sent

    async def aclose(self) -> None:
        await self.inner.aclose()


class NextStageClient:
    """POSTs the hand-off body to the next stage's `/v1/<stage>/jobs`."""

    def __init__(self, client: RemoteModelClient, path: str, *, attempts: int = 3, delay: float = 0.5):
        self._client = client
        self._path = path
        self._attempts = attempts
        self._delay = delay

    async def submit(self, payload: dict[str, Any]) -> None:
        """POST with retries (direct mode)."""
        await with_retry(lambda: self._client.post_json(self._path, payload), self._attempts, self._delay)

    async def send(self, payload: dict[str, Any]) -> None:
        """POST once (outbox mode)."""
        await self._client.post_json(self._path, payload)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
