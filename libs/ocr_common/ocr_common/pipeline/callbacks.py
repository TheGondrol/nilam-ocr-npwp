"""The two HTTP calls a stage makes when a job finishes: the callback to the orchestrator and the
hand-off to the next stage. `with_retry` is the retry policy of the direct (non-outbox) mode.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError

logger = logging.getLogger(__name__)


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
    ) -> bool:
        """Direct mode: build and send the callback with retries; returns False when it was skipped or gave up."""
        ...

    async def send(self, body: dict[str, Any]) -> None:
        """Outbox mode: send an already-built callback body once; raises `ServiceError` on failure."""
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
) -> dict[str, Any]:
    """The per-stage callback body. `error_code` is only present when set: `DOWNSTREAM_VALIDATION_ERROR`
    on a rejection, so a FAILED callback tells a rejected document from a stage that broke. `final: true`
    is only present on the DONE of the stage that ends the request (the last of its
    pipeline_name_sequence), whose `result` is then the request's answer."""
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
    return body


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
    ) -> bool:
        """Send `{request_id, stage, status, result, error_message[, error_code]}` with retries; False when
        skipped or failed."""
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

    async def send(self, body: dict[str, Any]) -> None:
        """Send one callback body without retries (the outbox relay retries)."""
        if self._client is None:
            logger.info("callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return
        await self._client.post_json(self._path, body)

    async def aclose(self) -> None:
        """Close the HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()


RESULT_COMPLETED = "completed"
RESULT_FAILED = "failed"
RESULT_FIELDS = ("nomor_npwp", "nama", "nama_badan")
_FINAL_STAGE = "SCORING"
_NAME_FIELDS = ("nama", "nama_badan")


def result_callback_body(stage_body: dict[str, Any]) -> dict[str, Any] | None:
    """The orchestrator's result callback (`ORCHESTRATION_CALLBACK_FORMAT=result`) for a per-stage callback
    body, or None when that stage event is not the end of the request.

    Completed by scoring (SCORING `DONE`, `result` = the final result)::

        {"request_id", "status": "completed",
         "result": {"nomor_npwp" | "nama" | "nama_badan": {"value": str, "confidence": float}},
         "guardrails": {...the guardrails report...}}

    `value` is "" when the field was not found and `confidence` is the trust model's probability that
    the value is correct (0.0 when not found); the name probability goes to whichever of `nama` /
    `nama_badan` holds the name. Completed by an earlier stage (a pipeline_name_sequence that ends before
    scoring; its DONE carries `final: true`): `result` is that stage's result as it is (the OCR result,
    or the structuring result) and `guardrails` is `{}`. Failed (any stage `FAILED`, including a
    rejection)::

        {"request_id", "status": "failed", "result": null, "guardrails": {},
         "error_code": "<STAGE>_FAILED" | "DOWNSTREAM_VALIDATION_ERROR", "error_message": str}
    """
    request_id, stage, status = stage_body["request_id"], stage_body["stage"], stage_body["status"]
    if status == "FAILED":
        return {
            "request_id": request_id,
            "status": RESULT_FAILED,
            "result": None,
            "guardrails": {},
            "error_code": stage_body.get("error_code") or f"{stage}_FAILED",
            "error_message": stage_body.get("error_message"),
        }
    final = stage_body.get("result")
    # A body without `final` was queued before pipeline_name_sequence existed: only SCORING ended a request.
    ends_request = stage_body.get("final", stage == _FINAL_STAGE)
    if status != "DONE" or not ends_request or not final:
        return None
    if stage != _FINAL_STAGE:
        return {"request_id": request_id, "status": RESULT_COMPLETED, "result": final, "guardrails": {}}
    fields = final.get("fields") or {}
    scoring = final.get("scoring") or {}
    probability = {"nomor_npwp": scoring.get("npwp_confidence")}
    probability.update(dict.fromkeys(_NAME_FIELDS, scoring.get("name_confidence")))
    result = {}
    for name in RESULT_FIELDS:
        value = (fields.get(name) or {}).get("value")
        found = value is not None and str(value).strip() != ""
        score = probability[name] if found else None
        result[name] = {"value": str(value) if found else "", "confidence": round(float(score or 0.0), 4)}
    return {
        "request_id": request_id,
        "status": RESULT_COMPLETED,
        "result": result,
        "guardrails": final.get("guardrails") or {},
    }


class ResultCallback:
    """The orchestrator's single result callback (`ORCHESTRATION_CALLBACK_FORMAT=result`): one POST per
    request when it ends, completed by the last stage of its pipeline_name_sequence or failed at any
    stage. It takes the same per-stage events as `OrchestrationCallback` (so the pipeline and the outbox
    are unchanged) and turns them into that body; the events that do not end a request (a `DONE` without
    `final`) are skipped."""

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
    ) -> bool:
        """Send the result callback with retries; False when this event is not final, or when it failed."""
        body = result_callback_body(
            stage_callback_body(
                request_id,
                stage,
                status,
                result=result,
                error_message=error_message,
                error_code=error_code,
                final=final,
            )
        )
        if body is None:
            return False
        if self._client is None:
            logger.info("result callback skipped (ORCHESTRATION_URL not set): %s %s", request_id, body["status"])
            return False
        client = self._client
        try:
            await with_retry(lambda: client.post_json(self._path, body), self._attempts, self._delay)
        except ServiceError as exc:
            logger.error("result callback failed: %s %s: %s", request_id, body["status"], exc.message)
            return False
        return True

    async def send(self, body: dict[str, Any]) -> None:
        """Outbox mode: turn a stored per-stage body into the result callback and send it once; nothing for
        an event that does not end the request."""
        result_body = result_callback_body(body)
        if result_body is None:
            return
        if self._client is None:
            logger.info("result callback skipped (ORCHESTRATION_URL not set): %s", body.get("request_id"))
            return
        await self._client.post_json(self._path, result_body)

    async def aclose(self) -> None:
        """Close the HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()


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
