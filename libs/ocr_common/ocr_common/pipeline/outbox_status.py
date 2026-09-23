"""What a service needs to expose GET /v1/<stage>/outbox: the response schema, the documented
examples, and the handler. The route itself is declared in the service (app/api/jobs.py) so that
every endpoint of a service is visible in one place."""

from typing import Any

from pydantic import BaseModel, Field

from ocr_common.errors import Conflict
from ocr_common.pipeline.outbox_sql import SqlOutbox
from ocr_common.pipeline.stage import StagePipeline
from ocr_common.web.envelope import envelope
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, Stage, SuccessEnvelope, error, success_examples


class OutboxStatus(BaseModel):
    """The `data` of `GET /v1/<stage>/outbox`."""

    enabled: bool = Field(
        ..., description="false when `PIPELINE_OUTBOX` is off: callbacks are sent directly and nothing is queued"
    )
    stage: Stage = Field(..., description="The stage whose messages this service's relay delivers", examples=["OCR"])
    pending: int = Field(
        ..., description="Messages not yet delivered, including the ones a relay is sending right now", examples=[0]
    )
    retrying: int = Field(..., description="Of `pending`: messages that already failed at least once", examples=[0])
    oldest_pending_seconds: float | None = Field(
        None,
        description=(
            "Age of the oldest pending message; null when nothing is pending. Growing past "
            "`PIPELINE_OUTBOX_STALE_AFTER_SECONDS` means the orchestrator or the next stage is not taking messages"
        ),
        examples=[None],
    )
    dead_letters: int = Field(
        ...,
        description=(
            "Messages given up on (a 4xx from the receiver, or 5xx for longer than "
            "`PIPELINE_OUTBOX_MAX_AGE_SECONDS`). They stay in `pipeline_outbox` with `failed_at` and `last_error` "
            "for inspection and are never retried; anything above 0 needs a human"
        ),
        examples=[0],
    )


class OutboxStatusResponse(SuccessEnvelope):
    """Envelope of `GET /v1/<stage>/outbox`."""

    data: OutboxStatus


class OutboxRelease(BaseModel):
    """The `data` of `POST /v1/<stage>/outbox/release`."""

    stage: Stage = Field(..., description="The stage whose dead letters were released", examples=["OCR"])
    request_id: str | None = Field(
        None, description="Only this request's dead letters were released; null = every dead letter of the stage"
    )
    released: int = Field(..., description="Dead letters put back in the queue, due immediately", examples=[1])


class OutboxReleaseResponse(SuccessEnvelope):
    """Envelope of `POST /v1/<stage>/outbox/release`."""

    data: OutboxRelease


OUTBOX_STATUS_SUMMARY = "Backlog of this stage's undelivered callbacks and hand-offs"
OUTBOX_STATUS_DESCRIPTION = (
    "Operational view of the transactional outbox (`PIPELINE_OUTBOX`): how many callbacks and hand-offs "
    "of this stage are still to be delivered, how old the oldest one is, and how many were given up on "
    "(dead letters). No other service watches the `pipeline_outbox` table, so this is where a monitor "
    "should look; the relay also logs a warning when the backlog is older than "
    "`PIPELINE_OUTBOX_STALE_AFTER_SECONDS` or a dead letter is waiting."
)


def outbox_status_responses(stage: str) -> dict[int | str, dict[str, Any]]:
    """`responses=` for the route: the documented examples, with the stage of the service filled in."""
    example = _example(stage)
    return {
        200: success_examples(
            "The backlog of this stage",
            idle=("Nothing waiting", envelope(200, "Success", example, REQUEST_ID_EXAMPLE)),
            backlog=(
                "The orchestrator has been down for a while",
                envelope(
                    200,
                    "Success",
                    {**example, "pending": 12, "retrying": 12, "oldest_pending_seconds": 754.3, "dead_letters": 1},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            disabled=(
                "`PIPELINE_OUTBOX` is off",
                envelope(200, "Success", {**example, "enabled": False}, REQUEST_ID_EXAMPLE),
            ),
        ),
        401: UNAUTHORIZED,
    }


OUTBOX_RELEASE_SUMMARY = "Send this stage's dead letters again"
OUTBOX_RELEASE_DESCRIPTION = (
    "Puts the dead letters of this stage back in the outbox queue, due immediately, so the relay delivers "
    "them again: use it after the receiver was fixed (for example the orchestrator's callback contract, "
    "or a next stage that was answering 4xx). Optional query `request_id` limits it to one request. "
    "A message that fails again goes through the same retry and dead-letter rules. "
    "`409` when `PIPELINE_OUTBOX` is off on this service."
)


def outbox_release_responses(stage: str) -> dict[int | str, dict[str, Any]]:
    """`responses=` for the release route."""
    return {
        200: success_examples(
            "Dead letters released",
            released=(
                "One request",
                envelope(
                    200,
                    "Success",
                    {"stage": stage, "request_id": REQUEST_ID_EXAMPLE, "released": 1},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            nothing=(
                "Nothing to release",
                envelope(200, "Success", {"stage": stage, "request_id": None, "released": 0}, REQUEST_ID_EXAMPLE),
            ),
        ),
        401: UNAUTHORIZED,
        409: error(409, "The outbox is off on this service", "PIPELINE_OUTBOX is off: there is no outbox to release"),
        422: error(
            422, "Validation Error", "query.request_id: Input should be a valid string", errors="VALIDATION_ERROR"
        ),
    }


async def outbox_release(pipeline: StagePipeline, request_id: str | None = None) -> dict[str, Any]:
    """The `data` of the release response; 409 when the service runs without an outbox."""
    if not isinstance(pipeline.outbox, SqlOutbox):
        raise Conflict("PIPELINE_OUTBOX is off: there is no outbox to release")
    released = await pipeline.outbox.release(pipeline.stage, request_id)
    return {"stage": pipeline.stage, "request_id": request_id, "released": released}


async def outbox_status(pipeline: StagePipeline) -> dict[str, Any]:
    """The `data` of the response: live numbers when the outbox is on, zeros with `enabled: false` otherwise."""
    data = {**_example(pipeline.stage), "enabled": isinstance(pipeline.outbox, SqlOutbox)}
    if isinstance(pipeline.outbox, SqlOutbox):
        stats = await pipeline.outbox.stats(pipeline.stage)
        data.update(
            pending=stats.pending,
            retrying=stats.retrying,
            oldest_pending_seconds=stats.oldest_pending_seconds,
            dead_letters=stats.dead_letters,
        )
    return data


def _example(stage: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "stage": stage,
        "pending": 0,
        "retrying": 0,
        "oldest_pending_seconds": None,
        "dead_letters": 0,
    }
