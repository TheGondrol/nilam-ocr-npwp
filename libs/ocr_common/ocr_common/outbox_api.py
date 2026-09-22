from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ocr_common.envelope import envelope
from ocr_common.jobs import StagePipeline
from ocr_common.outbox_sql import SqlOutbox
from ocr_common.request_id import get_request_id
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, Stage, SuccessEnvelope, success_examples
from ocr_common.security import verify_api_key


class OutboxStatus(BaseModel):
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
    data: OutboxStatus


def outbox_status_router(prefix: str, pipeline: Callable[[], StagePipeline]) -> APIRouter:
    router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])
    example: dict[str, Any] = {
        "enabled": True,
        "stage": "OCR",
        "pending": 0,
        "retrying": 0,
        "oldest_pending_seconds": None,
        "dead_letters": 0,
    }

    @router.get(
        f"{prefix}/outbox",
        response_model=OutboxStatusResponse,
        operation_id=f"get{prefix.rsplit('/', 1)[-1].capitalize()}OutboxStatus",
        summary="Backlog of this stage's undelivered callbacks and hand-offs",
        description=(
            "Operational view of the transactional outbox (`PIPELINE_OUTBOX`): how many callbacks and hand-offs "
            "of this stage are still to be delivered, how old the oldest one is, and how many were given up on "
            "(dead letters). No other service watches the `pipeline_outbox` table, so this is where a monitor "
            "should look; the relay also logs a warning when the backlog is older than "
            "`PIPELINE_OUTBOX_STALE_AFTER_SECONDS` or a dead letter is waiting."
        ),
        responses={
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
        },
    )
    async def outbox_status(request: Request):
        stage = pipeline()
        data: dict[str, Any] = {**example, "stage": stage.stage, "enabled": isinstance(stage.outbox, SqlOutbox)}
        if isinstance(stage.outbox, SqlOutbox):
            stats = await stage.outbox.stats(stage.stage)
            data.update(
                pending=stats.pending,
                retrying=stats.retrying,
                oldest_pending_seconds=stats.oldest_pending_seconds,
                dead_letters=stats.dead_letters,
            )
        return envelope(200, "Success", data, get_request_id(request))

    return router
