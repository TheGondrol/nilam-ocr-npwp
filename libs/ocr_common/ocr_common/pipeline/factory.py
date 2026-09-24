"""Builds the pipeline objects from settings. This is the only module a service's
composition root (app/dependencies.py) needs for the async pipeline."""

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.config import PipelineSettings
from ocr_common.pipeline.callbacks import NextStage, NextStageClient, OrchestrationCallback, ResultCallback
from ocr_common.pipeline.outbox import OutboxRelay
from ocr_common.pipeline.outcomes import build_stage_outcome
from ocr_common.pipeline.reaper import Resume, StaleJobReaper
from ocr_common.pipeline.repository import build_job_repository
from ocr_common.pipeline.results import StageResults
from ocr_common.pipeline.stage import StagePipeline
from ocr_common.testing_endpoints import TESTING_TABLE_PREFIX, testing_metrics_stage


def stage_client(base_url: str, api_key: str, timeout: float, name: str) -> RemoteModelClient:
    """A `RemoteModelClient` to another stage or the orchestrator, sending `X-API-Key` and passing 4xx through."""
    return RemoteModelClient(
        base_url, timeout, name=name, headers={"X-API-Key": api_key}, passthrough_client_errors=True
    )


def build_callback(settings: PipelineSettings) -> OrchestrationCallback | ResultCallback:
    """The callback to the orchestrator in the format it expects (`ORCHESTRATION_CALLBACK_FORMAT`): one per
    stage with `X-API-Key`, or one result callback per request with `X-Callback-Key`. Without
    `ORCHESTRATION_URL` it has no client and skips every call."""
    result_format = settings.orchestration_callback_format == "result"
    client = None
    if settings.orchestration_url and result_format:
        client = RemoteModelClient(
            settings.orchestration_url,
            settings.orchestration_timeout_seconds,
            name="orchestration result callback",
            headers={"X-Callback-Key": settings.orchestration_callback_key or ""},
            passthrough_client_errors=True,
        )
    elif settings.orchestration_url:
        client = stage_client(
            settings.orchestration_url,
            settings.orchestration_api_key or settings.api_key,
            settings.orchestration_timeout_seconds,
            "orchestration callback",
        )
    callback_class = ResultCallback if result_format else OrchestrationCallback
    return callback_class(
        client,
        settings.orchestration_callback_path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )


def build_stage_pipeline(
    settings: PipelineSettings,
    *,
    stage: str,
    table_prefix: str,
    next_stage: NextStage | None = None,
    testing: bool = False,
) -> StagePipeline:
    """The `StagePipeline` of a service from its settings: callback client, outbox, repository, outcome row.

    `testing=True` builds the pipeline behind the `-test` endpoints: the same work on the `testing_*` tables
    and `testing_pipeline_outbox`, with no callback and no write to the orchestrator's tables, so a load test
    never reaches the orchestrator. `next_stage` must then point at the next stage's `-test` endpoint."""
    lane_prefix = TESTING_TABLE_PREFIX if testing else ""
    outbox = None
    if settings.pipeline_outbox and settings.database_url:
        from ocr_common.pipeline.outbox_sql import SqlOutbox

        outbox = SqlOutbox(settings.database_url, lane_prefix)
    repository = build_job_repository(
        settings.database_url,
        f"{lane_prefix}{table_prefix}",
        lease_seconds=settings.pipeline_job_lease_seconds,
        outcome=None if testing else build_stage_outcome(settings, stage=stage),
        outbox=outbox,
        stage=stage,
    )
    return StagePipeline(
        stage=stage,
        repository=repository,
        callback=OrchestrationCallback(None, settings.orchestration_callback_path)
        if testing
        else build_callback(settings),
        next_stage_client=next_stage,
        outbox=outbox,
        callbacks=False if testing else settings.callbacks_enabled,
        metrics_stage=testing_metrics_stage(stage) if testing else None,
    )


def build_outbox_relay(settings: PipelineSettings, pipeline: StagePipeline) -> OutboxRelay | None:
    """The relay that delivers this stage's outbox messages; None when the outbox is off."""
    from ocr_common.pipeline.outbox_sql import SqlOutbox

    if not isinstance(pipeline.outbox, SqlOutbox):
        return None
    return OutboxRelay(
        pipeline.outbox,
        stage=pipeline.stage,
        callback=pipeline.callback,
        next_stage=pipeline.next_stage_client,
        callbacks=pipeline.callbacks,
        handoff_failed=pipeline.repository.handoff_failed,
        interval_seconds=settings.pipeline_outbox_interval_seconds,
        batch=settings.pipeline_outbox_batch,
        lease_seconds=settings.pipeline_outbox_lease_seconds,
        retry_delay_seconds=settings.pipeline_retry_delay_seconds,
        max_backoff_seconds=settings.pipeline_outbox_max_backoff_seconds,
        max_age_seconds=settings.pipeline_outbox_max_age_seconds,
        stale_after_seconds=settings.pipeline_outbox_stale_after_seconds,
        metrics_stage=pipeline.metrics_stage,
    )


def build_stale_job_reaper(
    settings: PipelineSettings, pipeline: StagePipeline, resume: Resume
) -> StaleJobReaper | None:
    """The per-process task that runs again jobs a dead process left PROCESSING. Only with a database:
    in-memory jobs die with the process anyway."""
    if not settings.database_url or not settings.pipeline_stale_jobs:
        return None
    return StaleJobReaper(
        pipeline.repository,
        resume,
        stage=pipeline.stage,
        interval_seconds=settings.pipeline_stale_job_interval_seconds,
        batch=settings.pipeline_stale_job_batch,
        metrics_stage=pipeline.metrics_stage,
    )


def build_stage_results(settings: PipelineSettings, *, testing: bool = False) -> StageResults | None:
    """Reader of earlier stages' results, for hand-offs that arrive by reference; None without a database.
    `testing=True` reads the `testing_*` tables."""
    if not settings.database_url:
        return None
    from ocr_common.pipeline.results_sql import SqlStageResults

    return SqlStageResults(settings.database_url, TESTING_TABLE_PREFIX if testing else "")


def build_next_stage_client(
    settings: PipelineSettings, *, base_url: str, api_key: str | None, timeout: float, path: str, name: str
) -> NextStageClient:
    """The client that hands jobs to the next stage at `base_url + path`."""
    return NextStageClient(
        stage_client(base_url, api_key or settings.api_key, timeout, name),
        path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )
