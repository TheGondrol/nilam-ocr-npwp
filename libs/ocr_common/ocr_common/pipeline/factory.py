"""Builds the pipeline objects from settings. This is the only module a service's
composition root (app/dependencies.py) needs for the async pipeline."""

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.config import PipelineSettings
from ocr_common.pipeline.callbacks import NextStage, NextStageClient, OrchestrationCallback
from ocr_common.pipeline.outbox import OutboxRelay
from ocr_common.pipeline.outcomes import build_stage_outcome
from ocr_common.pipeline.repository import build_job_repository
from ocr_common.pipeline.stage import StagePipeline


def stage_client(base_url: str, api_key: str, timeout: float, name: str) -> RemoteModelClient:
    return RemoteModelClient(
        base_url, timeout, name=name, headers={"X-API-Key": api_key}, passthrough_client_errors=True
    )


def build_stage_pipeline(
    settings: PipelineSettings, *, stage: str, table_prefix: str, next_stage: NextStage | None = None
) -> StagePipeline:
    client = None
    if settings.orchestration_url:
        client = stage_client(
            settings.orchestration_url,
            settings.orchestration_api_key or settings.api_key,
            settings.orchestration_timeout_seconds,
            "orchestration callback",
        )
    callback = OrchestrationCallback(
        client,
        settings.orchestration_callback_path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )
    outbox = None
    if settings.pipeline_outbox and settings.database_url:
        from ocr_common.pipeline.outbox_sql import SqlOutbox

        outbox = SqlOutbox(settings.database_url)
    repository = build_job_repository(
        settings.database_url,
        table_prefix,
        lease_seconds=settings.pipeline_job_lease_seconds,
        outcome=build_stage_outcome(settings, stage=stage),
        outbox=outbox,
        stage=stage,
    )
    return StagePipeline(
        stage=stage,
        repository=repository,
        callback=callback,
        next_stage_client=next_stage,
        outbox=outbox,
        callbacks=settings.callbacks_enabled,
    )


def build_outbox_relay(settings: PipelineSettings, pipeline: StagePipeline) -> OutboxRelay | None:
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
    )


def build_next_stage_client(
    settings: PipelineSettings, *, base_url: str, api_key: str | None, timeout: float, path: str, name: str
) -> NextStageClient:
    return NextStageClient(
        stage_client(base_url, api_key or settings.api_key, timeout, name),
        path,
        attempts=settings.pipeline_retry_attempts,
        delay=settings.pipeline_retry_delay_seconds,
    )
