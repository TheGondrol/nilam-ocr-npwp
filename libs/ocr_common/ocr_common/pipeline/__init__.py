"""The asynchronous pipeline shared by the stage services: claim a job, run it in the background,
store the result, notify the orchestrator, hand the job to the next stage.

Modules:
  stage           StagePipeline, the per-service orchestration of one job
  sequence        pipeline_name_sequence: which services a request runs, and in which order
  repository      job status storage: Protocol, in-memory, and (repository_sql) PostgreSQL
  callbacks       HTTP callback to the orchestrator and hand-off to the next stage, with retries
  runner          background tasks and graceful drain on shutdown
  outbox          transactional outbox: messages written with the result, delivered by OutboxRelay
  outcomes        the orchestrator's own outcome row (ORCHESTRATION_OUTCOME_TABLE)
  outbox_status   schema and handler behind each service's GET /v1/<stage>/outbox
  results         reading an earlier stage's stored result (hand-off by reference; SQL in results_sql)
  reaper          running again the jobs a dead process left PROCESSING
  schemas         Pydantic models of the payloads passed between stages and to the orchestrator
  factory         build_* helpers that wire all of the above from settings
  tables/database SQLAlchemy tables and engine (need the `db` extra)

Nothing imported here may import SQLAlchemy at module level: the orchestrator uses this package without the
`db` extra. The SQL implementations (*_sql modules, database, tables) are imported lazily by the
factory; tests/test_imports.py enforces it.
"""

from ocr_common.pipeline.callbacks import (
    NextStage,
    NextStageClient,
    OrchestrationCallback,
    StageCallback,
    with_retry,
)
from ocr_common.pipeline.factory import (
    build_next_stage_client,
    build_outbox_relay,
    build_stage_pipeline,
    build_stage_results,
    build_stale_job_reaper,
    stage_client,
)
from ocr_common.pipeline.outbox import OutboxMessage, OutboxRelay, callback_message, handoff_message
from ocr_common.pipeline.reaper import StaleJobReaper
from ocr_common.pipeline.repository import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PROCESSING,
    InMemoryJobRepository,
    JobRecord,
    JobRepository,
    StaleJob,
    build_job_repository,
)
from ocr_common.pipeline.results import StageResults, load_upstream
from ocr_common.pipeline.runner import CANCEL_GRACE_SECONDS, BackgroundRunner
from ocr_common.pipeline.sequence import (
    DEFAULT_SEQUENCE,
    EXTRACTION,
    GUARDRAILS,
    PIPELINE_NAMES,
    SCORING,
    STAGE_OF,
    STRUCTURING,
    InvalidSequence,
    chain,
    checked_sequence,
    next_service,
    stored,
    validate_sequence,
)
from ocr_common.pipeline.stage import (
    STAGE_OCR,
    STAGE_SCORING,
    STAGE_STRUCTURING,
    CallbackResult,
    HandoffPayload,
    Rejection,
    StagePipeline,
    Work,
)

__all__ = [
    "CANCEL_GRACE_SECONDS",
    "DEFAULT_SEQUENCE",
    "EXTRACTION",
    "GUARDRAILS",
    "PIPELINE_NAMES",
    "SCORING",
    "STAGE_OCR",
    "STAGE_OF",
    "STAGE_SCORING",
    "STAGE_STRUCTURING",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PROCESSING",
    "STRUCTURING",
    "BackgroundRunner",
    "CallbackResult",
    "HandoffPayload",
    "InMemoryJobRepository",
    "InvalidSequence",
    "JobRecord",
    "JobRepository",
    "NextStage",
    "NextStageClient",
    "OrchestrationCallback",
    "OutboxMessage",
    "OutboxRelay",
    "Rejection",
    "StageCallback",
    "StagePipeline",
    "StageResults",
    "StaleJob",
    "StaleJobReaper",
    "Work",
    "build_job_repository",
    "build_next_stage_client",
    "build_outbox_relay",
    "build_stage_pipeline",
    "build_stage_results",
    "build_stale_job_reaper",
    "callback_message",
    "chain",
    "checked_sequence",
    "handoff_message",
    "load_upstream",
    "next_service",
    "stage_client",
    "stored",
    "validate_sequence",
    "with_retry",
]
