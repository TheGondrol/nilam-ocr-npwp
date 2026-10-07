"""The tables of this repository, defined once here and used by the services, the Alembic migrations
and the tests. They all live in the schema `PIPELINE_SCHEMA` (`nilam_ocr_npwp`), not in `public`, and every name
starts with `TABLE_PREFIX` (`nilam_`): the callers name a table without it (`ocr`, `testing_`), the prefix is put
on here. The extraction stage's tables are `nilam_ocr_extraction_jobs` and `nilam_ocr_extraction_result` (0014):
`nilam_ocr_results` is the request's final answer, written by whichever service ends it.
`orchestration_outcome_table` and `orchestration_api_events_table` describe tables the orchestrator owns.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM

from ocr_common.pipeline.database import JSON_TYPE, PIPELINE_SCHEMA, TABLE_PREFIX
from ocr_common.testing_endpoints import TESTING_TABLE_PREFIX

PIPELINE_TABLE_PREFIXES = ("ocr", "structuring", "scoring")
# A stage whose tables are not `<prefix>_jobs` / `<prefix>_results` (the client's naming, 0014).
STAGE_TABLE_NAMES = {"ocr": ("ocr_extraction_jobs", "ocr_extraction_result")}


def stage_table_names(table_prefix: str) -> tuple[str, str]:
    """The jobs and results table names of `table_prefix` (`ocr`, `testing_ocr`, `structuring`, ...)."""
    stage = table_prefix.removeprefix(TESTING_TABLE_PREFIX)
    lane = table_prefix[: len(table_prefix) - len(stage)]
    jobs, results = STAGE_TABLE_NAMES.get(stage, (f"{stage}_jobs", f"{stage}_results"))
    return f"{TABLE_PREFIX}{lane}{jobs}", f"{TABLE_PREFIX}{lane}{results}"


def pipeline_tables(table_prefix: str, metadata: MetaData) -> tuple[Table, Table]:
    """The jobs and results tables of one stage on `metadata`: `nilam_<prefix>_jobs` and `nilam_<prefix>_results`,
    `nilam_ocr_extraction_jobs` and `nilam_ocr_extraction_result` for the extraction stage (`ocr`)."""
    jobs_name, results_name = stage_table_names(table_prefix)
    jobs = Table(
        jobs_name,
        metadata,
        Column("request_id", Text, primary_key=True),
        Column("status", Text, nullable=False),
        Column("error_message", Text, nullable=True),
        Column("attempts", Integer, nullable=False, server_default=text("1")),
        Column("input", JSON_TYPE, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(f"idx_{jobs_name}_status", "status"),
        Index(f"idx_{jobs_name}_ds", "ds"),
        schema=PIPELINE_SCHEMA,
    )
    results = Table(
        results_name,
        metadata,
        Column("request_id", Text, ForeignKey(jobs.c.request_id), primary_key=True),
        Column("result", JSON_TYPE, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(f"idx_{results_name}_ds", "ds"),
        schema=PIPELINE_SCHEMA,
    )
    return jobs, results


def outbox_table(metadata: MetaData, table_prefix: str = "") -> Table:
    """The `nilam_pipeline_outbox` table shared by the three stages (`nilam_testing_pipeline_outbox` with the
    testing prefix)."""
    name = f"{TABLE_PREFIX}{table_prefix}pipeline_outbox"
    return Table(
        name,
        metadata,
        Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
        Column("request_id", Text, nullable=False),
        Column("stage", Text, nullable=False),
        Column("kind", Text, nullable=False),
        Column("payload", JSON_TYPE, nullable=False),
        Column("attempts", Integer, nullable=False, server_default=text("0")),
        Column("next_attempt_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("failed_at", DateTime(timezone=True), nullable=True),
        Column("last_error", Text, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(
            f"idx_{name}_due",
            "stage",
            "next_attempt_at",
            "id",
            postgresql_where=text("failed_at IS NULL"),
            sqlite_where=text("failed_at IS NULL"),
        ),
        Index(
            f"idx_{name}_dead",
            "stage",
            postgresql_where=text("failed_at IS NOT NULL"),
            sqlite_where=text("failed_at IS NOT NULL"),
        ),
        Index(f"idx_{name}_request_id", "request_id"),
        schema=PIPELINE_SCHEMA,
    )


def guardrails_results_table(metadata: MetaData, table_prefix: str = "") -> Table:
    """`nilam_guardrails_results`: one row per guardrails verdict, written by the orchestrator NPWP, the rejected
    documents included (they never reach a stage table). Append-only: the same request_id sent again is
    judged again. `threshold_source` says whose threshold decided: `request` (the central orchestrator's,
    sent with the request) or `service` (the guardrails service's own). `pipeline_name_sequence` is the
    request's (null: the full pipeline), so the orchestrator's GET can answer a request that never reached a
    stage: guardrails only, or rejected here."""
    name = f"{TABLE_PREFIX}{table_prefix}guardrails_results"
    return Table(
        name,
        metadata,
        Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
        Column("request_id", Text, nullable=False),
        Column("passed", Boolean, nullable=False),
        Column("verdict", Text, nullable=True),
        Column("confidence", Float, nullable=True),
        Column("threshold", Float, nullable=True),
        Column("threshold_source", Text, nullable=False),
        Column("n_pages", Integer, nullable=True),
        Column("reason", Text, nullable=True),
        Column("pipeline_name_sequence", JSON_TYPE, nullable=True),
        Column("report", JSON_TYPE, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(f"idx_{name}_request_id", "request_id"),
        Index(f"idx_{name}_ds", "ds"),
        schema=PIPELINE_SCHEMA,
    )


def ocr_results_table(metadata: MetaData, table_prefix: str = "") -> Table:
    """`nilam_ocr_results`: the request's final answer, one row per request_id, in the shape of the extract-ocr
    answer. Written by the service that ends the request, in the same transaction as its own result: the last
    stage of the pipeline_name_sequence (completed, rejected by the structuring rules, or failed), or the
    orchestrator NPWP for guardrails (rejected, or the only service). `data` is the last service's result as it
    is (scoring: the contract's fields). `guardrails` is 0 passed, 1 rejected, null when the request left
    guardrails out. A request_id run again overwrites its row (`update_at`)."""
    name = f"{TABLE_PREFIX}{table_prefix}ocr_results"
    return Table(
        name,
        metadata,
        Column("status_code", Integer, nullable=False),
        Column("status_desc", Text, nullable=False),
        Column("message", Text, nullable=True),
        Column("data", JSON_TYPE, nullable=True),
        Column("errors", Text, nullable=True),
        Column("request_id", Text, primary_key=True),
        Column("guardrails", Integer, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("update_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        schema=PIPELINE_SCHEMA,
    )


def orchestration_outcome_table(name: str) -> Table:
    """The orchestrator's outcome table (`ORCHESTRATION_OUTCOME_TABLE`) as this code needs it; owned by them."""
    return Table(
        name,
        MetaData(),
        Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
        Column("request_id", Text, nullable=False, unique=True),
        Column("document_type", Text, nullable=False),
        Column("status_code", Integer, nullable=False),
        Column("downstream_status", Text, nullable=True),
        Column("downstream_stage", Text, nullable=True),
        Column("error_code", Text, nullable=True),
        Column("error_message", Text, nullable=True),
        Column("result_data", JSON_TYPE, nullable=True),
        Column("occurred_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
    )


# Labels of the orchestrator's enum types (schema `ocr` in the dev database), as they stand.
API_EVENT_ENDPOINTS = ("EXTRACT_OCR", "GET_OCR_RESULT")
API_EVENT_STATUSES = ("PENDING", "PROCESSING", "COMPLETED", "FAILED")
API_EVENT_STAGES = ("GUARDRAILS", "EXTRACTION", "SCORING", "STRUCTURING")


def orchestration_api_events_table(name: str) -> Table:
    """The orchestrator's API event log (`ORCHESTRATION_API_EVENTS_TABLE`, `schema.table` or `table`) as
    this code needs it; owned by them. Append-only: one row per event, no unique key on `request_id`.
    The enum types live in the table's schema and are never created from here."""
    schema, _, table_name = name.rpartition(".")
    schema = schema or None

    def enum(labels: tuple[str, ...], type_name: str) -> ENUM:
        return ENUM(*labels, name=type_name, schema=schema, create_type=False)

    return Table(
        table_name,
        MetaData(),
        Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
        Column("endpoint", enum(API_EVENT_ENDPOINTS, "endpoint"), nullable=False),
        Column("request_id", Text, nullable=False),
        Column("status_code", Integer, nullable=False),
        Column("error_code", Text, nullable=True),
        Column("result_data", JSON_TYPE, nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Column("downstream_status", enum(API_EVENT_STATUSES, "downstream_status"), nullable=True),
        Column("downstream_stage", enum(API_EVENT_STAGES, "downstream_stage"), nullable=True),
        Column("document_type", Text, nullable=True),
        schema=schema,
    )


def repo_metadata() -> MetaData:
    """Every table this repository migrates, for Alembic's autogenerate and `alembic check`: the stage tables
    the outbox, the guardrails verdicts and the final answers, again with the `testing_` prefix for the testing
    endpoints."""
    metadata = MetaData()
    for lane_prefix in ("", TESTING_TABLE_PREFIX):
        for table_prefix in PIPELINE_TABLE_PREFIXES:
            pipeline_tables(f"{lane_prefix}{table_prefix}", metadata)
        outbox_table(metadata, lane_prefix)
        guardrails_results_table(metadata, lane_prefix)
        ocr_results_table(metadata, lane_prefix)
    return metadata
