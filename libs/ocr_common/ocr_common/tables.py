from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    text,
)

from ocr_common.database import JSON_TYPE, Base

PIPELINE_TABLE_PREFIXES = ("ocr", "structuring", "scoring")


def pipeline_tables(table_prefix: str, metadata: MetaData) -> tuple[Table, Table]:
    jobs = Table(
        f"{table_prefix}_jobs",
        metadata,
        Column("request_id", Text, primary_key=True),
        Column("status", Text, nullable=False),
        Column("error_message", Text, nullable=True),
        Column("attempts", Integer, nullable=False, server_default=text("1")),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(f"idx_{table_prefix}_jobs_status", "status"),
        Index(f"idx_{table_prefix}_jobs_ds", "ds"),
    )
    results = Table(
        f"{table_prefix}_results",
        metadata,
        Column("request_id", Text, ForeignKey(jobs.c.request_id), primary_key=True),
        Column("result", JSON_TYPE, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        Column("ds", Text, nullable=False),
        Index(f"idx_{table_prefix}_results_ds", "ds"),
    )
    return jobs, results


OCR_NPWP_REQUESTS = Table(
    "ocr_npwp_requests",
    Base.metadata,
    Column("request_id", Text, primary_key=True),
    Column("status", Text, nullable=False),
    Column("result", JSON_TYPE, nullable=True),
    Column("guardrails", Double, nullable=True),
    Column("error_message", Text, nullable=True),
    Column("file_name", Text, nullable=True),
    Column("file_size_bytes", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("ds", Text, nullable=False),
    Index("idx_ocr_npwp_requests_status", "status"),
    Index("idx_ocr_npwp_requests_created_at", "created_at"),
    Index("idx_ocr_npwp_requests_ds", "ds"),
)


def outbox_table(metadata: MetaData) -> Table:
    return Table(
        "pipeline_outbox",
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
            "idx_pipeline_outbox_due",
            "stage",
            "next_attempt_at",
            "id",
            postgresql_where=text("failed_at IS NULL"),
            sqlite_where=text("failed_at IS NULL"),
        ),
        Index(
            "idx_pipeline_outbox_dead",
            "stage",
            postgresql_where=text("failed_at IS NOT NULL"),
            sqlite_where=text("failed_at IS NOT NULL"),
        ),
        Index("idx_pipeline_outbox_request_id", "request_id"),
    )


def orchestration_outcome_table(name: str) -> Table:
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


def repo_metadata() -> MetaData:
    metadata = MetaData()
    for table_prefix in PIPELINE_TABLE_PREFIXES:
        pipeline_tables(table_prefix, metadata)
    outbox_table(metadata)
    OCR_NPWP_REQUESTS.to_metadata(metadata)
    return metadata
