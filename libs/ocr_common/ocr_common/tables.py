from sqlalchemy import (
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


def repo_metadata() -> MetaData:
    metadata = MetaData()
    for table_prefix in PIPELINE_TABLE_PREFIXES:
        pipeline_tables(table_prefix, metadata)
    OCR_NPWP_REQUESTS.to_metadata(metadata)
    return metadata
