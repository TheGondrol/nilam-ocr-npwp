"""Prometheus metrics of the asynchronous pipeline: jobs by outcome and duration, stale jobs run again,
outbox deliveries, and the outbox backlog. They are exposed with the HTTP metrics at `/metrics`.

What to alert on: `pipeline_jobs_total{outcome!="done"}` rising, `pipeline_outbox_oldest_pending_seconds`
above `PIPELINE_OUTBOX_STALE_AFTER_SECONDS`, and `pipeline_outbox_dead_letters` above zero."""

from prometheus_client import Counter, Gauge, Histogram

from ocr_common.pipeline.outbox import OutboxStats

OUTCOME_DONE = "done"
OUTCOME_REJECTED = "rejected"
OUTCOME_FAILED = "failed"
OUTCOME_CRASHED = "crashed"
OUTCOME_INTERRUPTED = "interrupted"

JOBS = Counter("pipeline_jobs_total", "Jobs finished, by stage and outcome", ["stage", "outcome"])
JOB_DURATION = Histogram(
    "pipeline_job_duration_seconds",
    "Time from starting a job's work to storing its result",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
STALE_JOBS_RECLAIMED = Counter(
    "pipeline_stale_jobs_reclaimed_total", "Jobs a dead process left PROCESSING that were run again", ["stage"]
)
OUTBOX_DELIVERIES = Counter(
    "pipeline_outbox_deliveries_total",
    "Outbox delivery attempts, by message kind and what became of them",
    ["stage", "kind", "outcome"],
)
OUTBOX_PENDING = Gauge("pipeline_outbox_pending", "Outbox messages not yet delivered", ["stage"])
OUTBOX_RETRYING = Gauge("pipeline_outbox_retrying", "Of pending: messages that failed at least once", ["stage"])
OUTBOX_DEAD_LETTERS = Gauge("pipeline_outbox_dead_letters", "Outbox messages given up on", ["stage"])
OUTBOX_OLDEST_PENDING_SECONDS = Gauge(
    "pipeline_outbox_oldest_pending_seconds", "Age of the oldest pending outbox message; 0 when none", ["stage"]
)


def observe_outbox(stats: OutboxStats, stage: str | None = None) -> None:
    """Copies one reading of the backlog (taken by the relay while idle) into the gauges, labelled `stage`
    (the relay's metrics label; `stats.stage` by default)."""
    label = stage or stats.stage
    OUTBOX_PENDING.labels(label).set(stats.pending)
    OUTBOX_RETRYING.labels(label).set(stats.retrying)
    OUTBOX_DEAD_LETTERS.labels(label).set(stats.dead_letters)
    OUTBOX_OLDEST_PENDING_SECONDS.labels(label).set(stats.oldest_pending_seconds or 0.0)
