CREATE SCHEMA IF NOT EXISTS scoring;

CREATE TABLE IF NOT EXISTS scoring.jobs (
    request_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    error_message   TEXT,
    attempts        INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_jobs_status ON scoring.jobs (status);
CREATE INDEX IF NOT EXISTS idx_scoring_jobs_ds ON scoring.jobs (ds);

CREATE TABLE IF NOT EXISTS scoring.results (
    request_id      TEXT PRIMARY KEY REFERENCES scoring.jobs (request_id),
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_results_ds ON scoring.results (ds);
