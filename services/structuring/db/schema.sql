CREATE SCHEMA IF NOT EXISTS structuring;

CREATE TABLE IF NOT EXISTS structuring.jobs (
    request_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    error_message   TEXT,
    attempts        INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_structuring_jobs_status ON structuring.jobs (status);
CREATE INDEX IF NOT EXISTS idx_structuring_jobs_ds ON structuring.jobs (ds);

CREATE TABLE IF NOT EXISTS structuring.results (
    request_id      TEXT PRIMARY KEY REFERENCES structuring.jobs (request_id),
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_structuring_results_ds ON structuring.results (ds);
