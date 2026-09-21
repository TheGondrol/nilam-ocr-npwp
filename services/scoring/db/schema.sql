CREATE TABLE IF NOT EXISTS scoring_jobs (
    request_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    error_message   TEXT,
    attempts        INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_jobs_status ON scoring_jobs (status);
CREATE INDEX IF NOT EXISTS idx_scoring_jobs_ds ON scoring_jobs (ds);

CREATE TABLE IF NOT EXISTS scoring_results (
    request_id      TEXT PRIMARY KEY REFERENCES scoring_jobs (request_id),
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_results_ds ON scoring_results (ds);
