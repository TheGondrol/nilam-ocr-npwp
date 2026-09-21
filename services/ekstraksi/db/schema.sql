CREATE TABLE IF NOT EXISTS ocr_jobs (
    request_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    error_message   TEXT,
    attempts        INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr_jobs (status);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_ds ON ocr_jobs (ds);

CREATE TABLE IF NOT EXISTS ocr_results (
    request_id      TEXT PRIMARY KEY REFERENCES ocr_jobs (request_id),
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ocr_results_ds ON ocr_results (ds);

CREATE TABLE IF NOT EXISTS ocr_npwp_requests (
    request_id      TEXT PRIMARY KEY,

    status          TEXT NOT NULL,

    result          JSONB,
    guardrails      DOUBLE PRECISION,
    error_message   TEXT,

    file_name       TEXT,
    file_size_bytes INT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_status
    ON ocr_npwp_requests (status);
CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_created_at
    ON ocr_npwp_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_ocr_npwp_requests_ds
    ON ocr_npwp_requests (ds);
