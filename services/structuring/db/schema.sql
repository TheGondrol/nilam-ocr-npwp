-- Skema PostgreSQL service structuring (hanya kalau DATABASE_URL diisi).
--
-- Dijalankan manual lewat pgAdmin Query Tool atau psql, di database yang sama
-- dengan service lain; tiap service punya schema sendiri (ocr, structuring,
-- scoring). Aplikasi tidak membuat atau memigrasi skema sendiri. Aman
-- dijalankan berkali-kali.
--
-- Definisi kolom harus sama dengan build_tables() di ocr_common/jobs_sql.py.
--
-- structuring.jobs     satu baris per request_id. INSERT ... ON CONFLICT DO
--                      NOTHING saat job diterima (idempoten), lalu
--                      PROCESSING -> DONE | FAILED. Job FAILED boleh diklaim
--                      ulang; `attempts` menghitungnya.
-- structuring.results  field bernama hasil structuring, di-UPSERT saat DONE.
--
-- "ds": YYYYMMDD (UTC) dari created_at, konvensi partisi harian untuk
-- penarikan batch ke Big Data. Diisi aplikasi, bukan GENERATED column.
CREATE SCHEMA IF NOT EXISTS structuring;

CREATE TABLE IF NOT EXISTS structuring.jobs (
    request_id      TEXT PRIMARY KEY,
    -- PROCESSING | DONE | FAILED
    status          TEXT NOT NULL,
    -- Pesan kegagalan saat FAILED; NULL selain itu.
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
    -- {document_type, fields: {name: {value, confidence, source}}}
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_structuring_results_ds ON structuring.results (ds);
