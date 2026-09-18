-- Skema PostgreSQL service scoring (hanya kalau DATABASE_URL diisi).
--
-- Dijalankan manual lewat pgAdmin Query Tool atau psql, di database yang sama
-- dengan service lain; tiap service punya schema sendiri (ocr, structuring,
-- scoring). Aplikasi tidak membuat atau memigrasi skema sendiri. Aman
-- dijalankan berkali-kali.
--
-- Definisi kolom harus sama dengan build_tables() di ocr_common/jobs_sql.py.
--
-- scoring.jobs     satu baris per request_id. INSERT ... ON CONFLICT DO NOTHING
--                  saat job diterima (idempoten), lalu PROCESSING -> DONE |
--                  FAILED. Job FAILED boleh diklaim ulang; `attempts` menghitungnya.
-- scoring.results  laporan skor, di-UPSERT saat DONE. Hasil AKHIR request
--                  (field + skor + guardrails) dikirim ke Orkestrasi lewat
--                  callback dan disimpan di orkestrasi.requests.final_result,
--                  bukan di sini.
--
-- "ds": YYYYMMDD (UTC) dari created_at, konvensi partisi harian untuk
-- penarikan batch ke Big Data. Diisi aplikasi, bukan GENERATED column.
CREATE SCHEMA IF NOT EXISTS scoring;

CREATE TABLE IF NOT EXISTS scoring.jobs (
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

CREATE INDEX IF NOT EXISTS idx_scoring_jobs_status ON scoring.jobs (status);
CREATE INDEX IF NOT EXISTS idx_scoring_jobs_ds ON scoring.jobs (ds);

CREATE TABLE IF NOT EXISTS scoring.results (
    request_id      TEXT PRIMARY KEY REFERENCES scoring.jobs (request_id),
    -- {score, decision, field_scores, reasons}
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_results_ds ON scoring.results (ds);
