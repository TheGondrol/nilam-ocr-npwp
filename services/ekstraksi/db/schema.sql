-- Skema PostgreSQL service ekstraksi (hanya kalau DATABASE_URL diisi).
--
-- Dijalankan manual lewat pgAdmin Query Tool atau psql, sama seperti
-- ocr-nilam-db/schema.sql di nilam-ocr-orchestration. Aplikasi tidak membuat
-- atau memigrasi skema sendiri, hanya membaca/menulis lewat SQLAlchemy.
-- Aman dijalankan berkali-kali.
--
-- Tabel ocr_npwp_requests (kontrak lama): satu baris per request_id, DIPERBARUI di tempat mengikuti siklusnya:
-- pending (generate-request-id) -> completed | failed (extract-ocr).
-- Ini status operasional, bukan audit trail; audit per pemanggilan sudah
-- dicatat orchestrator di tabel orchestration_*.
--
-- "ds": YYYYMMDD (UTC) dari created_at, konvensi partisi harian untuk
-- penarikan batch ke Big Data. Diisi aplikasi dari nilai waktu yang sama
-- dengan created_at (lihat _now_with_ds), bukan GENERATED column, karena
-- to_char()/timezone hanya STABLE.

-- ---------------------------------------------------------------------------
-- Alur async (sequence diagram): schema `ocr`, milik ServiceOCR.
-- Definisi kolom harus sama dengan build_tables() di ocr_common/jobs_sql.py.
--
-- ocr.jobs     satu baris per request_id. INSERT ... ON CONFLICT DO NOTHING
--              saat job diterima (idempoten), lalu PROCESSING -> DONE | FAILED.
--              Job FAILED boleh diklaim ulang; `attempts` menghitungnya.
-- ocr.results  hasil OCR mentah (blok teks), di-UPSERT saat DONE.
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ocr;

CREATE TABLE IF NOT EXISTS ocr.jobs (
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

CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr.jobs (status);
CREATE INDEX IF NOT EXISTS idx_ocr_jobs_ds ON ocr.jobs (ds);

CREATE TABLE IF NOT EXISTS ocr.results (
    request_id      TEXT PRIMARY KEY REFERENCES ocr.jobs (request_id),
    -- {engine, model, elapsed_ms, full_text, blocks: [{text, confidence, bbox, page}]}
    result          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ocr_results_ds ON ocr.results (ds);

-- ---------------------------------------------------------------------------
-- Kontrak lama (generate-request-id -> extract-ocr -> get-ocr-result).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ocr_npwp_requests (
    request_id      TEXT PRIMARY KEY,

    -- pending | completed | failed
    status          TEXT NOT NULL,

    -- Field hasil OCR ({value, confidence} per field) saat completed; NULL selain itu.
    result          JSONB,
    -- Skor dokumen 0..1 (app scoring) saat completed; NULL selain itu.
    guardrails      DOUBLE PRECISION,
    -- Pesan kegagalan saat failed; NULL selain itu.
    error_message   TEXT,

    -- Metadata file yang disubmit. Isi file tidak pernah disimpan.
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
