-- Skema PostgreSQL untuk nilam-ocr-npwp (opsional; hanya kalau DATABASE_URL diisi).
--
-- Dijalankan manual lewat pgAdmin Query Tool atau psql, sama seperti
-- ocr-nilam-db/schema.sql di nilam-ocr-orchestration. Aplikasi tidak membuat
-- atau memigrasi skema sendiri, hanya membaca/menulis lewat SQLAlchemy
-- (src/repositories/request_repository.py). Aman dijalankan berkali-kali.
--
-- Satu baris per request_id, DIPERBARUI di tempat mengikuti siklusnya:
-- pending (generate-request-id) -> completed | failed (extract-ocr).
-- Ini status operasional, bukan audit trail; audit per pemanggilan sudah
-- dicatat orchestrator di tabel orchestration_*.
--
-- "ds": YYYYMMDD (UTC) dari created_at, konvensi partisi harian untuk
-- penarikan batch ke Big Data. Diisi aplikasi dari nilai waktu yang sama
-- dengan created_at (lihat _now_with_ds), bukan GENERATED column, karena
-- to_char()/timezone hanya STABLE.

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
