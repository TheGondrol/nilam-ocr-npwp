-- Tabel ini MILIK service orkestrasi, bukan repo ini. File ini ada untuk dua hal:
--   1. permintaan konkret ke tim orkestrasi: kolom dan constraint yang dibutuhkan pipeline;
--   2. menyiapkan tabel tiruannya di PostgreSQL lokal supaya alurnya bisa diuji (make db-external).
-- Migrasi Alembic di repo ini sengaja TIDAK menyentuh tabel ini.
-- Semua statement idempoten, jadi aman dijalankan ke tabel yang sudah ada maupun ke database kosong.

CREATE TABLE IF NOT EXISTS orchestration_extract_ocr (
    id                 BIGSERIAL PRIMARY KEY,
    request_id         TEXT NOT NULL,
    document_type      TEXT NOT NULL,
    status_code        INT NOT NULL,
    downstream_status  TEXT,
    downstream_stage   TEXT,
    error_code         TEXT,
    error_message      TEXT,
    result_data        JSONB,
    file_name          TEXT,
    file_size_bytes    INT,
    auth_token_id      BIGINT,
    datahub_validation JSONB,
    occurred_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds                 TEXT NOT NULL
);

-- Kolom baru yang diminta pipeline (di tabel yang sudah ada). Kontrak dengan orkestrasi: pipeline
-- meng-upsert downstream_status = 'processing' (saat job diklaim, downstream_stage = tahapnya),
-- 'completed' (scoring, dengan result_data) atau 'failed' (error_code = <TAHAP>_FAILED, error_message);
-- orkestrasi HANYA membaca downstream_status untuk menjawab polling client.
ALTER TABLE orchestration_extract_ocr ADD COLUMN IF NOT EXISTS downstream_status TEXT;
ALTER TABLE orchestration_extract_ocr ADD COLUMN IF NOT EXISTS downstream_stage TEXT;
ALTER TABLE orchestration_extract_ocr ADD COLUMN IF NOT EXISTS error_message TEXT;

-- WAJIB: pipeline menulis dengan UPSERT ON CONFLICT (request_id), jadi request_id harus unik.
-- Kalau satu request_id boleh punya beberapa baris (log per percobaan), beri tahu kami:
-- cara penulisannya harus diubah.
CREATE UNIQUE INDEX IF NOT EXISTS uq_orchestration_extract_ocr_request_id
    ON orchestration_extract_ocr (request_id);
