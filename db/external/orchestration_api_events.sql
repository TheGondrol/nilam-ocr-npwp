-- Tabel ini MILIK service orkestrasi, bukan repo ini. Isinya disalin dari database dev
-- (bribrain_ocr_nilam, 23 Sep 2026) supaya ORCHESTRATION_API_EVENTS_TABLE bisa diuji di
-- PostgreSQL lokal (make db-external). Migrasi Alembic di repo ini sengaja TIDAK menyentuh tabel ini.
--
-- Pipeline hanya MENAMBAH baris (append-only), satu per keadaan akhir request, dengan bentuk yang
-- sama seperti baris polling orkestrasi sendiri:
--   endpoint = GET_OCR_RESULT, status_code 200 / 422, downstream_status COMPLETED / FAILED,
--   downstream_stage = tahap terakhir atau tahap yang gagal,
--   result_data = {result, status, document_type, error_code, error_message, created_at, updated_at}.
-- Baris terbaru per request_id adalah keadaannya.
-- Semua statement idempoten.

CREATE SCHEMA IF NOT EXISTS ocr;

DO $$ BEGIN
    CREATE TYPE ocr.endpoint AS ENUM ('EXTRACT_OCR', 'GET_OCR_RESULT');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE ocr.downstream_status AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE ocr.downstream_stage AS ENUM ('GUARDRAILS', 'EXTRACTION', 'SCORING', 'STRUCTURING');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS ocr.orchestration_api_events (
    id                 BIGSERIAL PRIMARY KEY,
    endpoint           ocr.endpoint NOT NULL,
    request_id         TEXT NOT NULL,
    status_code        INT NOT NULL,
    error_code         TEXT,
    result_data        JSONB,
    auth_token_id      BIGINT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    ds                 TEXT NOT NULL,
    file_name          TEXT,
    file_size_bytes    INT,
    datahub_validation JSONB,
    downstream_status  ocr.downstream_status,
    downstream_stage   ocr.downstream_stage,
    document_type      TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_events_request_id ON ocr.orchestration_api_events (request_id);
CREATE INDEX IF NOT EXISTS idx_api_events_created_at ON ocr.orchestration_api_events (created_at);
CREATE INDEX IF NOT EXISTS idx_api_events_ds_endpoint_doctype
    ON ocr.orchestration_api_events (ds, endpoint, document_type);
