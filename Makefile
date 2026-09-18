.PHONY: install dev run test lint typecheck format openapi docker-up docker-up-db docker-down db-schema

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -r requirements-dev.txt

run:
	$(PY) -m uvicorn src.main:app --reload --port 8030

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m ty check src tests scripts

format:
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

# openapi.yaml adalah turunan dari kode; jalankan setelah mengubah route/schema.
openapi:
	API_KEY=x $(PY) -m scripts.export_openapi

docker-up:
	docker compose up -d --build

docker-up-db:
	docker compose -f docker-compose.yml -f docker-compose.db.yml up -d --build

docker-down:
	docker compose -f docker-compose.yml -f docker-compose.db.yml down

# Pasang skema ke PostgreSQL yang ditunjuk DATABASE_URL_PSQL (format psql, bukan
# +asyncpg), atau ke container postgres overlay kalau sedang jalan. Aman diulang.
DATABASE_URL_PSQL ?= postgresql://postgres:changeme@localhost:5433/bribrain_ocr_nilam
db-schema:
	@if docker ps --format '{{.Names}}' | grep -qx nilam-ocr-npwp-postgres; then \
		docker exec -i nilam-ocr-npwp-postgres psql -U "$${POSTGRES_USER:-postgres}" \
			-d "$${POSTGRES_DB:-bribrain_ocr_nilam}" -v ON_ERROR_STOP=1 < db/schema.sql; \
	else \
		psql "$(DATABASE_URL_PSQL)" -v ON_ERROR_STOP=1 -f db/schema.sql; \
	fi
