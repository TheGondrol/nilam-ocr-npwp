# Monorepo: libs/ocr_common (lib bersama) + services/<nama> (empat deployable).
#
#   make dev                pasang semua dependency + lib bersama (editable)
#   make test / lint / typecheck / openapi     semua service (+ lib)
#   make test-ekstraksi     satu service saja (juga lint-, typecheck-, openapi-, run-)
#   make run-guardrails     uvicorn --reload untuk satu service di port defaultnya
#   make weights            unduh bobot model dari GUARDRAILS_MODEL_URI (GCS/MinIO/https) sebelum build
#   make build              build keempat image
#   make up / down / ps / logs-<nama>          docker compose
#   make up-db              sama, plus PostgreSQL lokal (jobs/results ketiga tahap pipeline)
#   make smoke              uji pipeline async + kontrak lama lewat container yang jalan
#
# Package tiap service bernama `src` (sama seperti ocr-*), jadi pytest/ty
# harus dijalankan dari folder service masing-masing, bukan dari root.
SERVICES := guardrails ekstraksi structuring scoring
PY ?= python
PORT_guardrails := 8031
PORT_ekstraksi := 8030
PORT_structuring := 8032
PORT_scoring := 8033

dev:
	$(PY) -m pip install -r requirements-dev.txt

test: test-lib $(SERVICES:%=test-%)
lint:
	$(PY) -m ruff check .
format:
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .
typecheck: typecheck-lib $(SERVICES:%=typecheck-%)
openapi: $(SERVICES:%=openapi-%)

test-lib:
	cd libs/ocr_common && $(PY) -m pytest -q
typecheck-lib:
	cd libs/ocr_common && $(PY) -m ty check ocr_common tests

# Target per-service sengaja TIDAK di-.PHONY: make melewatkan pencarian
# implicit rule untuk target phony (lihat Makefile root nilam-ocr-orchestration).
test-%:
	cd services/$* && $(PY) -m pytest -q
typecheck-%:
	cd services/$* && $(PY) -m ty check src tests
openapi-%:
	cd services/$* && API_KEY=x ENVIRONMENT=local $(PY) -m ocr_common.openapi
run-%:
	cd services/$* && $(PY) -m uvicorn src.main:app --reload --port $(PORT_$*)

# Sumber bobot dibaca dari env (lihat scripts/fetch_weights.py). Tanpa env: lewati.
weights:
	$(PY) scripts/fetch_weights.py || [ $$? -eq 2 ]

build:
	docker compose build
up:
	docker compose up -d --build
up-%:
	docker compose up -d --build $*
up-db:
	docker compose -f docker-compose.yml -f docker-compose.db.yml up -d --build
down:
	docker compose -f docker-compose.yml -f docker-compose.db.yml down
ps:
	@docker ps --filter name=nilam-ocr- --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
logs-%:
	docker compose logs -f $*

smoke:
	$(PY) scripts/smoke_e2e.py

.PHONY: dev test lint format typecheck openapi test-lib typecheck-lib weights build up up-db down ps smoke
