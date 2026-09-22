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
openapi: $(SERVICES:%=openapi-%) openapi-gateway

test-lib:
	cd libs/ocr_common && $(PY) -m pytest -q
typecheck-lib:
	cd libs/ocr_common && $(PY) -m ty check ocr_common tests

test-%:
	cd services/$* && $(PY) -m pytest -q
typecheck-%:
	cd services/$* && $(PY) -m ty check app tests
openapi-%:
	cd services/$* && API_KEY=x ENVIRONMENT=local $(PY) -m ocr_common.web.openapi
openapi-gateway: $(SERVICES:%=openapi-%)
	$(PY) scripts/build_gateway_openapi.py
api-docs:
	$(PY) -m http.server 8088
run-%:
	cd services/$* && $(PY) -m uvicorn app.main:app --reload --port $(PORT_$*)

db-upgrade:
	$(PY) -m alembic -c db/alembic.ini upgrade head
db-check:
	$(PY) -m alembic -c db/alembic.ini check
db-revision:
	$(PY) -m alembic -c db/alembic.ini revision --autogenerate -m "$(m)"
db-external:
	$(PY) db/external/apply.py

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

.PHONY: dev test lint format typecheck openapi openapi-gateway api-docs test-lib typecheck-lib db-upgrade db-check db-revision db-external weights build up up-db down ps smoke
