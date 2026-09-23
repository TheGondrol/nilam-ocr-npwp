# Base image dipin ke digest (manifest list, semua arsitektur) supaya build bisa diulang.
# Memperbarui: docker buildx imagetools inspect python:3.11-slim  -> salin "Digest" ke semua Dockerfile.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9

WORKDIR /app

COPY db/requirements.lock ./db/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r db/requirements.lock

COPY libs/ocr_common ./libs/ocr_common
RUN pip install --no-cache-dir --no-deps --no-build-isolation ./libs/ocr_common

COPY db ./db

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["python", "-m", "alembic", "-c", "db/alembic.ini", "upgrade", "head"]
