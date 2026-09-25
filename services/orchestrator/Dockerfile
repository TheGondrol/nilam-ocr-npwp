# Base image dipin ke digest (manifest list, semua arsitektur) supaya build bisa diulang.
# Memperbarui: docker buildx imagetools inspect python:3.11-slim  -> salin "Digest" ke semua Dockerfile.
FROM python:3.11-slim@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9

WORKDIR /app

# Dependensi dari lock file ber-hash (make lock): versi transitif dan isi setiap wheel terkunci.
COPY services/orchestrator/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# ocr_common dipasang tanpa resolusi dependensi: semua dependensinya sudah ada di lock.
# Disalin setelah dependensi supaya perubahan ocr_common tidak mengulang unduhan di atas.
COPY libs/ocr_common ./libs/ocr_common
RUN pip install --no-cache-dir --no-deps --no-build-isolation ./libs/ocr_common

COPY services/orchestrator/app ./app

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PORT=8034
EXPOSE 8034

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8034\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
