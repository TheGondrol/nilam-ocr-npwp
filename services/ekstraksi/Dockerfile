FROM python:3.11-slim

WORKDIR /app

COPY libs/ocr_common ./libs/ocr_common
COPY services/ekstraksi/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt "./libs/ocr_common[db]"

COPY services/ekstraksi/src ./src

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PORT=8030
EXPOSE 8030

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8030\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
