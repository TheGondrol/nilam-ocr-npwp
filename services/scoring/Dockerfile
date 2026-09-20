FROM python:3.11-slim

WORKDIR /app

COPY libs/ocr_common ./libs/ocr_common
COPY services/scoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt "./libs/ocr_common[db]"

COPY services/scoring/src ./src
COPY services/scoring/weights ./weights
RUN test -f weights/trust_model.joblib || (echo "services/scoring/weights/trust_model.joblib tidak ada di build context" && exit 1)

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PORT=8033
EXPOSE 8033

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8033\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
