FROM python:3.11-slim

WORKDIR /app

COPY libs/ocr_common ./libs/ocr_common
COPY services/guardrails/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt ./libs/ocr_common

COPY services/guardrails/weights ./weights
RUN test -f weights/best_model.pt || (echo "services/guardrails/weights/best_model.pt tidak ada di build context" && exit 1)

COPY services/guardrails/app ./app

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PORT=8031 \
    GUARDRAILS_BACKEND=efficientnet \
    GUARDRAILS_MODEL_PATH=weights/best_model.pt \
    GUARDRAILS_DEVICE=cpu
EXPOSE 8031

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8031\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
