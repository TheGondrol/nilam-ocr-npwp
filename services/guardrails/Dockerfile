# Build dari ROOT repo (butuh libs/ocr_common):
#   docker build -f services/guardrails/Dockerfile -t nilam-ocr-guardrails .
#
# Bobot model IKUT ke dalam image: services/guardrails/weights/best_model.pt
# harus ada di build context (file ini di-gitignore, jadi di CI harus diunduh
# dulu dari storage artefak sebelum build). Inference CPU saja: wheel torch
# CPU dari index PyTorch, GUARDRAILS_DEVICE=cpu.
FROM python:3.11-slim

WORKDIR /app

# Lib bersama dulu, lalu requirements: layer ini hanya dibangun ulang kalau
# salah satunya berubah, bukan tiap kali kode service atau bobot disentuh.
COPY libs/ocr_common ./libs/ocr_common
COPY services/guardrails/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt ./libs/ocr_common

# Bobot di layer sendiri sebelum kode: ganti kode tidak menyalin ulang 16 MB bobot.
COPY services/guardrails/weights ./weights
RUN test -f weights/best_model.pt || (echo "services/guardrails/weights/best_model.pt tidak ada di build context" && exit 1)

COPY services/guardrails/src ./src

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PORT=8031 \
    GUARDRAILS_BACKEND=efficientnet \
    GUARDRAILS_MODEL_PATH=weights/best_model.pt \
    GUARDRAILS_DEVICE=cpu
EXPOSE 8031

# start-period lebih lama: model dimuat saat startup (beberapa detik di CPU).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8031\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
