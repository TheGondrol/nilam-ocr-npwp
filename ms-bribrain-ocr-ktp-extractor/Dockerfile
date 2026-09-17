# syntax=docker/dockerfile:1.5
###############################################################################
# Stage 1 – base: install OS deps + Python packages
###############################################################################
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.1-paddlepaddle3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 AS base

# Install Python and system dependencies (cache apt metadata/packages with BuildKit)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y python3 python3-pip python3-venv \
    ffmpeg libsm6 libxext6 libgomp1 curl telnet \
    && rm -rf /var/lib/apt/lists/*

# Point python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .python-version /app/

RUN uv lock

# Sync dependencies first
ENV UV_CACHE_DIR=/root/.cache/uv
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Install high-performance inference deps for PaddleOCR
# Skip during build by default; run inside a GPU-enabled container instead.
ARG INSTALL_HPI_DEPS=0
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${INSTALL_HPI_DEPS}" = "1" ]; then /app/.venv/bin/paddleocr install_hpi_deps gpu; else echo "Skipping HPI deps install at build time"; fi

# Copy application code
COPY src /app/src
COPY config.yaml /app/
COPY PaddleOCR_server.yaml /app/
COPY PaddleOCR_mobile.yaml /app/
COPY tests /app/tests
COPY autokernel /app/autokernel
COPY PaddleOCR2Pytorch /app/PaddleOCR2Pytorch

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

###############################################################################
# Stage 2 – e2e-test: run e2e tests (fails the build on error)
###############################################################################
FROM base AS e2e-test

RUN --mount=type=secret,id=env,target=/app/.env \
    if [ -f /app/.env ]; then \
        eval "$(python3 -c "\
import shlex; \
[print(f'export {line.partition(\"=\")[0].strip()}={shlex.quote(line.partition(\"=\")[2].strip().strip(chr(34)).strip(chr(39)))}') \
 for line in open('/app/.env') \
 if line.strip() and not line.strip().startswith('#') and '=' in line]")"; \
    fi && \
    /app/.venv/bin/pytest tests/e2e/ -m e2e -v --no-cov --tb=short 2>&1

###############################################################################
# Stage 3 – production: lean runtime image
###############################################################################
FROM base AS production

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8010/health/live || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8010"]
