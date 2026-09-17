# ===========================================================================
#  Stage 1: base – install system deps + Python deps + copy source
# ===========================================================================
FROM python:3.11.6-slim AS base

RUN apt-get update && apt-get install -y \
    htop \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock /app/

# Install ALL deps (main + test) so the test stage can reuse them
RUN uv sync --frozen --no-cache --all-extras

COPY . /app/

# Disable OpenVINO telemetry to prevent signal handler conflicts with uvicorn
ENV OPENVINO_TELEMETRY=0
ENV OPENVINO_LOG_LEVEL=0


# ===========================================================================
#  Stage 2: e2e-test – run e2e tests against the real running service
#
#  This stage starts the detector inside the container, waits for it
#  to become ready, and runs the e2e test suite.
#
#  To run this stage:
#    docker build --target e2e-test \
#      --network host \
#      --secret id=env,src=.env \
#      -t detector-test .
# ===========================================================================
FROM base AS e2e-test

RUN ln -sf /.env /app/.env

RUN chmod +x /app/scripts/run_e2e_tests.sh
RUN --mount=type=secret,id=env,target=/.env \
    /app/scripts/run_e2e_tests.sh


# ===========================================================================
#  Stage 3: production – lean final image
# ===========================================================================
FROM python:3.11.6-slim AS production

RUN apt-get update && apt-get install -y \
    htop \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache

COPY . /app/

# Disable OpenVINO telemetry to prevent signal handler conflicts with uvicorn
ENV OPENVINO_TELEMETRY=0
ENV OPENVINO_LOG_LEVEL=0

# Symlink .env file for K8s Vault compatibility
RUN ln -sf /.env /app/.env

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8060/health/live || exit 1

CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8060", "--timeout-keep-alive", "60"]
