
# ===========================================================================
#  Stage 1: base – install system deps + Python deps
# ===========================================================================
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y \
    telnet \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version /app/

# Install ALL deps (main + test) so the test stage can reuse them
RUN uv sync --frozen --no-cache --all-extras

COPY . /app/
RUN mkdir -p /app/logs


# ===========================================================================
#  Stage 2: e2e-test – run e2e tests against real downstream services
#
#  This stage starts the orchestrator inside the container, waits for it
#  to become healthy, and runs the e2e test suite.  All downstream
#  microservices must be reachable from the Docker build network.
#
#  To run this stage:
#    docker build --target e2e-test \
#      --network host \
#      --env-file .env \
#      -t orchestrator-test .
#
#  To SKIP tests and build production image only:
#    docker build -t orchestrator .
# ===========================================================================
FROM base AS e2e-test

# Symlink .env so config.yaml can resolve env vars
RUN ln -sf /.env /app/.env

# The test script:
#  1) starts uvicorn in background
#  2) polls /health until ready
#  3) runs pytest tests/e2e/ -m e2e
#  4) exits with test result code → fails the build if tests fail
RUN chmod +x /app/scripts/run_e2e_tests.sh
RUN --mount=type=secret,id=env,target=/.env \
    /app/scripts/run_e2e_tests.sh


# ===========================================================================
#  Stage 3: production – lean final image (no test deps)
# ===========================================================================
FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y \
    telnet \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --frozen --no-cache

COPY . /app/
RUN mkdir -p /app/logs

# Symlink .env file for K8s Vault compatibility
RUN ln -sf /.env /app/.env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uv", "run", "uvicorn", "src.main:app", "--port", "8000", "--host", "0.0.0.0", "--timeout-keep-alive", "60"]
