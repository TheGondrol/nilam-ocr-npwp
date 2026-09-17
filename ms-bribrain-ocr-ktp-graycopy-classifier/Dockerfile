# ===========================================================================
#  Stage 1: base – install system deps + Python deps + copy source
# ===========================================================================
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y \
    telnet \
    curl \
    htop \
    procps \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

# Install ALL deps (main + dev) so the test stage can reuse them
RUN uv sync --frozen --no-cache --all-extras

COPY . ./


# ===========================================================================
#  Stage 2: e2e-test – run e2e tests against the real running service
#
#  To run this stage:
#    docker build --target e2e-test \
#      --network host \
#      --secret id=env,src=.env \
#      -t graycopy-test .
# ===========================================================================
FROM base AS e2e-test

RUN ln -sf /.env /app/.env

RUN chmod +x /app/scripts/run_e2e_tests.sh
RUN --mount=type=secret,id=env,target=/.env \
    /app/scripts/run_e2e_tests.sh


# ===========================================================================
#  Stage 3: production – lean final image
# ===========================================================================
FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y \
    telnet \
    curl \
    htop \
    procps \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-cache

COPY . ./

# Symlink .env file for K8s Vault compatibility
RUN ln -sf /.env /app/.env

EXPOSE 8040

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8040/health/live || exit 1

CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8040", "--timeout-keep-alive", "60"]
