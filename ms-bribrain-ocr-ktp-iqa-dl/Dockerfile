###############################################################################
# Stage 1: base — install system deps, Python deps, copy source
###############################################################################
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

WORKDIR /app

# Fix potential stale GPG keys in cached layers, then install deps
# Ubuntu 22.04 ships Python 3.10 — no external PPA needed
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    python3-venv \
    telnet \
    curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml ./
COPY main.py ./
COPY config.yaml ./
COPY src/ ./src/

# Install dependencies using uv (with CUDA-enabled PyTorch)
# Install PyTorch with CUDA support first, then remaining dependencies
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --system -r pyproject.toml

# Create logs directory
RUN mkdir -p logs

###############################################################################
# Stage 2: e2e-test — run e2e tests (build fails if tests fail)
###############################################################################
FROM base AS e2e-test

# Install test dependencies
RUN uv pip install --system pytest pytest-asyncio httpx

# Copy test files and scripts
COPY tests/ ./tests/
COPY scripts/ ./scripts/
RUN chmod +x scripts/run_e2e_tests.sh

# Run e2e tests — this stage fails the build if any test fails
RUN --mount=type=secret,id=env,target=/app/.env \
    bash scripts/run_e2e_tests.sh

###############################################################################
# Stage 3: production — lean final image
###############################################################################
FROM base AS production

# Expose port
EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8100/health/live || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--timeout-keep-alive", "60"]
