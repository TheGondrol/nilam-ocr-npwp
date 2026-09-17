###############################################################################
# Stage 1: base — install system deps, Python deps, copy source
###############################################################################
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    telnet \
    curl \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install the application dependencies.
# insightface ships only a source distribution and has Cython extensions
# (cython_bbox, mesh_core_cython), so g++ is required to build it. We install
# the toolchain, sync, and purge it in the same layer to keep the image lean.
COPY pyproject.toml uv.lock /app/
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && uv sync --frozen --no-cache \
    && apt-get purge -y --auto-remove g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the application into the container
COPY . /app/

# Create logs directory
RUN mkdir -p /app/logs

# Pre-cache the InsightFace RetinaFace model pack so the first request
# doesn't trigger a network fetch at runtime.
RUN uv run python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'], allowed_modules=['detection']); \
    app.prepare(ctx_id=-1, det_size=(640, 640))"

###############################################################################
# Stage 2: e2e-test — run e2e tests (build fails if tests fail)
###############################################################################
FROM base AS e2e-test

# Install test dependencies
RUN uv sync --frozen --no-cache --extra test

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

EXPOSE 8070

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8070/health/live || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "src.main:app", "--port", "8070", "--host", "0.0.0.0", "--timeout-keep-alive", "60"]
