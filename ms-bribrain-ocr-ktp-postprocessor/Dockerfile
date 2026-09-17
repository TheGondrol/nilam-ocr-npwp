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
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache

# Copy application
COPY . /app/

# Create logs directory
RUN mkdir -p /app/logs

###############################################################################
# Stage 2: e2e-test — run e2e tests (build fails if tests fail)
###############################################################################
FROM base AS e2e-test

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
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8090/health/live || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090", "--timeout-keep-alive", "60"]
