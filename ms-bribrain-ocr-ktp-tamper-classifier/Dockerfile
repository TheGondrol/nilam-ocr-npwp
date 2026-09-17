###############################################################################
# Stage 1 – base: install OS deps + Python packages
###############################################################################
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

COPY pyproject.toml uv.lock .python-version README.md /app/

RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY . /app/

# Create logs directory
RUN mkdir -p /app/logs

###############################################################################
# Stage 2 – e2e-test: run e2e tests (fails the build on error)
###############################################################################
FROM base AS e2e-test

RUN uv sync --frozen --no-cache --extra test

RUN --mount=type=secret,id=env,target=/app/.env \
    if [ -f /app/.env ]; then \
        eval "$(python3 -c "\
import shlex; \
[print(f'export {line.partition(\"=\")[0].strip()}={shlex.quote(line.partition(\"=\")[2].strip().strip(chr(34)).strip(chr(39)))}') \
 for line in open('/app/.env') \
 if line.strip() and not line.strip().startswith('#') and '=' in line]")"; \
    fi && \
    /app/.venv/bin/pytest test/e2e/ -m e2e -v --no-cov --tb=short 2>&1

###############################################################################
# Stage 3 – production: lean runtime image
###############################################################################
FROM base AS production

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health/live || exit 1

CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--port", "8050", "--host", "0.0.0.0", "--timeout-keep-alive", "60"]
