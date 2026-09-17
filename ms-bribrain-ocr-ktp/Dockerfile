FROM python:3.11.6-slim

# Install telnet
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

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

EXPOSE 8000

# Run the application.
# CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "9000", "--host", "0.0.0.0"]
CMD ["uv", "run", "uvicorn", "main:app", "--port", "9000", "--host", "0.0.0.0", "--timeout-keep-alive", "60"]
