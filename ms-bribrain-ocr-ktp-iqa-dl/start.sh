#!/bin/bash

# Deploy locally using Docker

# Build Docker image (--no-cache avoids stale GPG signature issues)
echo "Building Docker image..."
docker build --no-cache -t ms-bribrain-ocr-ktp-iqa-dl:latest .

# Stop and remove existing container if running
echo "Stopping existing container (if any)..."
docker stop ms-bribrain-ocr-ktp-iqa-dl 2>/dev/null || true
docker rm ms-bribrain-ocr-ktp-iqa-dl 2>/dev/null || true

# Run container with GPU support
echo "Starting container..."
docker run -d \
  --name ms-bribrain-ocr-ktp-iqa-dl \
  --gpus all \
  -p 8100:8100 \
  --env-file .env \
  --restart always \
  ms-bribrain-ocr-ktp-iqa-dl:latest

echo "Container started successfully!"
echo "API available at: http://localhost:8100"
echo ""
echo "Check logs with: docker logs -f ms-bribrain-ocr-ktp-iqa-dl"
echo "Health check: curl http://localhost:8100/health"
