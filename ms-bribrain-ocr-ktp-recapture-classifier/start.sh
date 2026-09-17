#!/bin/bash
# Local Docker deployment script for Recapture Detection

# Variables
IMAGE_NAME="ms-bribrain-ocr-ktp-recapture-classifier"
CONTAINER_NAME="ms-bribrain-ocr-ktp-recapture-classifier"
PORT_MAPPING="8030:8030"

# Build the image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image."
    exit 1
fi

# Stop old container
echo "Stopping and removing old container (if any)..."
if docker ps -q -f name="$CONTAINER_NAME" > /dev/null 2>&1; then
    docker stop "$CONTAINER_NAME"
    echo "Stopped container: $CONTAINER_NAME"
fi

# Remove old container
if docker ps -a -q -f name="$CONTAINER_NAME" > /dev/null 2>&1; then
    docker rm "$CONTAINER_NAME"
    echo "Removed container: $CONTAINER_NAME"
fi

# Start container with GPU support
echo "Starting new container: $CONTAINER_NAME with GPU..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT_MAPPING" \
    --env-file .env \
    --restart always \
    "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to start new container."
    exit 1
fi

echo ""
echo "✓ Deployment successful!"
echo "Container $CONTAINER_NAME is running on http://localhost:8030"
echo "API Docs: http://localhost:8030/docs"
echo ""
echo "Test with: curl http://localhost:8030/health"
