#!/bin/bash
# Nexus deploy script (Bash) for ms-bribrain-ocr-ktp-detector.
# Single-container build/push and pull/deploy for this service.
# Pulls the Docker image from the BRI Nexus registry and runs it.
#
# Nexus image path convention:
#   registry.example.com/ocr-ktp/{dev|stg|prd}/{service_name}:{tag}
#
# Usage:
#   ./nexus_deploy.sh [env] [tag]
#     env   dev | stg | prd   (default: dev)
#     tag   image tag         (default: latest)
#
# Examples:
#   ./nexus_deploy.sh                # pulls ocr-ktp/dev/ms-bribrain-ocr-ktp-detector:latest
#   ./nexus_deploy.sh stg v1.2.0     # pulls ocr-ktp/stg/ms-bribrain-ocr-ktp-detector:v1.2.0

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables if needed)
# ---------------------------------------------------------------------------
NEXUS_REGISTRY="${NEXUS_REGISTRY:-registry.example.com}"
PROJECT="${PROJECT:-ocr-ktp}"
SERVICE_NAME="${SERVICE_NAME:-ms-bribrain-ocr-ktp-detector}"

CONTAINER_NAME="${CONTAINER_NAME:-ms-bribrain-ocr-ktp-detector}"
PORT_MAPPING="${PORT_MAPPING:-8060:8060}"
ENV_FILE="${ENV_FILE:-.env}"
GPU_FLAGS="${GPU_FLAGS:-}"
HEALTH_PATH="${HEALTH_PATH:-/health}"

DEPLOY_ENV="${1:-${DEPLOY_ENV:-dev}}"
TAG="${2:-${TAG:-latest}}"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
case "$DEPLOY_ENV" in
    dev|stg|prd) ;;
    *)
        echo "Error: invalid env '$DEPLOY_ENV'. Expected one of: dev, stg, prd."
        exit 1
        ;;
esac

IMAGE_REF="${NEXUS_REGISTRY}/${PROJECT}/${DEPLOY_ENV}/${SERVICE_NAME}:${TAG}"

echo "=========================================================="
echo " Nexus deploy"
echo "   Registry  : $NEXUS_REGISTRY"
echo "   Image     : $IMAGE_REF"
echo "   Container : $CONTAINER_NAME"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Login to Nexus
# Credentials are read from NEXUS_USERNAME / NEXUS_PASSWORD when set,
# otherwise docker will prompt interactively.
# ---------------------------------------------------------------------------
echo "Logging in to Nexus registry: $NEXUS_REGISTRY..."
if [ -n "${NEXUS_USERNAME:-}" ] && [ -n "${NEXUS_PASSWORD:-}" ]; then
    echo "$NEXUS_PASSWORD" | docker login "$NEXUS_REGISTRY" \
        --username "$NEXUS_USERNAME" --password-stdin
else
    docker login "$NEXUS_REGISTRY"
fi

# ---------------------------------------------------------------------------
# Pull the image
# ---------------------------------------------------------------------------
echo "Pulling image from Nexus: $IMAGE_REF..."
docker pull "$IMAGE_REF"

# ---------------------------------------------------------------------------
# Stop and remove old container
# ---------------------------------------------------------------------------
echo "Stopping and removing old container (if any)..."
EXISTING_CONTAINER=$(docker ps -q -f name="$CONTAINER_NAME")
if [ -n "$EXISTING_CONTAINER" ]; then
    docker stop "$CONTAINER_NAME"
    echo "Stopped container: $CONTAINER_NAME"
fi

EXISTING_CONTAINER=$(docker ps -a -q -f name="$CONTAINER_NAME")
if [ -n "$EXISTING_CONTAINER" ]; then
    docker rm "$CONTAINER_NAME"
    echo "Removed container: $CONTAINER_NAME"
fi

# ---------------------------------------------------------------------------
# Start the new container
# ---------------------------------------------------------------------------
echo "Starting new container: $CONTAINER_NAME..."
docker run -d \
    --name "$CONTAINER_NAME" \
    ${GPU_FLAGS} \
    -p "$PORT_MAPPING" \
    --env-file "$ENV_FILE" \
    --restart always \
    "$IMAGE_REF"

echo ""
echo "Deployment successful!"
echo "Container $CONTAINER_NAME is running on http://localhost:8060"
echo "API Docs: http://localhost:8060/docs"
echo ""
echo "Test with: curl http://localhost:8060${HEALTH_PATH}"
