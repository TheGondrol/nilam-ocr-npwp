#!/bin/bash
# Nexus build & push script (Bash) for ms-bribrain-ocr-ktp-postprocessor.
# Single-container build/push and pull/deploy for this service.
# Builds the Docker image and pushes it to the BRI Nexus registry.
#
# Nexus image path convention:
#   registry.example.com/ocr-ktp/{dev|stg|prd}/{service_name}:{tag}
#
# Usage:
#   ./nexus_build.sh [env] [tag]
#     env   dev | stg | prd   (default: dev)
#     tag   image tag         (default: latest)
#
# Examples:
#   ./nexus_build.sh                # -> ocr-ktp/dev/ms-bribrain-ocr-ktp-postprocessor:latest
#   ./nexus_build.sh stg v1.2.0     # -> ocr-ktp/stg/ms-bribrain-ocr-ktp-postprocessor:v1.2.0

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables if needed)
# ---------------------------------------------------------------------------
NEXUS_REGISTRY="${NEXUS_REGISTRY:-registry.example.com}"
PROJECT="${PROJECT:-ocr-ktp}"
SERVICE_NAME="${SERVICE_NAME:-ms-bribrain-ocr-ktp-postprocessor}"

DOCKERFILE="${DOCKERFILE:-Dockerfile}"
DOCKER_TARGET="${DOCKER_TARGET:-production}"

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

# Fully-qualified image reference:
#   registry/ocr-ktp/{env}/{service_name}:{tag}
IMAGE_REF="${NEXUS_REGISTRY}/${PROJECT}/${DEPLOY_ENV}/${SERVICE_NAME}:${TAG}"

echo "=========================================================="
echo " Nexus build & push"
echo "   Registry : $NEXUS_REGISTRY"
echo "   Image    : $IMAGE_REF"
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
# Build the image
# ---------------------------------------------------------------------------
echo "Building Docker image: $IMAGE_REF..."
docker build -f "$DOCKERFILE" ${DOCKER_TARGET:+--target "$DOCKER_TARGET"} -t "$IMAGE_REF" .

# ---------------------------------------------------------------------------
# Push to Nexus
# ---------------------------------------------------------------------------
echo "Pushing image to Nexus: $IMAGE_REF..."
docker push "$IMAGE_REF"

echo ""
echo "Build & push successful!"
echo "  Image: $IMAGE_REF"
echo ""
echo "Deploy it with: ./nexus_deploy.sh $DEPLOY_ENV $TAG"
