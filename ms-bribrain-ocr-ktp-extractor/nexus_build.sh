#!/bin/bash
# Nexus build & push script (Bash) for ms-bribrain-ocr-ktp-extract-fullpytorch.
# NOTE: start.sh deploys a multi-replica + Nginx + CUDA-MPS topology; this script runs a single container. Use start.sh for the full production topology.
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
#   ./nexus_build.sh                # -> ocr-ktp/dev/ms-bribrain-ocr-ktp-extract-fullpytorch:latest
#   ./nexus_build.sh stg v1.2.0     # -> ocr-ktp/stg/ms-bribrain-ocr-ktp-extract-fullpytorch:v1.2.0

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables if needed)
# ---------------------------------------------------------------------------
NEXUS_REGISTRY="${NEXUS_REGISTRY:-registry.example.com}"
PROJECT="${PROJECT:-ocr-ktp}"
SERVICE_NAME="${SERVICE_NAME:-ms-bribrain-ocr-ktp-extract-fullpytorch}"

DOCKERFILE="${DOCKERFILE:-Dockerfile.fullpytorch}"
DOCKER_TARGET="${DOCKER_TARGET:-}"

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

# Nginx image mirrored into Nexus so air-gapped deploy hosts can pull it
# without reaching Docker Hub. Pushed to the shared ocr-ktp/{env}/nginx:alpine
# path (no service name) so every service can reuse it.
NGINX_SOURCE_IMAGE="${NGINX_SOURCE_IMAGE:-nginx:alpine}"
NGINX_IMAGE_REF="${NEXUS_REGISTRY}/${PROJECT}/${DEPLOY_ENV}/nginx:alpine"

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
# BuildKit is required by Dockerfile.fullpytorch (syntax directive + cache mounts).
DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" ${DOCKER_TARGET:+--target "$DOCKER_TARGET"} -t "$IMAGE_REF" .

# ---------------------------------------------------------------------------
# Push to Nexus
# ---------------------------------------------------------------------------
echo "Pushing image to Nexus: $IMAGE_REF..."
docker push "$IMAGE_REF"

# ---------------------------------------------------------------------------
# Mirror nginx:alpine into Nexus (so air-gapped deploy hosts never need Docker Hub)
# ---------------------------------------------------------------------------
echo "Mirroring Nginx image into Nexus: ${NGINX_SOURCE_IMAGE} -> ${NGINX_IMAGE_REF}..."
docker pull "$NGINX_SOURCE_IMAGE"
docker tag "$NGINX_SOURCE_IMAGE" "$NGINX_IMAGE_REF"
docker push "$NGINX_IMAGE_REF"

echo ""
echo "Build & push successful!"
echo "  Image: $IMAGE_REF"
echo "  Nginx: $NGINX_IMAGE_REF"
echo ""
echo "Deploy it with: ./nexus_deploy.sh $DEPLOY_ENV $TAG"
