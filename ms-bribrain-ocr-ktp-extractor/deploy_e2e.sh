#!/bin/sh
# ---------------------------------------------------------------------------
# deploy_e2e.sh
#
# Build the extractor Docker image with e2e tests baked into the
# build pipeline, then deploy as a local container.
#
# The build will FAIL if e2e tests do not pass, so a successful deployment
# guarantees that the service passed its e2e suite.
#
# Usage:
#   sh deploy_e2e.sh              # build + deploy
#   sh deploy_e2e.sh --build-only # build (with tests) but don't deploy
# ---------------------------------------------------------------------------

set -eu

# -- Variables ---------------------------------------------------------------
IMAGE_NAME="ms-bribrain-ocr-extract"
CONTAINER_NAME="ms-bribrain-ocr-extract"
PORT_MAPPING="8010:8010"
ENV_FILE=".env"
BUILD_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=true ;;
        *) ;;
    esac
done

# -- Pre-flight checks ------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy .env.template to .env and fill in values."
    exit 1
fi

# -- Step 1: Run e2e tests ---------------------------------------------------
echo "=========================================="
echo "  Step 1: Running e2e tests"
echo "=========================================="
echo ""

docker build \
    --secret id=env,src="$ENV_FILE" \
    --target e2e-test \
    -t "${IMAGE_NAME}-e2e-test" \
    .

if [ $? -ne 0 ]; then
    echo ""
    echo "✘ E2E tests failed. Aborting deployment."
    exit 1
fi

echo ""
echo "✔ E2E tests passed."

# -- Step 2: Build production image ------------------------------------------
echo ""
echo "=========================================="
echo "  Step 2: Building production image"
echo "=========================================="
echo ""

docker build \
    --target production \
    -t "$IMAGE_NAME" \
    .

if [ $? -ne 0 ]; then
    echo ""
    echo "✘ Production build failed."
    exit 1
fi

echo ""
echo "✔ Production image built."

if [ "$BUILD_ONLY" = true ]; then
    echo "  --build-only flag set; skipping deployment."
    exit 0
fi

# -- Step 3: Stop & remove old container -------------------------------------
echo ""
echo "Stopping and removing old container (if any)..."

EXISTING=$(docker ps -q -f name="$CONTAINER_NAME")
if [ -n "$EXISTING" ]; then
    docker stop "$CONTAINER_NAME"
    echo "  Stopped container: $CONTAINER_NAME"
fi

EXISTING=$(docker ps -a -q -f name="$CONTAINER_NAME")
if [ -n "$EXISTING" ]; then
    docker rm "$CONTAINER_NAME"
    echo "  Removed container: $CONTAINER_NAME"
fi

# -- Step 4: Start new container ---------------------------------------------
echo ""
echo "Starting new container: $CONTAINER_NAME ..."

docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT_MAPPING" \
    --env-file "$ENV_FILE" \
    --restart always \
    "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "✘ Failed to start container."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Deployment successful!"
echo "=========================================="
echo ""
echo "  Container : $CONTAINER_NAME"
echo "  URL       : http://localhost:8010"
echo "  Docs      : http://localhost:8010/docs"
echo "  Health    : http://localhost:8010/health/live"
echo ""
echo "  Test with : curl http://localhost:8010/health"
