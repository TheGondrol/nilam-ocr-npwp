#!/bin/bash
# Deploy with optimized Hybrid backend: PaddleOCR detection + AutoKernel recognition.
#
# This script:
#   1. Verifies the in-repo autokernel/ directory has the required artifacts
#   2. Builds the image with autokernel deps (torch, triton)
#   3. Runs the container with GPU and hybrid backend
#
# The hybrid backend is ~3.6x faster than the previous autokernel mode:
#   - Detection-only PaddleOCR (~125ms vs ~410ms full pipeline)
#   - GPU preprocessing + batched inference + GPU argmax (~80ms vs ~350ms)
#   - Total: ~215ms per image
#
# Prerequisites:
#   - autokernel/workspace/ppocrv5/server_rec.pth must exist
#   - autokernel/workspace/graph_capture_eval/verification_result.json
#   - .env file with API_KEY (DATABASE_URL is optional for load testing)

set -euo pipefail

IMAGE_NAME="ms-bribrain-ocr-extract-hybrid"
CONTAINER_NAME="ms-bribrain-ocr-extract-hybrid"
PORT_MAPPING="${PORT_MAPPING:-8010:8010}"

# ── Validate prerequisites ──────────────────────────────────────────
REC_PTH="autokernel/workspace/ppocrv5/server_rec.pth"
VERIFICATION="autokernel/workspace/graph_capture_eval/verification_result.json"

missing=()
[[ -f "${REC_PTH}" ]] || missing+=("${REC_PTH}")
[[ -f "${VERIFICATION}" ]] || missing+=("${VERIFICATION}")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: Missing required artifacts:"
    printf '  %s\n' "${missing[@]}"
    echo ""
    echo "Run weight conversion first:"
    echo "  uv run --extra autokernel python scripts/prepare_ppocrv5_server_weights.py \\"
    echo "    --component rec --force \\"
    echo "    --rec-src ./src/models/server_models/ppocrv5_server_rec_source/inference.pdiparams"
    exit 1
fi

# Create .env if missing (allows load testing without a database)
if [[ ! -f .env ]]; then
    echo "No .env file found — creating minimal .env for load testing..."
    cat > .env <<'ENVEOF'
API_KEY=loadtest
DATABASE_URL=
ENVEOF
    echo "Created .env with API_KEY=loadtest (no database)."
fi

# ── Build the image ─────────────────────────────────────────────────
echo "Building Docker image: ${IMAGE_NAME}..."
DOCKER_BUILDKIT=1 docker build \
    --build-arg ENABLE_AUTOKERNEL=1 \
    --target production \
    -t "${IMAGE_NAME}" .

if [[ $? -ne 0 ]]; then
    echo "Error: Docker build failed."
    exit 1
fi

# ── Stop/remove old container ───────────────────────────────────────
EXISTING=$(docker ps -q -f name="^${CONTAINER_NAME}$")
if [[ -n "${EXISTING}" ]]; then
    echo "Stopping old container: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}"
fi
EXISTING=$(docker ps -a -q -f name="^${CONTAINER_NAME}$")
if [[ -n "${EXISTING}" ]]; then
    docker rm "${CONTAINER_NAME}"
fi

# ── Start container ─────────────────────────────────────────────────
echo "Starting container: ${CONTAINER_NAME} with GPU + Hybrid backend..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p ${PORT_MAPPING} \
    --restart unless-stopped \
    --env-file .env \
    -e APP_ENVIRO=onprem \
    -e OCR_BACKEND=hybrid \
    -e OCR_AUTOKERNEL_ENABLED=1 \
    -e AUTOKERNEL_ROOT=/app/autokernel \
    -e AUTOKERNEL_PPOCR_ROOT=/app/PaddleOCR2Pytorch \
    -e AUTOKERNEL_PPOCRV5_SERVER_REC_PTH=/app/autokernel/workspace/ppocrv5/server_rec.pth \
    -e AUTOKERNEL_WORKSPACE=/app/autokernel/workspace/graph_capture_eval \
    "${IMAGE_NAME}"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to start container."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Deployment successful!"
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      http://localhost:${PORT_MAPPING%%:*}"
echo "  Backend:   Hybrid (TextDetection + AutoKernel PyTorch rec)"
echo "  Expected:  ~215ms per image (3.6x faster than previous)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Wait for startup, then test with:"
echo "  curl http://localhost:${PORT_MAPPING%%:*}/health"
echo "  curl http://localhost:${PORT_MAPPING%%:*}/health/ready"
echo ""
echo "Load test:"
echo "  curl -X POST http://localhost:${PORT_MAPPING%%:*}/v1/ocr_extract \\"
echo "    -H 'X-API-Key: loadtest' \\"
echo "    -F 'file=@tests/data/good_data.png'"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
