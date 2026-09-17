#!/bin/bash
# Deploy with AutoKernel-optimized PP-OCRv5 recognizer.
#
# This script:
#   1. Verifies the in-repo autokernel/ directory has the required artifacts
#   2. Builds the image with autokernel deps and pre-converted .pth weights
#   3. Runs the container with GPU and autokernel enabled
#
# Prerequisites:
#   - autokernel/workspace/ppocrv5/server_det.pth and server_rec.pth must exist
#     (run: uv run --extra autokernel python scripts/prepare_ppocrv5_server_weights.py --component both --force)
#   - autokernel/workspace/graph_capture_eval/verification_result.json must exist
#   - .env file with DATABASE_URL and API_KEY

set -euo pipefail

IMAGE_NAME="ms-bribrain-ocr-extract-autokernel"
CONTAINER_NAME="ms-bribrain-ocr-extract-autokernel"
PORT_MAPPING="8010:8010"

# ── Validate prerequisites ──────────────────────────────────────────
DET_PTH="autokernel/workspace/ppocrv5/server_det.pth"
REC_PTH="autokernel/workspace/ppocrv5/server_rec.pth"
VERIFICATION="autokernel/workspace/graph_capture_eval/verification_result.json"

missing=()
[[ -f "${DET_PTH}" ]] || missing+=("${DET_PTH}")
[[ -f "${REC_PTH}" ]] || missing+=("${REC_PTH}")
[[ -f "${VERIFICATION}" ]] || missing+=("${VERIFICATION}")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: Missing required artifacts:"
    printf '  %s\n' "${missing[@]}"
    echo ""
    echo "Run weight conversion first:"
    echo "  uv run --extra autokernel python scripts/prepare_ppocrv5_server_weights.py \\"
    echo "    --component both --force \\"
    echo "    --det-src /home/jupyter/.paddlex/official_models/PP-OCRv5_server_det/inference.pdiparams \\"
    echo "    --rec-src ./src/models/server_models/ppocrv5_server_rec_source/inference.pdiparams"
    exit 1
fi

# ── Build the image ─────────────────────────────────────────────────
echo "Building Docker image: ${IMAGE_NAME}..."
DOCKER_BUILDKIT=1 docker build \
    --build-arg ENABLE_AUTOKERNEL=1 \
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
echo "Starting container: ${CONTAINER_NAME} with GPU + AutoKernel..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p ${PORT_MAPPING} \
    --restart always \
    --env-file .env \
    -e OCR_BACKEND=auto \
    -e OCR_AUTOKERNEL_ENABLED=1 \
    -e AUTOKERNEL_ROOT=/app/autokernel \
    -e AUTOKERNEL_PPOCR_ROOT=/app/PaddleOCR2Pytorch \
    -e AUTOKERNEL_PPOCRV5_SERVER_DET_PTH=/app/autokernel/workspace/ppocrv5/server_det.pth \
    -e AUTOKERNEL_PPOCRV5_SERVER_REC_PTH=/app/autokernel/workspace/ppocrv5/server_rec.pth \
    -e AUTOKERNEL_WORKSPACE=/app/autokernel/workspace/graph_capture_eval \
    "${IMAGE_NAME}"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to start container."
    exit 1
fi

echo ""
echo "Deployment successful!"
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      http://localhost:8010"
echo "  Backend:   AutoKernel PP-OCRv5 (11x recognizer speedup)"
echo ""
echo "Test with:"
echo "  curl http://localhost:8010/health"
echo "  curl http://localhost:8010/health/ready"
