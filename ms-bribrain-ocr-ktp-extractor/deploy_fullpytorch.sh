#!/bin/bash
# Deploy with fullpytorch backend: end-to-end PyTorch detection + recognition
# (fp16-mixed det, torch.compile rec).
#
# Mirrors scripts/run_server_fullpytorch.sh but runs inside a Docker container.
#
# Prerequisites:
#   - autokernel/workspace/ppocrv5/server_det.pth and server_rec.pth must exist
#   - tests/data/good_data_3.png (warmup image, already in repo)
#   - .env file with API_KEY (DATABASE_URL is optional for load testing)

set -euo pipefail

IMAGE_NAME="ms-bribrain-ocr-extract-fullpytorch"
CONTAINER_NAME="ms-bribrain-ocr-extract-fullpytorch"
PORT_MAPPING="${PORT_MAPPING:-8010:8010}"
OCR_WORKERS="${OCR_WORKERS:-2}"

# CUDA MPS is required — it lets multiple uvicorn workers share a single GPU
# efficiently. The script will start the daemon on the host if it isn't already
# running, and mounts the pipe/log dirs into the container.
CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-pipe}"
CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

# Torch Inductor cache (persistent across container restarts; shared across
# workers). First warmup populates it; subsequent starts hit the cache and
# drop cold-start from ~150s/worker to ~10–20s/worker.
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor-cache}"

# ── Validate prerequisites ──────────────────────────────────────────
DET_PTH="autokernel/workspace/ppocrv5/server_det.pth"
REC_PTH="autokernel/workspace/ppocrv5/server_rec.pth"

missing=()
[[ -f "${DET_PTH}" ]] || missing+=("${DET_PTH}")
[[ -f "${REC_PTH}" ]] || missing+=("${REC_PTH}")

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: Missing required artifacts:"
    printf '  %s\n' "${missing[@]}"
    echo ""
    echo "Run weight conversion first:"
    echo "  uv run --extra autokernel python scripts/prepare_ppocrv5_server_weights.py \\"
    echo "    --component both --force"
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

# ── Ensure CUDA MPS daemon is running on the host ───────────────────
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}" "${TORCHINDUCTOR_CACHE_DIR}"
if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "CUDA MPS daemon already running."
else
    echo "Starting CUDA MPS daemon..."
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "Error: nvidia-cuda-mps-control not found on host."
        echo "Install the NVIDIA driver tools or run as a user with GPU access."
        exit 1
    fi
    CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}" \
    CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY}" \
        nvidia-cuda-mps-control -d
    sleep 1
    if ! pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "Error: failed to start CUDA MPS daemon. Check ${CUDA_MPS_LOG_DIRECTORY}."
        exit 1
    fi
    echo "CUDA MPS daemon started."
fi

# ── Build the image ─────────────────────────────────────────────────
echo "Building Docker image: ${IMAGE_NAME}..."
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile.fullpytorch \
    -t "${IMAGE_NAME}" .

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
echo "Starting container: ${CONTAINER_NAME} with GPU + FullPyTorch backend (workers=${OCR_WORKERS})..."

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -p ${PORT_MAPPING} \
    --restart unless-stopped \
    --env-file .env \
    -v "${CUDA_MPS_PIPE_DIRECTORY}:${CUDA_MPS_PIPE_DIRECTORY}" \
    -v "${CUDA_MPS_LOG_DIRECTORY}:${CUDA_MPS_LOG_DIRECTORY}" \
    -v "${TORCHINDUCTOR_CACHE_DIR}:${TORCHINDUCTOR_CACHE_DIR}" \
    -e "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}" \
    -e "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY}" \
    -e "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}" \
    -e "TORCHINDUCTOR_FX_GRAPH_CACHE=1" \
    -e "TORCHINDUCTOR_AUTOGRAD_CACHE=1" \
    -e "TORCH_LOGS=recompiles" \
    -e HOME=/tmp \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e APP_ENVIRO=onprem \
    -e OCR_BACKEND=fullpytorch \
    -e OCR_AUTOKERNEL_ENABLED=1 \
    -e AUTOKERNEL_ROOT=/app/autokernel \
    -e AUTOKERNEL_PPOCR_ROOT=/app/PaddleOCR2Pytorch \
    -e AUTOKERNEL_PPOCRV5_SERVER_DET_PTH=/app/autokernel/workspace/ppocrv5/server_det.pth \
    -e AUTOKERNEL_PPOCRV5_SERVER_REC_PTH=/app/autokernel/workspace/ppocrv5/server_rec.pth \
    -e OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS=0 \
    -e OCR_AUTOKERNEL_DTYPE=float16 \
    -e OCR_AUTOKERNEL_DET_DTYPE=float16 \
    -e OCR_AUTOKERNEL_TORCH_COMPILE_REC=1 \
    -e OCR_AUTOKERNEL_TORCH_COMPILE_MODE=default \
    -e OCR_AUTOKERNEL_TORCH_COMPILE_DYNAMIC=0 \
    -e OCR_AUTOKERNEL_BUCKET_DEBUG_LOG=1 \
    -e OCR_AUTOKERNEL_REC_BATCH_SIZE=8 \
    -e OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN=1280 \
    -e OCR_AUTOKERNEL_DET_LIMIT_TYPE=max \
    -e OCR_AUTOKERNEL_WARMUP_IMAGE_PATH=/app/tests/data/good_data_3.png \
    -e OCR_WORKERS="${OCR_WORKERS}" \
    "${IMAGE_NAME}" \
    sh -c "exec uvicorn src.main:app --host 0.0.0.0 --port 8010 --workers ${OCR_WORKERS}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Deployment successful!"
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      http://localhost:${PORT_MAPPING%%:*}"
echo "  Backend:   FullPyTorch (fp16 det + compile-rec, batch=8)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Wait for startup (torch.compile warmup can take 30-60s), then test with:"
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
