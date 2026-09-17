#!/bin/bash
# Deploy with fullpytorch backend: end-to-end PyTorch detection + recognition
# (fp16-mixed det, torch.compile rec), with Nginx load balancing across replicas.
#
# Mirrors scripts/run_server_fullpytorch.sh but runs inside Docker containers
# behind an Nginx load balancer (all in Docker).
#
# Prerequisites:
#   - tests/data/good_data_3.png (warmup image, already in repo)
#   - .env file with API_KEY + SA_* GCS service-account vars
#     (DATABASE_URL is optional for load testing)
#     server_det.pth / server_rec.pth are pulled from GCS at startup.

set -euo pipefail

IMAGE_NAME="ms-bribrain-ocr-ktp-extract-fullpytorch"
CONTAINER_NAME="ms-bribrain-ocr-ktp-extract-fullpytorch"
NGINX_CONTAINER_NAME="ms-bribrain-ocr-ktp-fullpytorch-nginx"
NETWORK_NAME="ms-bribrain-ocr-ktp-fullpytorch-network"
CONTAINER_PORT=8010
REPLICA="${REPLICA:-2}"
NGINX_PORT="${NGINX_PORT:-80}"
OCR_WORKERS="${OCR_WORKERS:-2}"

## GPU device assignment. Comma-separated list of GPU indices (e.g. "0", "0,1",
# "1,2,3"). Replica i is pinned to GPU_DEVICES[i % len(GPU_DEVICES)] via
# --gpus '"device=<idx>"'.
#
# Default: every visible GPU on the host (queried via nvidia-smi), so replicas
# spread across the available hardware. Falls back to GPU 0 only if nvidia-smi
# is unavailable or reports no GPUs. Override by exporting GPU_DEVICES.
if [[ -z "${GPU_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    fi
    GPU_DEVICES="${GPU_DEVICES:-0}"
    echo "GPU_DEVICES not set; defaulting to detected GPUs: ${GPU_DEVICES}"
fi
IFS=',' read -r -a GPU_DEVICE_ARR <<< "${GPU_DEVICES}"
if [[ ${#GPU_DEVICE_ARR[@]} -eq 0 ]]; then
    echo "Error: GPU_DEVICES must contain at least one GPU index."
    exit 1
fi

# CUDA MPS is required — it lets multiple uvicorn workers share a single GPU
# efficiently. The script will start the daemon on the host if it isn't already
# running, and mounts the pipe/log dirs into the container.
CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-pipe}"
CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

# Torch Inductor cache (persistent across container restarts; shared across
# workers). First warmup populates it; subsequent starts hit the cache and
# drop cold-start from ~150s/worker to ~10–20s/worker.
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor-cache}"

# Helper: get container name for a given index
get_container_name() {
    local index=$1
    if [ "$index" -eq 0 ]; then
        echo "$CONTAINER_NAME"
    else
        echo "${CONTAINER_NAME}-${index}"
    fi
}

# server_det.pth / server_rec.pth are fetched from GCS at container
# startup by src/main.py (APP_ENVIRO=cloud, see env vars below), so no
# host-side .pth prerequisite check is needed.

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

# ── Ensure Docker network exists ────────────────────────────────────
echo ""
echo "Ensuring Docker network: ${NETWORK_NAME}..."
docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1 || docker network create "${NETWORK_NAME}"

# ── Stop/remove old replica containers ──────────────────────────────
echo "Stopping and removing old containers (if any)..."
for i in $(seq 0 $((REPLICA - 1))); do
    NAME=$(get_container_name "$i")
    EXISTING=$(docker ps -q -f name="^${NAME}$")
    if [[ -n "${EXISTING}" ]]; then
        docker stop "${NAME}"
        echo "Stopped container: ${NAME}"
    fi
    EXISTING=$(docker ps -a -q -f name="^${NAME}$")
    if [[ -n "${EXISTING}" ]]; then
        docker rm "${NAME}"
        echo "Removed container: ${NAME}"
    fi
done

# Also clean up any stale replicas from a previous larger REPLICA value
for stale in $(docker ps -a --format '{{.Names}}' | grep -E "^${CONTAINER_NAME}(-[0-9]+)?$" || true); do
    keep=0
    for i in $(seq 0 $((REPLICA - 1))); do
        if [[ "${stale}" == "$(get_container_name "$i")" ]]; then
            keep=1
            break
        fi
    done
    if [[ "${keep}" -eq 0 ]]; then
        docker rm -f "${stale}" >/dev/null 2>&1 || true
        echo "Removed stale container: ${stale}"
    fi
done

# ── Stop/remove old nginx container ─────────────────────────────────
EXISTING=$(docker ps -q -f name="^${NGINX_CONTAINER_NAME}$")
if [[ -n "${EXISTING}" ]]; then
    docker stop "${NGINX_CONTAINER_NAME}"
    echo "Stopped container: ${NGINX_CONTAINER_NAME}"
fi
EXISTING=$(docker ps -a -q -f name="^${NGINX_CONTAINER_NAME}$")
if [[ -n "${EXISTING}" ]]; then
    docker rm "${NGINX_CONTAINER_NAME}"
    echo "Removed container: ${NGINX_CONTAINER_NAME}"
fi

# ── Start replica containers ────────────────────────────────────────
echo ""
echo "Starting ${REPLICA} replica(s) with FullPyTorch backend (workers=${OCR_WORKERS} per replica)..."
for i in $(seq 0 $((REPLICA - 1))); do
    NAME=$(get_container_name "$i")
    GPU_IDX="${GPU_DEVICE_ARR[$((i % ${#GPU_DEVICE_ARR[@]}))]}"
    echo "Starting container: ${NAME} (GPU=${GPU_IDX})"

    docker run -d \
        --name "${NAME}" \
        --gpus "\"device=${GPU_IDX}\"" \
        --ipc=host \
        --network "${NETWORK_NAME}" \
        --user "$(id -u):$(id -g)" \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
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
        -e APP_ENVIRO=cloud \
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
        sh -c "exec uvicorn src.main:app --host 0.0.0.0 --port ${CONTAINER_PORT} --workers ${OCR_WORKERS}"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to start container ${NAME}."
        exit 1
    fi
done

# ── Generate Nginx config ───────────────────────────────────────────
echo ""
echo "Generating Nginx config..."

NGINX_CONF="upstream fullpytorch_backend {
    least_conn;
$(for i in $(seq 0 $((REPLICA - 1))); do
    NAME=$(get_container_name "$i")
    echo "    server ${NAME}:${CONTAINER_PORT} max_fails=3 fail_timeout=30s;"
done)
}

server {
    listen 80;
    server_name _;

    client_max_body_size 20M;
    client_body_buffer_size 10M;

    location / {
        proxy_pass http://fullpytorch_backend;
        proxy_http_version 1.1;

        proxy_set_header Connection \"\";
        proxy_set_header Host \\\$host;
        proxy_set_header X-Real-IP \\\$remote_addr;
        proxy_set_header X-Forwarded-For \\\$proxy_add_x_forwarded_for;

        proxy_connect_timeout 60;
        proxy_send_timeout 300;
        proxy_read_timeout 300;

        proxy_request_buffering off;
    }
}"

# ── Start Nginx container ───────────────────────────────────────────
echo "Starting Nginx container..."
docker run -d \
    --name "${NGINX_CONTAINER_NAME}" \
    --network "${NETWORK_NAME}" \
    -p ${NGINX_PORT}:80 \
    --restart always \
    nginx:alpine

if [ $? -ne 0 ]; then
    echo "Error: Failed to start Nginx container."
    exit 1
fi

# Write config into Nginx container and reload
echo "$NGINX_CONF" | docker exec -i "${NGINX_CONTAINER_NAME}" tee /etc/nginx/conf.d/default.conf > /dev/null

echo "Validating Nginx config..."
docker exec "${NGINX_CONTAINER_NAME}" nginx -t
if [ $? -ne 0 ]; then
    echo "Error: Nginx config validation failed!"
    exit 1
fi

echo "Reloading Nginx..."
docker exec "${NGINX_CONTAINER_NAME}" nginx -s reload

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Deployment successful!"
echo "  Backend:   FullPyTorch (fp16 det + compile-rec, batch=8)"
echo "  Replicas:  ${REPLICA} (workers=${OCR_WORKERS} per replica)"
echo "  GPUs:      ${GPU_DEVICES}"
echo "-----------------------------------------------------------------"
for i in $(seq 0 $((REPLICA - 1))); do
    NAME=$(get_container_name "$i")
    GPU_IDX="${GPU_DEVICE_ARR[$((i % ${#GPU_DEVICE_ARR[@]}))]}"
    echo "  ${NAME} (GPU=${GPU_IDX}) -> ${NAME}:${CONTAINER_PORT} (internal)"
done
echo "-----------------------------------------------------------------"
echo "  Nginx (port ${NGINX_PORT}) -> load balancing ${REPLICA} replicas"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Wait for startup (torch.compile warmup can take 30-60s per replica), then test with:"
echo "  curl http://localhost:${NGINX_PORT}/health"
echo "  curl http://localhost:${NGINX_PORT}/health/ready"
echo ""
echo "Load test:"
echo "  curl -X POST http://localhost:${NGINX_PORT}/v1/ocr_extract \\"
echo "    -H 'X-API-Key: loadtest' \\"
echo "    -F 'file=@tests/data/good_data.png'"
echo ""
echo "Logs:"
for i in $(seq 0 $((REPLICA - 1))); do
    NAME=$(get_container_name "$i")
    echo "  docker logs -f ${NAME}"
done
echo "  docker logs -f ${NGINX_CONTAINER_NAME}"
