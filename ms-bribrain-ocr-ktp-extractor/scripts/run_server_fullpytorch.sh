#!/bin/bash
# Launch the API locally with fullpytorch backend + fp16-mixed det + compile-rec.
set -eu
cd "$(dirname "$0")/.."
export OCR_BACKEND=fullpytorch
export OCR_AUTOKERNEL_ENABLED=1
export AUTOKERNEL_ROOT="${PWD}/autokernel"
export AUTOKERNEL_PPOCR_ROOT="${PWD}/PaddleOCR2Pytorch"
export AUTOKERNEL_PPOCRV5_SERVER_DET_PTH="${AUTOKERNEL_ROOT}/workspace/ppocrv5/server_det.pth"
export AUTOKERNEL_PPOCRV5_SERVER_REC_PTH="${AUTOKERNEL_ROOT}/workspace/ppocrv5/server_rec.pth"
export OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS=0
export OCR_AUTOKERNEL_DTYPE=float16
export OCR_AUTOKERNEL_DET_DTYPE=float16
export OCR_AUTOKERNEL_TORCH_COMPILE_REC=1
export OCR_AUTOKERNEL_TORCH_COMPILE_MODE=default
export OCR_AUTOKERNEL_REC_BATCH_SIZE=8
export OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN=1280
export OCR_AUTOKERNEL_DET_LIMIT_TYPE=max
export OCR_AUTOKERNEL_WARMUP_IMAGE_PATH="${PWD}/tests/data/good_data_3.png"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
WORKERS="${OCR_WORKERS:-1}"
exec .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8010 --workers "${WORKERS}"
