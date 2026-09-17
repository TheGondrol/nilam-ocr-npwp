#!/usr/bin/env bash
set -euo pipefail

export PATH="/app/.venv/bin:$PATH"

HPI_MARKER="/app/.hpi_deps_installed"

if [[ "${DISABLE_HPI_INSTALL:-0}" != "1" ]]; then
  if [[ ! -f "${HPI_MARKER}" ]]; then
    echo "Installing PaddleOCR HPI GPU deps (first run)..."
    /app/.venv/bin/paddleocr install_hpi_deps gpu
    touch "${HPI_MARKER}"
    echo "HPI GPU deps installed."
  else
    echo "HPI GPU deps already installed; skipping."
  fi
else
  echo "DISABLE_HPI_INSTALL=1 set; skipping HPI GPU deps install."
fi

exec "$@"
