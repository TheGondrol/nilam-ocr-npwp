#!/usr/bin/env bash
# Matikan tracker yang jalan di latar (kalau run.sh dijalankan di terminal yang
# sama, cukup Ctrl+C). Membunuh proses yang mendengarkan port backend & frontend.
#
#   tools/tracker/stop.sh            backend :8090 + frontend :5173
#   tools/tracker/stop.sh --stack    plus: hentikan keempat container, Postgres, dan Redis
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BACKEND_PORT="${PORT:-8090}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

kill_port() {
  local port="$1" pids=""
  if command -v taskkill >/dev/null 2>&1; then # Windows (Git Bash)
    pids=$(netstat -ano | grep -E "TCP\s+[0-9.]+:$port\s.*LISTENING" | awk '{print $5}' | sort -u)
    for pid in $pids; do taskkill //PID "$pid" //F //T >/dev/null 2>&1 && say "port $port: proses $pid dimatikan"; done
  else # Linux / WSL / macOS
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    for pid in $pids; do kill "$pid" 2>/dev/null && say "port $port: proses $pid dimatikan"; done
  fi
  [[ -z "$pids" ]] && say "port $port: tidak ada yang jalan"
}

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

if [[ -f "$HERE/.env" ]]; then
  set -a; . "$HERE/.env"; set +a
fi
if [[ "${TRACKER_TARGET:-local}" == "gke" ]]; then
  say "menutup port-forward ke ${GKE_NAMESPACE:-nilam-ocr-npwp}"
  for port in 9030 9031 9032 9033; do kill_port "$port"; done
fi

if [[ "${1:-}" == "--stack" ]]; then
  say "menghentikan container pipeline, Postgres, dan Redis (data tetap tersimpan)"
  (cd "$ROOT" && docker compose -f docker-compose.yml -f docker-compose.db.yml stop)
  docker stop nilam-ocr-redis >/dev/null 2>&1 && say "nilam-ocr-redis dihentikan"
fi
