#!/usr/bin/env bash
# Jalankan tracker (backend 8090 + Vite 5173) dari Git Bash / WSL / Linux.
#
#   tools/tracker/run.sh            backend + frontend; Ctrl+C mematikan semuanya
#   tools/tracker/run.sh --stack    plus: nyalakan Redis + keempat container + Postgres dulu
#
# Target diatur di tools/tracker/.env:
#   blok GKE aktif      -> port-forward ke namespace nilam-ocr-npwp, hasil tiap
#                          tahap diambil dengan polling (callback dari pod tidak
#                          bisa masuk ke laptop)
#   blok GKE dikomentari -> stack lokal seperti biasa, callback langsung
#
# Prasyarat yang diperiksa: Redis di 6379, keempat service /health, dan (lokal)
# tunnel SSH ke model OCR (8070). Yang tidak terpenuhi hanya diperingatkan,
# kecuali Redis.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BACKEND_PORT="${PORT:-8090}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-5434}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mperingatan:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mgagal:\033[0m %s\n' "$*" >&2; exit 1; }

up() { curl -s -m 3 -o /dev/null "$1"; }

# --- konfigurasi -------------------------------------------------------------
if [[ -f "$HERE/.env" ]]; then
  set -a; . "$HERE/.env"; set +a
fi
TRACKER_TARGET="${TRACKER_TARGET:-local}"

GUARDRAILS_URL="${GUARDRAILS_URL:-http://127.0.0.1:8031}"
EKSTRAKSI_URL="${EKSTRAKSI_URL:-http://127.0.0.1:8030}"
STRUCTURING_URL="${STRUCTURING_URL:-http://127.0.0.1:8032}"
SCORING_URL="${SCORING_URL:-http://127.0.0.1:8033}"
export GUARDRAILS_URL EKSTRAKSI_URL STRUCTURING_URL SCORING_URL

if [[ "${1:-}" == "--stack" && "$TRACKER_TARGET" == "gke" ]]; then
  die "--stack hanya untuk mode lokal; komentari blok GKE di .env dulu"
fi

if [[ "${1:-}" == "--stack" ]]; then
  say "menyalakan Redis + container pipeline + Postgres"
  docker start nilam-ocr-redis >/dev/null 2>&1 \
    || docker run -d --name nilam-ocr-redis --restart unless-stopped -p 127.0.0.1:6379:6379 redis:7.4-alpine >/dev/null
  (cd "$ROOT" && docker compose -f docker-compose.yml -f docker-compose.db.yml up -d)
fi

# --- prasyarat ---------------------------------------------------------------
docker exec nilam-ocr-redis redis-cli ping >/dev/null 2>&1 \
  || die "Redis tidak jalan. Jalankan dengan --stack, atau: docker start nilam-ocr-redis"

PF_PID=""
if [[ "$TRACKER_TARGET" == "gke" ]]; then
  NS="${GKE_NAMESPACE:-nilam-ocr-npwp}"
  WORKLOAD="${GKE_WORKLOAD:-deploy/nilam-ocr-npwp}"
  command -v kubectl >/dev/null 2>&1 || die "kubectl tidak ada di PATH"
  kubectl -n "$NS" get "$WORKLOAD" >/dev/null 2>&1 \
    || die "tidak bisa membaca $WORKLOAD di namespace $NS. Jalankan: gcloud container clusters get-credentials gc-ddb-dev-gke-cluster-01 --project ddb-kubecluster-dev-01 --location asia-southeast2"

  say "target GKE: $NS/$WORKLOAD -> 127.0.0.1:9030-9033 (port-forward)"
  kubectl -n "$NS" port-forward "$WORKLOAD" 9030:8030 9031:8031 9032:8032 9033:8033 >"$HERE/.port-forward.log" 2>&1 &
  PF_PID=$!

  for _ in $(seq 1 20); do
    up "$EKSTRAKSI_URL/health" && break
    sleep 0.5
  done
  up "$EKSTRAKSI_URL/health" \
    || die "port-forward tidak siap; lihat $HERE/.port-forward.log"
  say "hasil tiap tahap diambil dengan polling (TRACKER_POLL=${TRACKER_POLL:-0}); callback pod tetap ke Orkestrasi di cluster"
else
  say "target lokal: container di 127.0.0.1:803x"
  up "http://127.0.0.1:8070/health" \
    || warn "model OCR (:8070) tidak terjangkau; buka tunnel SSH ke VM (gcloud compute ssh ... -- -L 8070:localhost:8070)"
fi

for svc in "guardrails|$GUARDRAILS_URL" "ekstraksi|$EKSTRAKSI_URL" "structuring|$STRUCTURING_URL" "scoring|$SCORING_URL"; do
  name="${svc%%|*}"; url="${svc##*|}"
  up "$url/health" || warn "service $name ($url) tidak menjawab; tahap itu akan gagal"
done

python -c "import fastapi, httpx, redis" 2>/dev/null \
  || { say "memasang dependency backend"; python -m pip install -q -r "$HERE/backend/requirements.txt"; }
[[ -d "$HERE/frontend/node_modules" ]] \
  || { say "memasang dependency frontend"; (cd "$HERE/frontend" && npm install --no-audit --no-fund); }

# --- jalankan ------------------------------------------------------------------
cleanup() {
  say "mematikan tracker"
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" "${PF_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

say "backend  -> http://127.0.0.1:$BACKEND_PORT"
PORT="$BACKEND_PORT" python "$HERE/backend/app.py" &
BACKEND_PID=$!

say "frontend -> http://127.0.0.1:$FRONTEND_PORT"
(cd "$HERE/frontend" && npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort) &
FRONTEND_PID=$!

sleep 2
up "http://127.0.0.1:$BACKEND_PORT/api/health" || die "backend tidak menjawab; lihat log di atas"
say "siap. Buka http://127.0.0.1:$FRONTEND_PORT  (Ctrl+C untuk berhenti)"
wait
