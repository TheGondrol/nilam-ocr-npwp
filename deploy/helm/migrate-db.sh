#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAMESPACE="${NAMESPACE:-nilam-ocr-npwp}"
SECRET="${SECRET:-nilam-ocr-npwp-secrets}"
DB_PORT="${DB_PORT:-5432}"
IMAGE="${IMAGE:-nilam-ocr-migrate:local}"

usage() {
  cat <<EOF
Pemakaian: DB_HOST=<alamat> deploy/helm/migrate-db.sh [-y] [perintah alembic...]

Menjalankan migrasi database repo ini (db/migrations) ke database release.
Tanpa argumen: "upgrade head". Contoh lain: "current", "history", "heads".

DATABASE_URL diambil dari Secret $SECRET (namespace $NAMESPACE); host-nya diganti
dengan DB_HOST:DB_PORT (wajib di-set) karena laptop tidak bisa menjangkau DNS
cluster. Alembic dijalankan lewat image $IMAGE yang dibangun dari db/Dockerfile,
jadi laptop tidak perlu Python atau dependensinya.

Jalankan ini SEBELUM men-deploy image yang membutuhkan perubahan tabelnya.
EOF
}

die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ASSUME_YES=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ ${#ARGS[@]} -gt 0 ]] || ARGS=(upgrade head)

for tool in kubectl docker base64 sed; do
  command -v "$tool" >/dev/null || die "$tool tidak ditemukan di PATH"
done
[[ -n "${DB_HOST:-}" ]] || die "DB_HOST belum di-set: alamat PostgreSQL yang terjangkau dari laptop"

raw_url="$(kubectl -n "$NAMESPACE" get secret "$SECRET" -o jsonpath='{.data.DATABASE_URL}' | base64 -d)"
[[ -n "$raw_url" ]] || die "DATABASE_URL kosong di Secret $SECRET"
DATABASE_URL="$(printf '%s' "$raw_url" | sed -E "s#@[^/]+/#@$DB_HOST:$DB_PORT/#")"
export DATABASE_URL

echo "Target  : $(printf '%s' "$DATABASE_URL" | sed -E 's#//([^:]+):[^@]*@#//\1:***@#')"
echo "Perintah: alembic ${ARGS[*]}"

if [[ $ASSUME_YES -eq 0 ]]; then
  [[ -t 0 ]] || die "tidak ada terminal untuk konfirmasi; pakai -y"
  read -r -p "Jalankan? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "dibatalkan"
fi

docker build -q -f "$ROOT/db/Dockerfile" -t "$IMAGE" "$ROOT" >/dev/null
docker run --rm -e DATABASE_URL "$IMAGE" python -m alembic -c db/alembic.ini "${ARGS[@]}"

echo "Selesai."
