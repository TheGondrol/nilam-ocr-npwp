#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAMESPACE="${NAMESPACE:-nilam-ocr-npwp}"
SECRET="${SECRET:-nilam-ocr-npwp-secrets}"
DB_HOST="${DB_HOST:-34.50.114.49}"
DB_PORT="${DB_PORT:-5432}"
PSQL_IMAGE="${PSQL_IMAGE:-postgres:16-alpine}"
DB_SERVICES=(ekstraksi structuring scoring)

usage() {
  cat <<EOF
Pemakaian: deploy/helm/apply-schema.sh [-y] [service...]

Menjalankan services/<service>/db/schema.sql ke database release, dalam satu transaksi.
Tanpa argumen: ${DB_SERVICES[*]}. Semua statement memakai IF NOT EXISTS, jadi aman
dijalankan ulang, tetapi tidak menghapus atau memindahkan tabel lama.

DATABASE_URL diambil dari Secret $SECRET (namespace $NAMESPACE); host-nya diganti
dengan DB_HOST:DB_PORT (default $DB_HOST:$DB_PORT) karena laptop tidak bisa
menjangkau DNS cluster. psql dijalankan lewat Docker ($PSQL_IMAGE).
EOF
}

die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ASSUME_YES=0
SERVICES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "opsi tidak dikenal: $1" ;;
    *)
      [[ " ${DB_SERVICES[*]} " == *" $1 "* ]] || die "service tanpa database: $1 (pilihan: ${DB_SERVICES[*]})"
      SERVICES+=("$1"); shift ;;
  esac
done
[[ ${#SERVICES[@]} -gt 0 ]] || SERVICES=("${DB_SERVICES[@]}")

for tool in kubectl docker base64 sed; do
  command -v "$tool" >/dev/null || die "$tool tidak ditemukan di PATH"
done

FILES=()
for svc in "${SERVICES[@]}"; do
  f="$ROOT/services/$svc/db/schema.sql"
  [[ -f "$f" ]] || die "tidak ada: $f"
  FILES+=("$f")
done

raw_url="$(kubectl -n "$NAMESPACE" get secret "$SECRET" -o jsonpath='{.data.DATABASE_URL}' | base64 -d)"
[[ -n "$raw_url" ]] || die "DATABASE_URL kosong di Secret $SECRET"
PGURL="$(printf '%s' "$raw_url" | sed -E "s#^postgresql\+[a-z0-9]+://#postgresql://#; s#@[^/]+/#@$DB_HOST:$DB_PORT/#")"
export PGURL

echo "Target : $(printf '%s' "$PGURL" | sed -E 's#//([^:]+):[^@]*@#//\1:***@#')"
echo "File   :"
for f in "${FILES[@]}"; do echo "  ${f#"$ROOT"/}"; done

if [[ $ASSUME_YES -eq 0 ]]; then
  [[ -t 0 ]] || die "tidak ada terminal untuk konfirmasi; pakai -y"
  read -r -p "Jalankan? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "dibatalkan"
fi

cat "${FILES[@]}" | docker run --rm -i -e PGURL "$PSQL_IMAGE" \
  sh -c 'psql "$PGURL" -v ON_ERROR_STOP=1 --single-transaction -f -'

echo "Selesai."
