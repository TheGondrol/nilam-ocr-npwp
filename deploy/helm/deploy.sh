#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART="$ROOT/deploy/helm/nilam-ocr-npwp"
DEPLOY_ENV="${DEPLOY_ENV:-ddb-dev}"
VALUES="$CHART/values-$DEPLOY_ENV.yaml"
RELEASE="${RELEASE:-nilam-ocr-npwp}"
NAMESPACE="${NAMESPACE:-nilam-ocr-npwp}"
REGISTRY="${REGISTRY:-asia-southeast2-docker.pkg.dev/common-cicd-dev-01/gc-bribrain-dev-gar-temp-01}"
IMAGE_PREFIX="${IMAGE_PREFIX:-ms-bribrain-nilam-ocr-npwp}"
EXPECTED_CONTEXT="${EXPECTED_CONTEXT:-gke_ddb-kubecluster-dev-01_asia-southeast2_gc-ddb-dev-gke-cluster-01}"
TIMEOUT="${TIMEOUT:-10m}"
# Revisi release yang disimpan Helm (satu Secret sh.helm.release.v1.<release>.vN per revisi), untuk rollback.
HISTORY_MAX="${HISTORY_MAX:-5}"
ALL_SERVICES=(orchestrator guardrails ekstraksi structuring scoring)

usage() {
  cat <<EOF
Pemakaian: deploy/helm/deploy.sh [opsi] <service...|all>

Build image service yang disebut, push ke Artifact Registry, lalu helm upgrade
release $RELEASE dengan tag baru hanya untuk service tersebut. Tiap service adalah
Deployment sendiri ($RELEASE-<service>), jadi hanya pod service itu yang diganti.

Deploy (helm upgrade) hanya dari branch main yang sama persis dengan origin/main, tanpa
perubahan yang belum di-commit. Branch lain hanya boleh --build-only atau --dry-run.

Service: ${ALL_SERVICES[*]} (atau all)

Opsi:
  --tag TAG       pakai tag ini, bukan tag otomatis dari git (SHA pendek dari commit bersih).
                  Untuk deploy harus SHA commit di origin/main; commit selain HEAD hanya
                  bersama --skip-build
  --allow-dirty   izinkan build dari working tree yang belum di-commit; tag menjadi
                  <sha>-dirty-<waktu>. Hanya bersama --build-only: image ini tidak bisa di-deploy
  --skip-build    tanpa build/push; tag harus sudah ada di registry
  --build-only    build + push saja, tanpa helm upgrade
  --dry-run       tampilkan diff manifest terhadap release yang berjalan, tanpa apply
  -y, --yes       tanpa konfirmasi
  -h, --help      tampilkan bantuan ini

Environment: DEPLOY_ENV (default ddb-dev -> values-ddb-dev.yaml), RELEASE, NAMESPACE,
REGISTRY, EXPECTED_CONTEXT, TIMEOUT (default 10m), HISTORY_MAX (revisi release yang disimpan
untuk rollback, default 5).

Contoh:
  deploy/helm/deploy.sh scoring
  deploy/helm/deploy.sh ekstraksi structuring scoring
  deploy/helm/deploy.sh --dry-run --skip-build --tag 4d02eba all
EOF
}

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

TAG=""
ALLOW_DIRTY=0
SKIP_BUILD=0
BUILD_ONLY=0
DRY_RUN=0
ASSUME_YES=0
SERVICES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?--tag butuh nilai}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    all) SERVICES=("${ALL_SERVICES[@]}"); shift ;;
    -*) die "opsi tidak dikenal: $1" ;;
    *)
      [[ " ${ALL_SERVICES[*]} " == *" $1 "* ]] || die "service tidak dikenal: $1 (pilihan: ${ALL_SERVICES[*]})"
      [[ " ${SERVICES[*]-} " == *" $1 "* ]] || SERVICES+=("$1")
      shift ;;
  esac
done

[[ ${#SERVICES[@]} -gt 0 ]] || { usage; exit 1; }
[[ $SKIP_BUILD -eq 1 && $BUILD_ONLY -eq 1 ]] && die "--skip-build dan --build-only tidak bisa dipakai bersamaan"
[[ $SKIP_BUILD -eq 1 && -z "$TAG" ]] && die "--skip-build butuh --tag"
[[ -f "$VALUES" ]] || die "file values tidak ada: $VALUES"

for tool in git helm kubectl; do
  command -v "$tool" >/dev/null || die "$tool tidak ditemukan di PATH"
done
if [[ $SKIP_BUILD -eq 0 ]]; then
  for tool in docker gcloud; do
    command -v "$tool" >/dev/null || die "$tool tidak ditemukan di PATH"
  done
fi

cd "$ROOT"

# Deploy hanya dari main: yang jalan di cluster selalu commit yang sudah ada di origin/main, bukan
# branch fitur atau perubahan lokal. --build-only dan --dry-run tidak mengubah cluster, jadi boleh
# dari branch mana pun. Hotfix pun di-commit dan di-push ke main dulu, baru di-deploy.
if [[ $BUILD_ONLY -eq 0 && $DRY_RUN -eq 0 ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  [[ "$BRANCH" == main ]] \
    || die "deploy hanya boleh dari branch main (sekarang: $BRANCH). Merge dulu ke main, lalu: git checkout main && git pull --ff-only"
  git fetch --quiet origin main || die "git fetch origin main gagal"
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] \
    || die "main lokal ($(git rev-parse --short HEAD)) tidak sama dengan origin/main ($(git rev-parse --short origin/main)): pull commit yang belum ada di lokal, push commit lokal yang belum ada di origin."
  [[ -z "$(git status --porcelain -- libs services db deploy/helm)" ]] \
    || die "ada perubahan yang belum di-commit di libs/, services/, db/, atau deploy/helm/: yang di-deploy harus persis isi origin/main (--allow-dirty hanya bersama --build-only)."
  if [[ -n "$TAG" ]]; then
    TAG_COMMIT="$(git rev-parse --verify --quiet "${TAG%%-*}^{commit}" || true)"
    [[ "$TAG" != *-dirty* && -n "$TAG_COMMIT" ]] && git merge-base --is-ancestor "$TAG_COMMIT" origin/main \
      || die "tag $TAG bukan SHA commit di origin/main; hanya image dari main yang boleh di-deploy."
    [[ $SKIP_BUILD -eq 1 || "$TAG_COMMIT" == "$(git rev-parse HEAD)" ]] \
      || die "build baru selalu dari HEAD ($(git rev-parse --short HEAD)); tag commit lain hanya bersama --skip-build."
  fi
fi

if [[ -z "$TAG" ]]; then
  TAG="$(git rev-parse --short HEAD)"
  if [[ -n "$(git status --porcelain -- libs services db)" ]]; then
    [[ $ALLOW_DIRTY -eq 1 ]] || die "ada perubahan di libs/, services/, atau db/ yang belum di-commit: tag image harus SHA dari commit bersih supaya bisa dilacak dan dibangun ulang. Commit dulu, atau pakai --allow-dirty --build-only untuk uji coba."
    TAG="$TAG-dirty-$(date +%Y%m%d%H%M%S)"
  fi
fi

image_ref() { echo "$REGISTRY/$IMAGE_PREFIX-$1:$TAG"; }

if [[ $BUILD_ONLY -eq 0 ]]; then
  CONTEXT="$(kubectl config current-context 2>/dev/null || true)"
  [[ "$CONTEXT" == "$EXPECTED_CONTEXT" ]] \
    || die "kubectl context sekarang '$CONTEXT', diharapkan '$EXPECTED_CONTEXT'. Ganti context atau set EXPECTED_CONTEXT."
  helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1 \
    || die "release $RELEASE tidak ditemukan di namespace $NAMESPACE; install pertama kali lihat deploy/helm/README.md"
fi

log "Rencana"
echo "  service   : ${SERVICES[*]}"
echo "  tag       : $TAG"
echo "  registry  : $REGISTRY"
echo "  values    : ${VALUES#"$ROOT"/}"
if [[ $BUILD_ONLY -eq 0 ]]; then
  echo "  cluster   : $CONTEXT"
  echo "  release   : $RELEASE (namespace $NAMESPACE)"
fi
[[ "$TAG" == *-dirty-* ]] && echo "  PERINGATAN: image dibangun dari working tree yang belum di-commit (--allow-dirty); jangan dipakai di produksi"

if [[ $ASSUME_YES -eq 0 && $DRY_RUN -eq 0 ]]; then
  [[ -t 0 ]] || die "tidak ada terminal untuk konfirmasi; pakai -y"
  read -r -p "Lanjut? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "dibatalkan"
fi

if [[ $SKIP_BUILD -eq 0 && $DRY_RUN -eq 0 ]]; then
  if [[ " ${SERVICES[*]} " == *" guardrails "* && ! -f services/guardrails/weights/best_model.pt ]]; then
    die "services/guardrails/weights/best_model.pt tidak ada; jalankan 'make weights' dulu"
  fi

  for svc in "${SERVICES[@]}"; do
    log "Build $svc -> $(image_ref "$svc")"
    docker build --platform linux/amd64 -f "services/$svc/Dockerfile" -t "$(image_ref "$svc")" .
  done

  DOCKER_TMP_CONFIG="$(mktemp -d)"
  trap 'rm -rf "$DOCKER_TMP_CONFIG"' EXIT
  log "Login ke ${REGISTRY%%/*}"
  gcloud auth print-access-token \
    | docker --config "$DOCKER_TMP_CONFIG" login -u oauth2accesstoken --password-stdin "https://${REGISTRY%%/*}" >/dev/null

  for svc in "${SERVICES[@]}"; do
    log "Push $svc"
    docker --config "$DOCKER_TMP_CONFIG" push "$(image_ref "$svc")"
  done
fi

if [[ $BUILD_ONLY -eq 1 ]]; then
  log "Selesai (build-only). Deploy nanti dengan: deploy/helm/deploy.sh --skip-build --tag $TAG ${SERVICES[*]}"
  exit 0
fi

HELM_ARGS=(upgrade "$RELEASE" "$CHART" -n "$NAMESPACE" --reset-then-reuse-values -f "$VALUES" --timeout "$TIMEOUT"
  --history-max "$HISTORY_MAX")
for svc in "${SERVICES[@]}"; do
  HELM_ARGS+=(--set "services.$svc.image.tag=$TAG")
done

if [[ $DRY_RUN -eq 1 ]]; then
  log "Diff manifest (dry-run, tidak ada yang diubah)"
  current="$(mktemp)"; planned="$(mktemp)"
  helm get manifest "$RELEASE" -n "$NAMESPACE" > "$current"
  helm "${HELM_ARGS[@]}" --dry-run=server | sed -n '/^MANIFEST:/,/^NOTES:/p' | sed '1d;/^NOTES:/d' > "$planned"
  diff -u "$current" "$planned" && echo "  tidak ada perubahan"
  rm -f "$current" "$planned"
  exit 0
fi

if [[ "$(helm version --template '{{.Version}}')" == v3.* ]]; then
  HELM_ARGS+=(--wait --atomic)
else
  HELM_ARGS+=(--wait=watcher --rollback-on-failure)
fi

log "helm upgrade"
helm "${HELM_ARGS[@]}"

log "Status"
for svc in "${SERVICES[@]}"; do
  kubectl -n "$NAMESPACE" rollout status "deploy/$RELEASE-$svc" --timeout "$TIMEOUT"
done
kubectl -n "$NAMESPACE" get pods -l "app.kubernetes.io/instance=$RELEASE"
kubectl -n "$NAMESPACE" get deploy -l "app.kubernetes.io/instance=$RELEASE" \
  -o jsonpath='{range .items[*]}{"  "}{.metadata.labels.app\.kubernetes\.io/component}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
helm history "$RELEASE" -n "$NAMESPACE" --max 3
