#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART="$ROOT/deploy/helm/nilam-ocr-npwp"
DEPLOY_ENV="${DEPLOY_ENV:-shared-dev}"
VALUES="$CHART/values-$DEPLOY_ENV.yaml"
RELEASE="${RELEASE:-nilam-ocr-npwp}"
# Cluster tujuan per environment. shared-dev: API server privat, jadi kubectl/helm lewat tunnel SOCKS di bastion
# (IAP SSH) dengan kubeconfig sendiri; ~/.kube/config (context ddb) tidak disentuh.
case "$DEPLOY_ENV" in
  shared-dev)
    NAMESPACE="${NAMESPACE:-nilam}"
    GKE_PROJECT="${GKE_PROJECT:-shared-kubecluster-dev-01}"
    GKE_CLUSTER="${GKE_CLUSTER:-gc-bigpool-dev-gke-cluster-01}"
    GKE_REGION="${GKE_REGION:-asia-southeast2}"
    BASTION="${BASTION:-gc-shared-dev-gce-bastion-01}"
    BASTION_ZONE="${BASTION_ZONE:-asia-southeast2-a}"
    PROXY_PORT="${PROXY_PORT:-18090}"
    export KUBECONFIG="${SHARED_KUBECONFIG:-$HOME/.kube/config-$GKE_PROJECT}"
    DEFAULT_CONTEXT="gke_${GKE_PROJECT}_${GKE_REGION}_${GKE_CLUSTER}"
    ;;
  ddb-dev)
    NAMESPACE="${NAMESPACE:-nilam-ocr-npwp}"
    BASTION=""
    DEFAULT_CONTEXT="gke_ddb-kubecluster-dev-01_asia-southeast2_gc-ddb-dev-gke-cluster-01"
    ;;
  *) printf 'ERROR: DEPLOY_ENV tidak dikenal: %s (shared-dev | ddb-dev)\n' "$DEPLOY_ENV" >&2; exit 1 ;;
esac
# Nexus BRI: satu folder per aplikasi, satu image per service (<registry>/<prefix>-<service>:<tag>).
REGISTRY="${REGISTRY:-new-nexus.gcp.bri.co.id/bribrain/dev/nilam/nilam-ocr-npwp}"
IMAGE_PREFIX="${IMAGE_PREFIX:-ms-bribrain-nilam-ocr-npwp}"
# NEXUS_USERNAME / NEXUS_PASSWORD untuk push dan untuk pull secret di cluster (file rahasia, tidak di-commit).
NEXUS_ENV="${NEXUS_ENV:-nexus.env}"
# Generik, bukan khusus NPWP: dipakai bersama semua service dokumen NILAM OCR (NPWP, KTP, KK, ...) di namespace
# yang sama, jadi namanya tanpa nama dokumen.
PULL_SECRET="${PULL_SECRET:-nilam-ocr-nexus}"
EXPECTED_CONTEXT="${EXPECTED_CONTEXT:-$DEFAULT_CONTEXT}"
TIMEOUT="${TIMEOUT:-10m}"
# Python with ocr_common's dependencies (the repo's .venv), to fetch the models from GCS before a build.
# Default: the repo's .venv (Windows or Linux layout) when it exists, so the venv need not be activated.
if [[ -z "${PY:-}" ]]; then
  for candidate in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python"; do
    [[ -x "$candidate" ]] && { PY="$candidate"; break; }
  done
  PY="${PY:-python}"
fi
# Revisi release yang disimpan Helm (satu Secret sh.helm.release.v1.<release>.vN per revisi), untuk rollback.
HISTORY_MAX="${HISTORY_MAX:-5}"
ALL_SERVICES=(orchestrator guardrails extraction structuring scoring)

usage() {
  cat <<EOF
Pemakaian: deploy/helm/deploy.sh [opsi] <service...|all>

Build image service yang disebut, push ke Nexus ($REGISTRY), lalu helm upgrade
release $RELEASE dengan tag baru hanya untuk service tersebut. Tiap service adalah
Deployment sendiri (<release atau fullnameOverride>-<service>), jadi hanya pod service itu yang diganti.

Model guardrails dan scoring diambil dulu dari GCS (scripts/fetch_weights.py, kredensial dari wif.gcs.env
atau Application Default Credentials) lalu ikut di dalam image; SHA-256-nya jadi label image
nilam.model.sha256. Gagal mengambil model = build dibatalkan.

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
  --install       izinkan instalasi pertama kali (release belum ada di namespace); butuh Secret
                  \$existingSecret (nilam-ocr-npwp-secrets) sudah dibuat di namespace itu
  --update-pull-secret
                  tulis ulang pull secret Nexus \$PULL_SECRET dari \$NEXUS_ENV (mis. password Nexus berganti);
                  tanpa opsi ini pull secret yang sudah ada dibiarkan
  --replicas N    jumlah pod tiap service yang disebut, hanya untuk deploy ini (mis. 0: objek terpasang tanpa
                  pod, tidak memakan kuota namespace). Deploy berikutnya tanpa opsi ini kembali ke values
  --dry-run       tampilkan diff manifest terhadap release yang berjalan, tanpa apply
  -y, --yes       tanpa konfirmasi
  -h, --help      tampilkan bantuan ini

Kredensial Nexus dari \$NEXUS_ENV (default nexus.env di root repo): dipakai untuk push, dan untuk pull secret
\$PULL_SECRET yang dibuat/diperbarui di namespace sebelum helm upgrade.

Environment: DEPLOY_ENV (default shared-dev -> values-shared-dev.yaml, cluster gc-bigpool-dev-gke-cluster-01
namespace nilam lewat tunnel bastion; ddb-dev -> values-ddb-dev.yaml, cluster gc-ddb-dev-gke-cluster-01
namespace nilam-ocr-npwp), RELEASE, NAMESPACE, PROXY_PORT (tunnel shared-dev, default 18090), PY (Python .venv),
REGISTRY, IMAGE_PREFIX, NEXUS_ENV, PULL_SECRET, EXPECTED_CONTEXT, TIMEOUT (default 10m), HISTORY_MAX (revisi release yang disimpan
untuk rollback, default 5).

Contoh:
  deploy/helm/deploy.sh scoring
  deploy/helm/deploy.sh extraction structuring scoring
  deploy/helm/deploy.sh --dry-run --skip-build --tag 4d02eba all
EOF
}

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

TAG=""
ALLOW_DIRTY=0
SKIP_BUILD=0
BUILD_ONLY=0
INSTALL=0
UPDATE_PULL_SECRET=0
REPLICAS=""
RELEASE_EXISTS=1
DRY_RUN=0
ASSUME_YES=0
SERVICES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?--tag butuh nilai}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --install) INSTALL=1; shift ;;
    --update-pull-secret) UPDATE_PULL_SECRET=1; shift ;;
    --replicas) REPLICAS="${2:?--replicas butuh nilai}"; [[ "$REPLICAS" =~ ^[0-9]+$ ]] || die "--replicas harus angka: $REPLICAS"; shift 2 ;;
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
if [[ -n "$BASTION" && $BUILD_ONLY -eq 0 ]]; then
  command -v gcloud >/dev/null || die "gcloud tidak ditemukan di PATH (tunnel ke $GKE_CLUSTER lewat $BASTION)"
fi
if [[ $SKIP_BUILD -eq 0 ]]; then
  command -v docker >/dev/null || die "docker tidak ditemukan di PATH"
fi

cd "$ROOT"

# origin/main terbaru. Jaringan kantor memblokir SSH ke GitHub dan rute IPv6-nya (NAT64) sering di-reset, jadi
# kalau fetch biasa gagal, dicoba lagi lewat HTTPS ke alamat IPv4 GitHub (kredensial dari `gh` kalau ada,
# selain itu credential helper git biasa). Hasilnya tetap disimpan sebagai origin/main.
fetch_origin_main() {
  git fetch --quiet origin main 2>/dev/null && return 0
  local url ip helper=()
  url="$(git remote get-url origin)"
  [[ "$url" == git@github.com:* ]] && url="https://github.com/${url#git@github.com:}"
  [[ "$url" == https://github.com/* ]] || return 1
  ip="$(nslookup github.com 2>/dev/null | sed -n '/^Name:/,$p' | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n1)"
  command -v gh >/dev/null && helper=(-c credential.helper= -c "credential.helper=!gh auth git-credential")
  echo "  fetch lewat SSH gagal; mencoba HTTPS${ip:+ via IPv4 $ip}" >&2
  local attempt
  for attempt in 1 2 3; do
    if git ${helper[@]+"${helper[@]}"} ${ip:+-c "http.curloptResolve=github.com:443:$ip"} fetch --quiet "$url" main; then
      git update-ref refs/remotes/origin/main FETCH_HEAD
      return 0
    fi
    sleep 5
  done
  return 1
}

# Deploy hanya dari main: yang jalan di cluster selalu commit yang sudah ada di origin/main, bukan
# branch fitur atau perubahan lokal. --build-only dan --dry-run tidak mengubah cluster, jadi boleh
# dari branch mana pun. Hotfix pun di-commit dan di-push ke main dulu, baru di-deploy.
if [[ $BUILD_ONLY -eq 0 && $DRY_RUN -eq 0 ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  [[ "$BRANCH" == main ]] \
    || die "deploy hanya boleh dari branch main (sekarang: $BRANCH). Merge dulu ke main, lalu: git checkout main && git pull --ff-only"
  fetch_origin_main || die "git fetch origin main gagal (SSH dan HTTPS); cek koneksi ke github.com"
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

# NEXUS_USERNAME / NEXUS_PASSWORD dari $NEXUS_ENV (KEY=VALUE), tanpa menjalankan file itu sebagai skrip.
# Hanya dibaca kalau perlu: untuk push (build) dan untuk membuat pull secret yang belum ada.
nexus_value() { sed -n "s/^$1=//p" "$NEXUS_ENV" | tail -n1 | tr -d '\r'; }
NEXUS_USERNAME=""
NEXUS_PASSWORD=""
load_nexus_credentials() {
  [[ -n "$NEXUS_USERNAME" ]] && return 0
  [[ -f "$NEXUS_ENV" ]] || die "kredensial Nexus tidak ada: $NEXUS_ENV (NEXUS_USERNAME, NEXUS_PASSWORD)"
  NEXUS_USERNAME="$(nexus_value NEXUS_USERNAME)"
  NEXUS_PASSWORD="$(nexus_value NEXUS_PASSWORD)"
  [[ -n "$NEXUS_USERNAME" && -n "$NEXUS_PASSWORD" ]] || die "NEXUS_USERNAME / NEXUS_PASSWORD kosong di $NEXUS_ENV"
}
# Build: gagal di awal, bukan setelah build selesai.
[[ $SKIP_BUILD -eq 0 && $DRY_RUN -eq 0 ]] && load_nexus_credentials

# --- Tunnel (shared-dev) ----------------------------------------------------------------------------------------
# gcloud compute ssh -D: SOCKS5 di 127.0.0.1:$PROXY_PORT, keluar dari bastion ke API server privat. Di Windows
# gcloud menjalankan putty.exe sebagai child yang tidak ikut mati bersama gcloud, jadi yang dihentikan seluruh
# process tree-nya (taskkill /T).
TUNNEL_PID=""
TUNNEL_LOG=""
DOCKER_TMP_CONFIG=""
cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    if [[ -r "/proc/$TUNNEL_PID/winpid" ]]; then
      taskkill //F //T //PID "$(cat "/proc/$TUNNEL_PID/winpid")" >/dev/null 2>&1 || true
    else
      pkill -P "$TUNNEL_PID" 2>/dev/null || true
      kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    TUNNEL_PID=""
  fi
  [[ -n "$TUNNEL_LOG" ]] && rm -f "$TUNNEL_LOG"
  [[ -n "$DOCKER_TMP_CONFIG" ]] && rm -rf "$DOCKER_TMP_CONFIG"
  return 0
}
trap cleanup EXIT

port_open() { (echo > "/dev/tcp/127.0.0.1/$PROXY_PORT") 2>/dev/null; }

open_tunnel() {
  port_open && die "port $PROXY_PORT sudah dipakai (tunnel lama yang masih hidup? di Windows: proses putty.exe). Hentikan dulu atau set PROXY_PORT."
  log "Tunnel SOCKS lewat $BASTION (IAP) di 127.0.0.1:$PROXY_PORT"
  TUNNEL_LOG="$(mktemp)"
  # gcloud kadang timeout ke compute.googleapis.com (jaringan kantor: alamat IPv6 tanpa rute), jadi dicoba 3 kali.
  local attempt
  for attempt in 1 2 3; do
    gcloud compute ssh "$BASTION" --zone "$BASTION_ZONE" --project "$GKE_PROJECT" --tunnel-through-iap \
      --strict-host-key-checking=no -- -N -D "127.0.0.1:$PROXY_PORT" >"$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    for _ in $(seq 1 45); do
      port_open && break
      kill -0 "$TUNNEL_PID" 2>/dev/null || break
      sleep 2
    done
    port_open && break
    echo "  tunnel gagal (percobaan $attempt): $(grep -m1 -E 'ERROR|FATAL' "$TUNNEL_LOG" | cut -c1-200)" >&2
    cleanup
    TUNNEL_LOG="$(mktemp)"
  done
  port_open || { cat "$TUNNEL_LOG" >&2; die "tunnel ke $BASTION gagal 3 kali (perlu izin IAP + OS Login ke VM itu)"; }

  log "Kubeconfig $GKE_CLUSTER -> $KUBECONFIG"
  gcloud container clusters get-credentials "$GKE_CLUSTER" --region "$GKE_REGION" --project "$GKE_PROJECT" >/dev/null 2>&1 \
    || die "get-credentials $GKE_CLUSTER gagal"
  kubectl config set-cluster "$(kubectl config current-context)" --proxy-url="socks5://127.0.0.1:$PROXY_PORT" >/dev/null
  for attempt in 1 2 3; do
    # Bukan /readyz: API server GKE ini menjawabnya NotFound walau tunnel jalan.
    kubectl get namespace "$NAMESPACE" --request-timeout=30s >/dev/null 2>&1 && return 0
    echo "  API server belum menjawab lewat tunnel (percobaan $attempt)" >&2
    sleep 3
  done
  die "API server $GKE_CLUSTER tidak menjawab lewat tunnel"
}

if [[ $BUILD_ONLY -eq 0 ]]; then
  [[ -n "$BASTION" ]] && open_tunnel
  CONTEXT="$(kubectl config current-context 2>/dev/null || true)"
  [[ "$CONTEXT" == "$EXPECTED_CONTEXT" ]] \
    || die "kubectl context sekarang '$CONTEXT', diharapkan '$EXPECTED_CONTEXT'. Ganti context atau set EXPECTED_CONTEXT."
  kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || die "namespace $NAMESPACE tidak ada di $CONTEXT (atau tidak boleh dibaca)"
  if helm status "$RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
    RELEASE_EXISTS=1
  else
    RELEASE_EXISTS=0
    [[ $INSTALL -eq 1 ]] \
      || die "release $RELEASE belum ada di namespace $NAMESPACE; instalasi pertama: tambahkan --install (lihat deploy/helm/README.md)"
  fi
  # Secret aplikasi (API key, DATABASE_URL, callback key, ...) tidak dibuat chart; harus sudah ada.
  APP_SECRET="$(sed -n 's/^existingSecret: *//p' "$VALUES" | tail -n1)"
  [[ -n "$APP_SECRET" ]] || APP_SECRET="$(sed -n 's/^existingSecret: *//p' "$CHART/values.yaml" | tail -n1)"
  kubectl -n "$NAMESPACE" get secret "$APP_SECRET" >/dev/null 2>&1 \
    || die "Secret $APP_SECRET belum ada di namespace $NAMESPACE (API_KEY, DATABASE_URL, ORCHESTRATION_*; lihat deploy/helm/README.md)"
fi

log "Rencana"
echo "  service   : ${SERVICES[*]}"
echo "  tag       : $TAG"
echo "  registry  : $REGISTRY"
echo "  values    : ${VALUES#"$ROOT"/}"
if [[ $BUILD_ONLY -eq 0 ]]; then
  echo "  cluster   : $CONTEXT"
  echo "  release   : $RELEASE (namespace $NAMESPACE)$([[ $RELEASE_EXISTS -eq 0 ]] && echo ', INSTALASI PERTAMA')"
fi
[[ "$TAG" == *-dirty-* ]] && echo "  PERINGATAN: image dibangun dari working tree yang belum di-commit (--allow-dirty); jangan dipakai di produksi"

if [[ $ASSUME_YES -eq 0 && $DRY_RUN -eq 0 ]]; then
  [[ -t 0 ]] || die "tidak ada terminal untuk konfirmasi; pakai -y"
  read -r -p "Lanjut? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "dibatalkan"
fi

if [[ $SKIP_BUILD -eq 0 && $DRY_RUN -eq 0 ]]; then
  # The models are in GCS, one fixed path each, overwritten by a new model. Every guardrails / scoring build takes
  # the current one and bakes it in, so an image tag is also a model version: rolling back an image rolls back its
  # model. Its SHA-256 becomes the image label nilam.model.sha256 (docker inspect shows which model is inside).
  declare -A MODEL_SHA=()
  MODEL_SERVICES=()
  for svc in "${SERVICES[@]}"; do
    [[ "$svc" == guardrails || "$svc" == scoring ]] && MODEL_SERVICES+=("$svc")
  done
  if [[ ${#MODEL_SERVICES[@]} -gt 0 ]]; then
    log "Model dari GCS: ${MODEL_SERVICES[*]}"
    "$PY" -c "import google.auth" 2>/dev/null \
      || die "$PY tidak punya google-auth: aktifkan .venv (make dev) atau set PY=<python .venv>"
    fetched="$("$PY" scripts/fetch_weights.py "${MODEL_SERVICES[@]}")" \
      || die "model tidak bisa diambil dari GCS (lihat pesan di atas); build dibatalkan"
    while read -r name _ sha; do
      MODEL_SHA[$name]="$sha"
      echo "  $name: sha256 $sha"
    done <<< "$fetched"
  fi

  for svc in "${SERVICES[@]}"; do
    log "Build $svc -> $(image_ref "$svc")"
    labels=()
    [[ -n "${MODEL_SHA[$svc]:-}" ]] && labels=(--label "nilam.model.sha256=${MODEL_SHA[$svc]}")
    docker build --platform linux/amd64 ${labels[@]+"${labels[@]}"} -f "services/$svc/Dockerfile" \
      -t "$(image_ref "$svc")" .
  done

  DOCKER_TMP_CONFIG="$(mktemp -d)"  # dihapus oleh cleanup (trap EXIT)
  log "Login ke ${REGISTRY%%/*}"
  printf '%s' "$NEXUS_PASSWORD" \
    | docker --config "$DOCKER_TMP_CONFIG" login -u "$NEXUS_USERNAME" --password-stdin "${REGISTRY%%/*}" >/dev/null

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
if [[ $RELEASE_EXISTS -eq 0 ]]; then
  # Instalasi pertama: service yang tidak disebut tidak punya tag apa pun di cluster, jadi semuanya ikut.
  [[ ${#SERVICES[@]} -eq ${#ALL_SERVICES[@]} ]] || die "instalasi pertama harus semua service: deploy.sh --install ... all"
  HELM_ARGS+=(--install)
fi
for svc in "${SERVICES[@]}"; do
  HELM_ARGS+=(--set "services.$svc.image.tag=$TAG")
  # replicaCount hanya dipakai Deployment tanpa HPA; HPA (values.yaml produksi) tetap memegang jumlah pod.
  if [[ -n "$REPLICAS" ]]; then
    HELM_ARGS+=(--set "services.$svc.replicaCount=$REPLICAS")
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  log "Diff manifest (dry-run, tidak ada yang diubah)"
  current="$(mktemp)"; planned="$(mktemp)"
  [[ $RELEASE_EXISTS -eq 1 ]] && helm get manifest "$RELEASE" -n "$NAMESPACE" > "$current"
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

# Pull secret untuk Nexus (chart: imagePullSecrets). Dipakai bersama service dokumen lain di namespace ini, jadi
# yang sudah ada dibiarkan; hanya dibuat kalau belum ada. Password Nexus berganti: --update-pull-secret.
if kubectl -n "$NAMESPACE" get secret "$PULL_SECRET" >/dev/null 2>&1 && [[ $UPDATE_PULL_SECRET -eq 0 ]]; then
  log "Pull secret $PULL_SECRET sudah ada, dipakai apa adanya"
else
  load_nexus_credentials
  log "Pull secret $PULL_SECRET (${REGISTRY%%/*}): $([[ $UPDATE_PULL_SECRET -eq 1 ]] && echo diperbarui || echo dibuat)"
  kubectl -n "$NAMESPACE" create secret docker-registry "$PULL_SECRET" --docker-server="${REGISTRY%%/*}" \
    --docker-username="$NEXUS_USERNAME" --docker-password="$NEXUS_PASSWORD" --dry-run=client -o yaml \
    | kubectl apply -f - >/dev/null
fi

log "helm upgrade"
helm "${HELM_ARGS[@]}"

log "Status"
# Nama objek = <fullnameOverride atau release>-<service> (shared-dev: ms-bribrain-nilam-ocr-npwp-<service>).
OBJECT_PREFIX="$(sed -n 's/^fullnameOverride: *//p' "$VALUES" | tail -n1 | tr -d '"'"'"'\r')"
OBJECT_PREFIX="${OBJECT_PREFIX:-$RELEASE}"
for svc in "${SERVICES[@]}"; do
  kubectl -n "$NAMESPACE" rollout status "deploy/$OBJECT_PREFIX-$svc" --timeout "$TIMEOUT"
done
kubectl -n "$NAMESPACE" get pods -l "app.kubernetes.io/instance=$RELEASE"
kubectl -n "$NAMESPACE" get deploy -l "app.kubernetes.io/instance=$RELEASE" \
  -o jsonpath='{range .items[*]}{"  "}{.metadata.labels.app\.kubernetes\.io/component}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
helm history "$RELEASE" -n "$NAMESPACE" --max 3
