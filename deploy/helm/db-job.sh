#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAMESPACE="${NAMESPACE:-nilam-ocr-npwp}"
SECRET="${SECRET:-nilam-ocr-npwp-secrets}"
SOURCE_KEY="${SOURCE_KEY:-DATABASE_URL}"
TARGET_KEY="${TARGET_KEY:-DATABASE_URL_CLOUDSQL}"
ALEMBIC_KEY="${ALEMBIC_KEY:-DATABASE_URL}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-nilam-ocr-npwp}"
REGISTRY="${REGISTRY:-asia-southeast2-docker.pkg.dev/common-cicd-dev-01/gc-bribrain-dev-gar-temp-01}"
IMAGE_PREFIX="${IMAGE_PREFIX:-ms-bribrain-nilam-ocr-npwp}"
EXPECTED_CONTEXT="${EXPECTED_CONTEXT:-gke_ddb-kubecluster-dev-01_asia-southeast2_gc-ddb-dev-gke-cluster-01}"
# Node dev yang rusak (lihat affinity di values-ddb-dev.yaml): Job ini juga menghindarinya.
AVOID_NODE="${AVOID_NODE:-gke-gc-ddb-dev-gke-c-gc-dgv-dev-ndp-c-42b3b8a0-l5uf}"
TIMEOUT="${TIMEOUT:-30m}"

usage() {
  cat <<USAGE
Pemakaian:
  deploy/helm/db-job.sh [-y] copy [--check | --replace]
  deploy/helm/db-job.sh [-y] alembic <perintah alembic...>      # mis. upgrade head, current

Menjalankan pekerjaan database sebagai Job di namespace $NAMESPACE, dengan image dari db/Dockerfile
(commit bersih, tag = SHA, di-push ke $REGISTRY). Job diperlukan karena Cloud SQL hanya punya
private IP: laptop tidak bisa menjangkaunya, pod bisa. URL database dibaca pod dari Secret $SECRET
dan tidak pernah tampil di layar.

copy: menyalin semua tabel repo ini (schema nilam_ocr_npwp) dari Secret key $SOURCE_KEY ke
Secret key $TARGET_KEY (db/copy_database.py). Database sumber hanya dibaca (satu snapshot
read-only); tidak ada yang diubah atau dihapus di sana.
  (tanpa opsi)  salin baris yang belum ada di tujuan; aman diulang, service boleh tetap jalan
  --check       hanya cek koneksi, versi migrasi dan jumlah baris; tidak membuat atau menyalin apa pun
  --replace     kosongkan tabel tujuan dulu lalu salin persis; untuk cutover, semua service sudah di-scale 0

alembic: menjalankan Alembic ke database Secret key $ALEMBIC_KEY (pengganti migrate-db.sh untuk
database yang tidak terjangkau dari laptop).

Job memakai service account $SERVICE_ACCOUNT, jadi login IAM ke Cloud SQL lewat Workload Identity
kalau anotasinya terpasang (lihat deploy/helm/README.md, "Pindah ke Cloud SQL").
Environment: NAMESPACE, SECRET, SOURCE_KEY, TARGET_KEY, ALEMBIC_KEY, REGISTRY, TIMEOUT.
USAGE
}

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ASSUME_YES=0
while [[ $# -gt 0 && "$1" == -* ]]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "opsi tidak dikenal: $1" ;;
  esac
done
TASK="${1:-}"
[[ $# -gt 0 ]] && shift
case "$TASK" in
  copy)
    MODE=()
    for arg in "$@"; do
      case "$arg" in
        --check|--replace) [[ ${#MODE[@]} -eq 0 ]] || die "--check dan --replace tidak bisa bersamaan"; MODE=("$arg") ;;
        -y|--yes) ASSUME_YES=1 ;;
        *) die "opsi copy tidak dikenal: $arg" ;;
      esac
    done
    COMMAND=(python db/copy_database.py ${MODE[@]+"${MODE[@]}"})
    KEYS=("$SOURCE_KEY" "$TARGET_KEY")
    ;;
  alembic)
    [[ $# -gt 0 ]] || die "perintah alembic belum diberikan, mis.: alembic upgrade head"
    COMMAND=(python -m alembic -c db/alembic.ini "$@")
    KEYS=("$ALEMBIC_KEY")
    ;;
  *) usage; exit 1 ;;
esac

for tool in git kubectl docker gcloud; do
  command -v "$tool" >/dev/null || die "$tool tidak ditemukan di PATH"
done
cd "$ROOT"

CONTEXT="$(kubectl config current-context 2>/dev/null || true)"
[[ "$CONTEXT" == "$EXPECTED_CONTEXT" ]] || die "kubectl context sekarang '$CONTEXT', diharapkan '$EXPECTED_CONTEXT'"
[[ -z "$(git status --porcelain -- libs db)" ]] \
  || die "ada perubahan yang belum di-commit di libs/ atau db/: image Job harus dari commit bersih"
for key in "${KEYS[@]}"; do
  [[ -n "$(kubectl -n "$NAMESPACE" get secret "$SECRET" -o "jsonpath={.data.$key}")" ]] \
    || die "Secret $SECRET belum punya $key (lihat deploy/helm/README.md, \"Pindah ke Cloud SQL\")"
done

TAG="$(git rev-parse --short HEAD)"
IMAGE="$REGISTRY/$IMAGE_PREFIX-migrate:$TAG"
JOB="$IMAGE_PREFIX-db-$TASK-$(date +%Y%m%d%H%M%S)"

log "Rencana"
if [[ "$TASK" == copy ]]; then
  echo "  sumber  : Secret $SECRET/$SOURCE_KEY (hanya dibaca)"
  echo "  tujuan  : Secret $SECRET/$TARGET_KEY"
  echo "  mode    : ${MODE[*]:-salin baris yang belum ada}"
else
  echo "  database: Secret $SECRET/$ALEMBIC_KEY"
fi
echo "  perintah: ${COMMAND[*]}"
echo "  image   : $IMAGE"
echo "  job     : $NAMESPACE/$JOB"
if [[ $ASSUME_YES -eq 0 ]]; then
  [[ -t 0 ]] || die "tidak ada terminal untuk konfirmasi; pakai -y"
  read -r -p "Lanjut? [y/N] " answer
  [[ "$answer" == [yY]* ]] || die "dibatalkan"
fi

log "Build + push $IMAGE"
docker build --platform linux/amd64 -q -f db/Dockerfile -t "$IMAGE" . >/dev/null
DOCKER_TMP_CONFIG="$(mktemp -d)"
trap 'rm -rf "$DOCKER_TMP_CONFIG"' EXIT
gcloud auth print-access-token \
  | docker --config "$DOCKER_TMP_CONFIG" login -u oauth2accesstoken --password-stdin "https://${REGISTRY%%/*}" >/dev/null
docker --config "$DOCKER_TMP_CONFIG" push -q "$IMAGE" >/dev/null

# The command as a YAML flow sequence, each argument double-quoted.
COMMAND_YAML="[$(printf '"%s", ' "${COMMAND[@]}" | sed 's/, $//')]"
if [[ "$TASK" == copy ]]; then
  ENV_YAML="            - name: SOURCE_DATABASE_URL
              valueFrom:
                secretKeyRef: {name: $SECRET, key: $SOURCE_KEY}
            - name: TARGET_DATABASE_URL
              valueFrom:
                secretKeyRef: {name: $SECRET, key: $TARGET_KEY}"
else
  ENV_YAML="            - name: DATABASE_URL
              valueFrom:
                secretKeyRef: {name: $SECRET, key: $ALEMBIC_KEY}"
fi

log "Job $JOB"
kubectl -n "$NAMESPACE" apply -f - <<MANIFEST
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB
  labels:
    app.kubernetes.io/name: $IMAGE_PREFIX
    app.kubernetes.io/component: db-$TASK
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app.kubernetes.io/component: db-$TASK
    spec:
      restartPolicy: Never
      serviceAccountName: $SERVICE_ACCOUNT
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000  # appuser of db/Dockerfile, as podSecurityContext in values.yaml
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values: [$AVOID_NODE]
      containers:
        - name: $TASK
          image: $IMAGE
          command: $COMMAND_YAML
          env:
$ENV_YAML
            - name: HOME
              value: /tmp
          resources:
            requests: {cpu: 100m, memory: 256Mi}
            limits: {cpu: "1", memory: 512Mi}
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
MANIFEST

log "Menunggu Job (maksimal $TIMEOUT)"
kubectl -n "$NAMESPACE" wait --for=condition=Ready pod -l "job-name=$JOB" --timeout=5m >/dev/null 2>&1 || true
kubectl -n "$NAMESPACE" logs -f "job/$JOB" || true
if kubectl -n "$NAMESPACE" wait --for=condition=complete "job/$JOB" --timeout="$TIMEOUT" >/dev/null 2>&1; then
  log "Selesai: Job $JOB berhasil"
else
  kubectl -n "$NAMESPACE" get pods -l "job-name=$JOB"
  die "Job $JOB gagal atau belum selesai; log di atas, atau: kubectl -n $NAMESPACE logs job/$JOB"
fi
