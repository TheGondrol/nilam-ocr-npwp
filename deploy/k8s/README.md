# deploy/k8s — manifest GKE (Kustomize)

Empat Deployment + Service di satu namespace per lingkungan. Konfigurasi dari ConfigMap,
rahasia dari GCP Secret Manager lewat External Secrets Operator, alamat antar service lewat
nama Service (`http://structuring:8032`), jadi identik di semua lingkungan.

```
deploy/k8s/
├── base/
│   ├── common/          ServiceAccount (Workload Identity), ExternalSecret, NetworkPolicy
│   ├── guardrails/      Deployment · Service · PDB · ConfigMap (generator)
│   ├── ekstraksi/       sama
│   ├── structuring/     sama
│   └── scoring/         sama
└── overlays/
    ├── dev/             namespace nilam-ocr-dev, project edm-bribrain-dev-01, 1 replika
    ├── staging/         2 replika
    └── production/      2 replika + HPA guardrails (2..6, CPU 70%)
```

Status: ketiga overlay lolos `kubectl kustomize` dan pemeriksaan konsistensi (port, selector,
ConfigMap ber-hash, URL antar service, grace period). **Belum pernah di-apply ke cluster.**

## Yang harus diisi sebelum apply

Cari `REPLACE_ME`: `grep -rn REPLACE_ME deploy/k8s/overlays/<env>`.

| Yang diisi | Di mana | Catatan |
|---|---|---|
| Tag image | `images[].newTag` | Git SHA atau semver. Jangan `latest`: rollback jadi mustahil |
| Project Artifact Registry | `images[].newName` | `dev` sudah memakai `edm-bribrain-dev-01`; nama repo `nilam-ocr` adalah asumsi |
| `ORCHESTRATION_OUTCOME_TABLE` / `ORCHESTRATION_URL` | `configMapGenerator` (ekstraksi, structuring, scoring) | Cara hasil sampai ke Orkestrasi: tabel miliknya (default, `orchestration_extract_ocr`) dan/atau endpoint callback-nya. Salah satu wajib: tanpa keduanya pod menolak start. `ORCHESTRATION_URL` kosong = tidak ada callback sama sekali |
| `EKSTRAKSI_OCR_URL` | `configMapGenerator` ekstraksi | `dev` memakai `http://10.213.128.67:8070` (IP internal VM model); cluster dan VM harus satu VPC / di-peer |
| Namespace Orkestrasi | patch `NetworkPolicy allow-orchestration` | Tanpa ini Orkestrasi diblokir di lapis jaringan |
| GSA Workload Identity | patch `ServiceAccount` | Bind: `roles/iam.workloadIdentityUser` untuk `nilam-ocr-<env>/nilam-ocr` |

Rahasia di Secret Manager project yang sama (nama di `base/common/externalsecret.yaml`):
`nilam-ocr-api-key`, `nilam-ocr-database-url`, `nilam-ocr-orchestration-api-key`.

## Prasyarat cluster

- **External Secrets Operator** + `ClusterSecretStore` bernama `gcp-secret-manager`. Cek versi CRD;
  ESO lama memakai `external-secrets.io/v1beta1`, manifest ini `v1`.
- **GKE Dataplane V2** (atau network policy enforcement). Tanpa itu NetworkPolicy diabaikan diam-diam.
- **Workload Identity** aktif di node pool.
- **Cloud SQL (PostgreSQL)** terjangkau dari pod, dan migrasi [db/](../../db) sudah dijalankan
  (aplikasi tidak memigrasi sendiri; lihat `deploy/helm/migrate-db.sh`). Dengan Auth Proxy sebagai sidecar, host di `DATABASE_URL`
  adalah `127.0.0.1`; itu satu-satunya alamat localhost yang diizinkan pengaman konfigurasi.
- Namespace: `kubectl create namespace nilam-ocr-<env>`.

## Deploy

```bash
# 1. build + push (context = root repo, karena tiap image butuh libs/ocr_common)
TAG=$(git rev-parse --short HEAD)
REG=asia-southeast2-docker.pkg.dev/<project>/nilam-ocr
make weights                      # bobot guardrails harus ada SEBELUM build; build gagal kalau tidak
for s in guardrails ekstraksi structuring scoring; do
  docker build -f services/$s/Dockerfile -t $REG/$s:$TAG . && docker push $REG/$s:$TAG
done

# 2. lihat dulu apa yang akan diterapkan
kubectl kustomize deploy/k8s/overlays/dev | less
kubectl diff -k deploy/k8s/overlays/dev

# 3. apply
kubectl apply -k deploy/k8s/overlays/dev
kubectl -n nilam-ocr-dev rollout status deploy/guardrails deploy/ekstraksi deploy/structuring deploy/scoring
```

Pod yang `CrashLoopBackOff` tepat setelah apply hampir selalu pengaman konfigurasi yang bekerja:
`kubectl -n nilam-ocr-dev logs deploy/<nama> | grep "Value error"` menyebut variabel mana yang salah.

## Verifikasi setelah deploy

```bash
kubectl -n nilam-ocr-dev get pods,svc,hpa,externalsecret
kubectl -n nilam-ocr-dev port-forward svc/ekstraksi 8030:8030 &
curl localhost:8030/health     # backends: paddle + storage postgres
curl localhost:8030/ready      # {"status":"ready","checks":{"database":"ok"}}
```

Rantai penuh: `scripts/smoke_e2e.py` dengan `*_URL` diarahkan ke port-forward keempat service.

Rollback: `kubectl -n <ns> rollout undo deploy/<nama>`, atau apply ulang overlay dengan tag sebelumnya.

## Keputusan desain yang tertanam di manifest

- **`/health` untuk liveness, `/ready` untuk readiness.** Liveness tidak pernah menyentuh database:
  gangguan Cloud SQL sesaat membuat pod tidak menerima trafik, bukan di-restart massal.
- **`terminationGracePeriodSeconds: 45`** di tiga service pipeline = drain 30 s + preStop 5 s + cadangan.
  Default k8s (30 s) akan men-SIGKILL tepat saat job background masih diselesaikan.
- **`maxUnavailable: 0`**: pod lama baru dimatikan setelah pod baru `/ready`.
- **`GUARDRAILS_TORCH_THREADS` = `limits.cpu`.** Tanpa ini torch membuat thread sebanyak core node.
- **ConfigMap lewat generator** (nama ber-hash): mengubah konfigurasi otomatis memicu rolling update.
- **Rahasia per service**, bukan `envFrom` seluruh Secret: guardrails tidak menerima `DATABASE_URL`.
- **Pod ketat**: non-root (uid 1000), root filesystem read-only, tanpa capability, seccomp
  RuntimeDefault, `/tmp` emptyDir, `HOME=/tmp`. Sudah diuji dengan `docker run --read-only
  --cap-drop ALL --user 1000` pada image guardrails: inference jalan, nol error filesystem, ~300 MiB.
- **PDB `maxUnavailable: 1`**, bukan `minAvailable: 1`: yang terakhir menahan upgrade node saat replika 1.
- **HPA hanya guardrails** (satu-satunya yang CPU-bound), dan `replicas` dihapus dari Deployment-nya
  di production supaya `kubectl apply` tidak me-reset jumlah pod.

## Batas yang perlu diketahui

- Angka `resources` adalah titik awal dari pengukuran di laptop (guardrails ~300 MiB, lainnya jauh
  di bawah 256 MiB), bukan hasil load test di GKE. Sesuaikan setelah ada metrik nyata.
- Job pipeline hidup di memori pod. Shutdown rapi men-drain-nya; pod yang di-evict, OOM, atau node
  yang di-preempt kehilangan job yang sedang jalan (baris `jobs` tertinggal `PROCESSING`, tanpa
  callback). Lihat "Keterbatasan & Langkah Berikutnya" di [README.md](../../README.md).
- Satu `DATABASE_URL` (satu user) untuk tiga service. Lebih baik satu user per service dengan hak
  hanya atas schema-nya; itu perubahan di Secret + tiga entri ExternalSecret.
- Tidak ada Ingress: keempat service hanya dipanggil dari dalam cluster (Orkestrasi).
- Image guardrails 1,7 GB: pod baru butuh waktu tarik image saat scale-up. Pertimbangkan image
  streaming GKE, atau backend `remote` kalau service model guardrails sudah tersedia.
