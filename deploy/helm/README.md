# Helm chart `nilam-ocr-npwp`

Satu Deployment, satu pod, empat container: `guardrails` (8031), `ekstraksi` (8030), `structuring` (8032), `scoring` (8033). Satu Service `nilam-ocr-npwp` membuka keempat port itu. Pintu masuk pipeline adalah `ekstraksi` di port 8030.

Chart ini berdampingan dengan manifest Kustomize di [../k8s](../k8s), yang menjalankan tiap service sebagai Deployment terpisah.

## Yang sudah diverifikasi di `gc-ddb-dev-gke-cluster-01`

Dari pod di namespace `nilam-ocr-npwp`:

| Tujuan | Alamat | Hasil |
|---|---|---|
| Image di Artifact Registry | `common-cicd-dev-01/gc-bribrain-dev-gar-temp-01` | bisa ditarik tanpa `imagePullSecrets` |
| PostgreSQL (dalam cluster) | `postgres.ocr-dev.svc.cluster.local:5432` | terjangkau |
| PostgreSQL (LoadBalancer yang sama) | `34.50.114.49:5432` | terjangkau |
| Orkestrasi | `ocr-orchestration.ocr-dev.svc.cluster.local:80` | terjangkau |
| PaddleOCR | `10.213.128.67:8070` | terjangkau |

## Konfigurasi dinamis

Semua environment variable non-rahasia berasal dari `values`. Chart merender satu ConfigMap per service, dan hash ConfigMap itu dipasang sebagai annotation pod, jadi `helm upgrade` dengan nilai baru otomatis me-restart pod.

| Value | Isi |
|---|---|
| `environment` | `ENVIRONMENT` untuk semua service (`dev`, `staging`, `production`) |
| `image.registry`, `image.tag` | registry dan tag bersama; `services.<nama>.image.tag` menimpa per service |
| `orchestration.url` | wajib; `ekstraksi`, `structuring`, `scoring` menolak start tanpa ini |
| `commonEnv` | env var untuk keempat service |
| `services.<nama>.env` | env var per service; menimpa `commonEnv` dan nilai bawaan chart |
| `services.<nama>.upstreams` | service lain yang dipanggil; chart mengisi `<NAMA>_SERVICE_URL` otomatis |
| `services.<nama>.resources` | request dan limit per container |
| `services.<nama>.enabled` | matikan satu container |
| `existingSecret`, `secretKeys` | nama Secret dan nama key di dalamnya |

Tulis angka sebagai string (`"0.8"`, `"5242880"`), karena Helm mengubah angka besar menjadi notasi ilmiah.

URL antar-service memakai nama Service (`http://nilam-ocr-npwp:8032`), bukan `localhost`. Di luar `ENVIRONMENT=local`, service menolak `STRUCTURING_SERVICE_URL` dan `SCORING_SERVICE_URL` yang menunjuk ke localhost.

## Secret

Chart tidak membuat Secret. Buat sendiri sebelum atau sesudah install; pod menunggu sampai Secret ada.

```powershell
kubectl -n nilam-ocr-npwp create secret generic nilam-ocr-npwp-secrets `
  --from-literal=API_KEY='<api-key>' `
  --from-literal=DATABASE_URL='postgresql+asyncpg://<user>:<password>@postgres.ocr-dev.svc.cluster.local:5432/bribrain_ocr_nilam' `
  --from-literal=ORCHESTRATION_API_KEY='<opsional>'
```

`DATABASE_URL` harus berformat SQLAlchemy (`postgresql+asyncpg://`), bukan JDBC. Karakter khusus di password perlu di-URL-encode (`@` menjadi `%40`). `ORCHESTRATION_API_KEY` boleh dihilangkan.

Setelah mengubah Secret, restart pod: `kubectl -n nilam-ocr-npwp rollout restart deploy/nilam-ocr-npwp`.

## Skema database

Service tidak membuat tabel sendiri; tabel dipasang lewat migrasi Alembic ([db/](../../db)).
Dari laptop:

```bash
DB_HOST=<alamat postgres> ./migrate-db.sh          # upgrade head
DB_HOST=<alamat postgres> ./migrate-db.sh current  # revisi yang terpasang sekarang
```

Script mengambil `DATABASE_URL` dari Secret release, mengganti host-nya dengan `DB_HOST`,
lalu menjalankan Alembic di dalam image `db/Dockerfile`. Database yang tabelnya sudah
dipasang manual sebelum ada migrasi aman dijalankan: revisi baseline memakai
`CREATE TABLE IF NOT EXISTS`.

## Deploy perubahan kode

Install pertama tetap memakai perintah di bagian berikutnya. Setelah release ada, perubahan kode dideploy dengan [deploy.sh](deploy.sh) dari Git Bash, WSL, Linux atau macOS:

```bash
deploy/helm/deploy.sh --dry-run --skip-build --tag <tag> scoring
deploy/helm/deploy.sh scoring
deploy/helm/deploy.sh ekstraksi structuring scoring
deploy/helm/deploy.sh all
```

Script membangun image service yang disebut, mendorongnya ke Artifact Registry dengan tag `git rev-parse --short HEAD` (ditambah `-dirty-<waktu>` kalau `libs/` atau `services/` punya perubahan yang belum di-commit), lalu menjalankan `helm upgrade --reset-then-reuse-values -f values-ddb-dev.yaml --set services.<nama>.image.tag=<tag>`. Service lain tetap di tag lamanya. Upgrade menunggu pod siap dan otomatis rollback kalau gagal.

Keempat container ada dalam satu pod, jadi deploy satu service pun membuat pod baru berisi keempatnya. Pod lama baru berhenti setelah pod baru siap.

Perubahan `libs/ocr_common` masuk ke semua image; deploy `all`. Perubahan tabel dijalankan dulu dengan [migrate-db.sh](migrate-db.sh) sebelum deploy image yang membutuhkannya.

## Install dan upgrade

```powershell
helm upgrade --install nilam-ocr-npwp deploy/helm/nilam-ocr-npwp `
  -n nilam-ocr-npwp --create-namespace -f deploy/helm/nilam-ocr-npwp/values-ddb-dev.yaml
```

Mengganti tag image atau satu env var tanpa menyentuh file:

```powershell
helm upgrade nilam-ocr-npwp deploy/helm/nilam-ocr-npwp -n nilam-ocr-npwp --reuse-values `
  --set image.tag=<sha> --set services.scoring.env.SCORING_APPROVE_THRESHOLD="0.85"
```

Rollback: `helm -n nilam-ocr-npwp rollback nilam-ocr-npwp`.

## Akses cluster

`gcloud container clusters get-credentials gc-ddb-dev-gke-cluster-01 --project ddb-kubecluster-dev-01 --location asia-southeast2` butuh `gke-gcloud-auth-plugin`, yang dipasang dengan `gcloud components install gke-gcloud-auth-plugin` dari terminal Administrator.

## Cek cepat

```powershell
kubectl -n nilam-ocr-npwp get pods
kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp 8030:8030
curl http://localhost:8030/ready
```
