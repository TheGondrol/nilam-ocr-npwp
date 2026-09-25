# Helm chart `nilam-ocr-npwp`

Satu Deployment, Service, ConfigMap, PodDisruptionBudget, dan NetworkPolicy per service:
`orchestrator` (8034), `guardrails` (8031), `extraction` (8030), `structuring` (8032), `scoring`
(8033). Nama objeknya `<release>-<service>`, jadi di cluster dev: `nilam-ocr-npwp-orchestrator`, dst.
Tiap service bisa di-scale dan di-restart sendiri; guardrails (torch, CPU-bound) punya HPA opsional, dan
kalau ia kehabisan memori, tahap lain tidak ikut jatuh.

Service `nilam-ocr-npwp` (tanpa akhiran) adalah pintu masuk yang dipublikasikan ke Orkestrasi pusat
([integration.md](../../integration.md)): ia membuka port semua service ber-`entrypoint`, yaitu hanya
orchestrator (8034). Selector-nya mencakup semua pod release, tetapi port itu memakai `targetPort`
bernama (`orchestrator`), dan Kubernetes hanya memasukkan pod yang punya nama port itu ke endpoint
port tersebut. Karena nama service dipakai sebagai nama port, nama service maksimal 15 karakter, huruf
kecil/angka, tanpa `-`. URL antar service di dalam release memakai Service per komponen
(`http://nilam-ocr-npwp-structuring:8032`).

Ini satu-satunya jalur deploy; manifest Kustomize yang dulu ada di `deploy/k8s` sudah dihapus
dan polanya (Deployment per service, NetworkPolicy, PDB, HPA, ExternalSecret) dibawa ke sini.

## Yang sudah diverifikasi di `gc-ddb-dev-gke-cluster-01`

Dari pod di namespace `nilam-ocr-npwp`:

| Tujuan | Alamat | Hasil |
|---|---|---|
| Image di Artifact Registry | `common-cicd-dev-01/gc-bribrain-dev-gar-temp-01` | bisa ditarik tanpa `imagePullSecrets` |
| PostgreSQL (dalam cluster) | `postgres.ocr-dev.svc.cluster.local:5432` | terjangkau |
| PostgreSQL (LoadBalancer yang sama) | `34.50.114.49:5432` | terjangkau |
| Orkestrasi | `ocr-orchestration.ocr-dev.svc.cluster.local:80` | terjangkau |
| PaddleOCR | `10.213.128.67:8070` | terjangkau |

Cluster dev memakai `LEGACY_DATAPATH` tanpa network policy enforcement: NetworkPolicy chart ini
terpasang di sana tetapi belum ditegakkan. Di cluster dengan Dataplane V2 (atau Calico) ia
langsung berlaku, jadi `networkPolicy.clientNamespaces` harus terisi sebelum Orkestrasi memanggil.

## Objek yang dirender

| Objek | Per service | Value |
|---|---|---|
| Deployment `<release>-<svc>` | ya | `services.<svc>.replicaCount`, `resources`, `terminationGracePeriodSeconds` |
| Service `<release>-<svc>` | ya | `service.type`, `service.annotations` |
| Service `<release>` (pintu masuk) | tidak; port dari service ber-`entrypoint` | `service.entrypoint.enabled` |
| ConfigMap `<release>-<svc>` | ya | `environment`, `orchestration`, `commonEnv`, `services.<svc>.env`, `upstreams` |
| PodDisruptionBudget | ya | `podDisruptionBudget.{enabled,maxUnavailable}` |
| HorizontalPodAutoscaler | hanya yang `autoscaling.enabled` | `services.<svc>.autoscaling.{minReplicas,maxReplicas,targetCPUUtilizationPercentage}` |
| NetworkPolicy | default-deny + satu per service | `networkPolicy.{enabled,clientNamespaces}` |
| ServiceAccount | satu untuk semua | `serviceAccount.{create,name,annotations}` |
| ExternalSecret | opsional, mati | `externalSecret.*` |

Aturan NetworkPolicy dihitung dari `services.<svc>.upstreams`: guardrails dan extraction hanya
menerima dari orchestrator, `structuring` dari extraction dan orchestrator, `scoring` dari structuring
dan orchestrator (orchestrator membaca status tiap tahap). Service ber-`entrypoint` (hanya
orchestrator) menerima dari semua pod release dan dari namespace di `networkPolicy.clientNamespaces`.
Egress belum dibatasi (utang teknis di README utama).

## Konfigurasi dinamis

Semua environment variable non-rahasia berasal dari `values`. Chart merender satu ConfigMap per
service, dan hash isinya dipasang sebagai annotation pod, jadi `helm upgrade` dengan nilai baru
hanya me-restart pod service yang konfigurasinya berubah.

| Value | Isi |
|---|---|
| `environment` | `ENVIRONMENT` untuk semua service (`dev`, `staging`, `production`) |
| `image.registry`, `image.tag` | registry dan tag bersama; `services.<nama>.image.tag` menimpa per service |
| `orchestration.url` | callback ke Orkestrasi; kosong = hasil lewat `ORCHESTRATION_OUTCOME_TABLE` (salah satu wajib) |
| `commonEnv` | env var untuk kelima service (yang tidak dikenal sebuah service diabaikan) |
| `services.<nama>.env` | env var per service; menimpa `commonEnv` dan nilai bawaan chart |
| `services.<nama>.upstreams` | service lain yang dipanggil; chart mengisi `<NAMA>_SERVICE_URL` dan NetworkPolicy |
| `services.<nama>.entrypoint` | dipanggil dari luar release: ikut Service pintu masuk dan menerima `clientNamespaces` |
| `services.<nama>.pipeline` | tahap async: menerima `DATABASE_URL` dan konfigurasi Orkestrasi |
| `services.<nama>.replicaCount` | jumlah pod; diabaikan kalau `autoscaling.enabled` |
| `services.<nama>.resources` | request dan limit per container |
| `services.<nama>.enabled` | matikan satu service beserta semua objeknya |
| `existingSecret`, `secretKeys` | nama Secret dan nama key di dalamnya |

Tulis angka sebagai string (`"0.8"`, `"5242880"`), karena Helm mengubah angka besar menjadi notasi ilmiah.

URL antar-service memakai nama Service per komponen, bukan `localhost`. Di luar
`ENVIRONMENT=local`, service menolak `*_SERVICE_URL` yang menunjuk ke localhost (orchestrator: keempat
URL-nya; extraction dan structuring: tahap berikutnya).

## Secret

Dengan `externalSecret.enabled=false` (default) chart tidak membuat Secret. Buat sendiri sebelum
atau sesudah install; pod menunggu sampai Secret ada.

```powershell
kubectl -n nilam-ocr-npwp create secret generic nilam-ocr-npwp-secrets `
  --from-literal=API_KEY='<api-key>' `
  --from-literal=DATABASE_URL='postgresql+asyncpg://<user>:<password>@postgres.ocr-dev.svc.cluster.local:5432/bribrain_ocr_nilam' `
  --from-literal=ORCHESTRATION_API_KEY='<opsional>'
```

`DATABASE_URL` harus berformat SQLAlchemy (`postgresql+asyncpg://`), bukan JDBC. Karakter khusus
di password perlu di-URL-encode (`@` menjadi `%40`). `ORCHESTRATION_API_KEY` boleh dihilangkan.
`orchestrator` dan `guardrails` hanya menerima `API_KEY`; `DATABASE_URL` tidak pernah masuk ke pod-nya. Key opsional `API_KEYS`
(dipisah koma) diterima juga oleh semua service selama rotasi: tambahkan key baru di sana, pindahkan
pemanggil, lalu jadikan `API_KEY` dan hapus dari `API_KEYS`; tiap langkah cukup `rollout restart`.

Setelah mengubah Secret, restart pod yang memakainya:
`kubectl -n nilam-ocr-npwp rollout restart deploy -l app.kubernetes.io/instance=nilam-ocr-npwp`.

Cluster dengan External Secrets Operator dan `ClusterSecretStore` bisa memakai
`externalSecret.enabled=true`; Secret dengan nama `existingSecret` lalu diisi dari Secret Manager
(nama key di `externalSecret.remoteKeys`). Cluster dev tidak punya CRD ESO.

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

Install pertama tetap memakai perintah di bagian berikutnya. Setelah release ada, perubahan kode
dideploy dengan [deploy.sh](deploy.sh) dari Git Bash, WSL, Linux atau macOS:

```bash
deploy/helm/deploy.sh --dry-run --skip-build --tag <tag> scoring
deploy/helm/deploy.sh scoring
deploy/helm/deploy.sh extraction structuring scoring
deploy/helm/deploy.sh all
```

**Deploy hanya dari `main`.** Script menolak `helm upgrade` kalau checkout bukan branch `main`,
berbeda dengan `origin/main` (belum di-pull atau ada commit yang belum di-push), atau ada perubahan
yang belum di-commit di `libs/`, `services/`, `db/`, atau `deploy/helm/`. Jadi yang jalan di cluster
selalu commit yang sudah ada di `main`; hotfix pun di-commit dan di-push ke `main` dulu. `--tag` untuk
deploy harus SHA commit di `origin/main` (commit selain HEAD hanya bersama `--skip-build`, misalnya
kembali ke versi lama). Dari branch lain hanya `--build-only` dan `--dry-run` yang jalan.

Script membangun image service yang disebut, mendorongnya ke Artifact Registry dengan tag
`git rev-parse --short HEAD` (`--build-only --allow-dirty` dari working tree yang belum di-commit
menambah `-dirty-<waktu>`; image seperti itu tidak bisa di-deploy), lalu menjalankan `helm upgrade --reset-then-reuse-values
-f values-ddb-dev.yaml --set services.<nama>.image.tag=<tag>`. Service lain tetap di tag lamanya
dan pod-nya tidak disentuh: hanya Deployment service yang disebut yang rolling update. Upgrade
menunggu pod siap dan otomatis rollback kalau gagal.

Perubahan `libs/ocr_common` masuk ke semua image; deploy `all`. Perubahan tabel dijalankan dulu
dengan [migrate-db.sh](migrate-db.sh) sebelum deploy image yang membutuhkannya.

**Riwayat release.** Setiap `helm upgrade` atau `helm rollback` menyimpan satu revisi sebagai Secret
`sh.helm.release.v1.nilam-ocr-npwp.v<N>` di namespace; itulah yang dipakai `helm rollback`. `deploy.sh`
menyimpan 5 revisi terakhir (`--history-max`, ubah lewat env `HISTORY_MAX`); revisi yang lebih tua
dibuang Helm sendiri pada upgrade berikutnya, jadi Secret itu tidak perlu dihapus manual.

**Upgrade ke chart 0.3.0 (orchestrator sebagai pintu masuk): wajib `deploy.sh all`.** Values baru
mengubah ConfigMap guardrails (tanpa `*_SERVICE_URL` dan `PIPELINE_WAIT_SECONDS`), jadi annotation
`checksum/config` me-roll pod guardrails walau guardrails tidak disebut. Image guardrails lama menolak
start dengan ConfigMap itu (`ENVIRONMENT=production` dan `EXTRACTION_SERVICE_URL` default localhost),
lalu `--atomic` me-rollback seluruh release. Tag global `values-ddb-dev.yaml` juga tidak punya image
orchestrator. Sejak upgrade, entry Service `nilam-ocr-npwp` hanya membuka 8034: Orkestrasi pusat harus
pindah dari `:8031/v1/extract-ocr` ke `:8034/v1/extract-ocr` pada saat yang sama. `helm rollback` ke
revisi sebelumnya membuang port 8034 lagi, jadi Orkestrasi pusat harus kembali ke `:8031` kalau
rollback.

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

**Upgrade dari chart 0.1.x (satu pod, empat container).** Helm menghapus Deployment
`nilam-ocr-npwp` yang lama dan membuat empat Deployment baru dalam satu `helm upgrade`. Pod lama
hilang saat pod baru masih starting (guardrails butuh sekitar satu menit memuat model), jadi ada
jeda singkat tanpa layanan; lakukan di luar jam uji coba. Values lama tetap dipakai
(`--reset-then-reuse-values`), termasuk tag image per service.

## Akses cluster

`gcloud container clusters get-credentials gc-ddb-dev-gke-cluster-01 --project ddb-kubecluster-dev-01 --location asia-southeast2`
butuh `gke-gcloud-auth-plugin`, yang dipasang dengan `gcloud components install gke-gcloud-auth-plugin`
dari terminal Administrator.

## Cek cepat

```powershell
kubectl -n nilam-ocr-npwp get deploy,pods,svc,pdb,networkpolicy
kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp-orchestrator 8034:8034
curl http://localhost:8034/ready
```

`port-forward` harus ke Service per komponen (atau `deploy/nilam-ocr-npwp-<service>`): pada Service
pintu masuk `nilam-ocr-npwp`, kubectl memilih satu pod sembarang dari selector-nya.
