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

## Model dari GCS

Model guardrails dan scoring disimpan di GCS, satu path tetap per model:

- guardrails: `gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/guardrails/best_model.pt`
- scoring: `gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/scoring/trust_model.joblib`

**Model ikut ke dalam image saat build.** Setiap `./deploy.sh` yang membangun guardrails atau scoring mengambil
model terbaru dari GCS lebih dulu (`scripts/fetch_weights.py`, dicek MD5 dan SHA-256), lalu `docker build`
menyalinnya ke image dan memberi label `nilam.model.sha256`. Model gagal diambil = build dibatalkan. Container tidak
butuh GCS atau kredensial apa pun saat berjalan, jadi image yang sama jalan di GKE maupun Compute Engine.

Kredensial GCS di mesin build (`ocr_common.clients.gcp.google_credentials`), salah satu:

- `wif.gcs.env` di root repo (Entra ID workload identity dari Tim SEA; jangan di-commit), atau variabel
  `AZURE_*` / `GCP_*`-nya di environment;
- tanpa itu, Application Default Credentials: service account VM / runner CI yang punya akses ke bucket, atau
  `gcloud auth application-default login` di laptop.

`deploy.sh` butuh Python dari `.venv` repo (`make dev`); arahkan dengan `PY=...` kalau bukan `python`.

**Ganti model:** unggah menimpa objeknya, lalu build dan deploy service-nya:

```bash
python scripts/upload_model.py <file baru> gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/guardrails/best_model.pt
./deploy.sh guardrails
```

**Rollback:** setiap image membawa modelnya sendiri, jadi deploy image tag sebelumnya mengembalikan model lamanya
juga: `./deploy.sh --skip-build --tag <sha commit sebelumnya> guardrails`. Model yang ada di image:
`docker inspect --format '{{ index .Config.Labels "nilam.model.sha256" }}' <image>`. Objek GCS yang tertimpa sendiri
tidak bisa diambil kembali; build berikutnya memakai isi GCS saat itu.

**Opsi: ambil saat start alih-alih saat build.** Dengan `GUARDRAILS_MODEL_GCS_URI` / `SCORING_MODEL_GCS_URI` di env,
service mengunduh model sendiri saat start dan mengabaikan yang ada di image (ganti model cukup unggah + restart,
tanpa build). Kredensialnya sama: `gcpWif.*` + `AZURE_CLIENT_SECRET` di Secret, atau Application Default Credentials
(Workload Identity di GKE, service account VM di Compute Engine). Mati secara default; tidak dipakai di dev.

## Elastic APM

Kelima service sudah membawa agen Elastic APM (`libs/ocr_common/ocr_common/web/apm.py`), tapi **mati**
selama `apm.serverUrl` kosong: agen tidak dimuat, tidak ada yang dikirim, dan tidak ada env APM di pod.
Menyalakannya setelah server APM tersedia:

```bash
# token ATAU API key, sesuai server APM-nya (keduanya opsional di pod)
kubectl -n nilam-ocr-npwp patch secret nilam-ocr-npwp-secrets --type merge \
  -p '{"stringData":{"ELASTIC_APM_SECRET_TOKEN":"<token>"}}'
```

lalu isi `apm.serverUrl` (dan kalau perlu `apm.environment`, `apm.transactionSampleRate`,
`apm.sanitizeFieldNames`) di `values-ddb-dev.yaml`, commit ke main, dan `./deploy.sh all`. Di dev sudah terisi
(server `http://10.213.128.39:8200`, versi 9.2.0, environment `dev`, satu service name `nilam-ocr-npwp` untuk
kelima service; token sudah di Secret sejak 7 Okt 2026). Tanpa token di Secret, server APM menolak setiap kiriman
(401) walau service tetap jalan.

Yang terlihat di APM, dengan service name `nilam-ocr-npwp-<service>` (atau `apm.serviceName` kalau diisi):

- satu transaksi per request HTTP, dan satu transaksi `<STAGE> job` per job di latar belakang (yang berjalan
  sesudah jawaban `202`), hasilnya `done` / `failed` / `interrupted`;
- satu transaksi `<STAGE> callback` / `<STAGE> handoff` (type `outbox`) per pengiriman oleh relay outbox, dengan
  span HTTP ke Orkestrasi pusat / tahap berikutnya dan hasil `delivered` / `retry` / `dead` / `skipped` (`skipped`:
  tidak ada yang dikirim, mis. DONE tahap yang bukan akhir request di format result). Inilah bukti callback ke
  pusat benar-benar terkirim, dan jawaban HTTP-nya;
- satu trace per request: job meneruskan trace request HTTP yang mengantrekannya, dan pesan outbox menyimpan
  `traceparent` job-nya (tidak ikut dikirim), jadi pengiriman, request tahap berikutnya, dan job-nya ada di trace
  yang sama;
- span panggilan ke service lain dan ke model OCR (httpx) dan ke database (asyncpg / SQLAlchemy);
- label `request_id` di setiap transaksi, id yang sama dengan di log; log JSON membawa `trace.id` dan
  `transaction.id` selama transaksi aktif, jadi log dan trace bisa dicocokkan;
- metrik CPU dan memori proses.

Body dan header request **tidak pernah** dikirim (`CAPTURE_BODY=off`, `CAPTURE_HEADERS=false`): dokumen, nomor
NPWP, nama, dan API key tidak keluar ke server APM.

## Log ke Elasticsearch

Service tidak mengirim log ke Elasticsearch sendiri. Setiap service menulis satu baris JSON per log ke stderr,
dan **Filebeat 7.17 milik cluster** (DaemonSet `filebeat` di `kube-system`, dikelola tim platform, bukan chart
ini) mengambil output semua container di setiap node (`/var/log/containers/*.log`), menambahkan metadata
Kubernetes (`kubernetes.namespace`, `kubernetes.pod.name`, `kubernetes.container.name`, ...), lalu mengirimnya ke
Logstash `10.213.128.12:5040`, yang meneruskannya ke Elasticsearch. Tidak ada yang perlu dipasang atau
dinyalakan di chart ini.

Field tiap baris (`libs/ocr_common/ocr_common/web/logging.py`):

| Field | Isi |
|---|---|
| `@timestamp`, `log.level` | waktu dan level (ECS); `time` dan `severity` sama, untuk Cloud Logging |
| `message`, `logger`, `exception` | pesan, nama logger, stack trace kalau ada |
| `service`, `request_id` | service asal dan request / job yang sedang dikerjakan (`-` di luar request) |
| `trace.id`, `transaction.id` | trace Elastic APM yang aktif, untuk melompat dari log ke trace dan sebaliknya |
| `event.dataset: outbox`, `event.action`, `event.outcome` | satu event per pengiriman outbox: `delivered`, `skipped`, `retry`, `dead`, atau `released` (dead letter dilepas) |
| `outbox.*` | `id`, `kind` (callback / handoff), `stage`, `target` (tahap berikutnya / `orchestration`), `attempt`, `age_seconds`, dan saat gagal `error`, `status_code`, `retry_in_seconds` |

Isi dokumen (payload callback / hand-off, NPWP, nama) tidak pernah masuk log. Akses log probe yang berhasil
(`/health`, `/ready`, `/metrics`, beberapa detik sekali per pod) tidak ditulis, dan juga tidak menjadi transaksi APM.

Contoh pencarian di Kibana (KQL):

```text
kubernetes.namespace : "nilam-ocr-npwp" and request_id : "OCR_9cb01af2-..."     # semua log satu request
event.dataset : "outbox" and event.outcome : "failure"                          # callback / hand-off yang gagal
event.dataset : "outbox" and event.action : "dead"                              # dead letter, perlu dilepas
event.dataset : "outbox" and outbox.target : "orchestration" and outbox.status_code >= 400
```

**Perlu dipastikan dengan tim platform** (belum diverifikasi dari sisi repo ini): index tempat Logstash menulis
log namespace `nilam-ocr-npwp`, dan apakah Logstash mem-parse JSON di field `message`. Kalau tidak, field di atas
ada sebagai teks di dalam `message`, bukan field yang bisa dicari; minta filter `json { source => "message" }`
untuk namespace ini di pipeline Logstash (atau processor `decode_json_fields` di Filebeat).

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

Database yang tidak terjangkau dari laptop (Cloud SQL dengan private IP) dimigrasi lewat Job di
cluster: `./db-job.sh alembic upgrade head` (`alembic current`, dst.).

## Pindah ke Cloud SQL

Tujuan: instance `edm-bribrain-dev-01:asia-southeast2:gc-bribrain-dev-sql-psql-01`, database
`bribrain_ocr`, private IP `10.213.224.113` di shared VPC yang sama dengan cluster (laptop tidak bisa
menjangkaunya). Database lama tidak diubah atau dihapus: dibaca saja, dan tetap ada untuk rollback.

`DATABASE_URL` Cloud SQL memakai Cloud SQL Python Connector (`cloudsql_instance` di query, lihat
`libs/ocr_common/ocr_common/pipeline/database.py`):

```text
# login IAM (tanpa password): service account lewat GKE Workload Identity
postgresql+asyncpg://gc-bribrain-dev-sac-sql-01%40common-sec-dev-01.iam@/bribrain_ocr?cloudsql_instance=edm-bribrain-dev-01:asia-southeast2:gc-bribrain-dev-sql-psql-01
# login user/password Cloud SQL biasa
postgresql+asyncpg://<user>:<password>@/bribrain_ocr?cloudsql_instance=edm-bribrain-dev-01:asia-southeast2:gc-bribrain-dev-sql-psql-01
```

### 0. Prasyarat (tim platform / DBA)

Per 5 Okt 2026 instance ini belum mendukung login IAM: flag `cloudsql.iam_authentication` tidak aktif
dan semua user-nya `BUILT_IN`. Pilih salah satu:

- **Login IAM.** Aktifkan flag `cloudsql.iam_authentication=on`; daftarkan
  `gc-bribrain-dev-sac-sql-01@common-sec-dev-01.iam` sebagai user `CLOUD_IAM_SERVICE_ACCOUNT`; beri user
  itu `CREATE` di database `bribrain_ocr`; beri `roles/iam.workloadIdentityUser` di GSA
  `gc-bribrain-dev-sac-sql-01@common-sec-dev-01.iam.gserviceaccount.com` untuk
  `serviceAccount:ddb-kubecluster-dev-01.svc.id.goog[nilam-ocr-npwp/nilam-ocr-npwp]`. Lalu buka komentar
  `serviceAccount.annotations` di `values-ddb-dev.yaml`, commit ke main, dan `./deploy.sh all`.
  Catatan: GSA ini dipakai bersama untuk instance berisi banyak database tim lain; GSA khusus NPWP
  lebih aman.
- **User/password.** DBA membuat satu user khusus dengan `CREATE` di database `bribrain_ocr`.

Image semua service yang memakai database harus sudah berisi dukungan Cloud SQL (commit
`feat(db): Cloud SQL ...` atau lebih baru): `./deploy.sh all` dari main sebelum cutover.

### 1. Simpan URL tujuan di Secret

```bash
kubectl -n nilam-ocr-npwp patch secret nilam-ocr-npwp-secrets --type merge \
  -p '{"stringData":{"DATABASE_URL_CLOUDSQL":"<URL Cloud SQL di atas>"}}'
```

### 2. Cek dan salinan awal (service tetap jalan)

```bash
./db-job.sh copy --check   # koneksi, login, versi migrasi, jumlah baris; tidak mengubah apa pun
./db-job.sh copy           # membuat schema + tabel di Cloud SQL, menyalin baris; aman diulang
```

Salinan awal boleh diulang kapan saja: baris yang sudah ada di Cloud SQL dilewati. Di akhir, jumlah
baris tiap tabel dibandingkan dan `alembic check` memastikan strukturnya sama dengan migrasi.

### 3. Cutover (pipeline berhenti beberapa menit)

```bash
NS=nilam-ocr-npwp; SEL=app.kubernetes.io/instance=nilam-ocr-npwp
# catat replika, hentikan semua penulis (orchestrator + tahap)
kubectl -n $NS get deploy -l $SEL -o jsonpath='{range .items[*]}{.metadata.name}={.spec.replicas}{"\n"}{end}' > /tmp/replicas-before-cutover.txt
kubectl -n $NS scale deploy -l $SEL --replicas=0
# salinan final persis (tabel Cloud SQL dikosongkan dulu, lalu disalin ulang)
./db-job.sh copy --replace
# DATABASE_URL -> Cloud SQL; URL lama disimpan di DATABASE_URL_PREVIOUS untuk rollback
old=$(kubectl -n $NS get secret nilam-ocr-npwp-secrets -o jsonpath='{.data.DATABASE_URL}')
new=$(kubectl -n $NS get secret nilam-ocr-npwp-secrets -o jsonpath='{.data.DATABASE_URL_CLOUDSQL}')
kubectl -n $NS patch secret nilam-ocr-npwp-secrets --type merge -p "{\"data\":{\"DATABASE_URL_PREVIOUS\":\"$old\",\"DATABASE_URL\":\"$new\"}}"
# nyalakan lagi dengan replika semula
while IFS='=' read -r name n; do kubectl -n $NS scale deploy "$name" --replicas="$n"; done < /tmp/replicas-before-cutover.txt
```

Lalu cek `GET /health` tiap service (`database: ok`) dan satu request `extract-ocr` lewat orchestrator.
Migrasi berikutnya ke Cloud SQL: `./db-job.sh alembic upgrade head`, bukan `migrate-db.sh`.

### Rollback

Database lama tidak pernah diubah, jadi cukup kembalikan `DATABASE_URL` lalu restart:

```bash
prev=$(kubectl -n $NS get secret nilam-ocr-npwp-secrets -o jsonpath='{.data.DATABASE_URL_PREVIOUS}')
kubectl -n $NS patch secret nilam-ocr-npwp-secrets --type merge -p "{\"data\":{\"DATABASE_URL\":\"$prev\"}}"
kubectl -n $NS rollout restart deploy -l $SEL
```

Request yang masuk sesudah cutover hanya ada di Cloud SQL; salin balik dengan `copy_database.py`
(`SOURCE_DATABASE_URL` = Cloud SQL) kalau datanya perlu.

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
  --set image.tag=<sha> --set services.scoring.env.PIPELINE_DRAIN_TIMEOUT_SECONDS="45"
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
