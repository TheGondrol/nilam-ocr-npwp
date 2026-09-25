# Integrasi service OCR NPWP (nilam) di GKE

Dokumen untuk tim Orkestrasi/Gateway pusat. Isinya cara memanggil pipeline OCR NPWP yang
berjalan di cluster, apa yang dikirim balik, dan apa yang kami butuhkan dari kalian.

**Berubah sejak 24 September 2026: pintu masuknya pindah.** Kalian sekarang hanya memanggil satu
service, **orchestrator** NPWP di port **8034** (`POST /v1/extract-ocr`, plus `GET` status baru).
Kontraknya sama persis dengan `POST :8031/v1/extract-ocr` yang lama di guardrails; yang berubah hanya
port-nya. Guardrails (8031) dan ketiga tahap pipeline sekarang internal: `:8031/v1/extract-ocr` sudah
tidak ada, dan port 8030/8031 tidak lagi dibuka di Service `nilam-ocr-npwp`.

Semua angka dan contoh respons di dokumen ini diambil dari request sungguhan ke service
yang sedang berjalan, bukan karangan. Yang belum terbukti ditandai eksplisit.

## 1. Status

Sudah terbukti jalan di cluster:

- guardrails, ekstraksi, structuring, scoring `/ready` menjawab 200, koneksi database normal
- satu kartu NPWP asli lewat rantai penuh dalam ±6 detik, hasil
  `npwp_confidence` 0.9926 dan `name_confidence` 0.9953
- job tercatat `DONE` di ketiga tabel tahap

Belum pernah diuji: **callback ke service kalian**, karena alamatnya belum kami punya.
Lihat bagian 9.

Baru: pintu masuk tunggal `POST /v1/extract-ocr` di orchestrator, port 8034 (bagian 3 dan 4).
Angka di atas berasal dari alur sebelumnya; tahap OCR sampai scoring tidak berubah. Orchestrator
sudah diuji lokal (unit test dan rantai lengkap di Docker Compose); di cluster dev baru berlaku
setelah deploy berikutnya, yang kami kabari bersama waktu kalian pindah ke port 8034.

## 2. Akses

| Item | Nilai |
|---|---|
| Cluster | `gc-ddb-dev-gke-cluster-01` |
| Project | `ddb-kubecluster-dev-01`, region `asia-southeast2` |
| Namespace | `nilam-ocr-npwp` |
| Service (ClusterIP) | `nilam-ocr-npwp`: pintu masuk, hanya port 8034 (orchestrator) |

Satu-satunya alamat yang kalian pakai:

    http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8034

Di belakangnya, tiap service punya Deployment + Service sendiri (`nilam-ocr-npwp-<service>`):
orchestrator (8034), guardrails (8031), ekstraksi (8030), structuring (8032), scoring (8033). Selain
orchestrator semuanya internal: hanya dipanggil orchestrator atau tahap sebelumnya.

**Autentikasi.** Semua endpoint kecuali `/health` dan `/ready` memerlukan header:

    X-API-Key: changeme

Nilai itu sementara dan akan diganti sebelum dipakai serius; kami kabari kalau berubah.

**Jaringan.** NetworkPolicy di namespace kami mengizinkan ingress dari namespace `ocr-dev` hanya ke
orchestrator; guardrails dan ketiga tahap hanya menerima dari orchestrator dan dari tahap sebelumnya.
Cluster dev belum menegakkan NetworkPolicy, jadi saat ini tidak ada yang terblokir dan "internal" baru
berlaku di atas kertas; tetap panggil hanya orchestrator. Kalau namespace kalian bukan `ocr-dev`, beri
tahu kami supaya ditambahkan. Kami belum bisa menguji panggilan dari namespace kalian.

**Dari laptop** (untuk coba-coba), cukup forward orchestrator:

    kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp-orchestrator 8034:8034

## 3. Alur

Hanya **satu panggilan** dari sisi kalian, dan jawabannya mengikuti kontrak `extract-ocr`
kalian. Orchestrator memeriksa file, meminta guardrails menilainya, lalu menunggu pipeline sampai
`PIPELINE_WAIT_SECONDS` (default **15 detik**, dihitung sejak request diterima):

    POST :8034/v1/extract-ocr
      file > 2,5 MB              -> 413  message "Ukuran dokumen melebihi batas 2,5 MB, ..." (sebelum model)
      PDF > 2 halaman            -> 400  message "Jumlah halaman melebihi batas, ..."        (sebelum model)
      ditolak model guardrails   -> 200  job_status = failed, guardrails = 1, errors = null,
                                         message = "guardrails rejected"; tidak ada yang jalan, tidak ada callback
      ditolak aturan structuring -> 200  job_status = failed, guardrails = 1, errors = null,
                                         message = alasan penolakan dari aturan ML; scoring tidak jalan
      lolos, selesai tepat waktu -> 200  job_status = completed, data = {nomor_npwp, nama}, guardrails = 0
      lolos, gagal tepat waktu   -> 422  job_status = failed, guardrails = 0,
                                         errors = OCR_FAILED | STRUCTURING_FAILED | SCORING_FAILED
      lolos, belum selesai       -> 202  job_status = processing, data = null, guardrails = null
                                         hasil menyusul di callback SCORING

**`guardrails`: `1` = dokumen ditolak, `0` = lolos** (disepakati 25 September 2026; sebelumnya
kebalikannya). Dokumen yang ditolak dijawab **200**, bukan 400: bedakan dari hasil lewat
`guardrails` atau `job_status`, bukan lewat status HTTP.

Callback bersifat **opsional**: kalau kalian tidak menyediakan endpoint callback, kami
jalankan tanpa `ORCHESTRATION_URL`, tidak ada callback yang dikirim, dan hasil tiap request
sampai ke kalian lewat baris `request_id` di tabel `orchestration_extract_ocr` milik kalian
(`processing` saat tahap berjalan, `completed` + `result_data` dari scoring, `failed` +
`<TAHAP>_FAILED` kalau gagal, termasuk tahap yang tidak pernah bisa dihubungi). Di jalur 202
kalian tinggal membaca baris itu saat client polling. Kalau endpoint callback ada, setelah
dokumen lolos callback `OCR`, `STRUCTURING`, `SCORING` dikirim dalam semua
kasus; kalau hasil sudah diterima di respons 200, callback-nya boleh diabaikan. Di jalur 202,
callback SCORING membawa hasil akhir dalam bentuk internal (bagian 7); petakan ke `data`
dengan aturan yang sama seperti di bagian 4.

**Callback bisa datang tidak berurutan.** Sejak pengiriman callback dipindah ke outbox yang
tahan restart, tiap tahap mengirim callback-nya sendiri tanpa menunggu tahap lain, jadi
`SCORING` bisa tiba sebelum `OCR`. Perlakukan callback sebagai update status per tahap yang
idempoten (boleh datang dua kali), dan tentukan keadaan akhir dari `SCORING`/`DONE` atau
`FAILED` mana pun. Callback yang gagal di sisi kalian akan dikirim ulang, dan **tidak lagi
memperlambat pipeline**.

Rinciannya: jawaban `5xx`, timeout, atau host tidak terjangkau dikirim ulang dengan backoff
sampai 5 menit sekali, selama 24 jam. Jawaban **`4xx` tidak dikirim ulang**: callback-nya
disimpan sebagai *dead letter* di sisi kami dan harus dilepas manual, jadi jawab `4xx` hanya
kalau body-nya memang salah, bukan karena `request_id`-nya belum kalian kenal. Pengiriman
*at-least-once*: hand-off antar-tahap juga bisa terkirim dua kali; tahap berikutnya tidak
menjalankan ulang job yang sudah `DONE`/`PROCESSING`, tapi job yang sudah `FAILED` akan
dijalankan lagi (sama seperti kalau kalian mengirim ulang `request_id` itu).

Pasang HTTP timeout panggilan ini di atas `PIPELINE_WAIT_SECONDS`, mis. **30 detik** untuk
default 15 detik: pemeriksaan guardrails dan hand-off ke OCR bisa menambah waktu.

`request_id` dibuat oleh kalian dan menjadi kunci di semua tahap. Bebas formatnya,
string; contoh yang kami pakai saat uji: `REQ_a0e0fd34ed7a`.

Kalian **tidak** memanggil guardrails, ekstraksi, structuring, dan scoring sendiri (dan sejak
orchestrator ada, memang tidak bisa dari namespace kalian begitu NetworkPolicy ditegakkan). Keadaan
sebuah request kapan saja: `GET :8034/v1/extract-ocr/{request_id}` (bagian 4). Bagian 5 dan 6
menjelaskan apa yang terjadi di dalam, sebagai latar.

## 4. Service orchestrator (port 8034): pintu masuk

Memeriksa file, meminta guardrails (model EfficientNet, internal) menilai layak atau tidaknya
dokumen, dan kalau layak memulai pipeline lalu menunggu hasilnya sampai `PIPELINE_WAIT_SECONDS`.
Orchestrator sendiri tidak menyimpan apa pun dan tidak mengirim callback; callback datang dari
tahap-tahap pipeline (bagian 7).

### POST /v1/extract-ocr — jalankan pipeline

Kirim `request_id` plus dokumen sebagai `file` (multipart), **atau** sebagai `file_url`
supaya service ini yang mengunduh. Salah satu saja, tidak boleh dua-duanya.

    curl -X POST http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8034/v1/extract-ocr \
      -H "X-API-Key: changeme" \
      -F "request_id=REQ_001" \
      -F "document_type=npwp" \
      -F 'params={"nik": "3123456711950001", "refno": "PK19039Y8U"}' \
      -F "file=@npwp.jpg"

Path, field, dan bentuk jawabannya sama persis dengan `extract-ocr` yang dulu di guardrails port
8031; hanya port-nya yang berubah ke **8034**. Bentuk jawabannya mengikuti kontrak `extract-ocr`
kalian: envelope standar ditambah `document_type`, `job_status`, `guardrails`, dan `params`.

| Field | Wajib | Keterangan |
|---|---|---|
| `request_id` | ya | dibuat oleh kalian |
| `document_type` | tidak | default `npwp`; selain `npwp` dijawab 400 `UNSUPPORTED_DOCUMENT_TYPE` |
| `params` | tidak | JSON object atau string berkutip; tidak ditafsirkan, dikembalikan apa adanya di `params`. JSON tidak valid dijawab 422 `INVALID_PARAMS` |
| `file` / `file_url` | salah satu | JPEG, PNG, PDF, maksimal **2,5 MB** (lebih besar: **413**) dan maksimal **2 halaman** (lebih: **400**), keduanya dengan `message` berbahasa Indonesia yang bisa langsung ditampilkan ke pengguna, diperiksa sebelum model jalan (permintaan ML engineer, 23 Sep 2026). PDF dinilai per halaman |

`file_url` diunduh sekali di panggilan ini untuk penilaian guardrails, lalu **URL-nya** (bukan isi
file) diteruskan ke tahap OCR, yang mengunduhnya lagi, juga saat menjalankan ulang job yang
ditinggalkan pod yang mati. Presigned URL karena itu harus hidup lebih lama dari lease job
(`PIPELINE_JOB_LEASE_SECONDS`, 5 menit). Host-nya harus terdaftar di `FILE_URL_ALLOWED_HOSTS`
service (atau, kalau itu kosong, resolve ke alamat publik), dan redirect tidak diikuti.

Selesai dalam waktu tunggu, **200**:

    {
      "status_code": 200,
      "status_desc": "OK",
      "message": "OCR extraction completed successfully",
      "data": {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1}
      },
      "errors": null,
      "request_id": "REQ_001",
      "document_type": "npwp",
      "job_status": "completed",
      "guardrails": 0,
      "params": {"nik": "3123456711950001", "refno": "PK19039Y8U"}
    }

- `nama` = nama wajib pajak, atau nama badan pada kartu perusahaan.
- `confidence` = `1` kalau trust model ML memberi probabilitas benar minimal
  `FIELD_CONFIDENCE_THRESHOLD` (default 0.5), `0` kalau di bawahnya atau field tidak ditemukan.
- Flag dari aturan ekstraksi ML engineer **tidak** ada di `data`: flag itu internal, masuk sebagai
  input trust model. Dari 11 flag, hanya 2 yang ditoleransi (nama satu kata, huruf di nomor NPWP):
  nilainya tetap dikembalikan dan confidence-nya sudah memperhitungkan flag itu. Sembilan lainnya
  menolak dokumen (200, `guardrails: 1`, lihat di bawah).
- Pencocokan nama fuzzy terhadap nama nasabah (`refno`) dilakukan di sisi Orkestrasi, sesuai
  keputusan ML engineer; service ini tidak memakainya.

Belum selesai saat waktu tunggu habis, **202**; hasil menyusul lewat callback, atau baca dengan
`GET /v1/extract-ocr/{request_id}` (di bawah):

    {"status_code": 202, "status_desc": "Accepted", "message": "OCR job accepted; still processing",
     "data": null, "errors": null, "request_id": "REQ_001", "document_type": "npwp",
     "job_status": "processing", "guardrails": null, "params": {...}}

Ditolak model guardrails, **200** dengan `guardrails: 1`; tidak ada yang jalan dan tidak ada callback:

    {"status_code": 200, "status_desc": "OK", "message": "guardrails rejected",
     "data": null, "errors": null, "request_id": "REQ_001",
     "document_type": "npwp", "job_status": "failed", "guardrails": 1, "params": {...}}

Ditolak aturan structuring ML engineer (23 Sep 2026), juga **200** dengan bentuk yang sama;
`message` adalah alasan penolakan pertama dari aturan itu (bukan alasan flag yang ditoleransi),
dalam bahasa Indonesia, dan bisa langsung ditampilkan ke pengguna. Penolakan terjadi di tahap structuring, jadi scoring tidak jalan:

    {"status_code": 200, "status_desc": "OK",
     "message": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
     "data": null, "errors": null, "request_id": "REQ_001",
     "document_type": "npwp", "job_status": "failed", "guardrails": 1, "params": {...}}

Alasan yang menolak: dokumen blur / blank, bukan format standar NPWP, dokumen lain terdeteksi,
screenshot cek NPWP online, jumlah halaman melebihi batas, serta kode provinsi, kecamatan, tanggal
lahir, atau KPP pada nomor yang tidak valid. Kalau dokumen punya beberapa alasan, yang dipakai
adalah alasan penolakan pertama menurut urutan prioritas aturan ML. Kalau penolakan terjadi setelah
jawaban 202, keadaan akhirnya adalah callback `STRUCTURING` `FAILED` dengan alasan itu sebagai
`error_message`, dan baris 400 / `DOWNSTREAM_VALIDATION_ERROR` di tabel Orkestrasi. **Catatan:**
baris tabel dan callback itu belum ikut diubah ke 200; kabari kami kalau perlu disamakan.

Tahap pipeline gagal dalam waktu tunggu, **422**; request berakhir di sini, sama seperti
callback `FAILED`. `errors` menyebut tahapnya, `message` alasannya:

    {"status_code": 422, "status_desc": "Unprocessable Entity",
     "message": "ekstraksi OCR model is unavailable",
     "data": null, "errors": "OCR_FAILED", "request_id": "REQ_001",
     "document_type": "npwp", "job_status": "failed", "guardrails": 0, "params": {...}}

Error lain (envelope standar; `errors` sama dengan `message` kecuali disebut lain):

| Kode | `errors` | Arti | Pipeline jalan? |
|---|---|---|---|
| 400 | `UNSUPPORTED_DOCUMENT_TYPE` | `document_type` bukan `npwp` (dokumen yang *ditolak* bukan 400, lihat di atas) | tidak |
| 400 | = `message` | file kosong, format salah, PDF lebih dari 2 halaman (`Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP`), `file`/`file_url` dua-duanya / tidak ada, atau host `file_url` tidak diizinkan / tidak bisa diunduh | tidak |
| 413 | = `message` | file lebih dari 2,5 MB (`Ukuran dokumen melebihi batas 2,5 MB, pastikan hanya mengunggah dokumen NPWP`) | tidak |
| 401 | = `message` | `X-API-Key` salah | tidak |
| 422 | `INVALID_PARAMS` / `VALIDATION_ERROR` | `params` bukan JSON object / string, atau field wajib tidak dikirim | tidak |
| 503 / 504 | = `message` | guardrails atau modelnya tidak terjangkau / tidak menjawab (tidak dicoba ulang), atau tahap OCR tidak terjangkau / tidak menjawab (sudah dicoba ulang 3 kali) | tidak; kirim ulang aman |

**Idempoten.** `request_id` yang sama dikirim ulang: guardrails dicek lagi, tetapi pipeline
tidak menjalankan apa pun dua kali, kecuali percobaan sebelumnya berstatus `FAILED`, atau sudah
`PROCESSING` lebih lama dari lease (`PIPELINE_JOB_LEASE_SECONDS`, default 5 menit: prosesnya
mati di tengah jalan); dalam hal itu dijalankan ulang. Job yang masih jalan saat service
shutdown dilaporkan `FAILED` lewat callback, jadi cukup dikirim ulang. `request_id` yang sudah
selesai dijawab 200 dengan hasil yang tersimpan.

### GET /v1/extract-ocr/{request_id} — keadaan request sekarang

Kontrak yang sama dengan `POST`, tanpa menunggu: orchestrator membaca status tahap OCR, structuring,
dan scoring sekali, berurutan. Pakai untuk request yang dijawab 202, misalnya kalau callback-nya
tidak datang.

    curl http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8034/v1/extract-ocr/REQ_001 \
      -H "X-API-Key: changeme"

| Keadaan | HTTP | `job_status` | `errors` |
|---|---|---|---|
| selesai | 200 | `completed` + `data`, `guardrails: 0` | null |
| masih berjalan | 202 | `processing` | null |
| ditolak aturan structuring | 200 | `failed`, `guardrails: 1`, `message` = alasannya | null |
| satu tahap gagal | 422 | `failed`, `guardrails: 0` | `OCR_FAILED` / `STRUCTURING_FAILED` / `SCORING_FAILED` |
| tidak dikenal | 404 | – | = `message` |

- `params` selalu `null` di sini (tidak disimpan); `document_type` selalu `npwp`.
- **404** berarti tidak ada tahap yang punya job untuk `request_id` itu: dokumennya ditolak model
  guardrails (jawaban `POST`-nya, `guardrails: 1`, adalah jawaban final; tidak ada yang disimpan),
  ditolak sebelum dinilai, atau
  `POST`-nya masih berjalan di tahap penilaian guardrails.
- **503 / 504** berarti salah satu tahap tidak bisa dibaca saat itu; coba lagi.
- **Batasan:** serah terima antar tahap yang gagal permanen (setelah semua retry, atau menjadi dead
  letter di outbox kami) hanya tercatat di callback `FAILED` dan di tabel kalian; endpoint ini tetap
  menjawab 202 untuk request itu. Untuk keadaan final, callback dan tabel kalian yang berlaku.

## 5. Service ekstraksi (port 8030)

Tahap OCR. Backend OCR-nya PaddleOCR yang berjalan di VM terpisah; service ini yang
memanggilnya.

`POST /v1/ekstraksi/jobs` dipanggil oleh orchestrator, bukan oleh kalian. Setelah dijawab
202, di background: dokumen dibaca, OCR dijalankan, hasil disimpan, lalu job diserahkan ke
structuring, yang kemudian menyerahkan ke scoring.

### GET /v1/ekstraksi/jobs/{request_id} — status tahap OCR (internal)

Dibaca orchestrator untuk `GET /v1/extract-ocr/{request_id}`; dicantumkan sebagai latar.

    {"data": {"request_id": "REQ_001", "stage": "OCR", "status": "DONE",
              "error_message": null, "result": { … hasil OCR mentah … },
              "created_at": "...", "updated_at": "..."}}

`status` bernilai `PROCESSING`, `DONE`, atau `FAILED`. `404` berarti tahap ini belum
pernah menerima job dengan `request_id` tersebut.

### POST /v1/ekstraksi/extract — OCR mentah, sinkron (internal)

Menjalankan OCR saja dan langsung mengembalikan hasilnya. Tidak membuat job, tidak
mengirim callback, tidak menyentuh tahap lain. Untuk debugging kami.

## 6. Service structuring (port 8032) dan scoring (port 8033)

Keduanya dipanggil berantai oleh service sebelumnya dan internal; kalian tidak memanggilnya.

| Service | Endpoint | Siapa yang memanggil |
|---|---|---|
| guardrails | `POST /v1/guardrails/check` | orchestrator, untuk tiap dokumen |
| ekstraksi | `POST /v1/ekstraksi/jobs` | orchestrator, otomatis |
| structuring | `POST /v1/structuring/jobs` | ekstraksi, otomatis |
| structuring | `GET /v1/structuring/jobs/{request_id}` | orchestrator (status) |
| structuring | `POST /v1/structuring/structure` | debugging, sinkron |
| scoring | `POST /v1/scoring/jobs` | structuring, otomatis |
| scoring | `GET /v1/scoring/jobs/{request_id}` | orchestrator (status) |
| scoring | `POST /v1/scoring/confidence` | debugging, sinkron |

**structuring** mengubah baris OCR menjadi field bernama memakai aturan regex dan posisi
dari tim ML. Contoh isi `result` sungguhan:

    {
      "document_type": "npwp",
      "fields": {
        "nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 0.9999,
                       "source": "NPWP:09.254.294.3-407.000",
                       "signals": {"has_homoglyph": false, "candidate_count": 1}},
        "nama":       {"value": "KOJIB", "confidence": 1.0, "source": "KOJIB",
                       "signals": {"corrected": false}},
        "nama_badan": {"value": null, "confidence": 0.0, "source": null, "signals": null}
      }
    }

**scoring** menjalankan trust model dari tim ML dan menghasilkan dua angka confidence.
Tahap ini yang terakhir, dan hanya callback-nya yang membawa hasil akhir.

## 7. Kontrak callback

### Callback hasil (dipakai di dev sejak 24 Sep 2026)

Endpoint dari tim Orkestrasi, satu POST per request saat request selesai
(`ORCHESTRATION_CALLBACK_FORMAT=result`):

    POST http://ocr-orchestration.ocr-dev.svc.cluster.local/v1/ocr-callback
    X-Callback-Key: <ORCHESTRATION_CALLBACK_KEY>

Selesai (dikirim oleh scoring):

    {
      "request_id": "OCR_9cb01af2-493d-446d-b191-af120333f6d0",
      "status": "completed",
      "result": {
        "nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 0.9829},
        "nama":       {"value": "BUDI SANTOSO",         "confidence": 0.9512},
        "nama_badan": {"value": "",                     "confidence": 0.0}
      },
      "guardrails": {"passed": true, "reason": null, "document": {...}, "pages": [...]}
    }

`confidence` adalah probabilitas dari trust model bahwa nilainya benar, belum dibulatkan ke 0/1
seperti di respons `extract-ocr`. Field yang tidak ditemukan: `value` kosong dan `confidence` 0.0.
Probabilitas nama masuk ke `nama` atau `nama_badan`, mana pun yang berisi nama. `guardrails` adalah
laporan model guardrails untuk dokumen itu.

Callback dikirim oleh tahap-tahap pipeline sendiri (scoring, atau tahap yang berhenti), bukan oleh
orchestrator; alamat dan key-nya tidak berubah.

Gagal atau ditolak (dikirim oleh tahap yang berhenti):

    {
      "request_id": "OCR_9cb01af2-493d-446d-b191-af120333f6d0",
      "status": "failed",
      "result": null,
      "guardrails": {},
      "error_code": "DOWNSTREAM_VALIDATION_ERROR",
      "error_message": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
    }

`error_code` bernilai `DOWNSTREAM_VALIDATION_ERROR` kalau dokumen ditolak aturan structuring, atau
`OCR_FAILED` / `STRUCTURING_FAILED` / `SCORING_FAILED` kalau tahapnya gagal. Dokumen yang ditolak
model guardrails tidak mendapat callback, karena sudah dijawab 400 langsung di `extract-ocr`
orchestrator.
Jawaban 5xx dan timeout dikirim ulang (3 kali tanpa outbox, seperti di dev sekarang; sampai 24 jam
dengan `PIPELINE_OUTBOX`), 4xx tidak dikirim ulang.

### Callback per tahap (format lama, `ORCHESTRATION_CALLBACK_FORMAT=stage`)

Tiap tahap mem-POST ke `<ORCHESTRATION_URL><CALLBACK_PATH>`, dengan default path
`/v1/callbacks/stage`. Header `X-API-Key` ikut dikirim kalau kami diberi nilainya, dan header
`X-Request-ID` selalu berisi `request_id` request itu, sama dengan yang kami kirim ke service
kami sendiri, supaya log kalian dan log kami bisa dicocokkan.

Body untuk OCR dan STRUCTURING:

    {
      "request_id": "REQ_001",
      "stage": "OCR",
      "status": "DONE",
      "result": null,
      "error_message": null
    }

Dua hal yang paling sering disalahpahami:

1. **`result` selalu null untuk OCR dan STRUCTURING.** Hasil antara tidak dibuka ke luar;
   keadaan request dibaca lewat `GET /v1/extract-ocr/{request_id}` di orchestrator.
2. **Hanya callback SCORING yang membawa hasil akhir.**

Body callback SCORING saat sukses:

    {
      "request_id": "REQ_001",
      "stage": "SCORING",
      "status": "DONE",
      "error_message": null,
      "result": {
        "document_type": "npwp",
        "fields": {
          "nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 0.9999,
                         "source": "NPWP:09.254.294.3-407.000",
                         "signals": {"has_homoglyph": false, "candidate_count": 1}},
          "nama":       {"value": "KOJIB", "confidence": 1.0, "source": "KOJIB",
                         "signals": {"corrected": false}},
          "nama_badan": {"value": null, "confidence": 0.0, "source": null, "signals": null}
        },
        "scoring": {"npwp_confidence": 0.9926, "name_confidence": 0.9953},
        "guardrails": { … isi data dari langkah guardrails, dikembalikan apa adanya … }
      }
    }

`stage` bernilai `OCR`, `STRUCTURING`, atau `SCORING`. `status` bernilai `DONE` atau
`FAILED`.

**Kegagalan.** `status: FAILED` di tahap mana pun mengakhiri request; alasannya di
`error_message`, dan `result` bernilai null. Ada satu kasus khusus: kalau serah terima
ke tahap berikutnya gagal setelah retry, pengirim melaporkan `FAILED` dengan **nama
tahap berikutnya**, karena tahap itu tidak pernah menerima job dan tidak bisa melapor
untuk dirinya sendiri.

**Retry.** Callback yang gagal dicoba ulang 3 kali dengan jeda 0,5 detik. Setelah itu
menyerah dan hanya dicatat di log; job tetap dianggap selesai. Jadi callback bisa hilang,
dan rekonsiliasi di bagian 8 adalah jaring pengamannya.

Endpoint callback kalian cukup menjawab 2xx. Isi jawabannya tidak kami baca.

## 8. Rekonsiliasi kalau callback hilang

    GET :8034/v1/extract-ocr/{request_id}

Jawabannya kontrak `extract-ocr` yang sama dengan `POST` (bagian 4): 200 dengan `data` kalau
sudah selesai, 202 kalau masih berjalan, 400/422 kalau berhenti, 404 kalau tidak ada tahap yang
pernah menerima `request_id` itu. Endpoint status per tahap (`GET /v1/<tahap>/jobs/{request_id}`)
yang dulu kami sebut di sini sekarang internal; orchestrator yang membacanya untuk kalian.

## 9. Yang kami butuhkan dari kalian

**Satu hal: URL endpoint callback kalian.**

Saat ini `ORCHESTRATION_URL` di deployment kami sengaja menunjuk ke service kami sendiri,
supaya request uji coba kami tidak menembak service kalian dengan `request_id` palsu.
Begitu kalian kasih alamatnya, kami ganti dengan satu perintah.

Yang perlu kami tahu:

1. **URL dasar.** Kami menebak `http://ocr-orchestration.ocr-dev.svc.cluster.local`
   dari isi cluster. **Tolong dikonfirmasi**, kami belum pernah memvalidasinya.
2. **Path callback**, kalau bukan `/v1/callbacks/stage`.
3. **API key** endpoint callback kalian, kalau ada. Akan kami pasang sebagai
   `ORCHESTRATION_API_KEY` dan dikirim sebagai header `X-API-Key`.

Kontrak callback di bagian 7 adalah **usulan sepihak dari kami dan belum pernah
disepakati**. Kalau bentuknya perlu berbeda, bilang sekarang selagi murah diubah.

## 10. Amplop respons dan kode error

Semua respons memakai amplop yang sama.

Sukses: `status_code`, `status_desc`, `message`, `errors` (null), `request_id`, `data`.

Error: `data` selalu null, `errors` berisi kode yang bisa dibaca mesin, `message` berisi
penjelasan yang aman untuk di-log.

    {"status_code": 400, "status_desc": "Bad Request",
     "message": "Uploaded file is empty", "data": null,
     "errors": "VALIDATION_ERROR", "request_id": "REQ_001"}

| Kode | Arti |
|---|---|
| 400 | file bermasalah (kosong, tipe tidak didukung, lebih dari 2 halaman) atau intake salah |
| 413 | file lebih dari 2,5 MB |
| 401 | `X-API-Key` salah atau tidak ada |
| 404 | `request_id` tidak dikenal di tahap itu |
| 422 | body atau field tidak valid |
| 502/503 | model atau service tujuan tidak bisa dihubungi |

`extract-ocr` menambah `document_type`, `job_status`, `guardrails`, dan `params` ke amplop ini
(bagian 4).

## 11. Kontrak lama (sinkron): sudah dihapus

Sejak 24 September 2026, endpoint kontrak lama di service ekstraksi (port 8030) sudah
dihapus:

    POST /v1/generate-request-id
    POST /v1/extract-ocr                  (versi ekstraksi, port 8030)
    GET  /v1/get-ocr-result/{request_id}

Sejak orchestrator ada (24 September 2026), `POST /v1/extract-ocr` di guardrails (port 8031)
juga sudah dihapus. Pakai `POST /v1/extract-ocr` di orchestrator (port 8034, bagian 4), dan
`GET /v1/extract-ocr/{request_id}` untuk keadaan sebuah request (bagian 8).

## 12. Database

Pipeline menyimpan job dan hasil tiap tahap ke PostgreSQL yang sama dengan yang kalian
pakai, database `bribrain_ocr_nilam`. Semua tabel kami ada di schema `public`, sesuai
permintaan supaya seragam.

| Tabel | Isi |
|---|---|
| `ocr_jobs`, `ocr_results` | status dan hasil OCR mentah |
| `structuring_jobs`, `structuring_results` | field hasil penataan |
| `scoring_jobs`, `scoring_results` | confidence akhir |

Integrasi normal **tidak perlu menyentuh database ini**; semua yang dibutuhkan sudah ada
di callback dan di `GET /v1/extract-ocr/{request_id}`. Kami cantumkan supaya jelas tabel mana milik kami, dan
supaya kalian tahu tabel `orchestration_*` serta `auth_*` milik kalian tidak kami sentuh.

## 13. Spesifikasi lengkap

`api/gateway.openapi.yaml` di repo ini adalah spec OpenAPI yang berisi hanya yang kalian
pakai: `POST` dan `GET /v1/extract-ocr` di orchestrator, lengkap dengan skema dan contoh, plus
webhook `stageCallback` (format callback per tahap; format `result` yang dipakai di dev ada di
bagian 7). Itu sumber kebenaran paling detail; dokumen ini ringkasannya.

Swagger UI orchestrator juga hidup di `/docs`:

    kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp-orchestrator 8034:8034
    # buka http://127.0.0.1:8034/docs

Spec per service ada di `services/<nama>/openapi.yaml`.
