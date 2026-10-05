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

- guardrails, extraction, structuring, scoring `/ready` menjawab 200, koneksi database normal
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
orchestrator (8034), guardrails (8031), extraction (8030), structuring (8032), scoring (8033). Selain
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
      ditolak model guardrails   -> 400  errors = DOWNSTREAM_VALIDATION_ERROR, guardrails = 1
                                         tidak ada yang jalan, tidak ada callback
      ditolak aturan structuring -> 400  errors = DOWNSTREAM_VALIDATION_ERROR, guardrails = 1,
                                         message = alasan penolakan dari aturan ML; scoring tidak jalan
      lolos, selesai tepat waktu -> 200  data = {nomor_npwp, nama}, guardrails = 0
      lolos, gagal tepat waktu   -> 422  errors = OCR_FAILED | STRUCTURING_FAILED | SCORING_FAILED
      lolos, belum selesai       -> 202  data = null, guardrails = null
                                         hasil menyusul di callback SCORING

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

Kalian **tidak** memanggil guardrails, extraction, structuring, dan scoring sendiri (dan sejak
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
kalian: envelope standar ditambah `pipeline_last_stage` dan `guardrails`. Keadaan request dibaca
dari kode HTTP (juga di `status_code`): 200 selesai, 202 masih berjalan, 4xx / 5xx gagal atau ditolak.
Sejak 1 Oktober 2026 jawaban tidak lagi membawa `document_type`, `job_status`, dan `params`, dan field
`params` di request dihapus (kalau masih dikirim, diabaikan).

**`pipeline_last_stage`** bernilai `null` pada jawaban sukses (200, 202) dan menyebut service asal
**error**, dengan nama yang sama seperti di `pipeline_name_sequence`. Field ini ada di setiap jawaban
`POST` maupun `GET /v1/extract-ocr/{request_id}`:

| Jawaban | `pipeline_last_stage` |
|---|---|
| 200 selesai | `null` |
| 202 masih berjalan | `null` |
| 400 ditolak model guardrails | `guardrails` |
| 400 ditolak aturan structuring | `structuring` |
| 422 `<TAHAP>_FAILED` | service yang gagal (`OCR_FAILED` = `extraction`) |
| 400 / 500 / 503 / 504 saat memanggil sebuah service (tidak terjangkau, timeout, file tidak terbaca model) | service itu |
| ditolak orchestrator sendiri sebelum service pipeline mana pun dipanggil (API key, file, `pipeline_name_sequence`, threshold, `document_type`, request_id tidak dikenal pada `GET`) | `orchestrator` |

| Field | Wajib | Keterangan |
|---|---|---|
| `request_id` | ya | dibuat oleh kalian |
| `document_type` | tidak | default `npwp`; selain `npwp` dijawab 400 `UNSUPPORTED_DOCUMENT_TYPE` |
| `file` / `file_url` | salah satu | JPEG, PNG, PDF, maksimal **2,5 MB** (lebih besar: **413**) dan maksimal **2 halaman** (lebih: **400**), keduanya dengan `message` berbahasa Indonesia yang bisa langsung ditampilkan ke pengguna, diperiksa sebelum model jalan (permintaan ML engineer, 23 Sep 2026). PDF dinilai per halaman |
| `pipeline_name_sequence` | tidak | array of string: service yang dijalankan, berurutan. Default: keempatnya (lihat di bawah) |
| `guardrails_confidence_threshold` | tidak | JSON object per guardrails, `{"acc_rej": 0.8}`: pipeline ini punya satu guardrails, `acc_rej` (accept/reject), jadi hanya key itu yang valid. Nilainya angka di antara 0 dan 1 (eksklusif): threshold model guardrails untuk dokumen ini saja, pada probabilitas accept: halaman lolos kalau probabilitas accept ≥ threshold, ditolak kalau di bawahnya (`guardrails: 1`). Tidak dikirim: threshold guardrails sendiri (`GUARDRAILS_THRESHOLD_URL`, lalu `GUARDRAILS_THRESHOLD`, lalu 0.5) |
| `column_confidence_threshold` | tidak | JSON object, nilai 0–1, selalu sisi accept: `confidence` field itu `1` kalau probabilitas trust model ≥ nilainya, `0` kalau di bawahnya. Key `all_field` berlaku untuk semua field (`{"all_field": 0.8}`); key per field `nomor_npwp` / `nama` boleh dipakai sebagai gantinya atau bersamaan, dan key per field menang atas `all_field`. Field yang tidak disebut, atau field ini tidak dikirim: `FIELD_CONFIDENCE_THRESHOLD` (0.5) |

Threshold yang tidak bisa dibaca (bukan JSON object, nilai di luar rentang atau bukan angka, key guardrails selain `acc_rej`, atau key field selain `all_field` / `nomor_npwp` / `nama`) dijawab **422
`INVALID_THRESHOLD`** dengan `message` yang menyebut key atau nilai mana yang salah, dan tidak ada yang dijalankan. Contoh lengkap:

    request_id                       = OCR_361701a7-ad0f-46f7-9922-8eae7c99015e
    file_url                         = https://minio.example/ocr/abc.jpg
    pipeline_name_sequence           = ["guardrails","extraction","structuring","scoring"]
    guardrails_confidence_threshold  = {"acc_rej":0.8}
    column_confidence_threshold      = {"all_field":0.8}

Threshold guardrails yang dipakai tercatat di laporan guardrails (`document.threshold`).
`column_confidence_threshold` ikut disimpan bersama job, jadi `GET /v1/extract-ocr/{request_id}` menjawab dengan
`confidence` yang sama dengan jawaban `POST`-nya.

**Memilih service: `pipeline_name_sequence`.** Isinya nama service yang dijalankan, berurutan:
`guardrails`, `extraction`, `structuring`, `scoring`. Kirim sebagai field form berulang
(`pipeline_name_sequence=guardrails`, `pipeline_name_sequence=extraction`, ...) atau satu string
JSON array (`["guardrails", "extraction"]`). Tanpa field ini keempat service dijalankan, sama seperti
sebelumnya.

Aturannya: urutan tidak boleh diubah, `guardrails` boleh dilewati dari depan, dan service di
belakang boleh dipotong, tetapi service di tengah tidak boleh dilewati (setiap service setelah
guardrails butuh hasil service sebelumnya).

| `pipeline_name_sequence` | Boleh? | Isi `data` saat selesai |
|---|---|---|
| (tidak dikirim) atau `[guardrails, extraction, structuring, scoring]` | ya | `{nomor_npwp, nama}` seperti biasa |
| `[guardrails, extraction, structuring]` | ya | hasil structuring apa adanya: `{fields, flag, flag_reason, reject_reason, ...}` |
| `[guardrails, extraction]` | ya | hasil OCR apa adanya: `{full_text, blocks, ...}` |
| `[guardrails]` | ya | laporan guardrails apa adanya: `{passed, reason, document, pages}` |
| `[extraction, structuring, scoring]`, `[extraction, structuring]`, `[extraction]` | ya: guardrails dilewati (lihat di bawah) | seperti baris yang sama di atas |
| `[extraction, scoring]`, `[guardrails, structuring]` | tidak: ada yang dilewati di tengah | **422** `INVALID_PIPELINE_SEQUENCE` |
| `[structuring, scoring]`, `[scoring]` | tidak: tidak ada input untuk service pertama | **422** `INVALID_PIPELINE_SEQUENCE` |
| urutan terbalik, nama dobel, nama tidak dikenal (mis. `ekstraksi`) | tidak | **422** `INVALID_PIPELINE_SEQUENCE` |

Selain `data`, bentuk jawabannya tetap sama (`status_code`, `guardrails`, `pipeline_last_stage`,
dan seterusnya), begitu juga kode 200 / 202 / 400 / 422-nya. Service terakhir di urutan mengakhiri
request: hasilnya jadi `data`, dan tidak ada yang diteruskan ke service berikutnya. Aturan structuring
tetap bisa menolak (400) selama `structuring` ada di urutan. Request `[guardrails]` saja tidak
menjalankan tahap mana pun; putusannya disimpan, dan `GET /v1/extract-ocr/{request_id}` menjawabnya sama
dengan jawaban POST-nya (200 dengan laporan guardrails sebagai `data`).

**Melewati guardrails.** Urutan tanpa `guardrails` membuat dokumen tidak dinilai model guardrails dan
langsung masuk pipeline. Keputusannya sepenuhnya di kalian: kami tidak punya pengaturan yang
menolaknya. OCR (`extraction`) tidak bisa dilewati. Yang tetap berlaku:

- pengecekan file: tipe, 2,5 MB (413), dan 2 halaman (400);
- aturan structuring: dokumen blur / blank, kode wilayah salah, dan seterusnya tetap dijawab 400
  `DOWNSTREAM_VALIDATION_ERROR` dengan `guardrails: 1`. Nilai `guardrails: 1` dalam kasus ini hanya
  bisa berasal dari aturan structuring.

Model guardrails hanya bisa dilewati utuh: model itu hanya menilai diterima / ditolak, tanpa
pengecekan terpisah seperti blur atau terpotong. Trust model bekerja tanpa probabilitas guardrails
(input itu diisi seperti data yang hilang), jadi confidence bisa sedikit berbeda.

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
      "pipeline_last_stage": null,
      "guardrails": 0
    }

- `nama` = nama wajib pajak, atau nama badan pada kartu perusahaan.
- `confidence` = `1` kalau trust model ML memberi probabilitas benar minimal threshold field itu
  (`column_confidence_threshold`, kalau tidak dikirim `FIELD_CONFIDENCE_THRESHOLD`, default 0.5), `0` kalau
  di bawahnya atau field tidak ditemukan.
- Flag dari aturan extraction ML engineer **tidak** ada di `data`: flag itu internal, masuk sebagai
  input trust model. Dari 11 flag, hanya 2 yang ditoleransi (nama satu kata, huruf di nomor NPWP):
  nilainya tetap dikembalikan dan confidence-nya sudah memperhitungkan flag itu. Sembilan lainnya
  menolak dokumen dengan 400 (lihat di bawah).
- Pencocokan nama fuzzy terhadap nama nasabah (`refno`) dilakukan di sisi Orkestrasi, sesuai
  keputusan ML engineer; service ini tidak memakainya.

Belum selesai saat waktu tunggu habis, **202**; hasil menyusul lewat callback, atau baca dengan
`GET /v1/extract-ocr/{request_id}` (di bawah):

    {"status_code": 202, "status_desc": "Accepted", "message": "OCR job accepted; still processing",
     "data": null, "errors": null, "request_id": "REQ_001", "pipeline_last_stage": null,
     "guardrails": null}

Ditolak model guardrails, **400**; tidak ada yang jalan dan tidak ada callback:

    {"status_code": 400, "status_desc": "Bad Request",
     "message": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.99)",
     "data": null, "errors": "DOWNSTREAM_VALIDATION_ERROR", "request_id": "REQ_001",
     "pipeline_last_stage": "guardrails", "guardrails": 1}

Ditolak aturan structuring ML engineer (23 Sep 2026), juga **400** dengan bentuk yang sama;
`message` adalah alasan penolakan pertama dari aturan itu (bukan alasan flag yang ditoleransi),
dalam bahasa Indonesia, dan bisa langsung ditampilkan ke pengguna. Penolakan terjadi di tahap structuring, jadi scoring tidak jalan:

    {"status_code": 400, "status_desc": "Bad Request",
     "message": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
     "data": null, "errors": "DOWNSTREAM_VALIDATION_ERROR", "request_id": "REQ_001",
     "pipeline_last_stage": "structuring", "guardrails": 1}

Alasan yang menolak: dokumen blur / blank, bukan format standar NPWP, dokumen lain terdeteksi,
screenshot cek NPWP online, jumlah halaman melebihi batas, serta kode provinsi, kecamatan, tanggal
lahir, atau KPP pada nomor yang tidak valid. Kalau dokumen punya beberapa alasan, yang dipakai
adalah alasan penolakan pertama menurut urutan prioritas aturan ML. Kalau penolakan terjadi setelah
jawaban 202, keadaan akhirnya adalah callback `STRUCTURING` `FAILED` dengan alasan itu sebagai
`error_message`, dan baris 400 / `DOWNSTREAM_VALIDATION_ERROR` di tabel Orkestrasi.

Tahap pipeline gagal dalam waktu tunggu, **422**; request berakhir di sini, sama seperti
callback `FAILED`. `errors` menyebut tahapnya, `message` alasannya:

    {"status_code": 422, "status_desc": "Unprocessable Entity",
     "message": "extraction OCR model is unavailable",
     "data": null, "errors": "OCR_FAILED", "request_id": "REQ_001",
     "pipeline_last_stage": "extraction", "guardrails": 0}

Error lain (envelope standar). `errors` selalu kode yang stabil; `message` teks yang bisa berubah:

| Kode | `errors` | Arti | Pipeline jalan? |
|---|---|---|---|
| 400 | `UNSUPPORTED_DOCUMENT_TYPE` | `document_type` bukan `npwp` | tidak |
| 400 | `EMPTY_FILE` | file kosong | tidak |
| 400 | `UNSUPPORTED_FILE_TYPE` | tipe file bukan JPEG / PNG / PDF | tidak |
| 400 | `UNREADABLE_FILE` | file tidak bisa dibaca (PDF rusak / tanpa halaman, atau gambar yang tidak terbaca model guardrails; `pipeline_last_stage: guardrails`) | tidak |
| 400 | `TOO_MANY_PAGES` | PDF lebih dari 2 halaman (`Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP`) | tidak |
| 400 | `INVALID_FILE_SOURCE` | `file` dan `file_url` dua-duanya, atau tidak ada | tidak |
| 400 | `FILE_URL_REJECTED` | host `file_url` tidak diizinkan, atau tidak bisa diunduh | tidak |
| 413 | `FILE_TOO_LARGE` | file lebih dari 2,5 MB (`Ukuran dokumen melebihi batas 2,5 MB, pastikan hanya mengunggah dokumen NPWP`) | tidak |
| 401 | `UNAUTHORIZED` | `X-API-Key` salah atau tidak ada | tidak |
| 422 | `INVALID_PIPELINE_SEQUENCE` | `pipeline_name_sequence` melanggar aturan urutan; `message` menyebut alasannya | tidak |
| 422 | `INVALID_THRESHOLD` | `guardrails_confidence_threshold` / `column_confidence_threshold` tidak bisa dibaca; `message` menyebut alasannya | tidak |
| 422 | `VALIDATION_ERROR` | field wajib tidak dikirim | tidak |
| 500 | `DOWNSTREAM_SERVER_ERROR` | service internal (guardrails / extraction) menjawab tidak sesuai kontrak | tidak; kirim ulang aman |
| 500 | `INTERNAL_SERVER_ERROR` | kesalahan tak terduga di orchestrator | tidak; kirim ulang aman |
| 503 / 504 | `DOWNSTREAM_UNAVAILABLE` / `DOWNSTREAM_TIMEOUT` | guardrails atau modelnya tidak terjangkau / tidak menjawab (tidak dicoba ulang), atau tahap OCR tidak terjangkau / tidak menjawab (sudah dicoba ulang 3 kali) | tidak; kirim ulang aman |

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

| Keadaan | HTTP | isi | `errors` |
|---|---|---|---|
| selesai | 200 | `data` | null |
| masih berjalan | 202 | – | null |
| ditolak aturan structuring | 400 | `guardrails: 1`, `pipeline_last_stage: structuring` | `DOWNSTREAM_VALIDATION_ERROR` |
| ditolak model guardrails | 400 | `guardrails: 1`, `pipeline_last_stage: guardrails` | `DOWNSTREAM_VALIDATION_ERROR` |
| `[guardrails]` saja, lolos | 200 | laporan guardrails sebagai `data` | null |
| satu tahap gagal | 422 | `pipeline_last_stage`: tahap yang gagal | `OCR_FAILED` / `STRUCTURING_FAILED` / `SCORING_FAILED` |
| tidak dikenal | 404 | – | `REQUEST_ID_NOT_FOUND` |

- Request yang tidak punya job di tahap mana pun dijawab dari putusan guardrails terakhirnya
  (`nilam_guardrails_results`), sama seperti jawaban `POST`-nya: 400 kalau ditolak model guardrails, 200 dengan
  laporan guardrails kalau `[guardrails]` satu-satunya service-nya.
- **404** berarti tidak ada job dan tidak ada putusan guardrails yang tersimpan: ditolak sebelum dinilai
  (file, `pipeline_name_sequence`, threshold), lolos guardrails tapi serah terima ke extraction gagal
  (`POST`-nya dijawab 5xx), `POST`-nya masih dinilai guardrails, atau putusannya tidak sempat disimpan
  (database mati; penyimpanannya best-effort).
- **503 / 504** berarti salah satu tahap tidak bisa dibaca saat itu; coba lagi.
- **Batasan:** serah terima antar tahap yang gagal permanen (setelah semua retry, atau menjadi dead
  letter di outbox kami) hanya tercatat di callback `FAILED` dan di tabel kalian; endpoint ini tetap
  menjawab 202 untuk request itu. Untuk keadaan final, callback dan tabel kalian yang berlaku.

## 5. Service extraction (port 8030)

Tahap OCR. Backend OCR-nya PaddleOCR yang berjalan di VM terpisah; service ini yang
memanggilnya.

`POST /v1/extraction/jobs` dipanggil oleh orchestrator, bukan oleh kalian. Setelah dijawab
202, di background: dokumen dibaca, OCR dijalankan, hasil disimpan, lalu job diserahkan ke
structuring, yang kemudian menyerahkan ke scoring.

### GET /v1/extraction/jobs/{request_id} — status tahap OCR (internal)

Dibaca orchestrator untuk `GET /v1/extract-ocr/{request_id}`; dicantumkan sebagai latar.

    {"data": {"request_id": "REQ_001", "stage": "OCR", "status": "DONE",
              "error_message": null, "result": { … hasil OCR mentah … },
              "created_at": "...", "updated_at": "..."}}

`status` bernilai `PROCESSING`, `DONE`, atau `FAILED`. `404` berarti tahap ini belum
pernah menerima job dengan `request_id` tersebut.

### POST /v1/extraction/extract — OCR mentah, sinkron (internal)

Menjalankan OCR saja dan langsung mengembalikan hasilnya. Tidak membuat job, tidak
mengirim callback, tidak menyentuh tahap lain. Untuk debugging kami.

## 6. Service structuring (port 8032) dan scoring (port 8033)

Keduanya dipanggil berantai oleh service sebelumnya dan internal; kalian tidak memanggilnya.

| Service | Endpoint | Siapa yang memanggil |
|---|---|---|
| guardrails | `POST /v1/guardrails/check` | orchestrator, untuk tiap dokumen |
| extraction | `POST /v1/extraction/jobs` | orchestrator, otomatis |
| structuring | `POST /v1/structuring/jobs` | extraction, otomatis |
| structuring | `GET /v1/structuring/jobs/{request_id}` | orchestrator (status) |
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
(`ORCHESTRATION_CALLBACK_FORMAT=result`). Body dan aturan kirim ulang mengikuti kontrak mereka
"Callback Hasil OCR" (2 Okt 2026) dan jawaban mereka tanggal 5 Okt 2026:

    POST http://ocr-orchestration.ocr-dev.svc.cluster.local/v1/ocr-callback
    X-Callback-Key: <ORCHESTRATION_CALLBACK_KEY>

Callback dikirim oleh tahap-tahap pipeline sendiri (scoring, atau tahap yang berhenti), bukan oleh
orchestrator. Tahap tidak tahu apakah orchestrator menjawab 200 atau 202 (keduanya bisa terjadi di
sekitar `PIPELINE_WAIT_SECONDS`), jadi callback selalu dikirim; Orkestrasi menjawab callback untuk
request yang sudah dijawab 200 dengan `200 "Result already recorded"`.

Selesai: `result` sama persis dengan `data` jawaban 200 `extract-ocr` untuk request yang sama
(dengan urutan penuh: `nomor_npwp` dan `nama` dengan confidence 0/1 hasil threshold request itu;
`pipeline_name_sequence` yang berakhir sebelum `scoring`: hasil service terakhirnya apa adanya):

    {
      "request_id": "OCR_9cb01af2-493d-446d-b191-af120333f6d0",
      "status": "completed",
      "result": {
        "nomor_npwp": {"value": "09.254.294.3-407.000", "confidence": 1},
        "nama":       {"value": "BUDI SANTOSO",         "confidence": 1}
      },
      "guardrails": 0
    }

Ditolak aturan structuring setelah 202. Orkestrasi mengenali penolakan dari `result` null dengan
`guardrails` 1, dan meneruskan `message` ke kliennya:

    {
      "request_id": "OCR_9cb01af2-493d-446d-b191-af120333f6d0",
      "status": "completed",
      "result": null,
      "guardrails": 1,
      "message": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
      "error_code": "DOWNSTREAM_VALIDATION_ERROR"
    }

Gagal (dikirim oleh tahap yang gagal):

    {
      "request_id": "OCR_9cb01af2-493d-446d-b191-af120333f6d0",
      "status": "failed",
      "error_code": "STRUCTURING_FAILED",
      "message": "Internal error in STRUCTURING stage"
    }

`error_code` (`OCR_FAILED` / `STRUCTURING_FAILED` / `SCORING_FAILED`, semuanya 422 di jawaban
sinkron) ada di luar kontrak: Orkestrasi mengabaikannya untuk sekarang dan berencana meneruskannya
seperti jalur sinkron. Dokumen yang ditolak model guardrails dan request `[guardrails]` saja tidak
mendapat callback: keduanya selalu dijawab langsung di `extract-ocr` orchestrator.

Kirim ulang: 5xx, timeout, dan koneksi gagal dicoba ulang dengan backoff selama
`ORCHESTRATION_CALLBACK_MAX_AGE_SECONDS` (600 detik; Orkestrasi berhenti menunggu sejumlah detik
setelah 202 dan menjawab callback sesudahnya `409 RESULT_CONFLICT`). `409 RESULT_NOT_READY` (callback
tiba sebelum Orkestrasi mencatat 202-nya) dicoba ulang tiap 1,5 detik, maksimal 5 kali. 4xx lain
tidak dikirim ulang.

Saklar: `ORCHESTRATION_CALLBACK_ENABLED=false` (Helm `orchestration.callbackEnabled: false`) mematikan
callback walau `ORCHESTRATION_URL` terisi. Harus mengikuti mode NPWP di Orkestrasi: di mode `poll`
mereka menjawab setiap callback `409 CALLBACK_NOT_EXPECTED` dan membaca
`GET /v1/extract-ocr/{request_id}`.

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

1. **`result` null untuk OCR dan STRUCTURING, kecuali tahap itu yang terakhir.** Hasil antara
   tidak dibuka ke luar; keadaan request dibaca lewat `GET /v1/extract-ocr/{request_id}` di
   orchestrator. Kalau `pipeline_name_sequence` berakhir di tahap itu, callback `DONE`-nya membawa
   `"final": true` dan `result` berisi hasilnya apa adanya.
2. **Hanya callback dengan `"final": true` yang membawa hasil akhir.** Dengan urutan penuh, itu
   callback SCORING (juga bertanda `"final": true`).

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

**Endpoint threshold guardrails (kalian yang menyediakan).** Ambang model guardrails disimpan di
sisi kalian supaya bisa diubah tanpa deploy di sisi kami. Service guardrails kami membacanya
dengan:

    GET <URL kalian>/v1/thresholds/guardrails        (path bisa disesuaikan)
    X-API-Key: <opsional>

    {"threshold": 0.6}

- `threshold`: angka di antara 0 dan 1 (tidak termasuk 0 dan 1).

Ambang ini hanya satu, pada probabilitas **accept** dari model: dokumen lolos kalau probabilitas
accept **≥** `threshold`, dan ditolak kalau di bawahnya.

Contoh, keluaran model probabilitas accept 0,7 (berarti reject 0,3):

| Jawaban endpoint | Hasil |
|---|---|
| `{"threshold": 0.6}` | lolos (0,7 ≥ 0,6) |
| `{"threshold": 0.7}` | lolos (0,7 ≥ 0,7) |
| `{"threshold": 0.8}` | ditolak (0,7 < 0,8) |

Nilainya kami simpan 60 detik per pod, jadi perubahan di sisi kalian berlaku paling lambat
semenit kemudian. Kalau endpoint mati atau jawabannya tidak valid, kami tetap memakai ambang
terakhir yang pernah kalian berikan; kalau belum pernah ada, dipakai default (0,5).
Ambang yang dipakai tercatat di laporan guardrails (`document.threshold`). Endpoint ini belum
ada; sampai ada, kami memakai default.

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

`extract-ocr` menambah `pipeline_last_stage` dan `guardrails` ke amplop ini
(bagian 4).

## 11. Kontrak lama (sinkron): sudah dihapus

Sejak 24 September 2026, endpoint kontrak lama di service extraction (port 8030) sudah
dihapus:

    POST /v1/generate-request-id
    POST /v1/extract-ocr                  (versi extraction, port 8030)
    GET  /v1/get-ocr-result/{request_id}

Sejak orchestrator ada (24 September 2026), `POST /v1/extract-ocr` di guardrails (port 8031)
juga sudah dihapus. Pakai `POST /v1/extract-ocr` di orchestrator (port 8034, bagian 4), dan
`GET /v1/extract-ocr/{request_id}` untuk keadaan sebuah request (bagian 8).

## 12. Database

Pipeline menyimpan job dan hasil tiap tahap ke PostgreSQL yang sama dengan yang kalian
pakai, database `bribrain_ocr_nilam`. Semua tabel kami ada di schema `nilam_ocr_npwp` dan namanya
berawalan `nilam_` (sejak migrasi `0013`; sebelumnya `ocr_pipeline_npwp` tanpa awalan, dan sebelum itu
`public`), supaya terpisah dari tabel tim lain di `public`.

| Tabel | Isi |
|---|---|
| `nilam_ocr_npwp.nilam_ocr_jobs`, `nilam_ocr_npwp.nilam_ocr_results` | status dan hasil OCR mentah |
| `nilam_ocr_npwp.nilam_structuring_jobs`, `nilam_ocr_npwp.nilam_structuring_results` | field hasil penataan |
| `nilam_ocr_npwp.nilam_scoring_jobs`, `nilam_ocr_npwp.nilam_scoring_results` | confidence akhir |
| `nilam_ocr_npwp.nilam_guardrails_results` | setiap putusan guardrails, termasuk yang ditolak |

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
