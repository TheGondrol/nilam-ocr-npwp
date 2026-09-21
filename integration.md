# Integrasi service OCR NPWP (nilam) di GKE

Dokumen untuk tim Orkestrasi/Gateway. Isinya dokumentasi keempat service yang sudah
berjalan di cluster, cara memanggilnya, apa yang dikirim balik, dan satu hal yang kami
butuhkan dari kalian supaya callback nyambung.

Semua angka dan contoh respons di dokumen ini diambil dari request sungguhan ke service
yang sedang berjalan, bukan karangan. Yang belum terbukti ditandai eksplisit.

## 1. Status

Sudah terbukti jalan di cluster:

- keempat service `/ready` menjawab 200, koneksi database normal
- satu kartu NPWP asli lewat rantai penuh dalam ±6 detik, hasil
  `npwp_confidence` 0.9926 dan `name_confidence` 0.9953
- job tercatat `DONE` di ketiga tabel tahap

Belum pernah diuji: **callback ke service kalian**, karena alamatnya belum kami punya.
Itu satu-satunya potongan yang tersisa. Lihat bagian 9.

## 2. Akses

| Item | Nilai |
|---|---|
| Cluster | `gc-ddb-dev-gke-cluster-01` |
| Project | `ddb-kubecluster-dev-01`, region `asia-southeast2` |
| Namespace | `nilam-ocr-npwp` |
| Service (ClusterIP) | `nilam-ocr-npwp` |

Satu pod berisi empat container; satu Service membuka keempat portnya.

| Tahap | Port | Base URL dari namespace lain |
|---|---|---|
| ekstraksi | 8030 | `http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8030` |
| guardrails | 8031 | `http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8031` |
| structuring | 8032 | `http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8032` |
| scoring | 8033 | `http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8033` |

**Autentikasi.** Semua endpoint kecuali `/health` dan `/ready` memerlukan header:

    X-API-Key: changeme

Nilai itu sementara dan akan diganti sebelum dipakai serius; kami kabari kalau berubah.

**Jaringan.** Tidak ada NetworkPolicy di namespace kami, jadi dari sisi kami tidak ada
yang perlu di-whitelist. Kami belum bisa menguji panggilan dari namespace kalian.

**Dari laptop** (untuk coba-coba), semua port bisa di-forward sekaligus:

    kubectl -n nilam-ocr-npwp port-forward deploy/nilam-ocr-npwp 8030:8030 8031:8031 8032:8032 8033:8033

## 3. Alur yang direkomendasikan

Hanya dua panggilan dari sisi kalian. Sisanya datang sebagai callback.

    1. POST :8031/v1/guardrails/check     sinkron, tunggu jawabannya
         data.passed == false  -> berhenti, balas 422 ke client pakai data.reason
         data.passed == true   -> lanjut
    2. POST :8030/v1/ekstraksi/jobs       dijawab 202 seketika
         lalu jalan sendiri: OCR -> structuring -> scoring
    3. kalian menerima 3 callback: OCR, STRUCTURING, SCORING
         hasil akhir ada di callback SCORING

`request_id` dibuat oleh kalian dan menjadi kunci di semua tahap. Bebas formatnya,
string; contoh yang kami pakai saat uji: `REQ_a0e0fd34ed7a`.

Kalian **tidak perlu** memanggil structuring dan scoring sendiri. Kedua service itu
dipanggil berantai oleh service sebelumnya. Dokumentasinya tetap kami sertakan di
bagian 5 dan 6 supaya jelas apa yang terjadi dan bisa dipakai untuk debugging.

## 4. Service guardrails (port 8031)

Menilai layak atau tidaknya dokumen sebelum masuk OCR. Sinkron, tidak membuat job,
tidak mengirim callback. Model EfficientNet berjalan di dalam container.

### POST /v1/guardrails/check

Kirim `request_id` plus dokumen sebagai `file` (multipart), **atau** sebagai `file_url`
supaya service ini yang mengunduh. Salah satu saja, tidak boleh dua-duanya.

    curl -X POST http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8031/v1/guardrails/check \
      -H "X-API-Key: changeme" \
      -F "request_id=REQ_001" \
      -F "file=@npwp.jpg"

Format diterima: JPEG, PNG, PDF. Maksimal 5 MB. PDF dinilai per halaman.

Selalu menjawab **200** dengan laporan, termasuk saat dokumen ditolak. Yang menentukan
lanjut atau tidak adalah `data.passed`, bukan kode HTTP.

Contoh dokumen lolos:

    {
      "status_code": 200,
      "status_desc": "OK",
      "message": "OK",
      "errors": null,
      "request_id": "REQ_001",
      "data": {
        "passed": true,
        "reason": null,
        "document": {"verdict": "approve", "confidence": 0.9663,
                     "n_pages": 1, "n_approve": 1, "n_reject": 0},
        "pages": [{"page_index": 0, "proba_approve": 0.9663,
                   "proba_reject": 0.0337, "verdict": "approve"}]
      }
    }

Contoh dokumen ditolak (ini hasil sungguhan dari gambar polos):

    "data": {
      "passed": false,
      "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.99)",
      "document": {"verdict": "reject", "confidence": 0.9851,
                   "n_pages": 1, "n_approve": 0, "n_reject": 1},
      "pages": [{"page_index": 0, "proba_approve": 0.0149,
                 "proba_reject": 0.9851, "verdict": "reject"}]
    }

Kalau `passed` false, hentikan di situ dan pakai `data.reason` sebagai alasan ke client.

**Penting.** Seluruh isi `data` harus kalian simpan dan teruskan apa adanya ke langkah
berikutnya. Isinya dipakai lagi oleh scoring di tahap akhir sebagai salah satu masukan
trust model.

## 5. Service ekstraksi (port 8030)

OCR sekaligus pintu masuk rantai asinkron. Backend OCR-nya PaddleOCR yang berjalan di
VM terpisah; service ini yang memanggilnya.

### POST /v1/ekstraksi/jobs — mulai pipeline

    curl -X POST http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8030/v1/ekstraksi/jobs \
      -H "X-API-Key: changeme" \
      -F "request_id=REQ_001" \
      -F "document_type=npwp" \
      -F 'guardrails={"passed":true,"reason":null,"document":{...},"pages":[...]}' \
      -F "file=@npwp.jpg"

| Field | Wajib | Keterangan |
|---|---|---|
| `request_id` | ya | dibuat oleh kalian |
| `document_type` | ya | saat ini hanya `npwp` |
| `guardrails` | ya | isi `data` dari langkah guardrails, sebagai JSON string |
| `file` / `file_url` | salah satu | dokumen yang sama dengan yang dinilai guardrails |

Dijawab **202 seketika**, OCR belum jalan:

    {"status_code": 202, "status_desc": "Accepted", "message": "OK", "errors": null,
     "request_id": "REQ_001",
     "data": {"request_id": "REQ_001", "stage": "OCR",
              "status": "PROCESSING", "duplicate": false}}

Setelah itu, di background: dokumen dibaca, OCR dijalankan, hasil disimpan, callback
`OCR` dikirim, lalu job diserahkan ke structuring, yang kemudian menyerahkan ke scoring.

**Idempoten.** `request_id` yang sama dikirim ulang tetap dijawab 202 dengan
`duplicate: true` dan OCR tidak dijalankan dua kali, kecuali percobaan sebelumnya
berstatus `FAILED`; dalam hal itu dijalankan ulang.

**Catatan `file_url`.** URL diunduh di background, jadi presigned URL harus hidup lebih
lama daripada antrean terburuk. URL kedaluwarsa atau tidak terjangkau menjadi job
`FAILED`, bukan error 4xx, karena 202 sudah terlanjur dijawab.

### GET /v1/ekstraksi/jobs/{request_id} — status tahap OCR

    {"data": {"request_id": "REQ_001", "stage": "OCR", "status": "DONE",
              "error_message": null, "result": { … hasil OCR mentah … },
              "created_at": "...", "updated_at": "..."}}

`status` bernilai `PROCESSING`, `DONE`, atau `FAILED`. `404` berarti tahap ini belum
pernah menerima job dengan `request_id` tersebut.

### POST /v1/ekstraksi/extract — OCR mentah, sinkron

Menjalankan OCR saja dan langsung mengembalikan hasilnya. Tidak membuat job, tidak
mengirim callback, tidak menyentuh tahap lain. Berguna untuk debugging.

## 6. Service structuring (port 8032) dan scoring (port 8033)

Keduanya dipanggil berantai oleh service sebelumnya. Kalian normalnya hanya memakai
endpoint `GET` di bawah, untuk rekonsiliasi atau debugging.

| Service | Endpoint | Siapa yang memanggil |
|---|---|---|
| structuring | `POST /v1/structuring/jobs` | ekstraksi, otomatis |
| structuring | `GET /v1/structuring/jobs/{request_id}` | kalian, bila perlu |
| structuring | `POST /v1/structuring/structure` | debugging, sinkron |
| scoring | `POST /v1/scoring/jobs` | structuring, otomatis |
| scoring | `GET /v1/scoring/jobs/{request_id}` | kalian, bila perlu |
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

Tiap tahap mem-POST ke `<ORCHESTRATION_URL><CALLBACK_PATH>`, dengan default path
`/v1/callbacks/stage`. Header `X-API-Key` ikut dikirim kalau kami diberi nilainya.

Body untuk OCR dan STRUCTURING:

    {
      "request_id": "REQ_001",
      "stage": "OCR",
      "status": "DONE",
      "result": null,
      "error_message": null
    }

Dua hal yang paling sering disalahpahami:

1. **`result` selalu null untuk OCR dan STRUCTURING.** Kalau butuh hasil antaranya,
   ambil sendiri lewat `GET /v1/<tahap>/jobs/{request_id}`.
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

    GET :8030/v1/ekstraksi/jobs/{request_id}
    GET :8032/v1/structuring/jobs/{request_id}
    GET :8033/v1/scoring/jobs/{request_id}

Masing-masing mengembalikan `status`, `error_message`, dan `result` tahap itu bila sudah
`DONE`. `404` berarti tahap tersebut belum menerima job.

Cara ini sudah kami pakai dan terbukti jalan: saat menguji dari laptop, kami menarik
status tiap tahap dengan endpoint ini karena pod tidak bisa memanggil balik laptop.

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
| 400 | file bermasalah (kosong, kebesaran, tipe tidak didukung) atau intake salah |
| 401 | `X-API-Key` salah atau tidak ada |
| 404 | `request_id` tidak dikenal di tahap itu |
| 422 | body atau field tidak valid |
| 502/503 | model atau service tujuan tidak bisa dihubungi |

Guardrails adalah pengecualian: dokumen ditolak tetap dijawab 200, penolakannya ada di
`data.passed`.

## 11. Kontrak lama (sinkron)

Kalau kalian belum siap pindah ke alur asinkron, kontrak lama masih hidup di service
ekstraksi dan akan kami pertahankan sampai kalian pindah:

    POST /v1/generate-request-id          buat request_id
    POST /v1/extract-ocr                  jalankan seluruh rantai sekaligus, sinkron
    GET  /v1/get-ocr-result/{request_id}  ambil hasilnya

Perbedaannya: seluruh rantai dijalankan dalam satu panggilan yang ditunggu sampai
selesai, tanpa callback. Satu `request_id` hanya boleh dikirim sekali.

Alur asinkron di bagian 3 yang kami sarankan, karena tidak menahan koneksi selama OCR
berjalan.

## 12. Database

Pipeline menyimpan job dan hasil tiap tahap ke PostgreSQL yang sama dengan yang kalian
pakai, database `bribrain_ocr_nilam`. Semua tabel kami ada di schema `public`, sesuai
permintaan supaya seragam.

| Tabel | Isi |
|---|---|
| `ocr_jobs`, `ocr_results` | status dan hasil OCR mentah |
| `structuring_jobs`, `structuring_results` | field hasil penataan |
| `scoring_jobs`, `scoring_results` | confidence akhir |
| `ocr_npwp_requests` | dipakai kontrak lama di bagian 11 |

Integrasi normal **tidak perlu menyentuh database ini**; semua yang dibutuhkan sudah ada
di callback dan endpoint status. Kami cantumkan supaya jelas tabel mana milik kami, dan
supaya kalian tahu tabel `orchestration_*` serta `auth_*` milik kalian tidak kami sentuh.

## 13. Spesifikasi lengkap

`api/gateway.openapi.yaml` di repo ini adalah spec OpenAPI gabungan keempat service,
lengkap dengan skema, contoh, dan webhook `stageCallback`. Itu sumber kebenaran paling
detail; dokumen ini ringkasannya.

Swagger UI tiap service juga hidup di `/docs`:

    kubectl -n nilam-ocr-npwp port-forward deploy/nilam-ocr-npwp 8030:8030
    # buka http://127.0.0.1:8030/docs

Spec per service ada di `services/<nama>/openapi.yaml`.
