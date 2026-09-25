# tools/load-tester — mengukur laju request yang sanggup dilayani pipeline

Pembangkit beban memakai [k6](https://k6.io) yang dijalankan sebagai container Docker di laptop.
Angka yang dicari bukan satu "RPS maksimum", melainkan dua hal:

1. **Pintu masuk**: pada laju berapa jawaban `/v1/extract-ocr` masih 200 (pipeline selesai di dalam
   `PIPELINE_WAIT_SECONDS`), kapan bergeser ke 202, dan kapan mulai 5xx atau timeout.
2. **End-to-end**: berapa dokumen per menit yang benar-benar tuntas (callback SCORING DONE) dan p95
   waktunya dari submit sampai tuntas.

Keduanya dihitung oleh tracker ([../tracker](../tracker)) di menu **Load testing**: k6 melaporkan
tiap sampel ke `POST /api/loadtest/{run}/samples`, dan callback tahap untuk request berprefiks
`LT_` dihitung terpisah, tidak masuk daftar "Request terakhir".

## Menjalankan dari tracker

    tools/tracker/run.sh --stack     # stack lokal + tracker
    # buka http://127.0.0.1:5173 -> menu "Load testing" -> atur laju & durasi -> Mulai

Tracker menjalankan `docker run grafana/k6` di network compose (`ocr_default`) sehingga k6
memanggil `http://orchestrator:8034` langsung, dan melapor ke tracker lewat
`http://host.docker.internal:<PORT>`. Container diberi nama `nilam-lt-<run>` dan dihapus setelah
selesai. Env backend tracker yang terkait: `K6_IMAGE`, `K6_NETWORK`, `K6_TARGET`, `K6_TRACKER`.

File uji dipakai bergiliran dan berasal dari dua folder:

- `images/`: contoh yang ikut repo (`npwp1.jpg`, `npwp2.jpg`). Tidak pernah dihapus otomatis.
- `assets/`: semua unggahan dari tracker. Folder ini di-`.gitignore` karena isinya bisa dokumen
  nasabah asli, dan **dihapus otomatis begitu run yang memakainya berakhir**, baik selesai, gagal,
  maupun dihentikan. File yang diunggah tapi belum dipakai run tetap ada sampai dihapus.

Unggahan dikelola dari sidebar menu **Load testing**: unggah (klik atau seret, JPG/PNG/PDF, maks.
20 MB per file), lihat, hapus, dan centang file mana yang dipakai run berikutnya. Karena dihapus
setelah run, file yang sama perlu diunggah ulang untuk run berikutnya. Nama file dirapikan
(karakter selain huruf, angka, `._-` jadi `_`) dan tidak pernah menimpa file di kedua folder
(diberi akhiran `-1`, `-2`, ...). File yang dipakai run yang sedang berjalan tidak bisa dihapus.
File di atas 2,5 MB (default `MAX_UPLOAD_BYTES` service) ditandai karena akan dijawab 413; itu
sengaja dibiarkan bisa diunggah untuk menguji jalur penolakan. Endpoint-nya:
`POST /api/loadtest/images` (multipart, field `files`), `GET` dan `DELETE /api/loadtest/images/{nama}`,
dan `DELETE /api/loadtest/assets` untuk menghapus semua unggahan yang tertinggal (mis. setelah backend
mati di tengah run; ditolak 409 selama ada run berjalan).
Ringkasan k6 tiap run ditulis ke `out/<run>.json`.

## Menjalankan manual

    docker run --rm --network ocr_default --add-host host.docker.internal:host-gateway \
      -v "$PWD/tools/load-tester/k6:/scripts:ro" -v "$PWD/tools/load-tester/images:/images:ro" \
      -v "$PWD/tools/load-tester/out:/out" \
      -e RUN_ID=coba1 -e TARGET=http://orchestrator:8034 -e TRACKER=http://host.docker.internal:8090 \
      -e RATE=2 -e DURATION=60s -e MODE=constant -e IMAGES=npwp1.jpg,npwp2.jpg \
      grafana/k6 run /scripts/extract-ocr.js

Env skrip: `RUN_ID`, `TARGET`, `TRACKER` (kosong = tanpa lapor), `RATE`, `DURATION`, `MODE`
(`constant` | `ramp`), `IMAGES`, `WAIT_SECONDS`, `API_KEY`, `ENDPOINT` (default `/v1/extract-ocr`).

Load test ke **dev** (GKE) tanpa mengotori data Orkestrasi: port-forward orchestrator lalu tembak kembaran
`-test`-nya, yang menulis ke tabel `testing_*` dan tidak mengirim callback (lihat README utama,
"Endpoint Testing"):

    kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp-orchestrator 8034:8034
    docker run --rm --add-host host.docker.internal:host-gateway \
      -v "$PWD/tools/load-tester/k6:/scripts:ro" -v "$PWD/tools/load-tester/images:/images:ro" \
      -e ENDPOINT=/v1/extract-ocr-test -e TARGET=http://host.docker.internal:8034 -e API_KEY="$API_KEY" \
      -e RUN_ID=burst1 -e RATE=5 -e DURATION=60s -e IMAGES=npwp1.jpg \
      grafana/k6 run /scripts/extract-ocr.js

Di endpoint `-test`, `request_id` dibuat orchestrator (`TEST_<RUN_ID>_<uuid>`) dari field `run_id` yang dikirim
skrip; `LT_...` dari skrip diabaikan. Semua request satu run: `request_id LIKE 'TEST_burst1_%'`.

## Membaca hasilnya

- **200 / ditolak / 202 / 4xx / 5xx / timeout**: campuran jawaban pintu masuk. Bergesernya 200 ke 202
  adalah tanda pertama jenuh; 5xx dan timeout tanda kedua. Dokumen yang ditolak dijawab 400
  `DOWNSTREAM_VALIDATION_ERROR`, dan dihitung sendiri sebagai **ditolak**, bukan 4xx. Banyak penolakan berarti file ujinya yang
  ditolak model atau aturan ML (lihat alasannya di tracker), bukan tanda beban.
- **p50 / p95 extract-ocr**: lama koneksi ditahan orchestrator. Mendekati `PIPELINE_WAIT_SECONDS`
  berarti hampir semua jawaban 202.
- **Tuntas (SCORING DONE)** dan **p95 end-to-end**: kapasitas sesungguhnya. Bandingkan dengan jumlah
  dikirim; selisihnya masih antre, gagal, atau ditolak.
- **dropped_iterations** (dari k6): k6 tidak sanggup mempertahankan laju karena VU habis, artinya
  latensi sudah lebih panjang dari yang diperkirakan; laju yang tercapai lebih rendah dari yang
  diminta.
- **Backlog outbox**: kalau `PIPELINE_OUTBOX=true`, pesan yang menumpuk di `pending` menunjukkan
  relay atau penerima callback yang tertinggal, bukan tahap OCR.

## Batasan pengujian lokal

Model OCR dipanggil lewat tunnel SSH ke VM dan container berjalan di laptop, jadi angka absolutnya
tidak mewakili GKE. Yang bermakna dari pengujian lokal adalah bentuk kurvanya (kapan 200 menjadi
202, apakah ada 5xx, apakah outbox menumpuk) dan validasi alat ini sendiri sebelum diarahkan ke
cluster, di mana k6 sebaiknya dijalankan sebagai Job di dalam cluster, bukan lewat port-forward.

## Membersihkan data uji

Request uji berprefiks `LT_<run>_`. Tombol **Hapus run** di tracker menghapus catatan run di Redis
dan, kalau pemantau database hidup, baris `LT_<run>_%` dari `*_jobs`, `*_results`,
`pipeline_outbox`, dan `orchestration_extract_ocr`.
