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
memanggil `http://guardrails:8031` langsung, dan melapor ke tracker lewat
`http://host.docker.internal:<PORT>`. Container diberi nama `nilam-lt-<run>` dan dihapus setelah
selesai. Env backend tracker yang terkait: `K6_IMAGE`, `K6_NETWORK`, `K6_TARGET`, `K6_TRACKER`.

Gambar contoh diambil dari `images/` (tidak ikut repo; taruh beberapa foto NPWP di sana, dipakai
bergiliran). Ringkasan k6 tiap run ditulis ke `out/<run>.json`.

## Menjalankan manual

    docker run --rm --network ocr_default --add-host host.docker.internal:host-gateway \
      -v "$PWD/tools/load-tester/k6:/scripts:ro" -v "$PWD/tools/load-tester/images:/images:ro" \
      -v "$PWD/tools/load-tester/out:/out" \
      -e RUN_ID=coba1 -e TARGET=http://guardrails:8031 -e TRACKER=http://host.docker.internal:8090 \
      -e RATE=2 -e DURATION=60s -e MODE=constant -e IMAGES=npwp1.jpg,npwp2.jpg \
      grafana/k6 run /scripts/extract-ocr.js

Env skrip: `RUN_ID`, `TARGET`, `TRACKER` (kosong = tanpa lapor), `RATE`, `DURATION`, `MODE`
(`constant` | `ramp`), `IMAGES`, `WAIT_SECONDS`, `API_KEY`.

## Membaca hasilnya

- **200 / 202 / 4xx / 5xx / timeout**: campuran jawaban pintu masuk. Bergesernya 200 ke 202 adalah
  tanda pertama jenuh; 5xx dan timeout tanda kedua.
- **p50 / p95 extract-ocr**: lama koneksi ditahan guardrails. Mendekati `PIPELINE_WAIT_SECONDS`
  berarti hampir semua jawaban 202.
- **Tuntas (SCORING DONE)** dan **p95 end-to-end**: kapasitas sesungguhnya. Bandingkan dengan jumlah
  dikirim; selisihnya masih antre atau gagal.
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
