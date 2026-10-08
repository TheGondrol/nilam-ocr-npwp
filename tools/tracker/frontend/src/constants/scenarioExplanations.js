/**
 * Penjelasan teknis mendalam untuk setiap skenario pengujian ketahanan (Chaos Engineering).
 * Menjelaskan mekanisme internal di balik layar, alasan status akhir (PASS/WARN/FAIL/DONE/FAILED),
 * dan rekomendasi arsitektur untuk sistem produksi Bank BRI.
 */
export const SCENARIO_EXPLANATIONS = {
  'crash-inline': {
    title: 'Extraction Mati Mendadak (SIGKILL) - Dokumen Upload Inline',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'WARN (Job FAILED & Outbox Webhook FAILED)',
    mechanism:
      'Dokumen dikirim secara langsung via multipart form-data. File fisik gambar hanya tersimpan di memori RAM pod Extraction. Ketika pod terkena SIGKILL (simulasi OOMKilled / Node Evict), proses mati seketika dan data file di RAM lenyap selamanya.',
    whyStatus:
      'Ketika sistem Auto-Recovery berjalan setelah lease kedaluwarsa (~30 detik), sistem mendeteksi pekerjaan yang terputus (unowned job). Karena file di RAM sudah musnah dan tidak ada URL MinIO untuk mengunduh ulang, sistem secara sengaja menandai status FAILED (bukan digantung PROCESSING selamanya). Outbox Relay kemudian mengirimkan webhook callback FAILED ke Orkestrasi Pusat agar klien tahu bahwa request harus dikirim ulang.',
    recommendation:
      'Untuk lingkungan produksi transaksi kritis, sangat disarankan menggunakan skema file_url (MinIO/S3) seperti pada skenario "crash-file-url", sehingga jika pod restart/crash, sistem dapat self-healing mengunduh ulang gambar secara otomatis tanpa campur tangan nasabah.',
  },

  'crash-file-url': {
    title: 'Extraction Mati Mendadak (SIGKILL) - Dokumen File URL (MinIO)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'PASS (Job Pulih & Callback DONE)',
    mechanism:
      'Dokumen dikirim berupa referensi file_url yang tersimpan aman di Object Storage (MinIO/S3). Pod Extraction hanya memproses gambar dari URL tersebut. Saat pod dimatikan dengan SIGKILL, proses mati tetapi file fisik tetap aman di MinIO.',
    whyStatus:
      'Setelah lease habis (~30 detik), sistem Auto-Recovery mengambil alih pekerjaan yang terputus. Sistem membaca kembali file_url dari database, mengunduh ulang gambar dari MinIO (attempt 2), lalu menjalankan pipeline hingga Structuring dan Scoring. Hasil akhir selesai secara mulus dan dikirimkan sebagai Callback DONE.',
    recommendation:
      'Ini adalah standar emas arsitektur pipeline dokumen skala besar di cloud. Penggunaan file_url menjaga memori pod tetap hemat dan menjamin resiliensi 100% tanpa kehilangan data saat server mengalami gangguan.',
  },

  'stage-down': {
    title: 'Structuring Mati 25 Detik di Tengah Request (Handoff Failure)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'PASS (Handoff Tertahan di Outbox & Selesai Setelah Pulih)',
    mechanism:
      'Pod tahap berikutnya (Structuring) mati saat pod Extraction selesai melakukan OCR dan mencoba mengirimkan HTTP POST handoff. Sambungan HTTP ke port 8032 gagal.',
    whyStatus:
      'Extraction tidak panik atau membatalkan request. Berkat pola Transactional Outbox, payload handoff disimpan di PostgreSQL dan diulang secara berkala dengan exponential backoff. Begitu pod Structuring sehat kembali (/readyz), handoff berhasil terkirim dan pipeline lanjut hingga selesai.',
    recommendation:
      'Transactional Outbox mengisolasi kegagalan sementara antar mikroservis (loose coupling), mencegah cascading failure jika salah satu service hilir sedang deployment atau restart.',
  },

  'sigterm-drained': {
    title: 'Rolling Restart Extraction di Tengah Job Pendek (Graceful Drain)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'PASS (Job Selesai Sebelum Pod Berhenti)',
    mechanism:
      'Sinyal SIGTERM dikirim ke pod Extraction (meniru perintah kubectl rollout restart atau pod eviction saat maintenance node).',
    whyStatus:
      'Pod Extraction menangkap SIGTERM secara graceful. Service menghentikan penerimaan request baru tetapi tetap menuntaskan tugas OCR yang sedang aktif di memori (drain). Setelah job selesai dan dikirim ke Structuring, barulah pod berhenti dengan aman.',
    recommendation:
      'Pastikan terminationGracePeriodSeconds pada konfigurasi deployment Kubernetes diset lebih besar dari durasi rata-rata pemrosesan OCR dokumen terberat.',
  },

  'sigterm-interrupted': {
    title: 'Rolling Restart Extraction di Tengah Job Panjang (Drain Timeout Exceeded)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'WARN (Job FAILED Interrupted, Kiriman Ulang Selesai)',
    mechanism:
      'SIGTERM dikirim pada job yang membutuhkan waktu pemrosesan lebih lama daripada batas toleransi PIPELINE_DRAIN_TIMEOUT_SECONDS (30 detik).',
    whyStatus:
      'Karena batas waktu drain terlewati, pod terpaksa dihentikan paksa. Job ditandai FAILED dengan alasan "interrupted by a service shutdown" dan dilaporkan via webhook. Ketika Orkestrasi Pusat mengirim ulang dengan request_id yang sama, pipeline berjalan normal kembali.',
    recommendation:
      'Orkestrasi Pusat harus memperlakukan status FAILED akibat service shutdown sebagai transaksi yang boleh di-retry secara aman dengan request_id yang sama. Jadwalkan deployment pod di luar jam sibuk.',
  },

  'postgres-down': {
    title: 'Database PostgreSQL Mati 20 Detik (Cloud SQL Failover / Maintenance)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'PASS / WARN (Fast-Fail 5xx untuk Request Baru, Job Pulih)',
    mechanism:
      'Database PostgreSQL dimatikan selama 20 detik untuk mensimulasikan failover cluster HA atau koneksi database terputus total.',
    whyStatus:
      'Request baru yang masuk saat DB mati ditolak secara tegas dengan HTTP 5xx (Fast-Fail), mencegah request nasabah hilang tanpa jejak. Request yang sedang berjalan tertahan sementara, lalu setelah DB pulih, koneksi tersambung kembali dan status akhir dikirimkan ke webhook.',
    recommendation:
      'Klien pemanggil (API Gateway/Orchestrator) wajib menerapkan retry logic dengan jitter saat menerima HTTP 5xx selama jeda failover database (biasanya 10-30 detik pada Cloud SQL).',
  },

  'callback-down': {
    title: 'Orkestrasi Pusat Mati 20 Detik (Endpoint Callback 503)',
    category: 'Transactional Outbox',
    expectedOutcome: 'PASS (Callback Tertahan di Outbox & Terkirim Saat Pulih)',
    mechanism:
      'Hasil OCR dan Scoring sudah selesai, tetapi endpoint webhook milik Orkestrasi Pusat (Bank BRI) sedang down / 503 saat pengiriman hasil dilakukan.',
    whyStatus:
      'Hasil akhir tidak dibuang atau hilang. Worker Transactional Outbox Relay menahan payload di database lokal dan mengulang pengiriman dengan interval bertingkat. Begitu server pusat kembali online, callback langsung diterima tepat satu kali.',
    recommendation:
      'Pola Transactional Outbox menjamin garansi At-Least-Once Delivery; gangguan di server penerima callback tidak akan menyebabkan kehilangan data hasil verifikasi dokumen.',
  },

  'callback-flaky': {
    title: 'Orkestrasi Pusat Tersendat (Flaky: 503 Beberapa Kali Lalu 200)',
    category: 'Transactional Outbox',
    expectedOutcome: 'PASS (Terkirim Sukses Tepat Satu Kali Tanpa Duplikat)',
    mechanism:
      'Endpoint callback pusat mengalami fluktuasi sesaat (misalnya koneksi pool DB pusat penuh sesaat sehingga membalas 503 sebelum pulih).',
    whyStatus:
      'Outbox relay terus mencoba ulang pengiriman hingga berhasil. Sistem mencatat konfirmasi pengiriman sukses sehingga tidak ada duplikasi pesan berulang.',
    recommendation:
      'Meskipun sistem menjamin pengiriman, endpoint webhook di sisi Orkestrasi Pusat tetap disarankan bersifat idempoten untuk mengantisipasi network timeout semu.',
  },

  'callback-not-ready': {
    title: 'Callback Tiba Sebelum Pusat Mencatat 202-nya (Race Condition 409)',
    category: 'Transactional Outbox',
    expectedOutcome: 'PASS (409 Dikenali Sebagai Retryable & Diterima Sempurna)',
    mechanism:
      'Pipeline OCR selesai sangat cepat dalam hitungan milidetik, sehingga webhook callback tiba sebelum Orkestrasi Pusat selesai menulis state awal transaksi (Pusat merespons 409 RESULT_NOT_READY).',
    whyStatus:
      'Sistem secara cerdas membedakan HTTP 409 khusus ini dari error 4xx biasa. Alih-alih membuang pesan ke Dead Letter, Outbox Relay memberi jeda 1,5 detik lalu mengirim ulang hingga Pusat siap menerimanya.',
    recommendation:
      'Pemisahan penanganan error 409 RESULT_NOT_READY mencegah kegagalan prematur akibat selisih kecepatan asinkron antara orchestrator dan worker pipeline.',
  },

  'callback-slow': {
    title: 'Orkestrasi Pusat Lambat Menjawab Callback (> 10 Detik Timeout)',
    category: 'Transactional Outbox',
    expectedOutcome: 'WARN (Relay Mengirim Ulang Sesuai Kontrak Timeout)',
    mechanism:
      'Server webhook pusat memproses callback lebih lama dari batas ORCHESTRATION_TIMEOUT_SECONDS (10 detik).',
    whyStatus:
      'Karena timeout koneksi tercapai, Outbox Relay mengasumsikan pengiriman gagal di jaringan dan mengirimkan ulang callback. Jika respons sebelumnya sebenarnya sudah diproses oleh pusat, pusat akan menerima callback duplikat.',
    recommendation:
      'Endpoint callback di sisi Orkestrasi Pusat harus memproses payload secara asinkron (langsung jawab 200 OK begitu payload tiba di antrean) dan wajib menerapkan cek idempotensi berdasar request_id.',
  },

  'callback-rejected': {
    title: 'Orkestrasi Pusat Menolak Callback (HTTP 401 Unauthorized)',
    category: 'Transactional Outbox',
    expectedOutcome: 'WARN (Pindah ke Dead Letter Buffer, Bisa Direlease Manual)',
    mechanism:
      'Orkestrasi Pusat menolak callback karena autentikasi tidak valid (misalnya secret X-Callback-Key salah atau expired).',
    whyStatus:
      'Sistem tidak melakukan retry buta berulang kali untuk error 4xx permanen agar tidak membebani server. Pesan dipindahkan ke status Dead Letter dan dapat diperiksa di tabel Outbox serta dilepas via API /outbox/release setelah kunci diperbaiki.',
    recommendation:
      'Pantau tabel Transactional Outbox secara berkala untuk mendeteksi pesan Dead Letter akibat kesalahan konfigurasi credential webhook.',
  },

  baseline: {
    title: 'Jalur Normal Tanpa Gangguan (Baseline Benchmark)',
    category: 'Standar & Benchmark',
    expectedOutcome: 'PASS (200 Completed, 1 Callback, Bebas Antrean)',
    mechanism:
      'Menjalankan seluruh alur pipeline dari Client -> Guardrails -> Extraction -> Structuring -> Scoring -> Callback Webhook dalam kondisi semua service sehat.',
    whyStatus:
      'Semua tahap selesai tepat waktu, callback terkirim sekali, antrean outbox bersih (0 pending), dan tidak ada job tertinggal di database.',
    recommendation:
      'Gunakan metrik durasi dan konsumsi sumber daya skenario ini sebagai tolok ukur (baseline latency) performa standar sistem.',
  },

  resend: {
    title: 'Pusat Mengirim Ulang Request ID yang Sudah Selesai (Idempotency)',
    category: 'Idempotency & Keamanan',
    expectedOutcome: 'PASS (Dijawab dari Cache Database Tanpa Komputasi Ulang)',
    mechanism:
      'Klien atau Orkestrasi Pusat mengalami network timeout di sisinya, lalu mengirimkan kembali request POST dengan request_id yang persis sama.',
    whyStatus:
      'Sistem mendeteksi bahwa request_id ini sudah berstatus DONE di PostgreSQL. Sistem tidak menjalankan ulang OCR / model AI dari awal, melainkan langsung mengembalikan hasil tersimpan. Tidak ada callback ganda yang ditembakkan.',
    recommendation:
      'Idempotency melindungi server dari lonjakan beban komputasi ganda jika klien aplikasi sering melakukan auto-retry.',
  },

  concurrent: {
    title: 'Dua Request POST Bersamaan dengan Request ID Sama (Atomic Lock)',
    category: 'Idempotency & Keamanan',
    expectedOutcome: 'PASS (Hanya 1 Worker yang Mengklaim Job, 1 Callback)',
    mechanism:
      'Dua request POST dengan request_id yang identik tiba secara bersamaan dalam milidetik yang sama (simulasi double-click nasabah atau race condition gateway).',
    whyStatus:
      'Mekanisme atomic row lock database memastikan hanya satu instance yang berhasil mengklaim job ke status PROCESSING. Request kedua menunggu atau menerima respons yang sama, dan hanya tepat 1 callback akhir yang dikirimkan.',
    recommendation:
      'Menjamin integritas data transaksi dan efisiensi resource server terhadap serangan klik ganda.',
  },

  'entry-down': {
    title: 'Guardrails / Extraction Mati Saat Request Masuk (Fail-Fast Gate)',
    category: 'Pod Crash & Resiliensi',
    expectedOutcome: 'PASS (Fast-Fail 5xx Bersih, Tidak Ada Job Gantung)',
    mechanism:
      'Service hilir (Guardrails atau Extraction) sedang mati ketika request baru pertama kali tiba di gerbang Orchestrator.',
    whyStatus:
      'Orchestrator segera menolak request dengan HTTP 5xx dan mencantumkan secara transparan tahap mana yang tidak tersedia (pipeline_last_stage), tanpa meninggalkan pekerjaan menggantung setengah jalan di database.',
    recommendation:
      'Prinsip Fail-Fast memungkinkan Orkestrasi Pusat segera mengalihkan transaksi ke rute alternatif atau menginformasikan status sistem secara akurat tanpa membuat nasabah menunggu lama.',
  },
}
