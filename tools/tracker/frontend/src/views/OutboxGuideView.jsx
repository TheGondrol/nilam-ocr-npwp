import { useState, useEffect } from 'react'
import {
  Layers,
  Play,
  ExternalLink,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'

const OUTBOX_PROBLEMS = [
  {
    title: 'Hasil tersimpan, pesannya hilang',
    problem:
      'Begitu tahap selesai ia harus menyimpan hasil ke database DAN mengirim HTTP (hand-off ke tahap berikutnya, callback ke pusat). Keduanya tidak bisa atomik: pod mati di antaranya berarti hand-off atau callback hilang, request macet di tengah, pusat tidak pernah tahu.',
    solution:
      'Pesan ditulis ke nilam_pipeline_outbox di transaksi yang sama dengan hasil job: tersimpan bersama atau tidak sama sekali. Pengiriman dikerjakan relay sesudahnya.',
    scenarios: ['baseline', 'sigterm-drained'],
    look: [
      'Panel pipeline_outbox: baris QUEUED muncul di detik yang sama dengan hasil tahap, lalu CLAIMED dan DELIVERED.',
      'Timeline: commit hasil, klaim relay, hand-off, callback, berurutan.',
    ],
  },
  {
    title: 'Penerima sedang tidak bisa menerima',
    problem:
      'Pusat atau tahap berikutnya menjawab 5xx, timeout, atau tidak terjangkau. Dulu dicoba 3x dalam ~2 detik lalu menyerah: pesannya hilang.',
    solution:
      'Dicoba ulang dengan jeda berlipat (0,5 dtk, 1, 2, ... maks. 5 menit sekali): hand-off sampai 24 jam, callback sampai 600 dtk (batas tunggu pusat).',
    scenarios: ['callback-down', 'stage-down', 'callback-flaky'],
    look: [
      'Panel pipeline_outbox: baris RETRY dengan attempt bertambah dan next attempt makin jauh.',
      'Panel Callback ke Orkestrasi: kedatangan dijawab 503, lalu 200 begitu penerima hidup, tanpa tindakan manual.',
    ],
    manual: 'Manual: di menu Pipeline pilih "callback mati (503)", kirim dokumen, tunggu, lalu kembalikan ke normal.',
  },
  {
    title: 'Callback yang gagal tidak menahan pipeline',
    problem: 'Kalau pusat mati, tahap tidak boleh berhenti menyerahkan job ke tahap berikutnya.',
    solution: 'Hand-off dikirim lebih dulu dan diproses terpisah dari callback: pipeline tetap jalan, callback menumpuk lalu terkirim.',
    scenarios: ['callback-down'],
    look: ['Kartu structuring dan scoring tetap DONE sementara callback-nya masih RETRY.'],
  },
  {
    title: 'Beberapa replika, relay mati di tengah pengiriman',
    problem: 'Dua pod yang sama tidak boleh mengirim pesan yang sama bersamaan; pesan yang sedang dikirim pod yang mati tidak boleh hilang.',
    solution: 'Relay mengklaim baris dengan FOR UPDATE SKIP LOCKED dan lease 30 dtk; lease habis = baris diambil lagi replika lain.',
    scenarios: ['concurrent'],
    look: ['Satu klaim job dan satu callback akhir walau dua POST datang bersamaan.'],
    note: 'Lokal hanya satu replika per service: penguncian baris outbox dibuktikan test otomatis (libs/ocr_common/tests/test_outbox.py).',
  },
  {
    title: 'Rolling update (SIGTERM)',
    problem: 'Deploy mematikan pod lama saat job dan pengiriman sedang berjalan.',
    solution:
      'Pod menyelesaikan job dalam masa drain, relay menyelesaikan pengiriman yang sedang jalan lalu mencoba sekali lagi yang sudah jatuh tempo; sisanya aman di tabel.',
    scenarios: ['sigterm-drained', 'sigterm-interrupted'],
    look: ['Hand-off tetap terkirim setelah restart; job yang terlalu panjang FAILED "interrupted" dan pusat diberi tahu.'],
  },
  {
    title: 'Pesan yang memang ditolak (dead letter)',
    problem: 'Penerima menjawab 4xx (key salah, body ditolak): mencoba terus hanya membebani dan tidak akan berhasil.',
    solution:
      'Langsung jadi dead letter yang tetap tersimpan, terlihat di backlog dan log, dan bisa dikirim ulang setelah penyebabnya dibetulkan.',
    scenarios: ['callback-rejected'],
    look: [
      'Panel pipeline_outbox: baris DEAD dengan last_error; sidebar backlog: dead_letters bertambah.',
      'Tombol "Lepaskan dead letter" mengirimnya lagi.',
    ],
    manual: 'Manual: di menu Pipeline pilih "callback menolak (422)" atau "X-Callback-Key salah (401)".',
  },
  {
    title: 'Hand-off gagal permanen tetap dilaporkan',
    problem: 'Kalau tahap berikutnya tidak pernah menerima hand-off, request menggantung selamanya tanpa ada yang tahu.',
    solution: 'Hand-off yang jadi dead letter diganti callback FAILED atas nama tahap berikutnya: pusat tetap mendapat keadaan akhir.',
    scenarios: [],
    note: 'Belum ada skenario (perlu tahap yang menolak hand-off dengan 4xx); dibuktikan test test_a_handoff_the_next_stage_refuses_becomes_a_failed_callback.',
  },
  {
    title: 'Callback tiba sebelum pusat mencatat 202-nya',
    problem: 'Pipeline bisa selesai sangat cepat setelah jawaban 202; pusat menjawab callback-nya 409 RESULT_NOT_READY.',
    solution: 'Kode itu dikenali dan callback dikirim ulang tiap 1,5 dtk sampai 5x, bukan dianggap 4xx biasa (dead letter).',
    scenarios: ['callback-not-ready'],
    look: ['Panel Callback ke Orkestrasi: kedatangan dijawab 409 lalu 200, satu callback akhir diterima.'],
    manual: 'Manual: di menu Pipeline pilih "callback belum siap (409)".',
  },
  {
    title: 'Bisa dipantau',
    problem: 'Pesan yang tertahan atau mati tidak boleh hanya diketahui saat pusat komplain.',
    solution: 'Backlog per tahap (pending, retrying, umur tertua, dead letter), warning di log kalau basi, metrik, dan di dev transaksi APM + event log per pengiriman.',
    scenarios: [],
    look: [
      'Sidebar menu Pipeline: Backlog outbox tiap service (GET /v1/<tahap>/outbox).',
      'Dev: Elastic APM transaksi "<TAHAP> callback", Kibana event.dataset : "outbox".',
    ],
  },
]

const OUTBOX_LIMITS = [
  {
    title: 'Crash saat job sedang dikerjakan',
    problem: 'Pod mati sebelum hasilnya tersimpan: belum ada yang masuk outbox.',
    solution: 'Bukan outbox, melainkan mekanisme Auto-Recovery: job PROCESSING yang terputus dijalankan ulang setelah batas lease timeout. Untuk upload inline, file hilang bersama pod (job FAILED dengan permintaan kirim ulang); untuk file_url (MinIO) pipeline pulih mandiri.',
    scenarios: ['crash-file-url', 'crash-inline'],
  },
  {
    title: 'At-least-once: bisa duplikat',
    problem: 'Penerima yang lambat menjawab dianggap gagal, pesannya dikirim lagi padahal sudah diterima.',
    solution: 'Tidak dicegah oleh outbox: penerima harus idempoten per request_id (tahap kita sudah, pusat sudah diminta).',
    scenarios: ['callback-slow'],
  },
  {
    title: 'Database mati',
    problem: 'Hasil dan outbox sama-sama tidak bisa ditulis.',
    solution: 'Request baru ditolak 5xx. Temuan terbuka: request yang terputus bisa melapor FAILED lalu DONE.',
    scenarios: ['postgres-down'],
  },
]

function GuideCard({ item, byId, running, available, onRun, openRequest }) {
  return (
    <Card className="border-border bg-card shadow-xs p-4 space-y-3 hover:border-border/80 transition-all">
      <h3 className="text-sm font-semibold text-foreground">{item.title}</h3>

      <div className="grid grid-cols-[68px_1fr] gap-2 text-xs">
        <span className="font-semibold text-amber-500 dark:text-amber-400 uppercase text-[10px] tracking-wide">
          MASALAH
        </span>
        <p className="text-muted-foreground m-0 leading-relaxed">{item.problem}</p>
      </div>

      <div className="grid grid-cols-[68px_1fr] gap-2 text-xs">
        <span className="font-semibold text-emerald-500 dark:text-emerald-400 uppercase text-[10px] tracking-wide">
          OUTBOX
        </span>
        <p className="text-foreground m-0 leading-relaxed font-medium">{item.solution}</p>
      </div>

      {item.look && (
        <div className="grid grid-cols-[68px_1fr] gap-2 text-xs">
          <span className="font-semibold text-muted-foreground uppercase text-[10px] tracking-wide">
            LIHAT DI
          </span>
          <ul className="m-0 pl-4 text-muted-foreground space-y-0.5 list-disc">
            {item.look.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </div>
      )}

      {item.note && <div className="text-[11px] text-muted-foreground font-mono">{item.note}</div>}
      {item.manual && <div className="text-[11px] text-muted-foreground font-mono">{item.manual}</div>}

      {item.scenarios.length > 0 && (
        <div className="border-t border-border/40 pt-2.5 space-y-2">
          {item.scenarios.map((id) => {
            const scenario = byId[id]
            const last = scenario?.last
            const status = last?.status
            return (
              <div key={id} className="flex items-center justify-between gap-2 text-xs flex-wrap">
                <div className="flex items-center gap-1.5">
                  {status ? (
                    <Badge variant={status === 'pass' ? 'done' : status === 'warn' ? 'rejected' : 'failed'} className="text-[9px] px-1.5 py-0">
                      {status}
                    </Badge>
                  ) : (
                    <Badge variant="idle" className="text-[9px] px-1.5 py-0">BELUM</Badge>
                  )}
                  <span className="font-medium text-foreground">{scenario?.title ?? id}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    className="h-6 px-2 text-[10px] gap-1"
                    disabled={running || !available}
                    onClick={() => onRun([id])}
                  >
                    <Play size={10} />
                    <span>Jalankan Demo</span>
                  </Button>
                  {last?.request_ids?.[0] && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="xs"
                      className="h-6 px-2 text-[10px] text-muted-foreground gap-1 hover:text-foreground"
                      onClick={() => openRequest(last.request_ids[0])}
                    >
                      <span>Buka Request</span>
                      <ExternalLink size={10} />
                    </Button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}

export default function OutboxGuideView({ openRequest }) {
  const [data, setData] = useState(null)
  const [image, setImage] = useState('')
  const [error, setError] = useState(null)

  const load = () =>
    fetch('/api/scenarios')
      .then((r) => r.json())
      .then((d) => {
        setData(d)
        setImage((cur) => cur || d.images?.[0] || '')
      })
      .catch(() => {})

  useEffect(() => {
    load()
    const timer = setInterval(load, 1500)
    return () => clearInterval(timer)
  }, [])

  async function run(ids) {
    setError(null)
    const r = await fetch('/api/scenarios/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenarios: ids, image }),
    })
    if (!r.ok) setError((await r.json()).detail ?? r.statusText)
    load()
  }

  const byId = Object.fromEntries((data?.scenarios ?? []).map((s) => [s.id, s]))
  const running = Boolean(data?.running)
  const available = Boolean(data?.available)
  const allIds = [...new Set([...OUTBOX_PROBLEMS, ...OUTBOX_LIMITS].flatMap((item) => item.scenarios))].filter((id) => byId[id])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
      {/* LEFT COLUMN: Controls & Document Selection */}
      <div className="space-y-4">
        <Card className="border-border bg-card border-t-2 border-t-amber-500 shadow-xs">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="h-6 w-6 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-600 dark:text-amber-400">
                <Layers size={14} />
              </div>
              <span>Jalankan Demo Outbox</span>
            </CardTitle>
            <CardDescription className="text-xs">
              Buktikan kegunaan transactional outbox secara langsung
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-xs font-medium text-foreground">Dokumen Uji (Harus Lolos Validasi)</label>
              <select
                value={image}
                onChange={(e) => setImage(e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-background/80 px-3 py-1.5 text-xs text-foreground mt-1"
              >
                {(data?.images ?? []).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>

            <Button
              type="button"
              variant="default"
              disabled={running || !available}
              onClick={() => run(allIds)}
              className="w-full text-xs font-semibold gap-1.5"
            >
              <Play size={13} />
              <span>Jalankan Semua Demo ({allIds.length})</span>
            </Button>

            {running && (
              <div className="text-xs text-muted-foreground font-mono pt-1">
                Sedang berjalan: <b className="text-foreground">{data.running}</b>…
              </div>
            )}
            {error && (
              <Alert variant="destructive" className="py-2 text-xs">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      {/* RIGHT COLUMN: Guide Cards Grid */}
      <div className="space-y-6">
        {/* Intro Banner */}
        <Card className="border-border bg-card border-t-2 border-t-emerald-500 shadow-xs p-5 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded-md bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shrink-0">
                <Layers size={16} />
              </div>
              <h2 className="text-base font-bold text-foreground">
                Mengapa Transactional Outbox Dibutuhkan di Sistem Terdistribusi?
              </h2>
            </div>
            <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/30 text-emerald-600 dark:text-emerald-400">
              Reliability Architecture Pattern
            </Badge>
          </div>
          <div className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-1 border-t border-border/50">
            <p>
              <b className="text-rose-500 dark:text-rose-400">Tanpa Outbox:</b> Aplikasi melakukan <code>simpan hasil di DB</code> lalu <code>POST handoff / callback via HTTP</code>. Jika yang kedua gagal atau pod crash mendadak di antaranya, data tersimpan tapi downstream tidak pernah tahu.
            </p>
            <p>
              <b className="text-emerald-600 dark:text-emerald-400">Dengan Transactional Outbox:</b> Aplikasi menyimpan <code>hasil + antrean pesan</code> dalam <b>SATU transaksi atomik</b> di PostgreSQL. Relay terpisah mengirimkan pesan secara andal dengan garansi retry dan dead-letter handling.
            </p>
          </div>
        </Card>

        <div className="space-y-3">
          <h3 className="text-xs font-bold text-foreground tracking-wider uppercase font-mono">
            1. Masalah yang Ditangani Transactional Outbox
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {OUTBOX_PROBLEMS.map((item) => (
              <GuideCard
                key={item.title}
                item={item}
                byId={byId}
                running={running}
                available={available}
                onRun={run}
                openRequest={openRequest}
              />
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-xs font-bold text-foreground tracking-wider uppercase font-mono">
            2. Batasan: Hal yang Berada di Luar Ranah Outbox
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {OUTBOX_LIMITS.map((item) => (
              <GuideCard
                key={item.title}
                item={item}
                byId={byId}
                running={running}
                available={available}
                onRun={run}
                openRequest={openRequest}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
