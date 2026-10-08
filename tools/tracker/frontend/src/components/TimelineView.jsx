import { useState } from 'react'
import { Clock, ChevronDown, ChevronRight, Info } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import JsonViewer from './JsonViewer'

const LABELS = {
  GUARDRAILS: 'Orchestrator + guardrails',
  OCR: 'Extraction (OCR)',
  STRUCTURING: 'Structuring',
  SCORING: 'Scoring',
  ORKESTRASI: 'Orkestrasi',
}

const JOB_TABLES = {
  OCR: 'nilam_ocr_extraction_jobs',
  STRUCTURING: 'nilam_structuring_jobs',
  SCORING: 'nilam_scoring_jobs',
}

const JOB_PATHS = { OCR: '/v1/extraction/jobs', STRUCTURING: '/v1/structuring/jobs', SCORING: '/v1/scoring/jobs' }

// Urutan tahap yang menyimpan job; GUARDRAILS dijalankan orchestrator sendiri dan tidak punya job.
const PIPELINE = ['OCR', 'STRUCTURING', 'SCORING']
const STAGE_OF_SERVICE = { extraction: 'OCR', structuring: 'STRUCTURING', scoring: 'SCORING' }

const CONTRACT_LABEL = {
  pass: 'Sesuai Kontrak (PASS)',
  warn: 'Sesuai Kontrak dengan Catatan (WARN)',
  fail: 'MELANGGAR KONTRAK (FAIL)',
}

const CALLBACK_KIND = {
  completed: 'completed (selesai)',
  rejected: 'completed, result null (ditolak)',
  failed: 'failed',
  stage: 'format stage (lokal, bukan kontrak pusat)',
  invalid: 'tidak terbaca',
}

const PIPELINE_LAST_STAGE_RULE =
  'Sesuai kontrak extract-ocr, pipeline_last_stage hanya diisi kalau ada error atau penolakan, dan isinya nama service asal error itu (orchestrator, guardrails, extraction, structuring, atau scoring).'

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function label(stage) {
  return LABELS[stage] ?? stage
}

/** Tahap job yang mengakhiri request ini: tahap terakhir pipeline_name_sequence (tanpa sequence: SCORING). */
function lastJobStage(events) {
  const sequence = events.find((e) => e.type === 'client')?.sequence
  if (!sequence) return 'SCORING'
  const stages = sequence.map((s) => STAGE_OF_SERVICE[s]).filter(Boolean)
  return stages[stages.length - 1] ?? null
}

function nextStage(stage, events) {
  const sequence = events.find((e) => e.type === 'client')?.sequence
  const stages = sequence ? sequence.map((s) => STAGE_OF_SERVICE[s]).filter(Boolean) : PIPELINE
  const i = stages.indexOf(stage)
  return i >= 0 ? stages[i + 1] ?? null : null
}

function outboxWhat(m) {
  if (m.kind === 'handoff') return `handoff ${label(m.owner)} → ${label(m.target)}`
  return `callback ${label(m.owner)} → Orkestrasi`
}

function httpEntry(e) {
  const after = `setelah ${fmtMs(e.elapsed_ms)}`
  const pls = e.pipeline_last_stage
  if (e.http_status === 200) {
    return {
      title: `Orchestrator menjawab HTTP 200 ${after}: pipeline selesai sebelum batas tunggu ${e.wait_seconds} dtk, hasil final langsung ada di body jawaban.`,
      detail: `pipeline_last_stage = null karena tidak ada error. ${PIPELINE_LAST_STAGE_RULE} Service mana saja yang dijalankan sudah ditentukan pusat lewat pipeline_name_sequence, jadi tidak perlu diulang di jawaban.`,
    }
  }
  if (e.http_status === 202) {
    return {
      title: `Orchestrator menjawab HTTP 202 ${after}: batas tunggu ${e.wait_seconds} dtk habis sebelum tahap terakhir selesai.`,
      detail: `Pipeline tetap berjalan. Hasilnya menyusul lewat satu callback dari tahap terakhir, dan bisa dibaca kapan saja lewat GET /v1/extract-ocr/{request_id}. pipeline_last_stage = null karena belum ada error. ${PIPELINE_LAST_STAGE_RULE}`,
    }
  }
  return {
    title: `Orchestrator menjawab HTTP ${e.http_status}${e.errors ? ` ${e.errors}` : ''} ${after}.`,
    detail: pls
      ? `pipeline_last_stage = ${pls}: error atau penolakan ini berasal dari service ${pls}. ${PIPELINE_LAST_STAGE_RULE}`
      : PIPELINE_LAST_STAGE_RULE,
  }
}

function stageEntry(e, events) {
  const name = label(e.stage)
  if (e.stage === 'GUARDRAILS') {
    if (e.status === 'SKIPPED') {
      return { title: `${name}: model guardrails dilewati karena tidak ada di pipeline_name_sequence.` }
    }
    return {
      title: `${name}: ${e.status}${e.error_message ? ` · ${e.error_message}` : ''}.`,
      detail: 'Kartu orchestrator ditutup setelah jawaban HTTP-nya tercatat.',
    }
  }
  if (e.source === 'callback') {
    return { title: `Tracker menandai ${name} ${e.status} berdasarkan callback di atas.` }
  }
  const table = JOB_TABLES[e.stage]
  if (e.status === 'PROCESSING') {
    const i = PIPELINE.indexOf(e.stage)
    return {
      title: `${name} mulai bekerja: job PROCESSING tercatat di ${table}.`,
      detail:
        e.stage === 'OCR'
          ? 'Job ini dibuat orchestrator setelah guardrails lolos (POST /v1/extraction/jobs).'
          : `Job ini muncul karena handoff dari ${label(PIPELINE[i - 1])} sampai: itu bukti pesan handoff-nya terkirim.`,
    }
  }
  if (e.status === 'REJECTED') {
    return {
      title: `${name} menolak dokumen (aturan ML): ${e.error_message}.`,
      detail:
        'Dalam transaksi yang sama ditulis satu pesan outbox: callback penolakan (DOWNSTREAM_VALIDATION_ERROR). Tidak ada handoff, jadi tahap berikutnya tidak dijalankan.',
    }
  }
  if (e.status === 'FAILED') {
    return {
      title: `${name} gagal: ${e.error_message}.`,
      detail: 'Dalam transaksi yang sama ditulis satu pesan outbox: callback FAILED ke Orkestrasi.',
    }
  }
  if (e.status === 'DONE') {
    const final = e.stage === lastJobStage(events)
    const next = nextStage(e.stage, events)
    return {
      title: `${name} selesai: hasil, status DONE, dan pesan keluarnya disimpan dalam SATU transaksi database.`,
      detail: final
        ? `${name} adalah tahap terakhir request ini, jadi pesan keluarnya satu: callback hasil akhir ke Orkestrasi (ditulis ke nilam_pipeline_outbox, belum dikirim).`
        : `Pesan keluarnya dua, ditulis ke nilam_pipeline_outbox: handoff ke ${label(next)} dan callback DONE. Inilah inti pola outbox: kalau pod mati tepat setelah commit ini, pesannya tetap ada di tabel dan dikirim relay begitu service hidup lagi. Tidak mungkin hasil tersimpan tetapi ${label(next)} tidak pernah dipanggil.`,
    }
  }
  return { title: `${name}: ${e.status}` }
}

function outboxEntry(e, t0) {
  const m = e.message
  if (e.status === 'UNAVAILABLE') {
    return { title: `Tabel outbox tidak bisa dibaca: ${e.error_message ?? 'TRACKER_DATABASE_URL kosong'}.` }
  }
  const id = m.id != null ? ` #${m.id}` : ''
  const what = outboxWhat(m)
  switch (e.status) {
    case 'QUEUED':
      return {
        title: `Pesan${id} (${what}) terlihat di nilam_pipeline_outbox, menunggu dikirim.`,
        detail: 'Biasanya baris ini hanya hidup beberapa milidetik. Kalau sampai terbaca tracker, relay sedang sibuk atau penerimanya bermasalah.',
      }
    case 'CLAIMED':
      return {
        title: `Relay ${label(m.owner)} mengambil pesan${id} (${what}), percobaan ke-${m.attempts}.`,
        detail: 'Diambil dengan FOR UPDATE SKIP LOCKED: kalau ada beberapa replika, hanya satu yang mengirim pesan ini. Lease 30 dtk: kalau pod mati di tengah kirim, replika lain mengambil alih setelah lease habis.',
      }
    case 'RETRY': {
      const at = m.next_attempt_at ? (new Date(m.next_attempt_at).getTime() / 1000 - t0) * 1000 : null
      return {
        title: `Kirim pesan${id} (${what}) gagal: ${m.last_error}. Dicoba lagi pada t+${fmtMs(at)}.`,
        detail: 'Penerima menjawab 5xx atau tidak terjangkau: dicoba lagi dengan jeda yang terus membesar (1, 2, 4, … dtk, maks 5 mnt). Pipeline tidak tertahan, dan pesannya tetap aman di tabel selama menunggu.',
      }
    }
    case 'DEAD':
      return {
        title: `Pesan${id} (${what}) jadi DEAD LETTER setelah ${m.attempts} percobaan: ${m.last_error}.`,
        detail: 'Berhenti dicoba karena penerima menolak permanen (4xx selain 409 RESULT_NOT_READY) atau umurnya lewat batas (callback 600 dtk, handoff 24 jam). Barisnya tetap ada untuk diperiksa dan bisa dikirim ulang dengan tombol Lepaskan.',
      }
    case 'RELEASED':
      return { title: `Pesan${id} (${what}) dilepas dari dead letter dan segera dicoba lagi.` }
    case 'SKIPPED':
      return {
        title: `Relay ${label(m.owner)} memproses callback ${label(m.owner)} DONE: TIDAK dikirim, baris outbox dihapus.`,
        detail: 'Format callback result (kontrak pusat 2 Okt 2026) hanya mengirim satu callback per request, dari tahap terakhir. Setiap tahap tetap menulis pesan callback-nya; pengirimnya yang memutuskan, dan callback tahap tengah dilewati.',
      }
    case 'DELIVERED':
      if (e.source !== 'log') {
        return { title: `Pesan${id} (${what}) hilang dari tabel: sudah dikirim, lalu dihapus relay.` }
      }
      if (m.kind === 'handoff') {
        return {
          title: `Relay ${label(m.owner)} menyerahkan job ke ${label(m.target)} (POST ${JOB_PATHS[m.target] ?? '…'}, percobaan ke-${m.attempts}): diterima, baris outbox dihapus.`,
          detail: 'Dicatat dari log relay. Barisnya baru dihapus setelah penerima menjawab sukses; kalau gagal, pesan tetap di tabel dan dicoba lagi.',
        }
      }
      return {
        title: `Relay ${label(m.owner)} mengirim callback hasil ke Orkestrasi (percobaan ke-${m.attempts}): dijawab 2xx, baris outbox dihapus.`,
        detail: 'Dicatat dari log relay, sisi pengirim. Sisi penerimanya ada di baris CALLBACK.',
      }
    default:
      return { title: `Pesan${id}: ${e.status}` }
  }
}

function callbackEntry(e) {
  const by = `relay ${label(e.stage)}`
  let title
  if (e.format === 'result') {
    const what = e.error_code === 'DOWNSTREAM_VALIDATION_ERROR' ? 'ditolak (completed, result null)' : e.status === 'DONE' ? 'completed' : 'failed'
    title = `Orkestrasi (diperankan tracker) menerima callback hasil "${what}" dari ${by}, percobaan ke-${e.attempt}: dijawab ${e.http_status}.`
  } else {
    title = `Orkestrasi (diperankan tracker) menerima callback per tahap ${label(e.stage)} ${e.status}${e.final ? ' (final)' : ''}, percobaan ke-${e.attempt}: dijawab ${e.http_status}.`
  }
  const notes = []
  if (e.mode === 'key') notes.push('X-Callback-Key salah: ditolak 401, relay berhenti dan pesannya jadi dead letter.')
  else if (e.http_status >= 500) notes.push(`Simulasi ${e.mode}: relay akan mengulang dengan backoff.`)
  else if (e.http_status === 409) notes.push('409 RESULT_NOT_READY: relay mengulang beberapa kali.')
  else if (e.http_status >= 400) notes.push(`Simulasi ${e.mode}: penolakan permanen, relay berhenti dan pesannya jadi dead letter.`)
  if (e.duplicate) notes.push('DUPLIKAT: pesan yang sama sudah pernah diterima (at-least-once). Penerima harus idempoten.')
  if (e.format === 'result' && e.final && e.http_status < 300 && !e.duplicate) {
    notes.push('Ini satu-satunya callback request ini. Callback tetap dikirim walaupun POST sudah dijawab 200: service pipeline tidak tahu jawaban mana (200 atau 202) yang diterima pusat.')
  } else if (e.format !== 'result') {
    notes.push('Format stage: satu callback per tahap (default lokal lama). Di dev dipakai format result: satu callback per request.')
  }
  return { title, detail: notes.join(' ') }
}

function ocrResultEntry(e) {
  const r = e.row ?? {}
  const pls = r.pipeline_last_stage
  const title = `nilam_ocr_results: baris baru, HTTP ${r.status_code}${r.errors ? ` ${r.errors}` : ''}, guardrails ${r.guardrails ?? 'null'}, pipeline_last_stage ${pls ?? 'null'}.`
  let detail = 'Tabel ini mencatat setiap jawaban ke pusat (jawaban POST dan setiap callback yang terkirim), append-only.'
  if (r.status_code === 200 && !pls) detail = `Baris jawaban POST extract-ocr. ${detail}`
  if (r.status_code === 200 && pls) {
    detail = `Baris callback hasil yang terkirim. Di tabel log ini pipeline_last_stage berisi service pengirim callback (${pls}); di jawaban API, 200 selalu null.`
  }
  return { title, detail }
}

function describe(e, t0, events) {
  switch (e.type) {
    case 'client': {
      const sim = []
      if (e.slow) sim.push(`OCR ditunda ${e.slow} dtk (nama file delay${e.slow}s-…)`)
      if (e.callback_mode && e.callback_mode !== 'ok') sim.push(`callback orkestrasi ${e.callback_mode}`)
      if (e.sequence) sim.push(`pipeline_name_sequence ${e.sequence.join(' → ')}`)
      if (e.source === 'file_url') sim.push('dikirim sebagai file_url')
      if (e.origin) sim.push(e.origin.replace('scenario:', 'skenario '))
      const what = e.resend ? `KIRIM ULANG request_id yang sama (${e.filename})` : `mengirim ${e.filename} ke POST /v1/extract-ocr`
      return { title: `Orkestrasi (diperankan tracker) ${what}.`, detail: sim.join(' · ') || undefined }
    }
    case 'http':
      return httpEntry(e)
    case 'stage':
      return stageEntry(e, events)
    case 'outbox':
      return outboxEntry(e, t0)
    case 'callback':
      return callbackEntry(e)
    case 'ocr_results':
      return ocrResultEntry(e)
    case 'chaos':
      if (e.stage === 'DATABASE') return { title: e.status === 'DOWN' ? `Database tidak terbaca: ${e.error_message}` : 'Database terbaca lagi.' }
      return { title: `Gangguan: ${e.text}` }
    case 'pipeline':
      if (e.timeout) return { title: 'Pemantauan dihentikan: waktu habis.' }
      return e.dead_letters
        ? { title: `Selesai dengan ${e.dead_letters} dead letter di outbox.`, detail: 'Pesan itu tidak akan dicoba lagi sampai dilepas.' }
        : { title: 'Selesai: tidak ada pesan outbox yang tertinggal, semua sudah dikirim atau memang tidak perlu dikirim.' }
    default:
      return { title: `${e.stage} ${e.status}` }
  }
}

function actorOf(e) {
  switch (e.type) {
    case 'outbox':
      return e.source === 'log' ? 'OUTBOX · LOG' : 'OUTBOX · DB'
    case 'callback':
      return 'CALLBACK'
    case 'http':
      return 'HTTP'
    case 'ocr_results':
      return 'DB'
    case 'client':
      return 'CLIENT'
    case 'pipeline':
      return 'END'
    case 'stage':
      return e.source === 'db' ? `${e.stage} · DB` : e.stage
    default:
      return (e.stage || 'STAGE').toUpperCase()
  }
}

function ContractCheck({ report, body }) {
  const [open, setOpen] = useState(report.verdict === 'fail')
  const passed = report.checks.filter((c) => c.level === 'pass').length

  const verdictVariant = report.verdict === 'fail' ? 'destructive' : report.verdict === 'warn' ? 'warning' : 'done'

  return (
    <div className={`mt-2 rounded-lg border p-3 bg-background/60 transition-all ${report.verdict === 'fail' ? 'border-rose-500/40 bg-rose-500/5' : 'border-border/60'}`}>
      <div className="flex items-center justify-between cursor-pointer select-none" onClick={() => setOpen(!open)}>
        <div className="flex items-center gap-2 flex-wrap">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Badge variant={verdictVariant} className="text-[10px]">
            {CONTRACT_LABEL[report.verdict] ?? report.verdict}
          </Badge>
          <span className="text-xs font-semibold text-foreground">
            {CALLBACK_KIND[report.kind] ?? report.kind}
          </span>
          <span className="text-[11px] text-muted-foreground font-mono">
            · {passed}/{report.checks.length} aturan lulus
          </span>
        </div>
        <span className="text-[11px] text-muted-foreground">
          {open ? 'Tutup kontrak' : 'Lihat aturan'}
        </span>
      </div>

      {open && (
        <div className="mt-3 pt-3 border-t border-border/40 space-y-2">
          <ul className="space-y-1.5 m-0 p-0 list-none">
            {report.checks.map((c, i) => (
              <li key={i} className="flex items-baseline gap-2 text-xs">
                <Badge
                  variant={c.level === 'fail' ? 'destructive' : c.level === 'warn' ? 'warning' : 'done'}
                  className="text-[9px] px-1.5 py-0 uppercase font-mono shrink-0"
                >
                  {c.level}
                </Badge>
                <span className="font-medium text-foreground">{c.rule}</span>
                {c.detail && <span className="text-muted-foreground break-all"> · {c.detail}</span>}
              </li>
            ))}
          </ul>
          {body != null && (
            <div className="mt-2.5">
              <JsonViewer value={body} label="Payload Body Callback" defaultOpen={false} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ReadingGuide() {
  return (
    <div className="rounded-lg border border-border/70 bg-background/60 p-3 text-[11px] leading-relaxed text-muted-foreground space-y-1.5">
      <div className="flex items-center gap-1.5 font-semibold text-foreground text-xs">
        <Info size={13} />
        <span>Cara membaca</span>
      </div>
      <p className="m-0">
        Tiap baris: kalimat tebal = apa yang terjadi, kalimat abu-abu = artinya. Label di kiri = dari mana tracker
        mengetahuinya: <b>HTTP</b> jawaban orchestrator, <b>· DB</b> dibaca dari PostgreSQL (tiap 0,25 dtk),{' '}
        <b>OUTBOX · LOG</b> dicatat relay di log service-nya, <b>CALLBACK</b> diterima tracker yang berperan sebagai
        Orkestrasi pusat.
      </p>
      <p className="m-0">
        <b>Alur outbox di jalur normal:</b> tahap selesai → hasil + pesan keluar disimpan dalam satu transaksi → relay
        mengirim → baris dihapus. Baris hanya hidup beberapa milidetik, terlalu cepat untuk terbaca DB tiap 0,25 dtk,
        jadi buktinya diambil dari log relay. Status QUEUED, CLAIMED, RETRY, dan DEAD dari tabel baru muncul kalau
        pengiriman lambat atau gagal (lihat menu Skenario gangguan).
      </p>
    </div>
  )
}

export default function TimelineView({ events = [], t0 = 0 }) {
  const [filter, setFilter] = useState('all')
  const [showGuide, setShowGuide] = useState(true)

  // Event dari log relay membawa waktu log-nya sendiri dan bisa tiba sedikit terlambat: urutkan menurut waktu kejadian.
  const ordered = events.map((e, i) => ({ e, i })).sort((a, b) => a.e.ts - b.e.ts || a.i - b.i).map(({ e }) => e)

  const filteredEvents = ordered.filter((e) => {
    if (filter === 'all') return true
    if (filter === 'http') return e.type === 'http'
    if (filter === 'outbox') return e.type === 'outbox' || (e.type === 'stage' && e.source === 'db' && e.status !== 'PROCESSING')
    if (filter === 'callback') return e.type === 'callback'
    if (filter === 'stage') return e.type === 'stage' || e.type === 'chaos' || e.type === 'pipeline'
    return true
  })

  return (
    <Card className="border-border bg-card border-t-2 border-t-indigo-500 shadow-xs">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="h-7 w-7 rounded-md bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
              <Clock size={15} />
            </div>
            <CardTitle className="text-sm font-semibold">
              Timeline Event Real-Time (SSE)
            </CardTitle>
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
              Observability Stream
            </span>
          </div>
          <CardDescription className="text-xs">
            Urutan kejadian request ini dalam milidetik sejak dokumen dikirim (t+0)
          </CardDescription>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => setShowGuide(!showGuide)}
            className="h-6 px-2 text-[11px] text-muted-foreground"
          >
            {showGuide ? 'Sembunyikan panduan' : 'Cara membaca'}
          </Button>
          {/* Filter Tabs */}
          <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border/50">
            {[
              { id: 'all', label: 'Semua' },
              { id: 'http', label: 'HTTP' },
              { id: 'stage', label: 'Stage' },
              { id: 'outbox', label: 'Outbox' },
              { id: 'callback', label: 'Callback' },
            ].map((tab) => (
              <Button
                key={tab.id}
                type="button"
                variant={filter === tab.id ? 'secondary' : 'ghost'}
                size="xs"
                onClick={() => setFilter(tab.id)}
                className={`h-6 px-2 text-[11px] ${
                  filter === tab.id
                    ? 'bg-background shadow-xs font-semibold text-foreground border border-border/50'
                    : 'text-muted-foreground'
                }`}
              >
                {tab.label}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-2 space-y-3">
        {showGuide && <ReadingGuide />}

        {filteredEvents.length === 0 && (
          <div className="p-8 text-center text-xs text-muted-foreground">
            Belum ada event tercatat.
          </div>
        )}

        <div className="space-y-3 relative before:absolute before:inset-0 before:left-3 before:w-[1px] before:bg-border/60">
          {filteredEvents.map((e, i) => {
            const type = e.type
            const badgeVariant = type === 'outbox' ? 'processing' : type === 'callback' ? 'done' : type === 'http' ? 'rejected' : 'outline'
            const { title, detail } = describe(e, t0, events)

            return (
              <div key={i} className="flex items-start gap-3 relative pl-1">
                <div className="w-5 h-5 rounded-md bg-muted border border-border flex items-center justify-center font-mono text-[10px] font-medium text-muted-foreground shrink-0 z-10">
                  {i + 1}
                </div>
                <div className="flex-1 rounded-md border border-border bg-muted/20 p-3 space-y-1.5 transition-colors hover:bg-muted/30">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <Badge variant={badgeVariant} className="text-[10px] px-1.5 py-0 font-mono">
                        {actorOf(e)}
                      </Badge>
                      <span className="text-xs font-mono font-medium text-foreground">
                        t+{fmtMs((e.ts - t0) * 1000)}
                      </span>
                    </div>
                    {e.status && (
                      <span className="text-[11px] font-mono text-muted-foreground">
                        status: {e.status}
                      </span>
                    )}
                  </div>

                  <div className="text-xs text-foreground font-medium leading-relaxed">{title}</div>
                  {detail && <div className="text-[11px] text-muted-foreground leading-relaxed">{detail}</div>}

                  {e.type === 'callback' && e.contract && (
                    <ContractCheck report={e.contract} body={e.payload} />
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
