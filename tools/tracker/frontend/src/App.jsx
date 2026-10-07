import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const STAGES = ['GUARDRAILS', 'OCR', 'STRUCTURING', 'SCORING']
const LABELS = {
  GUARDRAILS: 'Orchestrator + guardrails',
  OCR: 'Extraction OCR',
  STRUCTURING: 'Structuring',
  SCORING: 'Scoring',
}
const OUTBOX_STAGES = ['OCR', 'STRUCTURING', 'SCORING']
const CALLBACK_MODES = [
  { value: 'ok', label: 'normal (2xx)', hint: 'callback diterima, baris outbox dihapus' },
  { value: 'down', label: 'mati (503)', hint: 'relay mengulang dengan backoff; pipeline tetap jalan' },
  { value: 'reject', label: 'menolak (422)', hint: 'relay berhenti: baris jadi dead letter' },
  { value: 'unauthorized', label: 'X-Callback-Key salah (401)', hint: 'sama dengan 422: dead letter, perlu release manual' },
  { value: 'slow', label: 'lambat (> timeout relay)', hint: 'dicatat lalu dijawab terlambat: relay mengirim ulang, pusat menerima duplikat' },
  { value: 'flaky', label: 'tersendat (503 lalu 200)', hint: 'beberapa kedatangan pertama tiap pesan 503, sesudahnya 200' },
  {
    value: 'not_ready',
    label: 'belum siap (409 RESULT_NOT_READY)',
    hint: 'pusat belum mencatat 202-nya: relay mengirim ulang tiap 1,5 dtk, lalu diterima',
  },
]
const SOURCES = [
  { value: 'upload', label: 'upload (multipart file)' },
  { value: 'file_url', label: 'file_url (seperti contoh cURL pusat)' },
]
// pipeline_name_sequence: service yang dijalankan satu request. Urutannya selalu guardrails -> extraction ->
// structuring -> scoring; guardrails boleh tidak ada di depan dan ujungnya boleh dipotong, tidak boleh melompat.
const SERVICE_OF_STAGE = { GUARDRAILS: 'guardrails', OCR: 'extraction', STRUCTURING: 'structuring', SCORING: 'scoring' }
const PIPELINE_NAMES = ['guardrails', 'extraction', 'structuring', 'scoring']
const SERVICE_LABELS = { guardrails: 'guardrails', extraction: 'extraction (OCR)', structuring: 'structuring', scoring: 'scoring' }
// Pintasan untuk toggle; kombinasi lain tinggal diklik sendiri.
const SEQUENCE_PRESETS = [
  { label: 'penuh', value: PIPELINE_NAMES },
  { label: 'tanpa guardrails', value: ['extraction', 'structuring', 'scoring'] },
  { label: 'guardrails saja', value: ['guardrails'] },
  { label: 'sampai extraction', value: ['guardrails', 'extraction'] },
  { label: 'sampai structuring', value: ['guardrails', 'extraction', 'structuring'] },
  { label: 'extraction saja', value: ['extraction'] },
]
const INVALID_SEQUENCE = 'INVALID_PIPELINE_SEQUENCE'

// Aturan yang sama dengan validate_sequence di orchestrator (ocr_common/pipeline/sequence.py), supaya UI bisa
// memberi tahu lebih dulu kalau sequence ini akan dijawab 422. null = valid.
function sequenceProblem(names) {
  if (!names.length) return 'pilih minimal satu service'
  const start = PIPELINE_NAMES.indexOf(names[0])
  if (start > PIPELINE_NAMES.indexOf('extraction')) return `${names[0]} tidak bisa di depan: butuh hasil ${PIPELINE_NAMES[start - 1]}`
  if (names.some((name, i) => name !== PIPELINE_NAMES[start + i])) return 'tidak boleh melompati service di tengah'
  return null
}

// Yang dikirim sebagai pipeline_name_sequence: null (field tidak dikirim, orchestrator memakai pipeline penuh)
// kalau keempatnya menyala, selain itu daftar service yang menyala dalam urutan pipeline.
function sequenceToSend(names) {
  return names.length === PIPELINE_NAMES.length ? null : names
}

// Toggle on/off per service. `names`: service yang menyala, selalu dalam urutan pipeline. `allowInvalid`: sequence
// yang melanggar aturan tetap boleh dikirim (form satu dokumen, supaya jawaban 422 orchestrator bisa dilihat).
function SequenceToggles({ names, onChange, allowInvalid }) {
  const toggle = (name) =>
    onChange(PIPELINE_NAMES.filter((n) => (n === name ? !names.includes(n) : names.includes(n))))
  const problem = sequenceProblem(names)
  const sent = sequenceToSend(names)
  return (
    <div className="sequence">
      <div className="sequence-label">pipeline_name_sequence</div>
      <div className="toggles">
        {PIPELINE_NAMES.map((name, i) => (
          <Fragment key={name}>
            {i > 0 && <span className="arrow">→</span>}
            <button
              type="button"
              className={`toggle ${names.includes(name) ? 'on' : ''}`}
              aria-pressed={names.includes(name)}
              onClick={() => toggle(name)}
            >
              {SERVICE_LABELS[name]}
            </button>
          </Fragment>
        ))}
      </div>
      <div className="presets">
        {SEQUENCE_PRESETS.map((p) => (
          <button key={p.label} type="button" className="link small-link" onClick={() => onChange(p.value)}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="hint small">
        dikirim: {sent ? <code>{JSON.stringify(sent)}</code> : 'tidak dikirim (orchestrator memakai pipeline penuh)'}
      </div>
      {problem && (
        <div className={allowInvalid && names.length ? 'warn small' : 'error small'}>
          {allowInvalid && names.length ? `akan ditolak orchestrator: 422 ${INVALID_SEQUENCE} (${problem})` : problem}
        </div>
      )}
    </div>
  )
}

function sequenceText(sequence) {
  return sequence ? sequence.join(' → ') : 'penuh'
}

// Service terakhir di pipeline_name_sequence (penuh: scoring). pipeline_last_stage null pada jawaban sukses.
function lastOfSequence(sequence) {
  return sequence && sequence.length ? sequence[sequence.length - 1] : 'scoring'
}

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function fmtSec(seconds) {
  if (seconds == null) return '-'
  return seconds < 60 ? `${seconds.toFixed(0)} dtk` : `${(seconds / 60).toFixed(1)} mnt`
}

// --- turunan dari daftar event ------------------------------------------------

// `sequence`: pipeline_name_sequence request ini (null = penuh). Tahap yang tidak ada di dalamnya tidak dijalankan.
function stageView(events, stage, sequence) {
  const mine = events.filter((e) => e.stage === stage && e.type === 'stage')
  const callbacks = events.filter((e) => e.stage === stage && e.type === 'callback')
  const http = stage === 'GUARDRAILS' ? events.find((e) => e.type === 'http') : null
  if (sequence && !sequence.includes(SERVICE_OF_STAGE[stage])) return { status: 'SKIPPED', callbacks, http }
  if (mine.length === 0 && callbacks.length === 0) return { status: 'PENDING', callbacks, http }
  const last = mine[mine.length - 1]
  // Penolakan aturan structuring datang dua kali (baris DB REJECTED dan callback FAILED) dalam urutan acak.
  const rejected = mine.find((e) => e.status === 'REJECTED')
  const status = rejected ? 'REJECTED' : last ? last.status : 'PROCESSING'
  const started = mine.find((e) => e.status === 'PROCESSING')?.ts
  const finished = last && ['DONE', 'FAILED', 'REJECTED'].includes(last.status) ? last.ts : null
  let elapsed = last?.elapsed_ms
  if (elapsed == null && started && finished) elapsed = (finished - started) * 1000
  const withResult = [...mine].reverse().find((e) => e.result)
  const db = mine.find((e) => e.source === 'db' && ['DONE', 'FAILED', 'REJECTED'].includes(e.status))
  const acknowledged = mine.find((e) => e.source === 'callback' && ['DONE', 'FAILED'].includes(e.status))
  return {
    status,
    elapsed,
    result: withResult?.result,
    error: last?.error_message,
    db,
    acknowledged,
    callbacks,
    http,
  }
}

function outboxView(events) {
  const rows = new Map()
  for (const e of events) {
    if (e.type !== 'outbox' || !e.message) continue
    const row = rows.get(e.message.id) ?? { history: [] }
    row.history.push({ status: e.status, ts: e.ts, message: e.message })
    row.state = e.status
    row.message = e.message
    rows.set(e.message.id, row)
  }
  return [...rows.values()].sort((a, b) => a.message.id - b.message.id)
}

function describe(e, t0) {
  const m = e.message
  switch (e.type) {
    case 'client': {
      const sim = []
      if (e.slow) sim.push(`OCR ditunda ${e.slow} dtk (nama file delay${e.slow}s-…)`)
      if (e.callback_mode && e.callback_mode !== 'ok') sim.push(`callback orkestrasi ${e.callback_mode}`)
      if (e.sequence) sim.push(`pipeline_name_sequence ${sequenceText(e.sequence)}`)
      if (e.source === 'file_url') sim.push('dikirim sebagai file_url')
      if (e.origin) sim.push(e.origin.replace('scenario:', 'skenario '))
      const what = e.resend ? `KIRIM ULANG request_id yang sama (${e.filename})` : `menerima ${e.filename}`
      return `Orkestrasi (tracker) ${what}${sim.length ? ` · ${sim.join(', ')}` : ''}`
    }
    case 'http':
      return `Orchestrator menjawab HTTP ${e.http_status}, job_status=${e.job_status ?? '-'}${
        e.errors ? ` (${e.errors})` : ''
      }, pipeline_last_stage=${e.pipeline_last_stage ?? 'null'} setelah ${fmtMs(e.elapsed_ms)} · batas tunggu ${e.wait_seconds} dtk`
    case 'stage':
      if (e.stage === 'GUARDRAILS' && e.status === 'SKIPPED') return `${LABELS.GUARDRAILS}: model guardrails dilewati (tidak ada di pipeline_name_sequence)`
      if (e.stage === 'GUARDRAILS') return `${LABELS.GUARDRAILS}: ${e.status}${e.error_message ? ` · ${e.error_message}` : ''}`
      if (e.status === 'REJECTED') return `${LABELS[e.stage]}: dokumen ditolak aturan ML · ${e.error_message}`
      if (e.source === 'db') {
        if (e.status === 'PROCESSING') return `${LABELS[e.stage]}: job diklaim (INSERT ${e.stage.toLowerCase()}_jobs PROCESSING)`
        if (e.status === 'DONE') return `${LABELS[e.stage]}: hasil tersimpan, job DONE (transaksi commit)`
        return `${LABELS[e.stage]}: job FAILED · ${e.error_message}`
      }
      if (e.source === 'callback') return `Orkestrasi mencatat ${e.stage} ${e.status} (callback diterima)`
      return `${LABELS[e.stage] ?? e.stage}: ${e.status}`
    case 'outbox': {
      if (e.status === 'UNAVAILABLE') return `Outbox tidak bisa dibaca: ${e.error_message ?? 'TRACKER_DATABASE_URL kosong'}`
      const what = m.kind === 'handoff' ? `handoff → ${m.target}` : m.message
      switch (e.status) {
        case 'QUEUED':
          return `Outbox: baris #${m.id} ${what} diantrekan oleh ${m.owner}, dalam transaksi yang sama dengan hasilnya`
        case 'CLAIMED':
          return `Relay ${m.owner} mengklaim #${m.id} (attempt ${m.attempts}, FOR UPDATE SKIP LOCKED, lease 30 dtk)`
        case 'DELIVERED':
          return `#${m.id} ${what} terkirim, baris dihapus`
        case 'RETRY': {
          const at = m.next_attempt_at ? ((new Date(m.next_attempt_at).getTime() / 1000 - t0) * 1000) : null
          return `#${m.id} ${what} gagal: ${m.last_error} · dicoba lagi pada t+${fmtMs(at)} (backoff)`
        }
        case 'DEAD':
          return `#${m.id} ${what} DEAD LETTER setelah ${m.attempts} attempt: ${m.last_error} · baris tetap ada`
        case 'RELEASED':
          return `#${m.id} ${what} dilepas dari dead letter, akan dicoba lagi`
        default:
          return `#${m.id} ${e.status}`
      }
    }
    case 'callback': {
      const kind =
        e.format === 'result' ? `Callback result (${e.status === 'DONE' ? 'completed' : 'failed'}, dari ${e.stage})` : `Callback ${e.stage} ${e.status}`
      const note = e.mode === 'key' ? ' (X-Callback-Key salah)' : e.mode !== 'ok' ? ` (simulasi ${e.mode})` : ''
      return `${kind}${e.final && e.format !== 'result' ? ' (final)' : ''} tiba (attempt ${e.attempt}) → orkestrasi menjawab ${e.http_status}${note}${
        e.duplicate ? ' · DUPLIKAT: pesan ini sudah pernah diterima' : ''
      }`
    }
    case 'chaos':
      if (e.stage === 'DATABASE') return e.status === 'DOWN' ? `Database tidak terbaca: ${e.error_message}` : 'Database terbaca lagi'
      return `Gangguan: ${e.text}`
    case 'pipeline':
      if (e.timeout) return 'Pemantauan dihentikan: waktu habis'
      return e.dead_letters ? `Selesai dengan ${e.dead_letters} dead letter di outbox` : 'Selesai: outbox kosong, semua pesan terkirim'
    default:
      return `${e.stage} ${e.status}`
  }
}

// --- komponen ------------------------------------------------------------------

function Json({ value, label = 'response' }) {
  const [open, setOpen] = useState(false)
  if (value == null) return null
  const text = JSON.stringify(value, null, 2)
  return (
    <div className="json">
      <button className="link" onClick={() => setOpen(!open)}>
        {open ? `sembunyikan ${label}` : `lihat ${label} (${text.length.toLocaleString()} karakter)`}
      </button>
      {open && <pre>{text}</pre>}
    </div>
  )
}

function Summary({ stage, result }) {
  if (!result) return null
  if (stage === 'OCR') {
    return (
      <div className="summary">
        <b>{result.blocks?.length ?? 0}</b> baris · engine {result.engine} · model {result.model ?? '-'} ·{' '}
        {fmtMs(result.elapsed_ms)} di model
        <ul className="lines">
          {(result.blocks ?? []).map((b, i) => (
            <li key={i}>
              <span className="conf">{b.confidence.toFixed(2)}</span> {b.text}
            </li>
          ))}
        </ul>
      </div>
    )
  }
  if (stage === 'STRUCTURING') {
    return (
      <table className="fields">
        <tbody>
          {Object.entries(result.fields ?? {}).map(([name, f]) => (
            <tr key={name}>
              <td className="name">{name}</td>
              <td className={f.value == null ? 'missing' : ''}>{f.value ?? '(tidak ditemukan)'}</td>
              <td className="conf">{f.confidence.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }
  if (stage === 'SCORING') {
    const s = result.scoring ?? result
    const pct = (v) => (v == null ? '(field kosong)' : `${(v * 100).toFixed(1)}%`)
    return (
      <div className="summary">
        <table className="fields">
          <tbody>
            <tr><td className="name">npwp_confidence</td><td><b>{pct(s.npwp_confidence)}</b></td></tr>
            <tr><td className="name">name_confidence</td><td><b>{pct(s.name_confidence)}</b></td></tr>
          </tbody>
        </table>
      </div>
    )
  }
  return null
}

function httpClass(status) {
  return status === 200 ? 'done' : status === 202 ? 'processing' : status === 400 ? 'rejected' : 'failed'
}

function httpNote(status, rejectedBy, body, sequence) {
  const last = lastOfSequence(sequence)
  if (status === 200) return last !== 'scoring' ? `200: data = hasil ${last} apa adanya` : '200: hasil lengkap ada di response'
  if (status === 202) return '202: hanya request_id, hasil menyusul lewat callback'
  if (status === 422 && body?.errors === INVALID_SEQUENCE) return '422: pipeline_name_sequence tidak valid, tidak ada yang jalan'
  if (status === 422) return '422: satu tahap gagal di dalam batas tunggu'
  if (status === 400) return rejectedBy === 'structuring' ? '400: ditolak aturan structuring' : '400: ditolak guardrails'
  return `HTTP ${status}`
}

function GuardrailsResponse({ http, t0 }) {
  if (!http) return null
  const cls = httpClass(http.http_status)
  const within = http.elapsed_ms / 1000 <= http.wait_seconds
  return (
    <div className={`http ${cls}`}>
      <div className="http-head">
        <span className={`pill ${cls}`}>HTTP {http.http_status}</span>
        <span className="label">job_status = {http.job_status ?? '-'}</span>
        {http.errors && <span className="error-code">{http.errors}</span>}
        <span className="elapsed">
          dijawab setelah {fmtMs(http.elapsed_ms)} {within ? '≤' : '>'} batas tunggu {http.wait_seconds} dtk
        </span>
      </div>
      <div className="meta">
        pipeline_name_sequence: <code>{sequenceText(http.sequence)}</code> · pipeline_last_stage:{' '}
        <code>{http.pipeline_last_stage ?? 'null'}</code>
      </div>
      <div className="meta">
        {http.http_status === 200 &&
          (lastOfSequence(http.sequence) !== 'scoring'
            ? `Request berhenti di ${lastOfSequence(http.sequence)} sesuai sequence: data adalah hasil ${lastOfSequence(http.sequence)} apa adanya, bukan field kontrak.`
            : 'Pipeline selesai di dalam batas tunggu: hasil langsung ada di respons, callback yang menyusul boleh diabaikan.')}
        {http.http_status === 202 && 'Batas tunggu habis sebelum tahap terakhir selesai: hasil menyusul lewat callback, dan bisa dibaca kapan saja lewat GET /v1/extract-ocr/{request_id} di orchestrator.'}
        {http.http_status === 422 &&
          (http.errors === INVALID_SEQUENCE
            ? 'pipeline_name_sequence tidak valid: orchestrator menolak sebelum menjalankan apa pun.'
            : 'Satu tahap gagal di dalam batas tunggu.')}
        {http.http_status === 400 &&
          (http.rejected_by === 'structuring'
            ? 'Lolos guardrails, lalu ditolak aturan structuring ML: OCR dan structuring sudah jalan, scoring tidak.'
            : 'Dokumen ditolak guardrails: tidak ada tahap yang dijalankan.')}
      </div>
      <Json value={http.body} label="response extract-ocr" />
    </div>
  )
}

function Stage({ stage, view, index, t0, last }) {
  const cls = `stage ${view.status.toLowerCase()}`
  const rel = (ts) => `t+${fmtMs((ts - t0) * 1000)}`
  const rejected = view.callbacks.filter((c) => !c.accepted)
  const lastCallback = view.callbacks[view.callbacks.length - 1]
  if (view.status === 'SKIPPED') {
    return (
      <div className={cls}>
        <div className="stage-head">
          <span className="index">{index + 1}</span>
          <span className="label">{LABELS[stage]}</span>
          <span className="pill skipped">SKIPPED</span>
        </div>
        <div className="hint small">
          Tidak dijalankan: <code>{SERVICE_OF_STAGE[stage]}</code> tidak ada di pipeline_name_sequence.
          {stage === 'GUARDRAILS' && ' Pengecekan file di orchestrator tetap jalan.'}
        </div>
      </div>
    )
  }
  return (
    <div className={cls}>
      <div className="stage-head">
        <span className="index">{index + 1}</span>
        <span className="label">{LABELS[stage]}</span>
        <span className={`pill ${view.status.toLowerCase()}`}>{view.status}</span>
        {last && <span className="tag" title="Hasil tahap ini adalah data jawaban, dan callback DONE-nya membawa final: true">tahap terakhir</span>}
        {view.elapsed != null && <span className="elapsed">{fmtMs(view.elapsed)}</span>}
      </div>
      {view.error && <div className="error">{view.error}</div>}
      {stage !== 'GUARDRAILS' && (
        <ul className="facts">
          <li>
            <span className="k">hasil di DB</span>
            {view.db ? `${view.db.status} pada ${rel(view.db.ts)}` : 'belum'}
          </li>
          <li>
            <span className="k">orkestrasi tahu</span>
            {view.acknowledged
              ? `callback diterima ${rel(view.acknowledged.ts)}${lastCallback ? ` (attempt ${lastCallback.attempt})` : ''}`
              : rejected.length
                ? `belum: ${rejected.length} callback dijawab ${rejected[rejected.length - 1].http_status}, relay mengulang`
                : 'belum'}
          </li>
        </ul>
      )}
      <Summary stage={stage} result={view.result} />
      <Json value={view.result} />
    </div>
  )
}

const OUTBOX_STATE = {
  QUEUED: 'processing',
  CLAIMED: 'processing',
  RETRY: 'rejected',
  DELIVERED: 'done',
  DEAD: 'failed',
  RELEASED: 'processing',
}

function OutboxPanel({ rows, t0, onRelease, releasing, unavailable }) {
  const dead = rows.filter((r) => r.state === 'DEAD').length
  const rel = (ts) => `t+${fmtMs((ts - t0) * 1000)}`
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>pipeline_outbox untuk request ini</h2>
        {dead > 0 && (
          <button className="small" disabled={releasing} onClick={onRelease}>
            {releasing ? 'melepas…' : `Lepaskan ${dead} dead letter`}
          </button>
        )}
      </div>
      {unavailable && <div className="error">{unavailable}</div>}
      {!unavailable && rows.length === 0 && <p className="hint">Belum ada baris. Baris muncul saat sebuah tahap meng-commit hasilnya.</p>}
      {rows.length > 0 && (
        <table className="outbox">
          <thead>
            <tr>
              <th>#</th>
              <th>ditulis oleh</th>
              <th>pesan</th>
              <th>status</th>
              <th>attempt</th>
              <th>riwayat</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.message.id} className={OUTBOX_STATE[r.state]}>
                <td className="mono">{r.message.id}</td>
                <td>{r.message.owner}</td>
                <td>
                  <span className={`kind ${r.message.kind}`}>{r.message.kind}</span>{' '}
                  {r.message.kind === 'handoff' ? `→ ${r.message.target}` : r.message.message}
                </td>
                <td>
                  <span className={`pill ${OUTBOX_STATE[r.state]}`}>{r.state}</span>
                  {r.message.last_error && r.state !== 'DELIVERED' && <div className="error small">{r.message.last_error}</div>}
                </td>
                <td className="mono">{r.message.attempts}</td>
                <td className="history">
                  {r.history.map((h, i) => (
                    <span key={i}>
                      {h.status.toLowerCase()} {rel(h.ts)}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <ul className="legend">
        <li><b>QUEUED</b>: ditulis dalam transaksi yang sama dengan hasil tahap. Pod mati setelah ini pun pesannya tidak hilang.</li>
        <li><b>CLAIMED</b>: relay tahap itu mengambilnya (<code>FOR UPDATE SKIP LOCKED</code>, lease 30 dtk) dan mengirim; handoff dulu, baru callback.</li>
        <li><b>DELIVERED</b>: penerima menjawab 2xx, baris dihapus. <b>RETRY</b>: 5xx / timeout, dicoba lagi dengan backoff sampai 24 jam. <b>DEAD</b>: 4xx atau lewat umur; baris menetap sampai dilepas.</li>
      </ul>
    </section>
  )
}

function Timeline({ events, t0 }) {
  return (
    <section className="panel">
      <h2>Timeline</h2>
      <ol className="timeline">
        {events.map((e, i) => (
          <li key={i} className={`ev ${e.type} ${(e.status ?? '').toLowerCase()}`}>
            <span className="t">t+{fmtMs((e.ts - t0) * 1000)}</span>
            <span className="who">{e.type === 'outbox' ? 'outbox' : e.type === 'callback' ? 'callback' : e.type === 'http' ? 'http' : e.stage.toLowerCase()}</span>
            <div className="what">
              {describe(e, t0)}
              {e.type === 'callback' && e.contract && <ContractCheck report={e.contract} body={e.payload} />}
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

const CONTRACT_LABEL = { pass: 'sesuai kontrak', warn: 'sesuai kontrak, ada catatan', fail: 'MELANGGAR kontrak' }
const CALLBACK_KIND = {
  completed: 'completed (selesai)',
  rejected: 'completed, result null (ditolak)',
  failed: 'failed',
  stage: 'format stage (lokal, bukan kontrak pusat)',
  invalid: 'tidak terbaca',
}

// Hasil cek kontrak satu callback (backend/contract.py) dan body JSON yang benar-benar tiba. Terbuka sendiri
// kalau melanggar.
function ContractCheck({ report, body }) {
  const passed = report.checks.filter((c) => c.level === 'pass').length
  return (
    <details className={`contract ${report.verdict}`} open={report.verdict === 'fail'}>
      <summary>
        <span className={`pill level-${report.verdict}`}>{CONTRACT_LABEL[report.verdict] ?? report.verdict}</span>{' '}
        {CALLBACK_KIND[report.kind] ?? report.kind} · {passed}/{report.checks.length} cek lulus · body JSON
      </summary>
      <ul className="contract-checks">
        {report.checks.map((c, i) => (
          <li key={i}>
            <span className={`pill level-${c.level}`}>{c.level}</span>
            <span>
              {c.rule}
              {c.detail && <span className="detail"> · {c.detail}</span>}
            </span>
          </li>
        ))}
      </ul>
      {body != null && (
        <div className="json">
          <pre>{JSON.stringify(body, null, 2)}</pre>
        </div>
      )}
    </details>
  )
}

function Backlog({ overview }) {
  if (!overview) return null
  return (
    <div className="backlog">
      {OUTBOX_STAGES.map((stage) => {
        const s = overview[stage] ?? {}
        const bad = (s.dead_letters ?? 0) > 0 || (s.oldest_pending_seconds ?? 0) > 300
        return (
          <div key={stage} className={`backlog-item ${bad ? 'bad' : ''}`}>
            <span className="k">{stage}</span>
            {s.error ? (
              <span className="v">?</span>
            ) : s.enabled === false ? (
              <span className="v">outbox mati</span>
            ) : (
              <span className="v">
                {s.pending ?? 0} pending · {s.retrying ?? 0} retry · {s.dead_letters ?? 0} dead
                {s.oldest_pending_seconds != null && ` · tertua ${fmtSec(s.oldest_pending_seconds)}`}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

// --- load testing ----------------------------------------------------------------

// Dokumen yang ditolak dijawab 400 DOWNSTREAM_VALIDATION_ERROR, dihitung sendiri: bukan request yang salah.
const STATUS_BUCKETS = [
  { key: '200', label: '200 selesai', cls: 'done' },
  { key: 'rejected', label: '400 ditolak', cls: 'rejected' },
  { key: '202', label: '202 menyusul', cls: 'processing' },
  { key: '4xx', label: '4xx', cls: 'failed' },
  { key: '5xx', label: '5xx', cls: 'failed' },
  { key: 'timeout', label: 'timeout', cls: 'timeout' },
]

function pct(n, total) {
  return total ? `${((100 * n) / total).toFixed(1)}%` : '-'
}

function fmtRate(v) {
  return v == null ? '-' : `${v.toFixed(2)} rps`
}

function runStatusClass(status) {
  return status === 'running' ? 'processing' : status === 'finished' ? 'done' : status === 'failed' ? 'failed' : 'idle'
}

function Tile({ k, v, s, cls }) {
  return (
    <div className={`tile ${cls ?? ''}`}>
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {s && <div className="s">{s}</div>}
    </div>
  )
}

// Default MAX_UPLOAD_BYTES service (ocr_common/config.py); file lebih besar akan dijawab 413.
const SERVICE_UPLOAD_LIMIT = 2.5 * 1024 * 1024

function fmtBytes(n) {
  if (n == null) return ''
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function TestFiles({ config, selected, setSelected, onChanged }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef(null)
  const files = config?.image_files ?? []

  async function upload(list) {
    if (!list?.length) return
    setBusy(true)
    setError(null)
    const form = new FormData()
    for (const f of list) form.append('files', f)
    try {
      const r = await fetch('/api/loadtest/images', { method: 'POST', body: form })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? r.statusText)
      await onChanged()
      setSelected((prev) => [...prev.filter((n) => body.images.includes(n)), ...body.saved])
    } catch (err) {
      setError(String(err.message ?? err))
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }

  async function remove(name) {
    if (!window.confirm(`Hapus ${name} dari folder images?`)) return
    setError(null)
    const r = await fetch(`/api/loadtest/images/${encodeURIComponent(name)}`, { method: 'DELETE' })
    const body = await r.json().catch(() => ({}))
    if (!r.ok) {
      setError(body.detail ?? r.statusText)
      return
    }
    setSelected((prev) => prev.filter((n) => n !== name))
    onChanged()
  }

  function drop(e) {
    e.preventDefault()
    setDragging(false)
    if (!busy) upload(e.dataTransfer.files)
  }

  const allSelected = files.length > 0 && files.every((f) => selected.includes(f.name))
  return (
    <div className="testfiles">
      <div className="testfiles-head">
        <span>file uji ({selected.length}/{files.length} dipakai)</span>
        {files.length > 1 && (
          <button type="button" className="link" onClick={() => setSelected(allSelected ? [] : files.map((f) => f.name))}>
            {allSelected ? 'kosongkan' : 'pilih semua'}
          </button>
        )}
      </div>
      <ul className="images">
        {files.map(({ name, size }) => (
          <li key={name}>
            <label>
              <input
                type="checkbox"
                checked={selected.includes(name)}
                onChange={(e) => setSelected(e.target.checked ? [...selected, name] : selected.filter((n) => n !== name))}
              />
              <span className="fname" title={name}>
                {name}
              </span>
            </label>
            <span className={size > SERVICE_UPLOAD_LIMIT ? 'fsize over' : 'fsize'} title={size > SERVICE_UPLOAD_LIMIT ? 'di atas batas 2,5 MB service: akan dijawab 413' : ''}>
              {fmtBytes(size)}
            </span>
            <a href={`/api/loadtest/images/${encodeURIComponent(name)}`} target="_blank" rel="noreferrer">
              lihat
            </a>
            <button type="button" className="link danger" onClick={() => remove(name)}>
              hapus
            </button>
          </li>
        ))}
      </ul>
      {config && files.length === 0 && <div className="hint small">Belum ada file uji. Unggah minimal satu.</div>}
      <label
        className={`dropzone ${dragging ? 'over' : ''} ${busy ? 'busy' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <input ref={input} type="file" multiple accept=".jpg,.jpeg,.png,.pdf" disabled={busy} onChange={(e) => upload(e.target.files)} />
        {busy ? 'Mengunggah…' : 'Unggah file uji: klik atau seret ke sini'}
        <span className="hint small">
          JPG, PNG, atau PDF, maks. {fmtBytes(config?.max_image_bytes)} per file. Disimpan ke {config?.load_tester_dir ?? 'load-tester'}/images.
        </span>
      </label>
      {error && <div className="error">{error}</div>}
    </div>
  )
}

function LoadTest({ overview, nav }) {
  const [config, setConfig] = useState(null)
  const [runs, setRuns] = useState([])
  const [current, setCurrent] = useState(null)
  const [detail, setDetail] = useState(null)
  const [rate, setRate] = useState(1)
  const [duration, setDuration] = useState(60)
  const [mode, setMode] = useState('constant')
  const [images, setImages] = useState([])
  const [sequence, setSequence] = useState(PIPELINE_NAMES)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)

  const loadRuns = () =>
    fetch('/api/loadtest')
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => {})

  const loadConfig = () =>
    fetch('/api/loadtest/config')
      .then((r) => r.json())
      .then((c) => {
        setConfig(c)
        return c
      })
      .catch(() => null)

  useEffect(() => {
    loadConfig().then((c) => c && setImages(c.images))
    loadRuns()
    const timer = setInterval(loadRuns, 2000)
    return () => clearInterval(timer)
  }, [])

  // Detail run yang dipilih: tiap detik selama berjalan, tiap 5 detik setelah selesai.
  const detailStatus = detail?.status
  useEffect(() => {
    if (!current) {
      setDetail(null)
      return
    }
    let stopped = false
    const tick = () =>
      fetch(`/api/loadtest/${current}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then((d) => {
          if (!stopped) setDetail(d)
        })
        .catch(() => {})
    tick()
    const timer = setInterval(tick, detailStatus === 'running' || !detailStatus ? 1000 : 5000)
    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [current, detailStatus])

  useEffect(() => {
    if (!current && runs[0]) setCurrent(runs[0].run_id)
  }, [runs, current])

  async function start(e) {
    e.preventDefault()
    setStarting(true)
    setError(null)
    try {
      const r = await fetch('/api/loadtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rate,
          duration_seconds: duration,
          mode,
          images,
          pipeline_name_sequence: sequenceToSend(sequence),
        }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? r.statusText)
      setCurrent(body.run_id)
      loadRuns()
    } catch (err) {
      setError(String(err.message ?? err))
    } finally {
      setStarting(false)
    }
  }

  async function stop(run) {
    await fetch(`/api/loadtest/${run}/stop`, { method: 'POST' })
    loadRuns()
  }

  async function remove(run) {
    if (!window.confirm(`Hapus run ${run} beserta baris LT_${run}_* di database?`)) return
    await fetch(`/api/loadtest/${run}`, { method: 'DELETE' })
    if (current === run) setCurrent(null)
    loadRuns()
  }

  const running = runs.some((r) => r.status === 'running')
  const d = detail
  const sent = d?.sent ?? 0
  const waitSeconds = config?.wait_seconds ?? 15

  return (
    <div className="app">
      <aside>
        <h1>NPWP pipeline tracker</h1>
        {nav}
        <form onSubmit={start}>
          <label className="field">
            laju (request/detik)
            <input type="number" min="0.1" max={config?.max_rate ?? 50} step="0.1" value={rate} onChange={(e) => setRate(Number(e.target.value))} />
          </label>
          <label className="field">
            durasi (detik)
            <input type="number" min="1" max={config?.max_duration ?? 1800} value={duration} onChange={(e) => setDuration(Number(e.target.value))} />
          </label>
          <label className="field">
            pola
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="constant">tetap</option>
              <option value="ramp">naik bertahap</option>
            </select>
          </label>
          <SequenceToggles names={sequence} onChange={setSequence} />
          <TestFiles config={config} selected={images} setSelected={setImages} onChanged={loadConfig} />
          <div className="hint small">
            Tuntas = callback DONE tahap terakhir sequence (<code>final: true</code>); untuk &quot;guardrails saja&quot;, jawaban 200 itu sendiri.
            Mis. &quot;extraction saja&quot; mengukur kapasitas OCR tanpa guardrails, structuring, dan scoring.
          </div>
          <div className="hint small">
            k6 ({config?.k6_image ?? 'grafana/k6'}
            {config && !config.k6_image_ready ? ', image belum ada: docker pull dulu' : ''}) jalan di network <code>{config?.network}</code> dan
            memanggil <code>{config?.target}</code>. Koneksi ditahan sampai {waitSeconds} dtk, jadi ~{Math.ceil(rate * (waitSeconds + 10))} VU
            disiapkan.
          </div>
          <button disabled={starting || running || images.length === 0 || sequenceProblem(sequence) !== null}>
            {running ? 'Ada run yang berjalan' : starting ? 'Menyalakan k6…' : 'Mulai load test'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}

        <h2>Backlog outbox tiap service</h2>
        <Backlog overview={overview} />

        <h2>Run</h2>
        <ul className="requests runs">
          {runs.map((r) => (
            <li key={r.run_id} className={r.run_id === current ? 'active' : ''}>
              <button className="link" onClick={() => setCurrent(r.run_id)}>
                <span className={`dot ${runStatusClass(r.status)}`} />
                <span className="rid">
                  {r.run_id} · {r.rate} rps · {r.duration_seconds} dtk
                  {r.pipeline_name_sequence ? ` · ${sequenceText(r.pipeline_name_sequence)}` : ''}
                </span>
                <span className="meta">
                  <span className={`pill ${runStatusClass(r.status)}`}>{r.status}</span> {r.sent} dikirim · {r.counts?.['200'] ?? 0}×200 ·{' '}
                  {r.counts?.['202'] ?? 0}×202 · {r.completed} tuntas{r.rejected > 0 && ` · ${r.rejected} ditolak`}
                </span>
              </button>
            </li>
          ))}
          {runs.length === 0 && <li className="hint small">Belum ada run.</li>}
        </ul>
      </aside>
      <main>
        {!d && <p className="hint">Atur laju dan durasi, lalu mulai. Hasil tiap run tersimpan di Redis sampai dihapus.</p>}
        {d && (
          <>
            <div className="head">
              <code>run {d.run_id}</code>
              <span className="meta">
                {d.rate} rps · {d.duration_seconds} dtk · {d.mode === 'ramp' ? 'naik bertahap' : 'tetap'} · sequence{' '}
                {sequenceText(d.pipeline_name_sequence)} · {d.images?.join(', ')}
              </span>
              <span className={`pill ${d.status === 'running' ? 'live' : runStatusClass(d.status)}`}>{d.status}</span>
              <span className="actions">
                {d.status === 'running' && (
                  <button className="small ghost" onClick={() => stop(d.run_id)}>
                    Hentikan
                  </button>
                )}
                {d.status !== 'running' && (
                  <button className="small danger" onClick={() => remove(d.run_id)}>
                    Hapus run
                  </button>
                )}
              </span>
            </div>

            <div className="tiles">
              <Tile k="dikirim" v={sent} s={`laju tercapai ${fmtRate(d.achieved_rate)} dari ${d.rate} rps`} />
              {STATUS_BUCKETS.map((b) => (
                <Tile key={b.key} k={b.label} v={d.counts?.[b.key] ?? 0} s={pct(d.counts?.[b.key] ?? 0, sent)} cls={b.cls} />
              ))}
            </div>
            <div className="mix">
              {STATUS_BUCKETS.map((b) => (
                <span key={b.key} className={`b${b.key}`} style={{ width: sent ? `${(100 * (d.counts?.[b.key] ?? 0)) / sent}%` : 0 }} title={b.label} />
              ))}
            </div>
            <div className="legend">
              {STATUS_BUCKETS.map((b) => (
                <span key={b.key}>
                  <i className={`b${b.key}`} />
                  {b.label}
                </span>
              ))}
            </div>

            <div className="panel">
              <h2>Pintu masuk: orchestrator /v1/extract-ocr</h2>
              <div className="kv">
                <span className="k">p50</span>
                <span>{fmtMs(d.extract?.p50)}</span>
                <span className="k">p95</span>
                <span>{fmtMs(d.extract?.p95)}</span>
                <span className="k">maks</span>
                <span>{fmtMs(d.extract?.max)}</span>
                <span className="k">batas tunggu</span>
                <span>{waitSeconds} dtk · di atas ini jawabannya 202</span>
              </div>
            </div>

            <div className="panel">
              <h2>
                End-to-end: submit sampai{' '}
                {d.pipeline_name_sequence?.length === 1 && d.pipeline_name_sequence[0] === 'guardrails'
                  ? 'jawaban 200 (guardrails saja)'
                  : `callback DONE ${d.pipeline_name_sequence ? d.pipeline_name_sequence[d.pipeline_name_sequence.length - 1] : 'scoring'} (final)`}
              </h2>
              <div className="tiles">
                <Tile k="tuntas" v={d.completed} s={pct(d.completed, sent)} cls="done" />
                <Tile k="ditolak" v={d.rejected ?? 0} s="model guardrails / aturan structuring" cls="rejected" />
                <Tile k="gagal" v={d.failed} s="callback FAILED, bukan penolakan" cls="failed" />
                {d.contract && (
                  <Tile
                    k="callback melanggar kontrak"
                    v={d.contract.fail}
                    s={`${d.contract.pass} sesuai, ${d.contract.warn} dengan catatan (bentuk body saja)`}
                    cls={d.contract.fail ? 'failed' : 'done'}
                  />
                )}
                <Tile k="dalam proses" v={d.in_flight} s="pipeline jalan, belum tuntas / gagal / ditolak" cls="processing" />
                <Tile
                  k="dokumen / menit"
                  v={d.completed_per_minute == null ? '-' : d.completed_per_minute.toFixed(1)}
                  s="tuntas per menit sejak kiriman pertama"
                />
                <Tile k="p50 e2e" v={fmtMs(d.e2e?.p50)} />
                <Tile k="p95 e2e" v={fmtMs(d.e2e?.p95)} />
              </div>
              {Object.keys(d.callbacks ?? {}).length > 0 && (
                <div className="kv">
                  {Object.entries(d.callbacks)
                    .sort()
                    .map(([k, v]) => (
                      <Fragment key={k}>
                        <span className="k">callback {k}</span>
                        <span>{v}</span>
                      </Fragment>
                    ))}
                </div>
              )}
            </div>

            {d.k6 && (
              <div className="panel">
                <h2>Ringkasan k6</h2>
                <div className="kv">
                  <span className="k">iterasi</span>
                  <span>{d.k6.iterations ?? '-'}</span>
                  <span className="k">dropped iterations</span>
                  <span>
                    {d.k6.dropped_iterations}
                    {d.k6.dropped_iterations > 0 && ' · k6 kehabisan VU, laju yang diminta tidak tercapai'}
                  </span>
                  <span className="k">VU maks</span>
                  <span>{d.k6.vus_max ?? '-'}</span>
                  <span className="k">http_req_failed</span>
                  <span>{d.k6.http_req_failed_rate == null ? '-' : pct(d.k6.http_req_failed_rate, 1)}</span>
                </div>
              </div>
            )}

            {d.rejections?.length > 0 && (
              <div className="panel">
                <h2>Alasan penolakan</h2>
                <ul className="lines">
                  {d.rejections.map((x) => (
                    <li key={x.reason}>
                      <span className="conf">{x.count}×</span> {x.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(d.last_errors?.length > 0 || d.failures?.length > 0) && (
              <div className="panel">
                <h2>Error terakhir</h2>
                <ul className="lines">
                  {d.last_errors.map((e, i) => (
                    <li key={`h${i}`}>
                      <span className="conf">HTTP {e.status}</span> {e.request_id} · {e.message ?? '-'}
                    </li>
                  ))}
                  {d.failures.map((f, i) => (
                    <li key={`f${i}`}>
                      <span className="conf">FAILED</span> {f.request_id} · {f.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {d.log_tail && (
              <div className="panel">
                <h2>Log k6</h2>
                <pre className="log">{d.log_tail}</pre>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

// "Kenapa outbox?": tiap masalah yang ditangani transactional outbox, apa yang dilakukannya, dan skenario gangguan
// yang membuktikannya di stack lokal. Untuk menjelaskan ke tim tanpa membaca kode.
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
    solution: 'Bukan outbox, tapi reaper job basi: job PROCESSING tanpa pemilik dijalankan ulang setelah lease. Upload inline sudah hilang (job FAILED, minta kirim ulang); file_url pulih sendiri.',
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
  const [open, setOpen] = useState(null)
  return (
    <section className="panel guide-card">
      <h3>{item.title}</h3>
      <div className="guide-row">
        <span className="k">masalah</span>
        <p>{item.problem}</p>
      </div>
      <div className="guide-row">
        <span className="k">outbox</span>
        <p>{item.solution}</p>
      </div>
      {item.look && (
        <div className="guide-row">
          <span className="k">lihat di</span>
          <ul>
            {item.look.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
        </div>
      )}
      {item.note && <p className="hint small">{item.note}</p>}
      {item.manual && <p className="hint small">{item.manual}</p>}
      {item.scenarios.length > 0 && (
        <div className="guide-demos">
          {item.scenarios.map((id) => {
            const scenario = byId[id]
            const last = scenario?.last
            return (
              <div key={id} className="guide-demo">
                <div className="guide-demo-head">
                  {last ? <LevelPill level={last.status} /> : <span className="pill idle">belum</span>}
                  <span className="name">{scenario?.title ?? id}</span>
                  <button className="small" disabled={running || !available} onClick={() => onRun([id])}>
                    Jalankan
                  </button>
                  {last && (
                    <button className="link small-link" onClick={() => setOpen(open === id ? null : id)}>
                      {open === id ? 'tutup laporan' : 'laporan terakhir'}
                    </button>
                  )}
                </div>
                {scenario?.expect && <div className="hint small">diharapkan: {scenario.expect}</div>}
                {open === id && last && <RunReport run={last} openRequest={openRequest} />}
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}

function OutboxGuide({ nav, openRequest }) {
  const [data, setData] = useState(null)
  const [image, setImage] = useState('')
  const [error, setError] = useState(null)

  const load = () =>
    fetch('/api/scenarios')
      .then((r) => r.json())
      .then((d) => {
        setData(d)
        setImage((current) => current || d.images[0] || '')
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
  const card = (item) => (
    <GuideCard key={item.title} item={item} byId={byId} running={running} available={available} onRun={run} openRequest={openRequest} />
  )

  return (
    <div className="app">
      <aside>
        <h1>NPWP pipeline tracker</h1>
        {nav}
        <p className="hint">
          Tiap kartu: masalah yang muncul karena tahap harus menyimpan hasil dan mengirim HTTP sekaligus, apa yang dilakukan transactional
          outbox, dan skenario gangguan yang membuktikannya di stack lokal.
        </p>
        {data && !available && (
          <div className="error">Skenario hanya di mode lokal (docker compose). Komentari blok GKE di tools/tracker/.env.</div>
        )}
        <label className="field wide">
          dokumen uji (harus lolos guardrails dan aturan structuring)
          <select value={image} onChange={(e) => setImage(e.target.value)}>
            {(data?.images ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <div className="actions">
          <button className="small" disabled={running || !available} onClick={() => run(allIds)}>
            Jalankan semua demo ({allIds.length})
          </button>
        </div>
        {running && <p className="hint small">Sedang berjalan: {data.running}. Detail langkahnya di menu Skenario gangguan.</p>}
        {error && <div className="error">{error}</div>}
      </aside>
      <main>
        <section className="panel guide-intro">
          <h2>Kenapa outbox?</h2>
          <p>
            Tanpa outbox: <code>simpan hasil</code> lalu <code>POST hand-off / callback</code>. Kalau yang kedua gagal atau pod mati di antaranya,
            hasilnya ada tapi tidak ada yang tahu. Dengan outbox: <code>simpan hasil + pesan</code> dalam satu transaksi, lalu relay mengirim pesan
            itu sampai diterima, dicoba ulang, atau dicatat sebagai dead letter.
          </p>
        </section>
        <h2 className="guide-heading">Yang ditangani</h2>
        <div className="guide-grid">{OUTBOX_PROBLEMS.map(card)}</div>
        <h2 className="guide-heading">Yang tidak ditangani outbox</h2>
        <div className="guide-grid">{OUTBOX_LIMITS.map(card)}</div>
      </main>
    </div>
  )
}

function Nav({ view, setView }) {
  return (
    <div className="nav">
      <button className={view === 'pipeline' ? 'active' : ''} onClick={() => setView('pipeline')}>
        Pipeline
      </button>
      <button className={view === 'loadtest' ? 'active' : ''} onClick={() => setView('loadtest')}>
        Load testing
      </button>
      <button className={view === 'scenarios' ? 'active' : ''} onClick={() => setView('scenarios')}>
        Skenario gangguan
      </button>
      <button className={view === 'outbox' ? 'active' : ''} onClick={() => setView('outbox')}>
        Kenapa outbox?
      </button>
    </div>
  )
}

export default function App() {
  const [view, setView] = useState(() => {
    try {
      // #scenarios / #loadtest / #outbox membuka menu itu langsung (tautan, screenshot headless).
      const stored = window.location.hash.slice(1) || localStorage.getItem('tracker.view')
      return ['loadtest', 'scenarios', 'outbox'].includes(stored) ? stored : 'pipeline'
    } catch (_) {
      return 'pipeline'
    }
  })
  const [overview, setOverview] = useState(null)
  const [focus, setFocus] = useState(null)
  useEffect(() => {
    try {
      localStorage.setItem('tracker.view', view)
    } catch (_) {
      /* abaikan */
    }
  }, [view])
  useEffect(() => {
    const load = () => fetch('/api/outbox').then((r) => r.json()).then(setOverview).catch(() => {})
    load()
    const timer = setInterval(load, 2000)
    return () => clearInterval(timer)
  }, [])
  const nav = <Nav view={view} setView={setView} />
  if (view === 'loadtest') return <LoadTest overview={overview} nav={nav} />
  const openRequest = (rid) => {
    setFocus({ rid, at: Date.now() })
    setView('pipeline')
  }
  if (view === 'outbox') return <OutboxGuide nav={nav} openRequest={openRequest} />
  if (view === 'scenarios') return <Scenarios nav={nav} overview={overview} openRequest={openRequest} />
  return <Pipeline nav={nav} overview={overview} focus={focus} />
}

function Pipeline({ nav, overview: sharedOverview, focus }) {
  const [requests, setRequests] = useState([])
  const [current, setCurrent] = useState(null)
  const [events, setEvents] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [live, setLive] = useState(false)
  const [slow, setSlow] = useState(false)
  const [slowSeconds, setSlowSeconds] = useState(20)
  const [sendNames, setSendNames] = useState(PIPELINE_NAMES)
  const [source, setSource] = useState('upload')
  const [resendId, setResendId] = useState('')
  const [sim, setSim] = useState({ callback: 'ok', wait_seconds: 15, database: null })
  const overview = sharedOverview
  const [releasing, setReleasing] = useState(false)
  const fileRef = useRef()
  const sourceRef = useRef()

  const loadRequests = () => fetch('/api/requests').then((r) => r.json()).then(setRequests)
  const loadOverview = () => {}
  useEffect(() => {
    loadRequests()
    fetch('/api/simulation').then((r) => r.json()).then(setSim)
  }, [])
  // Dibuka dari menu skenario: tampilkan request itu.
  useEffect(() => {
    if (focus?.rid) setCurrent(focus.rid)
  }, [focus])

  // Buka SSE untuk request yang dipilih: replay dari awal, lalu live.
  useEffect(() => {
    if (!current) return
    setEvents([])
    sourceRef.current?.close()
    const source = new EventSource(`/api/requests/${current}/events`)
    sourceRef.current = source
    setLive(true)
    source.addEventListener('stage', (msg) => {
      setEvents((prev) => [...prev, JSON.parse(msg.data)])
    })
    source.addEventListener('end', () => {
      source.close()
      setLive(false)
      loadRequests()
      loadOverview()
    })
    source.onerror = () => {
      setLive(false)
    }
    return () => source.close()
  }, [current])

  async function submit(e) {
    e.preventDefault()
    const file = fileRef.current.files[0]
    if (!file) return
    setBusy(true)
    setError(null)
    const form = new FormData()
    form.append('file', file)
    form.append('slow_seconds', slow ? String(slowSeconds) : '0')
    const sent = sequenceToSend(sendNames)
    form.append('pipeline_name_sequence', sent ? JSON.stringify(sent) : '')
    form.append('source', source)
    form.append('request_id', resendId.trim())
    try {
      const r = await fetch('/api/requests', { method: 'POST', body: form })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? r.statusText)
      if (body.request_id === current) {
        // Kiriman ulang request yang sedang dilihat: buka SSE lagi supaya event barunya terlihat.
        setCurrent(null)
        setTimeout(() => setCurrent(body.request_id), 0)
      } else {
        setCurrent(body.request_id)
      }
      loadRequests()
    } catch (err) {
      setError(String(err.message ?? err))
    } finally {
      setBusy(false)
    }
  }

  // Pilih request lebih dulu, lalu upload: supaya timeline terlihat sejak detik pertama,
  // request_id dibuat di backend, jadi UI baru bisa berlangganan setelah POST selesai.
  // Untuk kasus 202 (≥15 dtk) itu terlalu lama, maka daftar request di-refresh
  // lebih awal dan request terbaru dipilih otomatis selama masih menunggu.
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(async () => {
      const rows = await fetch('/api/requests').then((r) => r.json())
      setRequests(rows)
      if (rows[0] && rows[0].request_id !== current && !rows[0].ended) setCurrent(rows[0].request_id)
    }, 700)
    return () => clearInterval(timer)
  }, [busy, current])

  async function setCallbackMode(mode) {
    const r = await fetch('/api/simulation', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ callback: mode }),
    })
    setSim(await r.json())
  }

  async function release() {
    setReleasing(true)
    try {
      await fetch(`/api/requests/${current}/outbox/release`, { method: 'POST' })
      if (!live) {
        // SSE sudah ditutup oleh event END: buka lagi supaya event pelepasan dan retry-nya terlihat.
        const id = current
        setCurrent(null)
        setTimeout(() => setCurrent(id), 0)
      }
    } finally {
      setReleasing(false)
    }
  }

  const clientEvent = events.find((e) => e.type === 'client')
  const httpEvent = events.find((e) => e.type === 'http')
  // Sequence yang ditolak (422) tidak menjalankan apa pun: kartu tampil apa adanya, tidak ada yang "dilewati".
  const sequence = httpEvent?.errors === INVALID_SEQUENCE ? null : clientEvent?.sequence ?? null
  const lastStage = [...STAGES].reverse().find((stage) => !sequence || sequence.includes(SERVICE_OF_STAGE[stage]))
  const t0 = clientEvent?.ts ?? events[0]?.ts ?? 0
  const outboxRows = useMemo(() => outboxView(events), [events])
  const unavailable = events.find((e) => e.type === 'outbox' && e.status === 'UNAVAILABLE')
  const ended = events.find((e) => e.type === 'pipeline')
  const total = ended ? (ended.ts - t0) * 1000 : events.length > 1 ? (events[events.length - 1].ts - t0) * 1000 : null

  return (
    <div className="app">
      <aside>
        <h1>NPWP pipeline tracker</h1>
        {nav}
        <form onSubmit={submit}>
          <input ref={fileRef} type="file" accept="image/jpeg,image/png,application/pdf" />
          <label className="check">
            <input type="checkbox" checked={slow} onChange={(e) => setSlow(e.target.checked)} />
            pipeline lambat: OCR ditunda
            <input
              type="number"
              min="1"
              max="120"
              value={slowSeconds}
              disabled={!slow}
              onChange={(e) => setSlowSeconds(Number(e.target.value))}
            />
            dtk
          </label>
          <div className="hint small">
            batas tunggu orchestrator {sim.wait_seconds} dtk: di bawah itu jawabannya 200 + hasil, di atas itu 202 dan hasil menyusul lewat callback. Nama file diberi awalan <code>delay{slowSeconds}s-</code>; hook ini hanya hidup di ENVIRONMENT=local.
          </div>
          <SequenceToggles names={sendNames} onChange={setSendNames} allowInvalid />
          <div className="hint small">
            Service yang dijalankan request ini. Tahap terakhir mengakhiri request: hasilnya jadi <code>data</code> apa adanya dan callback DONE-nya
            membawa <code>final: true</code>. Pengecekan file di orchestrator selalu jalan, dan aturan structuring tetap menolak kalau structuring jalan.
          </div>
          <label className="field wide">
            dokumen dikirim sebagai
            <select value={source} onChange={(e) => setSource(e.target.value)} disabled={Boolean(sim.target) && sim.target !== 'local'}>
              {SOURCES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field wide">
            request_id (kosong = baru; isi = kirim ulang)
            <input value={resendId} placeholder="REQ_…" onChange={(e) => setResendId(e.target.value)} />
          </label>
          {current && (
            <button type="button" className="link small-link" onClick={() => setResendId(current)}>
              pakai request_id yang sedang dilihat
            </button>
          )}
          <button disabled={busy || sendNames.length === 0}>
            {busy ? `Menunggu orchestrator (maks. ${sim.wait_seconds} dtk)…` : resendId.trim() ? 'Kirim ulang request_id ini' : 'Kirim dokumen'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}

        <h2>Simulasi orkestrasi</h2>
        <div className="radios">
          {CALLBACK_MODES.map((m) => (
            <label key={m.value} className={`radio ${sim.callback === m.value ? 'active' : ''}`}>
              <input type="radio" name="cb" checked={sim.callback === m.value} onChange={() => setCallbackMode(m.value)} />
              <span>
                callback {m.label}
                <small>{m.hint}</small>
              </span>
            </label>
          ))}
        </div>
        <div className="hint small">
          Berlaku untuk callback yang datang mulai sekarang. Matikan, kirim dokumen, lalu nyalakan lagi: pesan yang tertahan terkirim pada retry berikutnya.
          Lambat = dijawab setelah {sim.callback_slow_seconds ?? 12} dtk; tersendat = {sim.callback_flaky_failures ?? 2}× 503 per pesan.
          {sim.callback_key ? ' /v1/ocr-callback memeriksa X-Callback-Key (TRACKER_CALLBACK_KEY).' : ''}
          {sim.database === false && <div className="error">Pemantau outbox mati: {sim.database_error}</div>}
        </div>

        <h2>Backlog outbox tiap service</h2>
        <Backlog overview={overview} />

        <h2>Request terakhir</h2>
        <ul className="requests">
          {requests.map((r) => (
            <li key={r.request_id} className={r.request_id === current ? 'active' : ''}>
              <button className="link" onClick={() => setCurrent(r.request_id)}>
                <span className={`dot ${(r.status ?? '').toLowerCase()}`} />
                <span className="rid">{r.request_id}</span>
                <span className="meta">
                  {r.http_status ? (
                    <>
                      <span className={`pill ${httpClass(r.http_status)}`}>HTTP {r.http_status}</span> {r.job_status ?? ''}
                      {r.elapsed_ms != null ? ` · ${fmtMs(r.elapsed_ms)}` : ''}
                    </>
                  ) : (
                    `${r.stage ?? ''} ${r.status ?? ''}`
                  )}
                  {r.slow ? ` · lambat ${r.slow} dtk` : ''}
                  {r.sequence ? ` · ${sequenceText(r.sequence)}` : ''}
                </span>
              </button>
              {r.body && (
                <div className="resp">
                  <span className="meta">{httpNote(r.http_status, r.rejected_by, r.body, r.sequence)}</span>
                  <Json value={r.body} label="response" />
                </div>
              )}
            </li>
          ))}
        </ul>
      </aside>
      <main>
        {!current && <p className="hint">Kirim foto NPWP untuk melihat tiap tahap, baris outbox, dan callback-nya berjalan.</p>}
        {current && (
          <>
            <div className="head">
              <code>{current}</code>
              {clientEvent && (
                <span className="meta">
                  {clientEvent.filename} · {(clientEvent.size / 1024).toFixed(0)} KB
                </span>
              )}
              <span className={`pill ${live ? 'live' : 'idle'}`}>{live ? 'LIVE' : 'selesai'}</span>
              {total != null && <span className="elapsed">total {fmtMs(total)}</span>}
            </div>
            <GuardrailsResponse http={httpEvent} t0={t0} />
            <div className="stages">
              {STAGES.map((stage, i) => (
                <Stage
                  key={stage}
                  stage={stage}
                  index={i}
                  view={stageView(events, stage, sequence)}
                  t0={t0}
                  last={Boolean(sequence) && !sequence.includes('scoring') && stage === lastStage}
                />
              ))}
            </div>
            <OutboxPanel
              rows={outboxRows}
              t0={t0}
              onRelease={release}
              releasing={releasing}
              unavailable={unavailable ? describe(unavailable, t0) : null}
            />
            <Timeline events={events} t0={t0} />
          </>
        )}
      </main>
    </div>
  )
}

// --- skenario gangguan ("shit happens") ---------------------------------------------

const LEVEL_LABEL = { pass: 'PASS', warn: 'WARN', fail: 'FAIL', info: 'INFO', error: 'ERROR', running: 'JALAN', stopped: 'DIHENTIKAN' }
const CHAOS_SERVICES = ['orchestrator', 'guardrails', 'extraction', 'structuring', 'scoring', 'postgres']

function LevelPill({ level }) {
  return <span className={`pill level-${level}`}>{LEVEL_LABEL[level] ?? level}</span>
}

function fmtClock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString() : ''
}

function RunReport({ run, openRequest }) {
  const [showSteps, setShowSteps] = useState(false)
  return (
    <div className="run-report">
      <div className="meta">
        {fmtClock(run.started_at)}
        {run.ended_at ? ` · ${fmtSec(run.ended_at - run.started_at)}` : ' · berjalan…'}
        {run.request_ids.length > 0 && ' · '}
        {run.request_ids.map((rid) => (
          <button key={rid} className="link rid-link" onClick={() => openRequest(rid)}>
            {rid}
          </button>
        ))}
      </div>
      {run.checks.length > 0 && (
        <table className="checks">
          <tbody>
            {run.checks.map((c, i) => (
              <tr key={i} className={`check-${c.level}`}>
                <td>
                  <LevelPill level={c.level} />
                </td>
                <td>
                  <div>{c.label}</div>
                  {c.detail && <div className="meta detail">{c.detail}</div>}
                </td>
                <td className="meta t">t+{c.t} dtk</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <button className="link small-link" onClick={() => setShowSteps(!showSteps)}>
        {showSteps ? 'sembunyikan' : 'lihat'} langkah ({run.steps.length})
      </button>
      {(showSteps || run.status === 'running' || run.status === 'error') && (
        <ol className="steps">
          {run.steps.map((s, i) => (
            <li key={i} className={s.text.startsWith('ERROR') ? 'error' : ''}>
              <span className="meta">t+{s.t}</span> {s.text}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function ChaosPanel({ chaos, reload }) {
  const [service, setService] = useState('structuring')
  const [action, setAction] = useState('stop')
  const [seconds, setSeconds] = useState(20)
  const [error, setError] = useState(null)

  async function disrupt(e) {
    e.preventDefault()
    setError(null)
    const r = await fetch('/api/chaos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ service, action, seconds }),
    })
    if (!r.ok) setError((await r.json()).detail ?? r.statusText)
    reload()
  }

  async function startNow(name) {
    await fetch(`/api/chaos/${name}/start`, { method: 'POST' })
    reload()
  }

  if (!chaos) return null
  const disrupted = new Set(chaos.disruptions.map((d) => d.service))
  return (
    <>
      <div className="backlog">
        {chaos.containers.map((c) => (
          <div key={c.service} className={`backlog-item ${c.status !== 'running' ? 'bad' : ''}`}>
            <span className="k">{c.service}</span>
            <span className="v">
              {c.status}
              {c.health ? ` · ${c.health}` : ''}
              {c.status !== 'running' && c.status !== 'missing' && (
                <>
                  {' '}
                  <button className="link" onClick={() => startNow(c.service)}>
                    nyalakan
                  </button>
                </>
              )}
              {disrupted.has(c.service) && ' · diganggu'}
            </span>
          </div>
        ))}
      </div>
      <form onSubmit={disrupt} className="chaos-form">
        <label className="field wide">
          service
          <select value={service} onChange={(e) => setService(e.target.value)}>
            {CHAOS_SERVICES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="field wide">
          cara
          <select value={action} onChange={(e) => setAction(e.target.value)}>
            <option value="stop">SIGTERM (rolling restart, drain dulu)</option>
            <option value="kill">SIGKILL (OOM / node hilang)</option>
          </select>
        </label>
        <label className="field">
          mati selama (dtk)
          <input type="number" min="1" max="600" value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} />
        </label>
        <button disabled={!chaos.available}>Ganggu</button>
      </form>
      <div className="hint small">
        Untuk dicoba manual sambil mengirim dokumen di menu Pipeline; gangguannya ikut tercatat di timeline request yang sedang hidup. SIGTERM menunggu
        grace {chaos.stop_grace_seconds} dtk seperti terminationGracePeriodSeconds di chart.
      </div>
      {error && <div className="error">{error}</div>}
    </>
  )
}

function Scenarios({ nav, overview, openRequest }) {
  const [data, setData] = useState(null)
  const [chaos, setChaos] = useState(null)
  const [image, setImage] = useState('')
  const [error, setError] = useState(null)
  const [open, setOpen] = useState({})

  const load = () =>
    fetch('/api/scenarios')
      .then((r) => r.json())
      .then((d) => {
        setData(d)
        setImage((current) => current || d.images[0] || '')
      })
      .catch(() => {})
  const loadChaos = () =>
    fetch('/api/chaos')
      .then((r) => r.json())
      .then(setChaos)
      .catch(() => {})

  useEffect(() => {
    load()
    loadChaos()
    const timer = setInterval(() => {
      load()
      loadChaos()
    }, 1500)
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

  async function stopRun() {
    await fetch('/api/scenarios/stop', { method: 'POST' })
    load()
  }

  const running = Boolean(data?.running)
  const settings = data?.settings
  const scenarios = data?.scenarios ?? []
  const totals = { pass: 0, warn: 0, fail: 0, error: 0, never: 0 }
  for (const s of scenarios) {
    const status = s.last?.status
    if (!status) totals.never += 1
    else if (status in totals) totals[status] += 1
  }
  const longLease = settings && settings.job_lease_seconds > 60

  return (
    <div className="app">
      <aside>
        <h1>NPWP pipeline tracker</h1>
        {nav}
        {data && !data.available && (
          <div className="error">Skenario hanya di mode lokal (docker compose). Komentari blok GKE di tools/tracker/.env.</div>
        )}
        <label className="field wide">
          dokumen uji (harus lolos guardrails dan aturan structuring)
          <select value={image} onChange={(e) => setImage(e.target.value)}>
            {(data?.images ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <div className="actions">
          <button className="small" disabled={running || !data?.available} onClick={() => run(null)}>
            Jalankan semua
          </button>
          {running && (
            <button className="small danger" onClick={stopRun}>
              Hentikan
            </button>
          )}
        </div>
        {settings && (
          <>
            <h2>Pengaturan yang dipakai</h2>
            <ul className="facts">
              <li>
                <span className="k">batas tunggu</span>
                {settings.wait_seconds} dtk
              </li>
              <li>
                <span className="k">grace SIGTERM</span>
                {settings.stop_grace_seconds} dtk
              </li>
              <li>
                <span className="k">drain job</span>
                {settings.drain_seconds} dtk
              </li>
              <li>
                <span className="k">lease job</span>
                {settings.job_lease_seconds} dtk
              </li>
              <li>
                <span className="k">reaper tiap</span>
                {settings.stale_job_interval_seconds} dtk
              </li>
              <li>
                <span className="k">callback lambat</span>
                {settings.callback_slow_seconds} dtk
              </li>
            </ul>
            {longLease && (
              <div className="hint small warn-box">
                Lease {settings.job_lease_seconds} dtk: skenario crash dan database menunggu reaper &gt; {fmtSec(settings.job_lease_seconds)}. Untuk latihan isi{' '}
                <code>PIPELINE_JOB_LEASE_SECONDS=30</code> dan <code>PIPELINE_STALE_JOB_INTERVAL_SECONDS=5</code> di services/*/.env, lalu{' '}
                <code>TRACKER_JOB_LEASE_SECONDS=30</code> dan <code>TRACKER_STALE_JOB_INTERVAL_SECONDS=5</code> di tools/tracker/.env.
              </div>
            )}
          </>
        )}
        <h2>Gangguan manual</h2>
        <ChaosPanel chaos={chaos} reload={loadChaos} />
        <h2>Backlog outbox tiap service</h2>
        <Backlog overview={overview} />
      </aside>
      <main>
        <div className="head">
          <h2 className="inline">Uji kesiapan: apa yang terjadi kalau sesuatu rusak</h2>
          <span className="meta">
            {totals.pass} PASS · {totals.warn} WARN · {totals.fail} FAIL
            {totals.error ? ` · ${totals.error} ERROR` : ''}
            {totals.never ? ` · ${totals.never} belum dijalankan` : ''}
          </span>
        </div>
        <p className="hint small">
          Tiap skenario mengirim dokumen sungguhan lewat orchestrator, membuat gangguannya pada saat yang tepat, lalu memeriksa database, outbox, dan callback.
          PASS = sesuai harapan; WARN = perilakunya benar tapi harus diketahui tim atau Orkestrasi pusat; FAIL = harus diperbaiki sebelum go-live. Satu run pada
          satu waktu; di akhir tiap skenario container dinyalakan lagi dan callback kembali normal.
        </p>
        {error && <div className="error">{error}</div>}
        {scenarios.map((s) => {
          const last = s.last
          const status = last?.status
          const expanded = open[s.id] ?? ['running', 'fail', 'error', 'warn'].includes(status)
          return (
            <section key={s.id} className={`panel scenario scenario-${status ?? 'never'}`}>
              <div className="panel-head">
                <div>
                  <div className="scenario-title">
                    {status ? <LevelPill level={status} /> : <span className="pill idle">BELUM</span>} {s.title}
                    {s.needs_lease && <span className="tag">butuh lease pendek</span>}
                  </div>
                  <div className="meta">{s.simulates}</div>
                  <div className="meta">
                    <b>Siap berarti:</b> {s.expect}
                  </div>
                </div>
                <div className="actions">
                  {last && (
                    <button className="link" onClick={() => setOpen({ ...open, [s.id]: !expanded })}>
                      {expanded ? 'tutup' : 'hasil'}
                    </button>
                  )}
                  <button className="small" disabled={running || !data?.available} onClick={() => run([s.id])}>
                    Jalankan
                  </button>
                </div>
              </div>
              {last && expanded && <RunReport run={last} openRequest={openRequest} />}
            </section>
          )
        })}
      </main>
    </div>
  )
}
