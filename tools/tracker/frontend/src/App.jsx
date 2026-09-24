import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const STAGES = ['GUARDRAILS', 'OCR', 'STRUCTURING', 'SCORING']
const LABELS = {
  GUARDRAILS: 'Orchestrator + guardrails',
  OCR: 'Ekstraksi OCR',
  STRUCTURING: 'Structuring',
  SCORING: 'Scoring',
}
const OUTBOX_STAGES = ['OCR', 'STRUCTURING', 'SCORING']
const CALLBACK_MODES = [
  { value: 'ok', label: 'normal (2xx)', hint: 'callback diterima, baris outbox dihapus' },
  { value: 'down', label: 'mati (503)', hint: 'relay mengulang dengan backoff; pipeline tetap jalan' },
  { value: 'reject', label: 'menolak (422)', hint: 'relay berhenti: baris jadi dead letter' },
]

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function fmtSec(seconds) {
  if (seconds == null) return '-'
  return seconds < 60 ? `${seconds.toFixed(0)} dtk` : `${(seconds / 60).toFixed(1)} mnt`
}

// --- turunan dari daftar event ------------------------------------------------

function stageView(events, stage) {
  const mine = events.filter((e) => e.stage === stage && e.type === 'stage')
  const callbacks = events.filter((e) => e.stage === stage && e.type === 'callback')
  const http = stage === 'GUARDRAILS' ? events.find((e) => e.type === 'http') : null
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
      return `Orkestrasi (tracker) menerima ${e.filename}${sim.length ? ` · simulasi: ${sim.join(', ')}` : ''}`
    }
    case 'http':
      return `Orchestrator menjawab HTTP ${e.http_status}, job_status=${e.job_status ?? '-'}${
        e.errors ? ` (${e.errors})` : ''
      } setelah ${fmtMs(e.elapsed_ms)} · batas tunggu ${e.wait_seconds} dtk`
    case 'stage':
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
    case 'callback':
      return `Callback ${e.stage} ${e.status} tiba (attempt ${e.attempt}) → orkestrasi menjawab ${e.http_status}${
        e.mode !== 'ok' ? ' (simulasi)' : ''
      }`
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

function httpNote(status, rejectedBy) {
  if (status === 200) return '200: hasil lengkap ada di response'
  if (status === 202) return '202: hanya request_id, hasil menyusul lewat callback'
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
        {http.http_status === 200 && 'Pipeline selesai di dalam batas tunggu: hasil langsung ada di respons, callback yang menyusul boleh diabaikan.'}
        {http.http_status === 202 && 'Batas tunggu habis sebelum SCORING selesai: hasil menyusul lewat callback, dan bisa dibaca kapan saja lewat GET /v1/extract-ocr/{request_id} di orchestrator.'}
        {http.http_status === 422 && 'Satu tahap gagal di dalam batas tunggu.'}
        {http.http_status === 400 &&
          (http.rejected_by === 'structuring'
            ? 'Lolos guardrails, lalu ditolak aturan structuring ML: OCR dan structuring sudah jalan, scoring tidak.'
            : 'Dokumen ditolak guardrails: tidak ada tahap yang dijalankan.')}
      </div>
      <Json value={http.body} label="response extract-ocr" />
    </div>
  )
}

function Stage({ stage, view, index, t0 }) {
  const cls = `stage ${view.status.toLowerCase()}`
  const rel = (ts) => `t+${fmtMs((ts - t0) * 1000)}`
  const rejected = view.callbacks.filter((c) => !c.accepted)
  const lastCallback = view.callbacks[view.callbacks.length - 1]
  return (
    <div className={cls}>
      <div className="stage-head">
        <span className="index">{index + 1}</span>
        <span className="label">{LABELS[stage]}</span>
        <span className={`pill ${view.status.toLowerCase()}`}>{view.status}</span>
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
            <span className="what">{describe(e, t0)}</span>
          </li>
        ))}
      </ol>
    </section>
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

const STATUS_BUCKETS = [
  { key: '200', label: '200 selesai', cls: 'done' },
  { key: '202', label: '202 menyusul', cls: 'processing' },
  { key: '4xx', label: '4xx', cls: 'rejected' },
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
        body: JSON.stringify({ rate, duration_seconds: duration, mode, images }),
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
          <TestFiles config={config} selected={images} setSelected={setImages} onChanged={loadConfig} />
          <div className="hint small">
            k6 ({config?.k6_image ?? 'grafana/k6'}
            {config && !config.k6_image_ready ? ', image belum ada: docker pull dulu' : ''}) jalan di network <code>{config?.network}</code> dan
            memanggil <code>{config?.target}</code>. Koneksi ditahan sampai {waitSeconds} dtk, jadi ~{Math.ceil(rate * (waitSeconds + 10))} VU
            disiapkan.
          </div>
          <button disabled={starting || running || images.length === 0}>
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
                </span>
                <span className="meta">
                  <span className={`pill ${runStatusClass(r.status)}`}>{r.status}</span> {r.sent} dikirim · {r.counts?.['200'] ?? 0}×200 ·{' '}
                  {r.counts?.['202'] ?? 0}×202 · {r.completed} tuntas
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
                {d.rate} rps · {d.duration_seconds} dtk · {d.mode === 'ramp' ? 'naik bertahap' : 'tetap'} · {d.images?.join(', ')}
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
              <h2>End-to-end: submit sampai callback SCORING DONE</h2>
              <div className="tiles">
                <Tile k="tuntas" v={d.completed} s={pct(d.completed, sent)} cls="done" />
                <Tile k="gagal" v={d.failed} s="callback FAILED" cls="failed" />
                <Tile k="dalam proses" v={d.in_flight} s="dikirim, belum ada SCORING DONE" cls="processing" />
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

function Nav({ view, setView }) {
  return (
    <div className="nav">
      <button className={view === 'pipeline' ? 'active' : ''} onClick={() => setView('pipeline')}>
        Pipeline
      </button>
      <button className={view === 'loadtest' ? 'active' : ''} onClick={() => setView('loadtest')}>
        Load testing
      </button>
    </div>
  )
}

export default function App() {
  const [view, setView] = useState(() => {
    try {
      return localStorage.getItem('tracker.view') === 'loadtest' ? 'loadtest' : 'pipeline'
    } catch (_) {
      return 'pipeline'
    }
  })
  const [overview, setOverview] = useState(null)
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
  return <Pipeline nav={nav} overview={overview} />
}

function Pipeline({ nav, overview: sharedOverview }) {
  const [requests, setRequests] = useState([])
  const [current, setCurrent] = useState(null)
  const [events, setEvents] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [live, setLive] = useState(false)
  const [slow, setSlow] = useState(false)
  const [slowSeconds, setSlowSeconds] = useState(20)
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
    form.append('document_type', 'npwp')
    form.append('slow_seconds', slow ? String(slowSeconds) : '0')
    try {
      const r = await fetch('/api/requests', { method: 'POST', body: form })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? r.statusText)
      setCurrent(body.request_id)
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
          <button disabled={busy}>{busy ? `Menunggu orchestrator (maks. ${sim.wait_seconds} dtk)…` : 'Kirim dokumen'}</button>
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
                </span>
              </button>
              {r.body && (
                <div className="resp">
                  <span className="meta">{httpNote(r.http_status, r.rejected_by)}</span>
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
            <GuardrailsResponse http={events.find((e) => e.type === 'http')} t0={t0} />
            <div className="stages">
              {STAGES.map((stage, i) => (
                <Stage key={stage} stage={stage} index={i} view={stageView(events, stage)} t0={t0} />
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
