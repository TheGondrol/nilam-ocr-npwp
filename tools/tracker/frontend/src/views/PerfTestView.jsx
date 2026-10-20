import { useState, useEffect, useRef } from 'react'
import {
  Gauge,
  Square,
  Trash2,
  FileDown,
  Clock,
  Server,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription } from '@/components/ui/alert'
import SequenceToggles, { PIPELINE_NAMES, sequenceProblem, sequenceToSend, sequenceText } from '../components/SequenceToggles'
import { TestFilesManager } from './LoadTestView'

const PROFILE_ORDER = ['smoke', 'load', 'stress', 'spike', 'soak']
const ACTIVE = ['starting', 'running', 'draining']
const STATUS_TEXT = {
  starting: 'menyiapkan',
  running: 'berjalan',
  draining: 'menunggu antrean',
  finished: 'selesai',
  aborted: 'dihentikan otomatis',
  stopped: 'dihentikan',
  failed: 'gagal',
  interrupted: 'terputus',
}
const TONE_CLASS = {
  ok: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
  bad: 'border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-400',
  warn: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400',
  info: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-400',
}
const TONE_ICON = { ok: CheckCircle2, bad: XCircle, warn: AlertTriangle, info: Info }
// SLO di UI dalam satuan yang enak diisi: persen dan detik; dikirim ke backend dalam satuan aslinya.
const SLO_FIELDS = [
  { key: 'error_rate', label: 'Error server maks (%)', toUi: (v) => v * 100, fromUi: (v) => v / 100, step: 0.1 },
  { key: 'entry_p95_ms', label: 'p95 pintu masuk maks (detik)', toUi: (v) => v / 1000, fromUi: (v) => v * 1000, step: 1 },
  { key: 'e2e_p95_ms', label: 'p95 end-to-end maks (detik)', toUi: (v) => v / 1000, fromUi: (v) => v * 1000, step: 1 },
  { key: 'min_completion', label: 'Rasio tuntas min (%)', toUi: (v) => v * 100, fromUi: (v) => v / 100, step: 0.5 },
  { key: 'max_restarts', label: 'Restart/OOM maks', toUi: (v) => v, fromUi: (v) => v, step: 1 },
  // Queueing theory: di atas ~75% waktu tunggu melonjak (rho / (1 - rho)); kapasitas stress dihitung di bawah ini.
  { key: 'max_cpu_utilization', label: 'Utilisasi CPU maks (%)', toUi: (v) => v * 100, fromUi: (v) => v / 100, step: 5 },
]

function fmtMs(ms) {
  if (ms == null) return '-'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}
function fmtPct(v) {
  return v == null ? '-' : `${(v * 100).toFixed(1)}%`
}
function fmtDuration(s) {
  if (s == null) return '-'
  s = Math.round(s)
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
}
function sloValue(key, v) {
  if (v == null) return '-'
  if (key === 'error_rate' || key === 'min_completion' || key === 'max_cpu_utilization') return fmtPct(v)
  if (key.endsWith('_ms')) return fmtMs(v)
  return String(v)
}

// Rencana beban di sisi UI, hanya untuk estimasi durasi dan puncak sebelum run (backend yang menentukan).
function estimate(profile, p) {
  const n = (k) => Number(p[k] ?? 0)
  if (profile === 'smoke') return { seconds: n('duration_seconds'), peak: n('base_rate') }
  if (profile === 'load' || profile === 'soak') return { seconds: n('ramp_seconds') + n('duration_seconds'), peak: n('base_rate') }
  if (profile === 'stress') {
    const steps = n('step_rate') > 0 ? Math.floor((n('peak_rate') - n('start_rate')) / n('step_rate') + 1e-9) + 1 : 1
    return { seconds: steps * (n('step_seconds') + 10), peak: n('start_rate') + (steps - 1) * n('step_rate'), steps }
  }
  if (profile === 'spike') return { seconds: 25 + n('baseline_seconds') + n('spike_seconds') + n('recovery_seconds'), peak: n('peak_rate') }
  return { seconds: 0, peak: 0 }
}

function LineChart({ title, series, hlines = [], xMax, yFmt = (v) => v.toFixed(1), height = 150 }) {
  const W = 600
  const H = height
  const pad = { l: 40, r: 8, t: 8, b: 18 }
  const values = [...series.flatMap((s) => s.points.map((p) => p[1]).filter((v) => v != null)), ...hlines.map((h) => h.value)]
  const yMax = Math.max(1e-9, ...values) * 1.1 || 1
  const x = (t) => pad.l + ((W - pad.l - pad.r) * Math.min(t, xMax)) / (xMax || 1)
  const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - Math.min(v / yMax, 1))
  const path = (pts) => {
    let d = ''
    let pen = false
    for (const [t, v] of pts) {
      if (v == null) {
        pen = false
        continue
      }
      d += `${pen ? 'L' : 'M'}${x(t).toFixed(1)},${y(v).toFixed(1)} `
      pen = true
    }
    return d
  }
  return (
    <div className="space-y-1">
      <div className="text-[11px] font-semibold text-foreground">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={y(yMax * f)} y2={y(yMax * f)} stroke="currentColor" className="text-border" strokeWidth="0.5" />
            <text x={pad.l - 4} y={y(yMax * f) + 3} textAnchor="end" fontSize="9" className="fill-muted-foreground">
              {yFmt(yMax * f)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <text key={f} x={x(xMax * f)} y={H - 4} textAnchor="middle" fontSize="9" className="fill-muted-foreground">
            {((xMax * f) / 60).toFixed(1)}m
          </text>
        ))}
        {hlines.map((h) => (
          <line key={h.label} x1={pad.l} x2={W - pad.r} y1={y(h.value)} y2={y(h.value)} stroke={h.color} strokeDasharray="4 3" strokeWidth="1" />
        ))}
        {series.map((s) => (
          <path key={s.label} d={path(s.points)} fill="none" stroke={s.color} strokeWidth="1.8" strokeDasharray={s.dashed ? '5 3' : undefined} />
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 text-[10px] text-muted-foreground">
        {[...series, ...hlines].map((s) => (
          <span key={s.label} className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  )
}

function Tile({ label, value, sub, tone }) {
  return (
    <Card className={`p-3 border ${tone ? TONE_CLASS[tone] : 'bg-muted/20 border-border'}`}>
      <div className="text-[10px] font-mono font-medium opacity-80">{label}</div>
      <div className="text-lg font-bold font-mono mt-0.5">{value}</div>
      {sub && <div className="text-[10px] opacity-70 mt-0.5">{sub}</div>}
    </Card>
  )
}

export default function PerfTestView() {
  const [config, setConfig] = useState(null)
  const [runs, setRuns] = useState([])
  const [current, setCurrent] = useState(null)
  const [detail, setDetail] = useState(null)
  const [profile, setProfile] = useState('smoke')
  const [params, setParams] = useState({})
  const [slo, setSlo] = useState({})
  const [images, setImages] = useState([])
  const [sequence, setSequence] = useState(PIPELINE_NAMES)
  const [note, setNote] = useState('')
  const [ack, setAck] = useState(false)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)

  const loadRuns = () => fetch('/api/perftest').then((r) => r.json()).then(setRuns).catch(() => {})
  const loadConfig = () =>
    fetch('/api/perftest/config')
      .then((r) => r.json())
      .then((c) => {
        setConfig(c)
        return c
      })
      .catch(() => null)

  useEffect(() => {
    loadConfig().then((c) => {
      if (!c) return
      setImages(c.image_files.filter((f) => !f.upload).map((f) => f.name))
      setSlo(Object.fromEntries(SLO_FIELDS.map((f) => [f.key, f.toUi(c.slo[f.key])])))
      setParams({ ...c.profiles.smoke.defaults })
    })
    loadRuns()
    const timer = setInterval(loadRuns, 3000)
    return () => clearInterval(timer)
  }, [])

  function selectProfile(name) {
    setProfile(name)
    if (config) setParams({ ...config.profiles[name].defaults })
    setAck(false)
  }

  useEffect(() => {
    if (!current) return
    let stopped = false
    const tick = () =>
      fetch(`/api/perftest/${current}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then((d) => !stopped && setDetail(d))
        .catch(() => {})
    tick()
    const timer = setInterval(tick, 3000)
    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [current])

  const info = config?.profiles?.[profile]
  const est = estimate(profile, params)
  const running = runs.some((r) => ACTIVE.includes(r.status))
  const needsAck = info?.heavy

  // Run yang berakhir menghapus unggahannya: muat ulang daftar file supaya yang sudah dihapus tidak tampil lagi.
  const wasRunning = useRef(false)
  useEffect(() => {
    if (wasRunning.current && !running) loadConfig()
    wasRunning.current = running
  }, [running])

  async function start(e) {
    e.preventDefault()
    setStarting(true)
    setError(null)
    const payload = {
      profile,
      params: Object.fromEntries(Object.entries(params).map(([k, v]) => [k, Number(v)])),
      slo: Object.fromEntries(SLO_FIELDS.map((f) => [f.key, f.fromUi(Number(slo[f.key]))])),
      images,
      pipeline_name_sequence: sequenceToSend(sequence),
      note,
    }
    try {
      const r = await fetch('/api/perftest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail ?? r.statusText)
      setCurrent(d.run_id)
      loadRuns()
    } catch (err) {
      setError(String(err.message ?? err))
      loadConfig()
    } finally {
      setStarting(false)
    }
  }

  async function stop(run) {
    if (!window.confirm(`Hentikan run ${run}? Laporan tetap dibuat dari data yang sudah terkumpul.`)) return
    await fetch(`/api/perftest/${run}/stop`, { method: 'POST' }).catch(() => {})
    loadRuns()
  }

  async function remove(run) {
    if (!window.confirm(`Hapus run ${run} beserta laporannya?`)) return
    const cleanup = window.confirm('Hapus juga baris TEST_' + run + '_* dari tabel nilam_testing_* di database dev?')
    const r = await fetch(`/api/perftest/${run}?cleanup_db=${cleanup}`, { method: 'DELETE' })
    if (!r.ok) {
      const d = await r.json().catch(() => ({}))
      window.alert(d.detail ?? r.statusText)
      return
    }
    if (current === run) setCurrent(null)
    loadRuns()
  }

  const shown = current && detail?.meta?.run_id === current ? detail : null
  const m = shown?.meta
  const res = shown?.result
  const s = res?.summary
  const windows = res?.windows ?? []
  const xMax = Math.max(60, ...windows.map((w) => w.t + 30), m?.plan ? m.plan.reduce((a, p) => a + p.seconds, 0) : 0)
  const totalSeconds = m?.plan?.reduce((a, p) => a + p.seconds, 0) ?? 0
  const progress = m?.elapsed_seconds && totalSeconds ? Math.min(1, m.elapsed_seconds / totalSeconds) : m && !ACTIVE.includes(m.status) ? 1 : 0

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6 items-start">
      <div className="space-y-4">
        <Card className="border-border bg-card border-t-2 border-t-violet-500 shadow-xs">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="h-6 w-6 rounded-md bg-violet-500/10 border border-violet-500/25 flex items-center justify-center text-violet-600 dark:text-violet-400">
                <Gauge size={14} />
              </div>
              <span>Performance Test GKE</span>
            </CardTitle>
            <CardDescription className="text-xs space-y-1">
              <span className="block">k6 sebagai Job di cluster, menembak endpoint -test (tabel nilam_testing_*, tanpa callback).</span>
              {config && (
                <span className="block font-mono text-[10px]">
                  {config.context.split('_').pop()} / {config.namespace} ·{' '}
                  <span className={config.cluster_ok ? 'text-emerald-500' : 'text-rose-500'}>
                    {config.cluster_ok ? 'kubectl ok' : 'kubectl tidak bisa membuat Job'}
                  </span>{' '}
                  · DB relay: {config.db_relay ?? 'dibuat saat run'}
                </span>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {config && !config.cluster_ok && (
              <Alert variant="destructive" className="mb-3 py-2 text-xs">
                <AlertDescription>{config.cluster_error || 'Tidak punya izin membuat Job di namespace ini.'}</AlertDescription>
              </Alert>
            )}
            <form onSubmit={start} className="space-y-3">
              <div className="grid grid-cols-5 gap-1">
                {PROFILE_ORDER.map((name) => (
                  <Button
                    key={name}
                    type="button"
                    size="xs"
                    variant={profile === name ? 'default' : 'outline'}
                    onClick={() => selectProfile(name)}
                    className="text-[11px] px-1"
                  >
                    {config?.profiles?.[name]?.label.split(' ')[0] ?? name}
                  </Button>
                ))}
              </div>
              {info && (
                <div className="text-[11px] text-muted-foreground leading-snug p-2 rounded-md bg-muted/40 border border-border/50">
                  <div className="font-semibold text-foreground">{info.question}</div>
                  <div className="mt-0.5">{info.summary}</div>
                </div>
              )}

              {info &&
                Object.entries(info.params).map(([key, label]) => (
                  <div key={key}>
                    <label className="text-xs font-medium text-foreground">{label}</label>
                    <Input
                      type="number"
                      step={key.endsWith('rate') ? '0.1' : '1'}
                      min={key.endsWith('rate') ? '0.1' : '0'}
                      value={params[key] ?? ''}
                      onChange={(e) => setParams({ ...params, [key]: e.target.value })}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                ))}
              <div className="text-[11px] font-mono text-muted-foreground">
                Estimasi: {fmtDuration(est.seconds)} · puncak {est.peak.toFixed(1)} rps
                {est.steps ? ` · ${est.steps} step` : ''} · ~{Math.round(est.peak * est.seconds * 0.6)} request
              </div>

              <details className="rounded-md border border-border/60 p-2">
                <summary className="text-xs font-semibold cursor-pointer">Kriteria lulus (SLO)</summary>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  {SLO_FIELDS.map((f) => (
                    <div key={f.key}>
                      <label className="text-[10px] text-muted-foreground">{f.label}</label>
                      <Input
                        type="number"
                        step={f.step}
                        value={slo[f.key] ?? ''}
                        onChange={(e) => setSlo({ ...slo, [f.key]: e.target.value })}
                        className="h-7 text-xs"
                      />
                    </div>
                  ))}
                </div>
              </details>

              <SequenceToggles names={sequence} onChange={setSequence} />
              <TestFilesManager config={config} selected={images} setSelected={setImages} onChanged={loadConfig} />

              <div>
                <label className="text-xs font-medium text-foreground">Catatan (masuk laporan)</label>
                <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="mis. setelah rilis f2dde8e" className="mt-1 h-8 text-xs" />
              </div>

              {needsAck && (
                <label className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 p-2 rounded-md border border-amber-500/30 bg-amber-500/5">
                  <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} className="mt-0.5" />
                  <span>
                    Tes ini membebani cluster dev yang dipakai bersama (dan VM model OCR). Sudah dikoordinasikan dengan
                    tim yang memakai cluster.
                  </span>
                </label>
              )}

              <Button
                type="submit"
                disabled={starting || running || !images.length || sequenceProblem(sequence) !== null || (needsAck && !ack) || (config && !config.cluster_ok)}
                className="w-full text-xs font-semibold"
              >
                {running ? 'Ada Run yang Berjalan' : starting ? 'Membuat Job k6…' : `Mulai ${info?.label ?? ''}`}
              </Button>
            </form>
            {error && (
              <Alert variant="destructive" className="mt-3 py-2 text-xs">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card className="border-border bg-card border-t-2 border-t-zinc-400 dark:border-t-zinc-600 shadow-xs">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold flex items-center gap-2">
              <Clock size={13} />
              <span>Riwayat Run ({runs.length})</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5 max-h-[340px] overflow-y-auto pr-1">
            {runs.map((r) => (
              <div
                key={r.run_id}
                onClick={() => setCurrent(r.run_id)}
                className={`p-2.5 rounded-lg border cursor-pointer space-y-1 ${
                  r.run_id === current ? 'bg-accent border-border' : 'bg-background/40 border-border/50 hover:bg-muted/40'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold">
                    {r.run_id} · {config?.profiles?.[r.profile]?.label ?? r.profile}
                  </span>
                  <Badge
                    variant={ACTIVE.includes(r.status) ? 'processing' : r.status === 'finished' ? 'done' : r.status === 'failed' ? 'failed' : 'rejected'}
                    className="text-[10px] px-1.5 py-0"
                  >
                    {STATUS_TEXT[r.status] ?? r.status}
                  </Badge>
                </div>
                <div className="text-[11px] text-muted-foreground font-mono">
                  {new Date(r.created_at * 1000).toLocaleString()} · {r.sent} req
                </div>
                {r.verdict && <div className={`text-[11px] font-semibold ${TONE_CLASS[r.verdict.tone]?.split(' ').slice(-2).join(' ')}`}>{r.verdict.title}</div>}
              </div>
            ))}
            {!runs.length && <div className="text-center py-6 text-xs text-muted-foreground">Belum ada run.</div>}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        {!shown && (
          <Card className="border-border bg-card p-10 text-sm text-muted-foreground space-y-3">
            <div className="font-semibold text-foreground">Urutan yang disarankan</div>
            <ol className="list-decimal pl-5 space-y-1 text-xs">
              <li><b>Smoke</b>: pastikan semuanya hidup dan alat ukur bekerja.</li>
              <li><b>Load</b>: beban normal yang diharapkan; hasilnya jadi baseline antar rilis.</li>
              <li><b>Stress</b>: cari laju maksimum yang masih memenuhi SLO dan service yang jenuh lebih dulu.</li>
              <li><b>Spike</b>: lonjakan mendadak dan waktu pulihnya.</li>
              <li><b>Soak</b>: kestabilan jangka panjang (kebocoran memori, degradasi).</li>
            </ol>
            <div className="text-xs">Setiap run selesai menghasilkan laporan PDF otomatis (tombol unduh di panel run).</div>
          </Card>
        )}

        {m && res && (
          <>
            <Card className="p-3.5 border-border border-t-2 border-t-violet-500 space-y-2">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <div className="font-mono text-sm font-bold">
                    {m.run_id} · {config?.profiles?.[m.profile]?.label ?? m.profile}
                  </div>
                  <div className="text-xs text-muted-foreground font-mono">
                    {STATUS_TEXT[m.status] ?? m.status}
                    {m.status_detail ? ` · ${m.status_detail}` : ''} · sequence: {sequenceText(m.pipeline_name_sequence)} · VU{' '}
                    {m.vus?.join('/')}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="xs" className="gap-1" onClick={() => window.open(`/api/perftest/${m.run_id}/report.pdf`, '_blank')}>
                    <FileDown size={12} />
                    {ACTIVE.includes(m.status) ? 'PDF sementara' : 'Unduh PDF'}
                  </Button>
                  {ACTIVE.includes(m.status) ? (
                    <Button variant="outline" size="xs" className="gap-1" onClick={() => stop(m.run_id)}>
                      <Square size={12} />
                      Hentikan
                    </Button>
                  ) : (
                    <Button variant="destructive" size="xs" className="gap-1" onClick={() => remove(m.run_id)}>
                      <Trash2 size={12} />
                      Hapus
                    </Button>
                  )}
                </div>
              </div>
              <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-violet-500 transition-all" style={{ width: `${progress * 100}%` }} />
              </div>
              {(m.error || m.report_error || m.db_note) && (
                <div className="text-[11px] text-rose-500 font-mono">{m.error || m.report_error || `database: ${m.db_note}`}</div>
              )}
            </Card>

            <div className={`p-3 rounded-lg border ${TONE_CLASS[res.verdict.tone]}`}>
              <div className="text-sm font-bold">{res.verdict.title}</div>
              <div className="text-xs mt-0.5">{res.verdict.body}</div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
              <Tile label="DIKIRIM" value={s.sent} sub={s.achieved_rate == null ? '' : `${s.achieved_rate.toFixed(2)} rps`} />
              <Tile label="ERROR SERVER" value={fmtPct(s.error_rate)} sub={`${s.counts['5xx']} 5xx · ${s.counts.timeout} timeout`} tone={s.counts['5xx'] + s.counts.timeout ? 'bad' : undefined} />
              <Tile label="202 MENYUSUL" value={s.counts['202']} sub={`${s.counts['200']} langsung 200`} />
              <Tile label="P95 PINTU MASUK" value={fmtMs(s.entry.p95)} sub={`p50 ${fmtMs(s.entry.p50)}`} />
              <Tile label="P95 END-TO-END" value={fmtMs(s.e2e.p95)} sub={`p50 ${fmtMs(s.e2e.p50)}`} />
              <Tile label="TUNTAS" value={s.completed} sub={`${s.in_flight} belum · ${s.failed} gagal · ${s.rejected} ditolak`} />
            </div>

            <Card className="p-4 border-border space-y-1.5">
              <div className="text-xs font-semibold">Kriteria lulus (SLO)</div>
              {res.slo.map((row) => {
                const tone = row.ok ? 'ok' : row.ok === false ? 'bad' : 'warn'
                const Icon = TONE_ICON[tone]
                return (
                  <div key={row.key} className="flex items-center justify-between text-xs border-b border-border/40 py-1">
                    <span className="flex items-center gap-1.5">
                      <Icon size={13} className={TONE_CLASS[tone].split(' ').slice(-2).join(' ')} />
                      {row.label}
                      {row.note && <span className="text-[10px] text-muted-foreground">({row.note})</span>}
                    </span>
                    <span className="font-mono">
                      {sloValue(row.key, row.actual)} <span className="text-muted-foreground">/ {sloValue(row.key, row.target)}</span>
                    </span>
                  </div>
                )
              })}
            </Card>

            <Card className="p-4 border-border grid grid-cols-1 xl:grid-cols-2 gap-4">
              <LineChart
                title="Laju (rps): rencana, dikirim, tuntas"
                xMax={xMax}
                series={[
                  { label: 'rencana', color: '#71717a', dashed: true, points: windows.map((w) => [w.t + 15, w.target_rate]) },
                  { label: 'dikirim', color: '#2563eb', points: windows.map((w) => [w.t + 15, w.rate]) },
                  { label: 'tuntas', color: '#16a34a', points: windows.map((w) => [w.t + 15, w.done_rate]) },
                ]}
              />
              <LineChart
                title="Latensi (detik)"
                xMax={xMax}
                yFmt={(v) => `${v.toFixed(0)}s`}
                series={[
                  { label: 'p95 pintu masuk', color: '#2563eb', points: windows.map((w) => [w.t + 15, w.entry_p95 == null ? null : w.entry_p95 / 1000]) },
                  { label: 'p95 end-to-end', color: '#16a34a', points: windows.map((w) => [w.t + 15, w.e2e_p95 == null ? null : w.e2e_p95 / 1000]) },
                ]}
                hlines={[
                  { label: 'SLO pintu masuk', value: res.slo_targets.entry_p95_ms / 1000, color: '#93c5fd' },
                  { label: 'SLO e2e', value: res.slo_targets.e2e_p95_ms / 1000, color: '#86efac' },
                ]}
              />
            </Card>

            <Card className="p-4 border-border space-y-2">
              <div className="text-xs font-semibold">Temuan</div>
              {res.findings.map(([tone, text], i) => {
                const Icon = TONE_ICON[tone] ?? Info
                return (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <Icon size={13} className={`mt-0.5 shrink-0 ${TONE_CLASS[tone]?.split(' ').slice(-2).join(' ')}`} />
                    <span>{text}</span>
                  </div>
                )
              })}
            </Card>

            <Card className="p-4 border-border space-y-2 overflow-x-auto">
              <div className="text-xs font-semibold">Per fase</div>
              <table className="w-full text-[11px] font-mono">
                <thead className="text-muted-foreground">
                  <tr className="text-left">
                    <th className="py-1">Fase</th><th>rps</th><th>kirim</th><th>error</th><th>202</th><th>p95 masuk</th><th>p95 e2e</th><th>tuntas</th><th>CPU maks</th>
                  </tr>
                </thead>
                <tbody>
                  {res.phases.map((p) => {
                    const cpuOver = p.cpu_util_max != null && p.cpu_util_max > res.slo_targets.max_cpu_utilization
                    return (
                      <tr key={p.name} className={`border-t border-border/40 ${p.kind === 'ramp' ? 'text-muted-foreground' : ''}`}>
                        <td className="py-1">{p.label}</td>
                        <td>{p.achieved_rate == null ? '-' : p.achieved_rate.toFixed(2)}</td>
                        <td>{p.sent}</td>
                        <td>{fmtPct(p.error_rate)}</td>
                        <td>{fmtPct(p.share_202)}</td>
                        <td>{fmtMs(p.entry_p95)}</td>
                        <td>{fmtMs(p.e2e_p95)}</td>
                        <td>{p.completed_of_sent}/{p.sent}</td>
                        <td className={cpuOver && p.kind === 'hold' ? 'text-rose-500' : ''} title="utilisasi rata-rata pod tersibuk, service paling sibuk">
                          {p.cpu_util_max == null ? '-' : `${Math.round(p.cpu_util_max * 100)}% ${p.cpu_busiest}`}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </Card>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <Card className="p-4 border-border space-y-2 overflow-x-auto">
                <div className="text-xs font-semibold flex items-center gap-1.5"><Server size={13} />Resource (pod tertinggi vs limit)</div>
                <table className="w-full text-[11px] font-mono">
                  <thead className="text-muted-foreground"><tr className="text-left"><th className="py-1">service</th><th>CPU rata2</th><th>CPU maks</th><th>mem</th><th>restart</th></tr></thead>
                  <tbody>
                    {Object.entries(res.resources).filter(([, r]) => r.cpu_max_m != null || r.restarts).map(([svc, r]) => (
                      <tr key={svc} className="border-t border-border/40">
                        <td className="py-1">{svc} ×{r.replicas ?? '-'}</td>
                        <td className={r.cpu_pct_avg > res.slo_targets.max_cpu_utilization ? 'text-rose-500' : ''}>{fmtPct(r.cpu_pct_avg)}</td>
                        <td className={r.cpu_pct_max >= 0.9 ? 'text-amber-500' : ''}>{fmtPct(r.cpu_pct_max)}</td>
                        <td className={r.mem_pct_max >= 0.85 ? 'text-amber-500' : ''}>{fmtPct(r.mem_pct_max)}</td>
                        <td className={r.restarts || r.oom ? 'text-rose-500' : ''}>{r.restarts}/{r.oom}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
              <Card className="p-4 border-border space-y-2 overflow-x-auto">
                <div className="text-xs font-semibold">Per tahap pipeline</div>
                <table className="w-full text-[11px] font-mono">
                  <thead className="text-muted-foreground"><tr className="text-left"><th className="py-1">tahap</th><th>n</th><th>p50</th><th>p95</th><th>jeda p95</th></tr></thead>
                  <tbody>
                    {Object.entries(res.stages).map(([name, st]) => (
                      <tr key={name} className="border-t border-border/40">
                        <td className="py-1">{name}</td>
                        <td>{st.count}</td>
                        <td>{st.p50 == null ? '-' : fmtMs(st.p50 * 1000)}</td>
                        <td>{st.p95 == null ? '-' : fmtMs(st.p95 * 1000)}</td>
                        <td>{st.gap_p95 == null ? '-' : fmtMs(st.gap_p95 * 1000)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>

            {m.log_tail && (
              <Card className="p-4 border-border space-y-2">
                <div className="text-xs font-semibold">Log k6 (selain baris sampel)</div>
                <pre className="p-3 text-[11px] font-mono bg-black/40 text-slate-300 rounded-lg overflow-x-auto max-h-56">{m.log_tail}</pre>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
