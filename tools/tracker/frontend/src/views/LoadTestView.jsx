import { useState, useEffect, useRef } from 'react'
import {
  BarChart3,
  Square,
  Trash2,
  UploadCloud,
  Clock,
  Zap,
  CheckCircle2,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription } from '@/components/ui/alert'
import SequenceToggles, {
  PIPELINE_NAMES,
  sequenceProblem,
  sequenceToSend,
  sequenceText,
} from '../components/SequenceToggles'

const STATUS_BUCKETS = [
  { key: '200', label: '200 Selesai', color: '#10b981' },
  { key: 'rejected', label: '400 Ditolak', color: '#f59e0b' },
  { key: '202', label: '202 Menyusul', color: '#38bdf8' },
  { key: '4xx', label: '4xx Klien', color: '#f87171' },
  { key: '5xx', label: '5xx Server', color: '#ef4444' },
  { key: 'timeout', label: 'Timeout', color: '#64748b' },
]

const SERVICE_UPLOAD_LIMIT = 2.5 * 1024 * 1024

function pct(n, total) {
  return total ? `${((100 * n) / total).toFixed(1)}%` : '-'
}

function fmtRate(v) {
  return v == null ? '-' : `${v.toFixed(2)} rps`
}

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function fmtBytes(n) {
  if (n == null) return ''
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function TestFilesManager({ config, selected, setSelected, onChanged }) {
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

  const allSelected = files.length > 0 && files.every((f) => selected.includes(f.name))

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-xs">
        <span className="font-semibold text-foreground">
          File Uji ({selected.length}/{files.length} dipakai)
        </span>
        {files.length > 1 && (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => setSelected(allSelected ? [] : files.map((f) => f.name))}
            className="h-5 px-1.5 text-[10px]"
          >
            {allSelected ? 'Kosongkan' : 'Pilih Semua'}
          </Button>
        )}
      </div>

      <div className="max-h-36 overflow-y-auto space-y-1 pr-1">
        {files.map(({ name, size }) => (
          <div
            key={name}
            className="flex items-center justify-between p-1.5 px-2.5 rounded-md bg-muted/40 border border-border/50 text-xs"
          >
            <label className="flex items-center gap-2 cursor-pointer overflow-hidden flex-1">
              <input
                type="checkbox"
                checked={selected.includes(name)}
                onChange={(e) => setSelected(e.target.checked ? [...selected, name] : selected.filter((n) => n !== name))}
                className="rounded border-border text-primary"
              />
              <span className="truncate font-mono text-[11px]">{name}</span>
            </label>
            <div className="flex items-center gap-2 shrink-0">
              <span className={`text-[10px] font-mono ${size > SERVICE_UPLOAD_LIMIT ? 'text-amber-400' : 'text-muted-foreground'}`}>
                {fmtBytes(size)}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={() => remove(name)}
                className="h-5 w-5 p-0 text-rose-400 hover:text-rose-300"
              >
                <Trash2 size={12} />
              </Button>
            </div>
          </div>
        ))}
      </div>

      <label
        className={`flex items-center justify-center gap-2 p-3 border border-dashed rounded-lg bg-background/40 cursor-pointer transition-colors ${
          dragging ? 'border-primary bg-accent/40' : 'border-border/70 hover:bg-muted/30'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (!busy) upload(e.dataTransfer.files)
        }}
      >
        <input ref={input} type="file" multiple accept=".jpg,.jpeg,.png,.pdf" disabled={busy} onChange={(e) => upload(e.target.files)} className="hidden" />
        <UploadCloud size={16} className="text-muted-foreground" />
        <span className="text-[11px] text-muted-foreground">
          {busy ? 'Mengunggah…' : 'Unggah file uji (.jpg, .png, .pdf)'}
        </span>
      </label>
      {error && <div className="text-xs text-rose-400">{error}</div>}
    </div>
  )
}

export default function LoadTestView() {
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

  const loadRuns = () => fetch('/api/loadtest').then((r) => r.json()).then(setRuns).catch(() => {})
  const loadConfig = () => fetch('/api/loadtest/config').then((r) => r.json()).then((c) => { setConfig(c); return c }).catch(() => null)

  useEffect(() => {
    loadConfig().then((c) => c && setImages(c.images || []))
    loadRuns()
    const timer = setInterval(loadRuns, 2000)
    return () => clearInterval(timer)
  }, [])

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
    const timer = setInterval(tick, 1000)
    return () => {
      stopped = true
      clearInterval(timer)
    }
  }, [current])

  async function start(e) {
    e.preventDefault()
    setStarting(true)
    setError(null)
    const payload = {
      rate: Number(rate),
      duration_seconds: Number(duration),
      mode,
      images,
      pipeline_name_sequence: sequenceToSend(sequence),
    }
    try {
      const r = await fetch('/api/loadtest', {
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
    } finally {
      setStarting(false)
    }
  }

  async function stop(runId) {
    await fetch(`/api/loadtest/${runId}/stop`, { method: 'POST' }).catch(() => {})
    loadRuns()
  }

  async function remove(runId) {
    if (!window.confirm(`Hapus data sesi run ${runId}?`)) return
    await fetch(`/api/loadtest/${runId}`, { method: 'DELETE' }).catch(() => {})
    if (current === runId) setCurrent(null)
    loadRuns()
  }

  const running = runs.some((r) => r.status === 'running')
  const d = detail
  const sent = d?.sent ?? 0
  const waitSeconds = d?.wait_seconds ?? 15

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
      {/* LEFT COLUMN: Controls & Runs History */}
      <div className="space-y-4">
        <Card className="border-border bg-card border-t-2 border-t-sky-500 shadow-xs">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="h-6 w-6 rounded-md bg-sky-500/10 border border-sky-500/25 flex items-center justify-center text-sky-600 dark:text-sky-400">
                <Zap size={14} />
              </div>
              <span>Konfigurasi Load Testing</span>
            </CardTitle>
            <CardDescription className="text-xs">
              Uji beban k6 container lokal ke endpoint /v1/extract-ocr
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={start} className="space-y-3">
              <div>
                <label className="text-xs font-medium text-foreground">Laju Request (RPS / req/sec)</label>
                <Input
                  type="number"
                  min="0.1"
                  step="0.1"
                  max="50"
                  value={rate}
                  onChange={(e) => setRate(e.target.value)}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-foreground">Durasi Uji (Detik)</label>
                <Input
                  type="number"
                  min="5"
                  max="600"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-foreground">Pola Beban Trafik</label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="flex h-9 w-full rounded-md border border-input bg-background/80 px-3 py-1.5 text-xs text-foreground mt-1"
                >
                  <option value="constant">Beban Tetap (Constant Rate)</option>
                  <option value="ramp">Naik Bertahap (Ramp Up)</option>
                </select>
              </div>

              <SequenceToggles names={sequence} onChange={setSequence} />

              <TestFilesManager config={config} selected={images} setSelected={setImages} onChanged={loadConfig} />

              <Button
                type="submit"
                variant="default"
                disabled={starting || running || images.length === 0 || sequenceProblem(sequence) !== null}
                className="w-full text-xs font-semibold gap-1.5"
              >
                {running ? 'Ada Run yang Sedang Berjalan' : starting ? 'Menyalakan k6 Container…' : 'Mulai Load Testing'}
              </Button>
            </form>
            {error && (
              <Alert variant="destructive" className="mt-3 py-2 text-xs">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Runs History */}
        <Card className="border-border bg-card border-t-2 border-t-zinc-400 dark:border-t-zinc-600 shadow-xs">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-semibold flex items-center gap-2">
              <div className="h-6 w-6 rounded-md bg-muted border border-border flex items-center justify-center text-muted-foreground">
                <Clock size={13} />
              </div>
              <span>Daftar Sesi Run ({runs.length})</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5 max-h-[300px] overflow-y-auto pr-1">
            {runs.map((r) => {
              const active = r.run_id === current
              return (
                <div
                  key={r.run_id}
                  onClick={() => setCurrent(r.run_id)}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer text-left space-y-1 ${
                    active
                      ? 'bg-accent border-border text-foreground'
                      : 'bg-background/40 border-border/50 hover:bg-muted/40'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-foreground">{r.run_id}</span>
                    <Badge variant={r.status === 'running' ? 'processing' : r.status === 'finished' ? 'done' : 'failed'} className="text-[10px] px-1.5 py-0">
                      {r.status}
                    </Badge>
                  </div>
                  <div className="text-[11px] text-muted-foreground font-mono flex gap-1.5">
                    <span>{r.rate} rps · {r.duration_seconds}s</span>
                    <span>· {r.sent} sent</span>
                    <span className="text-emerald-400">· {r.completed} done</span>
                  </div>
                </div>
              )
            })}
            {runs.length === 0 && (
              <div className="text-center py-6 text-xs text-muted-foreground">
                Belum ada sesi run.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* RIGHT COLUMN: Load Test Dashboard Metrics */}
      <div className="space-y-4">
        {!d && (
          <Card className="border-border bg-card p-12 text-center">
            <div className="max-w-md mx-auto space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border bg-muted text-muted-foreground mx-auto">
                <BarChart3 size={18} />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-foreground">
                  Pilih atau Mulai Sesi Load Testing
                </h3>
                <p className="text-xs text-muted-foreground leading-normal">
                  Atur parameter laju dan durasi di panel samping untuk mengeksekusi k6 runner di container Docker lokal.
                </p>
              </div>
            </div>
          </Card>
        )}

        {d && (
          <>
            {/* Header */}
            <Card className="border-border bg-card border-t-2 border-t-blue-500 shadow-xs p-3.5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-md bg-blue-500/10 border border-blue-500/25 flex items-center justify-center text-blue-600 dark:text-blue-400 shrink-0">
                    <Zap size={16} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-bold text-foreground">
                        RUN: {d.run_id}
                      </span>
                      <Badge variant="outline" className="text-[10px] font-mono border-blue-500/30 text-blue-600 dark:text-blue-400">
                        k6 Runner Session
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground font-mono mt-0.5">
                      {d.rate} rps · {d.duration_seconds}s · {d.mode} · sequence: {sequenceText(d.pipeline_name_sequence)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={d.status === 'running' ? 'processing' : 'done'} className="text-[10px] font-mono">
                    {d.status}
                  </Badge>
                  {d.status === 'running' ? (
                    <Button type="button" variant="outline" size="xs" onClick={() => stop(d.run_id)} className="h-7 px-2.5 text-xs gap-1">
                      <Square size={12} />
                      <span>Hentikan</span>
                    </Button>
                  ) : (
                    <Button type="button" variant="destructive" size="xs" onClick={() => remove(d.run_id)} className="h-7 px-2.5 text-xs gap-1">
                      <Trash2 size={12} />
                      <span>Hapus</span>
                    </Button>
                  )}
                </div>
              </div>
            </Card>

            {/* KPI Metrics Tiles */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
              <Card className="p-3 bg-muted/20 border-border">
                <div className="text-[10px] font-mono font-medium text-muted-foreground">TOTAL DIKIRIM</div>
                <div className="text-lg font-bold font-mono text-foreground mt-0.5">{sent}</div>
                <div className="text-[10px] text-muted-foreground mt-0.5">Rate: {fmtRate(d.achieved_rate)}</div>
              </Card>

              {STATUS_BUCKETS.map((b) => {
                const count = d.counts?.[b.key] ?? 0
                return (
                  <Card key={b.key} className="p-3.5 bg-background/60 border-border/70">
                    <div className="text-[10px] font-mono font-semibold text-muted-foreground">{b.label}</div>
                    <div className="text-xl font-bold font-mono mt-1" style={{ color: b.color }}>{count}</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">{pct(count, sent)} total</div>
                  </Card>
                )
              })}
            </div>

            {/* Stacked Distribution Bar */}
            <Card className="p-4 bg-card border-border border-t-2 border-t-amber-500 shadow-xs space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold flex items-center gap-2">
                  <div className="h-6 w-6 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-600 dark:text-amber-400">
                    <BarChart3 size={13} />
                  </div>
                  <span className="text-foreground">Distribusi Respons HTTP Orchestrator</span>
                </div>
                <Badge variant="outline" className="text-[10px] font-mono border-amber-500/30 text-amber-600 dark:text-amber-400">
                  Status Code Breakdown
                </Badge>
              </div>
              <div className="h-4 rounded-full bg-muted/60 overflow-hidden flex">
                {STATUS_BUCKETS.map((b) => {
                  const count = d.counts?.[b.key] ?? 0
                  const widthPct = sent ? (100 * count) / sent : 0
                  return (
                    <div
                      key={b.key}
                      style={{ width: `${widthPct}%`, background: b.color }}
                      title={`${b.label}: ${count} (${pct(count, sent)})`}
                      className="transition-all"
                    />
                  )
                })}
              </div>
              <div className="flex gap-4 flex-wrap text-xs pt-1">
                {STATUS_BUCKETS.map((b) => (
                  <div key={b.key} className="flex items-center gap-1.5 text-[11px]">
                    <span className="w-2 h-2 rounded-xs" style={{ background: b.color }} />
                    <span className="text-muted-foreground">{b.label} ({d.counts?.[b.key] ?? 0})</span>
                  </div>
                ))}
              </div>
            </Card>

            {/* Latency and End-to-end Panels */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="p-4 bg-card border-border border-t-2 border-t-sky-500 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs flex items-center gap-2">
                    <div className="h-6 w-6 rounded-md bg-sky-500/10 border border-sky-500/25 flex items-center justify-center text-sky-600 dark:text-sky-400">
                      <Clock size={13} />
                    </div>
                    <span>Pintu Masuk: /v1/extract-ocr</span>
                  </CardTitle>
                  <Badge variant="outline" className="text-[10px] font-mono border-sky-500/30 text-sky-600 dark:text-sky-400">
                    Ingress Latency
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">P50 LATENCY</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{fmtMs(d.extract?.p50)}</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">P95 LATENCY</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{fmtMs(d.extract?.p95)}</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">MAX LATENCY</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{fmtMs(d.extract?.max)}</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">BATAS TUNGGU</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{waitSeconds}s</div>
                  </div>
                </div>
              </Card>

              <Card className="p-4 bg-card border-border border-t-2 border-t-emerald-500 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs flex items-center gap-2">
                    <div className="h-6 w-6 rounded-md bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 size={13} />
                    </div>
                    <span>End-to-End Pipeline Tuntas</span>
                  </CardTitle>
                  <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/30 text-emerald-600 dark:text-emerald-400">
                    Full Pipeline
                  </Badge>
                </div>
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">TUNTAS SELESAI</div>
                    <div className="text-sm font-bold font-mono text-emerald-400 mt-0.5">{d.completed}</div>
                    <div className="text-[10px] text-muted-foreground">{pct(d.completed, sent)}</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">DOKUMEN / MENIT</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">
                      {d.completed_per_minute == null ? '-' : d.completed_per_minute.toFixed(1)}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">P50 E2E</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{fmtMs(d.e2e?.p50)}</div>
                  </div>
                  <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                    <div className="text-[10px] text-muted-foreground font-mono">P95 E2E</div>
                    <div className="text-sm font-bold font-mono text-foreground mt-0.5">{fmtMs(d.e2e?.p95)}</div>
                  </div>
                </div>
              </Card>
            </div>

            {/* k6 raw terminal tail */}
            {d.log_tail && (
              <Card className="p-4 bg-card border-border border-t-2 border-t-zinc-400 dark:border-t-zinc-600 shadow-xs space-y-2">
                <CardTitle className="text-xs flex items-center gap-2">
                  <div className="h-6 w-6 rounded-md bg-muted border border-border flex items-center justify-center text-muted-foreground">
                    <Clock size={13} />
                  </div>
                  <span>Log Eksekusi k6 Container</span>
                </CardTitle>
                <pre className="p-3 text-[11px] font-mono leading-relaxed bg-black/40 text-slate-300 rounded-lg overflow-x-auto max-h-64 border border-border/40">
                  {d.log_tail}
                </pre>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}
