import { useState, useEffect, useMemo } from 'react'
import {
  ShieldAlert,
  Flame,
  Play,
  Square,
  ExternalLink,
  Search,
  RefreshCw,
  SlidersHorizontal,
  Film,
  Clock,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableRow, TableCell } from '@/components/ui/table'
import { Alert, AlertDescription } from '@/components/ui/alert'
import TopologyMap from '../components/TopologyMap'
import ScenarioFileManager from '../components/ScenarioFileManager'
import ScenarioReplayModal from '../components/ScenarioReplayModal'
import ScenarioExplanationPopover from '../components/ScenarioExplanationPopover'

const LEVEL_LABEL = {
  pass: 'PASS',
  warn: 'WARN',
  fail: 'FAIL',
  info: 'INFO',
  error: 'ERROR',
  running: 'JALAN',
  stopped: 'DIHENTIKAN',
}

const CHAOS_SERVICES = ['orchestrator', 'guardrails', 'extraction', 'structuring', 'scoring', 'postgres']

const SCENARIO_CATEGORIES = {
  'callback-down': 'outbox',
  'callback-flaky': 'outbox',
  'callback-not-ready': 'outbox',
  'callback-slow': 'outbox',
  'callback-rejected': 'outbox',
  'stage-down': 'crash',
  'sigterm-drained': 'crash',
  'sigterm-interrupted': 'crash',
  'crash-inline': 'crash',
  'crash-file-url': 'crash',
  'postgres-down': 'database',
  'resend': 'idempotency',
  'concurrent': 'idempotency',
  'entry-down': 'crash',
}

const CATEGORY_TABS = [
  { id: 'all', label: 'Semua Skenario' },
  { id: 'outbox', label: 'Outbox & Callback' },
  { id: 'crash', label: 'Pod Crash & Drain' },
  { id: 'database', label: 'Database Failover' },
  { id: 'idempotency', label: 'Idempotensi & Concurrency' },
]

function fmtSec(seconds) {
  if (seconds == null) return '-'
  return seconds < 60 ? `${seconds.toFixed(0)} dtk` : `${(seconds / 60).toFixed(1)} mnt`
}

function fmtClock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString() : ''
}

function badgeVariantOf(status) {
  switch (status?.toLowerCase()) {
    case 'pass': return 'done'
    case 'warn': return 'rejected'
    case 'fail':
    case 'error': return 'failed'
    case 'running': return 'processing'
    default: return 'idle'
  }
}

function RunReport({ run, openRequest, onOpenReplay }) {
  const [showSteps, setShowSteps] = useState(false)
  return (
    <div className="mt-3.5 border-t border-border/50 pt-3 space-y-2.5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap font-mono">
          <span>Dimulai: {fmtClock(run.started_at)}</span>
          <span>· Durasi: {run.ended_at ? fmtSec(run.ended_at - run.started_at) : 'sedang berjalan…'}</span>
          {run.options && (
            <span className="text-foreground/80">
              · Injeksi: t+{run.options.chaos_delay_seconds ?? 5}s · Mati: {run.options.outage_seconds ?? 10}s
            </span>
          )}
          {run.request_ids?.length > 0 && <span>· Request:</span>}
          {run.request_ids?.map((rid) => (
            <Button
              key={rid}
              type="button"
              variant="ghost"
              size="xs"
              className="h-5 px-1.5 text-[11px] font-mono text-muted-foreground gap-1 hover:text-foreground"
              onClick={() => openRequest(rid)}
            >
              <span>{rid}</span>
              <ExternalLink size={10} />
            </Button>
          ))}
        </div>

        {onOpenReplay && (
          <Button
            type="button"
            variant="secondary"
            size="xs"
            className="h-6 px-2.5 text-[11px] font-semibold gap-1.5 bg-primary/10 text-primary border border-primary/25 hover:bg-primary/20 transition-all shadow-xs"
            onClick={() => onOpenReplay(run)}
          >
            <Film size={11} />
            <span>Putar Replay Animasi</span>
          </Button>
        )}
      </div>

      {run.checks?.length > 0 && (
        <Table>
          <TableBody>
            {run.checks.map((c, i) => (
              <TableRow key={i}>
                <TableCell className="w-16 py-1.5">
                  <Badge variant={badgeVariantOf(c.level)} className="text-[10px] px-1.5 py-0 font-mono">
                    {LEVEL_LABEL[c.level] ?? c.level}
                  </Badge>
                </TableCell>
                <TableCell className="py-1.5">
                  <div className="font-semibold text-foreground text-xs">{c.label}</div>
                  {c.detail && <div className="text-[11px] text-muted-foreground mt-0.5">{c.detail}</div>}
                </TableCell>
                <TableCell className="w-20 text-right font-mono text-muted-foreground text-[11px] py-1.5">
                  t+{c.t}s
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <Button
        type="button"
        variant="ghost"
        size="xs"
        className="text-[11px] h-6 px-2 text-muted-foreground hover:text-foreground"
        onClick={() => setShowSteps(!showSteps)}
      >
        {showSteps ? 'Sembunyikan Langkah' : `Lihat Jejak Eksekusi (${run.steps?.length ?? 0} langkah)`}
      </Button>

      {(showSteps || run.status === 'running' || run.status === 'error') && (
        <ol className="m-0 pl-4 space-y-1 text-xs text-muted-foreground max-h-48 overflow-y-auto font-mono list-decimal border-l border-border/40 ml-1">
          {(run.steps ?? []).map((s, i) => (
            <li key={i} className={`py-0.5 ${s.text?.startsWith('ERROR') ? 'text-rose-400 font-semibold' : 'text-slate-300'}`}>
              <span className="text-muted-foreground mr-1.5 font-sans">t+{s.t}s</span>
              <span>{s.text}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function ManualDisruptionForm({ chaos, onDisrupt }) {
  const [service, setService] = useState('structuring')
  const [action, setAction] = useState('stop')
  const [seconds, setSeconds] = useState(20)

  const handleSubmit = (e) => {
    e.preventDefault()
    onDisrupt(service, action, seconds)
  }

  return (
    <Card className="border-border bg-card border-t-2 border-t-amber-500 shadow-xs">
      <CardHeader className="pb-3">
        <CardTitle className="text-xs font-semibold flex items-center gap-2">
          <div className="h-6 w-6 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-600 dark:text-amber-400">
            <Flame size={13} />
          </div>
          <span>Picukan Gangguan Pod Khusus</span>
        </CardTitle>
        <CardDescription className="text-xs">
          Simulasikan crash mandiri di luar alur pengujian otomatis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="text-xs font-medium text-foreground">Target Service Pod</label>
            <select
              value={service}
              onChange={(e) => setService(e.target.value)}
              className="flex h-8 w-full rounded-md border border-input bg-background/80 px-2.5 py-1 text-xs text-foreground mt-1"
            >
              {CHAOS_SERVICES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-foreground">Metode Gangguan</label>
            <select
              value={action}
              onChange={(e) => setAction(e.target.value)}
              className="flex h-8 w-full rounded-md border border-input bg-background/80 px-2.5 py-1 text-xs text-foreground mt-1"
            >
              <option value="stop">SIGTERM (Graceful shutdown, drain outbox relay)</option>
              <option value="kill">SIGKILL (Crash mendadak / OOM killed)</option>
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-foreground">Durasi Mati (Detik)</label>
            <Input
              type="number"
              min="1"
              max="600"
              value={seconds}
              className="h-8 text-xs mt-1"
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </div>

          <Button
            type="submit"
            variant="default"
            className="w-full text-xs font-semibold gap-1.5 bg-amber-600 hover:bg-amber-700 text-white"
            disabled={!chaos?.available}
          >
            <Flame size={14} />
            <span>Eksekusi Gangguan</span>
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

export default function ScenariosView({ openRequest }) {
  const [data, setData] = useState(null)
  const [chaos, setChaos] = useState(null)
  const [image, setImage] = useState('')
  const [error, setError] = useState(null)
  const [open, setOpen] = useState({})
  const [activeCategory, setActiveCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedReplay, setSelectedReplay] = useState(null)
  const [chaosDelaySeconds, setChaosDelaySeconds] = useState(5)
  const [outageSeconds, setOutageSeconds] = useState(10)

  const loadScenarios = () =>
    fetch('/api/scenarios')
      .then((r) => r.json())
      .then((d) => {
        setData(d)
        setImage((cur) => cur || d.images?.[0] || '')
      })
      .catch(() => {})

  const loadChaos = () =>
    fetch('/api/chaos')
      .then((r) => r.json())
      .then(setChaos)
      .catch(() => {})

  useEffect(() => {
    loadScenarios()
    loadChaos()
    const timer = setInterval(() => {
      loadScenarios()
      loadChaos()
    }, 1500)
    return () => clearInterval(timer)
  }, [])

  const handleDisrupt = async (service, action, seconds) => {
    setError(null)
    try {
      const res = await fetch('/api/chaos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ service, action, seconds }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || res.statusText)
      }
      loadChaos()
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  const handleStartContainer = async (service) => {
    try {
      await fetch(`/api/chaos/${service}/start`, { method: 'POST' })
      loadChaos()
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  const runScenarios = async (ids, overrideOptions = {}) => {
    setError(null)
    try {
      const res = await fetch('/api/scenarios/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenarios: ids,
          image,
          chaos_delay_seconds: overrideOptions.chaos_delay_seconds ?? Number(chaosDelaySeconds),
          outage_seconds: overrideOptions.outage_seconds ?? Number(outageSeconds),
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || res.statusText)
      }
      loadScenarios()
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  const stopRun = async () => {
    await fetch('/api/scenarios/stop', { method: 'POST' })
    loadScenarios()
  }

  const running = Boolean(data?.running)
  const scenarios = data?.scenarios ?? []

  // Stats calculation
  const totals = { pass: 0, warn: 0, fail: 0, error: 0, never: 0 }
  for (const s of scenarios) {
    const status = s.last?.status
    if (!status) totals.never += 1
    else if (status in totals) totals[status] += 1
  }

  // Active running scenario object
  const activeRunningScenario = scenarios.find((s) => s.id === data?.running)
  const latestStep = activeRunningScenario?.last?.steps?.[activeRunningScenario?.last?.steps?.length - 1]?.text

  const [runningElapsed, setRunningElapsed] = useState(0)
  useEffect(() => {
    if (!running) {
      setRunningElapsed(0)
      return
    }
    const start = Date.now()
    const timer = setInterval(() => {
      setRunningElapsed((Date.now() - start) / 1000)
    }, 200)
    return () => clearInterval(timer)
  }, [running])

  // Filtered scenarios
  const filteredScenarios = useMemo(() => {
    return scenarios.filter((s) => {
      // Category match
      const cat = SCENARIO_CATEGORIES[s.id] || 'other'
      if (activeCategory !== 'all' && cat !== activeCategory) {
        return false
      }
      // Search query match
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase()
        const matchTitle = s.title?.toLowerCase().includes(q)
        const matchSim = s.simulates?.toLowerCase().includes(q)
        const matchExpect = s.expect?.toLowerCase().includes(q)
        const matchId = s.id?.toLowerCase().includes(q)
        if (!matchTitle && !matchSim && !matchExpect && !matchId) {
          return false
        }
      }
      return true
    })
  }, [scenarios, activeCategory, searchQuery])

  return (
    <div className="space-y-6">
      {/* 1. RUNNING STATE HERO BANNER (STICKY & FROZEN ON SCROLL) */}
      {running && (
        <div className="sticky top-16 z-40 transition-all duration-300 animate-in fade-in slide-in-from-top-2">
          <Card className="border-border/80 bg-card/95 backdrop-blur-xl border-t-2 border-t-primary shadow-2xl overflow-hidden relative ring-1 ring-primary/25">
            <div className="absolute inset-0 bg-primary/5 animate-pulse pointer-events-none" />
            <CardContent className="p-3.5 sm:p-4 flex items-center justify-between gap-4 flex-wrap relative z-10">
              <div className="flex items-center gap-3 min-w-0">
                <div className="h-9 w-9 rounded-xl bg-primary/10 border border-primary/25 flex items-center justify-center text-primary shrink-0 animate-spin">
                  <RefreshCw size={18} />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="processing" className="text-[10px] animate-pulse font-mono tracking-wide">
                      EKSEKUSI BERJALAN
                    </Badge>
                    <span className="font-mono text-xs font-bold text-foreground truncate max-w-xs sm:max-w-md">
                      {data.running}
                    </span>
                    {runningElapsed > 0 && (
                      <Badge variant="outline" className="text-[10px] font-mono text-muted-foreground border-border/70">
                        {runningElapsed.toFixed(1)}s
                      </Badge>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 truncate">
                    {latestStep ? (
                      <span className="text-foreground/90 font-mono text-[11px]">
                        Langkah aktif: {latestStep}
                      </span>
                    ) : (
                      activeRunningScenario?.title || 'Menjalankan skenario pengujian ketahanan…'
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  onClick={stopRun}
                  className="gap-1.5 text-xs font-semibold shadow-xs"
                >
                  <Square size={13} />
                  <span>Hentikan Skenario</span>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 2. TOPOLOGY ARCHITECTURE MAP */}
      <TopologyMap
        chaos={chaos}
        onDisrupt={handleDisrupt}
        onStart={handleStartContainer}
        activeDisruptionService={activeRunningScenario ? 'structuring' : null}
      />


      {/* 3. MAIN WORKSPACE (2 Columns: Controls + Scenarios Suite) */}
      <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
        {/* LEFT COLUMN: Document Selection & Manual Disruptions */}
        <div className="space-y-5">
          {/* Document File Manager */}
          <Card className="border-border bg-card border-t-2 border-t-sky-500 shadow-xs">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <div className="h-6 w-6 rounded-md bg-sky-500/10 border border-sky-500/25 flex items-center justify-center text-sky-600 dark:text-sky-400">
                  <SlidersHorizontal size={13} />
                </div>
                <span>Dokumen Uji &amp; Peluncur</span>
              </CardTitle>
              <CardDescription className="text-xs">
                Pilih atau unggah gambar NPWP yang akan digunakan dalam simulasi
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <ScenarioFileManager
                images={data?.images ?? []}
                imageFiles={data?.image_files ?? []}
                selectedImage={image}
                onSelectImage={setImage}
                onImagesChanged={loadScenarios}
              />

              <div className="pt-2 border-t border-border/50">
                <Button
                  type="button"
                  variant="default"
                  disabled={running || !data?.available || !image}
                  onClick={() => runScenarios(null)}
                  className="w-full text-xs font-semibold gap-1.5"
                >
                  <Play size={13} />
                  <span>Jalankan Seluruh Skenario ({scenarios.length})</span>
                </Button>
              </div>

              {error && (
                <Alert variant="destructive" className="py-2 text-xs">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Parameter Gangguan (Chaos Tuning) Card */}
          <Card className="border-border bg-card border-t-2 border-t-amber-500 shadow-xs">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-6 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-500">
                    <SlidersHorizontal size={13} />
                  </div>
                  <span>Parameter Gangguan</span>
                </div>
                <Badge variant="outline" className="text-[10px] font-mono text-amber-500 border-amber-500/30">
                  Tuning
                </Badge>
              </CardTitle>
              <CardDescription className="text-xs">
                Sesuaikan durasi mati service dan detik injeksi agar representatif saat demonstrasi
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Parameter 1: Detik Injeksi Gangguan */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label className="font-semibold text-foreground flex items-center gap-1.5">
                    <Clock size={12} className="text-sky-500" />
                    <span>Detik Injeksi Gangguan:</span>
                  </label>
                  <span className="font-mono font-bold text-sky-500 bg-sky-500/10 px-1.5 py-0.5 rounded text-[11px]">
                    t+{chaosDelaySeconds}s
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-snug">
                  Waktu jeda sejak request dikirim sebelum pod/DB dimatikan.
                </p>
                <div className="flex items-center gap-1.5 pt-1 flex-wrap">
                  {[2, 3, 5, 8, 10].map((sec) => (
                    <Button
                      key={sec}
                      type="button"
                      variant={chaosDelaySeconds === sec ? 'default' : 'outline'}
                      size="xs"
                      onClick={() => setChaosDelaySeconds(sec)}
                      className={`h-6 text-[11px] font-mono px-2 ${
                        chaosDelaySeconds === sec ? 'font-bold shadow-xs' : ''
                      }`}
                    >
                      {sec}s{sec === 5 ? ' (Default)' : ''}
                    </Button>
                  ))}
                  <input
                    type="number"
                    min="1"
                    max="60"
                    value={chaosDelaySeconds}
                    onChange={(e) => setChaosDelaySeconds(Math.max(1, Number(e.target.value)))}
                    className="h-6 w-14 rounded border border-border bg-background px-1.5 text-[11px] font-mono text-center focus:ring-1 focus:ring-primary"
                    title="Input detik kustom"
                  />
                </div>
              </div>

              {/* Parameter 2: Durasi Mati Service (Outage) */}
              <div className="space-y-1.5 pt-2 border-t border-border/50">
                <div className="flex items-center justify-between text-xs">
                  <label className="font-semibold text-foreground flex items-center gap-1.5">
                    <Flame size={12} className="text-amber-500" />
                    <span>Durasi Mati Pod / DB:</span>
                  </label>
                  <span className="font-mono font-bold text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded text-[11px]">
                    {outageSeconds}s
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-snug">
                  Berapa lama pod atau database tetap mati sebelum dinyalakan lagi.
                </p>
                <div className="flex items-center gap-1.5 pt-1 flex-wrap">
                  {[5, 10, 15, 20, 30].map((sec) => (
                    <Button
                      key={sec}
                      type="button"
                      variant={outageSeconds === sec ? 'default' : 'outline'}
                      size="xs"
                      onClick={() => setOutageSeconds(sec)}
                      className={`h-6 text-[11px] font-mono px-2 ${
                        outageSeconds === sec ? 'font-bold shadow-xs' : ''
                      }`}
                    >
                      {sec}s{sec === 10 ? ' (Ideal)' : ''}
                    </Button>
                  ))}
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={outageSeconds}
                    onChange={(e) => setOutageSeconds(Math.max(1, Number(e.target.value)))}
                    className="h-6 w-14 rounded border border-border bg-background px-1.5 text-[11px] font-mono text-center focus:ring-1 focus:ring-primary"
                    title="Input detik kustom"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Manual Chaos Disruption Panel */}
          <ManualDisruptionForm chaos={chaos} onDisrupt={handleDisrupt} />
        </div>

        {/* RIGHT COLUMN: Scenarios Suite & Reports */}
        <div className="space-y-4">
          {/* Top Summary Banner */}
          <Card className="border-border bg-card border-t-2 border-t-violet-500 shadow-xs p-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-md bg-violet-500/10 border border-violet-500/25 flex items-center justify-center text-violet-600 dark:text-violet-400 shrink-0">
                  <ShieldAlert size={16} />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-foreground">
                    Uji Kesiapan &amp; Ketahanan Sistem (Readiness Test)
                  </h2>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Evaluasi otomatis: Sigterm drain, Sigkill OOM, DB timeout, dan transactional outbox retry
                  </p>
                </div>
              </div>

              <div className="flex gap-1.5 flex-wrap">
                <Badge variant="done" className="text-[10px] font-mono">
                  {totals.pass} PASS
                </Badge>
                <Badge variant="rejected" className="text-[10px] font-mono">
                  {totals.warn} WARN
                </Badge>
                <Badge variant="failed" className="text-[10px] font-mono">
                  {totals.fail} FAIL
                </Badge>
                {totals.never > 0 && (
                  <Badge variant="idle" className="text-[10px] font-mono">
                    {totals.never} BELUM
                  </Badge>
                )}
              </div>
            </div>
          </Card>

          {/* Filtering Bar: Category Tabs & Search Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              {/* Category Pills */}
              <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none">
                {CATEGORY_TABS.map((tab) => (
                  <Button
                    key={tab.id}
                    type="button"
                    variant={activeCategory === tab.id ? 'secondary' : 'ghost'}
                    size="xs"
                    onClick={() => setActiveCategory(tab.id)}
                    className={`h-7 px-2.5 text-xs font-medium rounded-lg transition-all ${
                      activeCategory === tab.id
                        ? 'bg-secondary text-foreground shadow-xs font-semibold'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <span>{tab.label}</span>
                  </Button>
                ))}
              </div>

              {/* Search Bar */}
              <div className="relative w-full sm:w-56">
                <Input
                  type="text"
                  placeholder="Cari skenario…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-8 pl-8 text-xs bg-background/80"
                />
                <Search size={13} className="absolute left-2.5 top-2.5 text-muted-foreground" />
              </div>
            </div>
          </div>

          {/* Scenarios Cards List */}
          <div className="space-y-3">
            {filteredScenarios.map((s) => {
              const last = s.last
              const status = last?.status || 'never'
              const expanded = open[s.id] ?? ['running', 'fail', 'error', 'warn'].includes(status)
              const isRunningThis = data?.running === s.id

              return (
                <Card
                  key={s.id}
                  className={`border bg-card shadow-xs transition-all duration-200 ${
                    isRunningThis
                      ? 'border-t-2 border-t-primary ring-1 ring-primary/40 bg-primary/5'
                      : status === 'pass'
                      ? 'border-t-2 border-t-emerald-500'
                      : status === 'warn'
                      ? 'border-t-2 border-t-amber-500'
                      : status === 'fail'
                      ? 'border-t-2 border-t-rose-500'
                      : 'border-t-2 border-t-zinc-400 dark:border-t-zinc-600'
                  }`}
                >
                  <CardContent className="p-4 space-y-2.5">
                    <div className="flex items-start justify-between gap-3 flex-wrap sm:flex-nowrap">
                      <div className="space-y-1.5 flex-1 min-w-0">
                        {/* Status + Title + Meta badges */}
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge variant={badgeVariantOf(status)} className="text-[10px] font-mono px-1.5 py-0">
                            {isRunningThis ? 'RUNNING' : LEVEL_LABEL[status] ?? status.toUpperCase()}
                          </Badge>
                          <span className="font-semibold text-sm text-foreground">
                            {s.title}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground">
                            ({s.id})
                          </span>
                          <ScenarioExplanationPopover scenarioId={s.id} />
                          {s.needs_lease && (
                            <Badge variant="outline" className="text-[10px] font-mono text-amber-500 border-amber-500/30">
                              butuh lease pendek
                            </Badge>
                          )}
                        </div>

                        {/* Simulation description */}
                        <div className="text-xs text-muted-foreground leading-relaxed flex items-start gap-1.5">
                          <span className="font-semibold text-foreground shrink-0 text-[11px] uppercase tracking-wide">
                            Simulasi:
                          </span>
                          <span>{s.simulates}</span>
                        </div>

                        {/* Architecture expectation */}
                        <div className="text-[11px] text-muted-foreground leading-relaxed flex items-start gap-1.5">
                          <span className="font-semibold text-emerald-600 dark:text-emerald-400 shrink-0 uppercase tracking-wide">
                            Ekspektasi:
                          </span>
                          <span className="font-medium text-foreground">{s.expect}</span>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2 shrink-0">
                        {last && (
                          <>
                            <Button
                              type="button"
                              variant="secondary"
                              size="xs"
                              onClick={() => setSelectedReplay(last)}
                              className="h-7 text-xs gap-1.5 font-semibold bg-primary/10 text-primary border border-primary/25 hover:bg-primary/20 shadow-xs"
                              title="Putar kanvas animasi replay skenario ini"
                            >
                              <Film size={12} />
                              <span>Replay</span>
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="xs"
                              onClick={() => setOpen({ ...open, [s.id]: !expanded })}
                              className="h-7 text-xs"
                            >
                              {expanded ? 'Tutup' : 'Lihat Hasil'}
                            </Button>
                          </>
                        )}
                        <Button
                          type="button"
                          variant="default"
                          size="xs"
                          disabled={running || !data?.available || !image}
                          onClick={() => runScenarios([s.id])}
                          className="h-7 text-xs gap-1"
                        >
                          <Play size={11} />
                          <span>Jalankan</span>
                        </Button>
                      </div>
                    </div>

                    {/* Detailed Report */}
                    {last && expanded && (
                      <RunReport
                        run={last}
                        openRequest={openRequest}
                        onOpenReplay={setSelectedReplay}
                      />
                    )}
                  </CardContent>
                </Card>
              )
            })}

            {filteredScenarios.length === 0 && (
              <div className="text-center py-12 text-xs text-muted-foreground">
                Tidak ada skenario yang sesuai dengan pencarian atau kategori ini.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Interactive Animated Replay Canvas Modal */}
      {selectedReplay && (
        <ScenarioReplayModal
          run={selectedReplay}
          onClose={() => setSelectedReplay(null)}
          openRequest={openRequest}
        />
      )}
    </div>
  )
}

