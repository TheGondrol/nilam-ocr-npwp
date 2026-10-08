import { useState, useEffect, useRef, useMemo } from 'react'
import {
  Send,
  Radio,
  Database,
  Clock,
  Search,
  XCircle,
  Copy,
  Check,
  RefreshCw,
  Zap,
  UploadCloud,
  FileText,
  AlertCircle,
  ShieldCheck,
  Cpu,
  CheckCheck,
  Globe,
  Trash2,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Alert, AlertDescription } from '@/components/ui/alert'
import SequenceToggles, {
  PIPELINE_NAMES,
  sequenceToSend,
  sequenceText,
} from '../components/SequenceToggles'
import PipelineStepper from '../components/PipelineStepper'
import OutboxTable from '../components/OutboxTable'
import TimelineView from '../components/TimelineView'
import JsonViewer from '../components/JsonViewer'
import OcrResultsCard from '../components/OcrResultsCard'

const STAGES = ['GUARDRAILS', 'OCR', 'STRUCTURING', 'SCORING']
const LABELS = {
  GUARDRAILS: 'Guardrails & Validasi',
  OCR: 'Extraction (PaddleOCR)',
  STRUCTURING: 'Structuring (NPWP Rules)',
  SCORING: 'Scoring & Confidence',
}

const STAGE_THEMES = {
  GUARDRAILS: {
    border: 'border-t-amber-500',
    icon: ShieldCheck,
    tagClass: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/25',
    tag: 'VALIDASI DOKUMEN',
  },
  OCR: {
    border: 'border-t-sky-500',
    icon: FileText,
    tagClass: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
    tag: 'OCR TEXT EXTRACTION',
  },
  STRUCTURING: {
    border: 'border-t-violet-500',
    icon: Cpu,
    tagClass: 'bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/25',
    tag: 'DATA STRUCTURING',
  },
  SCORING: {
    border: 'border-t-emerald-500',
    icon: CheckCheck,
    tagClass: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/25',
    tag: 'CONFIDENCE & MATCHING',
  },
}

const SERVICE_OF_STAGE = {
  GUARDRAILS: 'guardrails',
  OCR: 'extraction',
  STRUCTURING: 'structuring',
  SCORING: 'scoring',
}

const CALLBACK_MODES = [
  { value: 'ok', label: '200 OK (Normal)', hint: 'Orkestrasi pusat menerima callback seketika.' },
  { value: 'slow', label: '200 Slow (12s)', hint: 'Orkestrasi menjawab setelah timeout relay (10s): memicu duplikat.' },
  { value: 'flaky', label: '503 Flaky (2x)', hint: 'Dua kedatangan pertama 503, lalu 200: menguji exponential backoff.' },
  { value: 'not_ready', label: '409 Not Ready (2x)', hint: '409 RESULT_NOT_READY 2x lalu 200: orchestrator belum catat 202.' },
  { value: 'down', label: '503 Down', hint: 'Gagal terus-menerus: relay berhenti setelah MAX_ATTEMPTS (dead letter).' },
  { value: 'reject', label: '422 Reject', hint: 'Pusat menolak payload: relay langsung menandai dead letter tanpa retry.' },
  { value: 'unauthorized', label: '401 Unauthorized', hint: 'X-Callback-Key salah: ditolak permanen.' },
]

const SOURCES = [
  { value: 'upload', label: 'Multipart Upload (file biner)' },
  { value: 'file_url', label: 'URL Berbagi (file_url download)' },
]

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function fmtSec(sec) {
  if (sec == null) return ''
  return sec < 60 ? `${Math.round(sec)} dtk` : `${(sec / 60).toFixed(1)} mnt`
}

function badgeVariantOf(status) {
  switch (String(status)?.toUpperCase()) {
    case '200':
    case 'DONE': return 'done'
    case '202':
    case 'PROCESSING': return 'processing'
    case '400':
    case 'REJECTED': return 'rejected'
    case '422':
    case '500':
    case 'FAILED': return 'failed'
    default: return 'idle'
  }
}

function stageView(events, stage, sequence) {
  const service = SERVICE_OF_STAGE[stage]
  if (sequence && !sequence.includes(service)) {
    return { status: 'SKIPPED', callbacks: [], http: null }
  }
  const mine = events.filter((e) => e.stage === stage)
  const callbacks = events.filter((e) => e.type === 'callback' && e.stage === stage)
  const http = events.find((e) => e.type === 'http' && e.stage === stage)
  const last = mine[mine.length - 1]
  const rejected = mine.find((e) => e.status === 'REJECTED')
  const status = rejected ? 'REJECTED' : last ? last.status : 'PROCESSING'
  const started = mine.find((e) => e.status === 'PROCESSING')?.ts
  const finished = last && ['DONE', 'FAILED', 'REJECTED'].includes(last.status) ? last.ts : null
  let elapsed = last?.elapsed_ms
  if (elapsed == null && started && finished) elapsed = (finished - started) * 1000
  const withResult = [...mine].reverse().find((e) => e.result)
  const db = mine.find((e) => e.source === 'db' && ['DONE', 'FAILED', 'REJECTED'].includes(e.status))
  const acknowledged = mine.find((e) => e.source === 'callback' && ['DONE', 'FAILED'].includes(e.status))
  // Format callback result: callback tahap tengah ditulis ke outbox lalu dilewati relay (log: skipped).
  const callbackSkipped = events.some(
    (e) => e.type === 'outbox' && e.status === 'SKIPPED' && e.message?.owner === stage && e.message?.kind === 'callback',
  )
  return {
    callbackSkipped,
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

const OUTBOX_TERMINAL = ['DELIVERED', 'SKIPPED', 'DEAD']

function outboxView(events) {
  // Event dari tabel membawa id baris. Event dari log relay (DELIVERED / SKIPPED) tidak: ia menutup baris tabel
  // yang masih terbuka dengan pengirim, jenis, dan tujuan yang sama, atau jadi baris sendiri kalau barisnya
  // terhapus sebelum sempat terbaca dari tabel (jalur normal).
  const rows = []
  const ordered = [...events].sort((a, b) => a.ts - b.ts)
  for (const e of ordered) {
    if (e.type !== 'outbox' || !e.message) continue
    const m = e.message
    let row =
      m.id != null
        ? rows.find((r) => r.message.id === m.id)
        : rows.find(
            (r) =>
              !OUTBOX_TERMINAL.includes(r.state) &&
              r.message.owner === m.owner &&
              r.message.kind === m.kind &&
              r.message.target === m.target,
          )
    if (!row) {
      row = { history: [] }
      rows.push(row)
    }
    row.history.push({ status: e.status, ts: e.ts, message: m })
    row.state = e.status
    row.message = { ...row.message, ...m, id: row.message?.id ?? m.id }
    row.fromLog = row.fromLog || e.source === 'log'
  }
  return rows
}

function Summary({ stage, result }) {
  if (!result) return null
  if (stage === 'OCR') {
    return (
      <div className="mt-2.5 space-y-2">
        <div className="flex gap-1.5 flex-wrap">
          <Badge variant="done">{result.blocks?.length ?? 0} BARIS</Badge>
          <Badge variant="processing">ENGINE: {result.engine}</Badge>
          {result.model && <Badge variant="idle">MODEL: {result.model}</Badge>}
          {result.elapsed_ms && (
            <span className="font-mono text-[11px] text-muted-foreground self-center">
              {fmtMs(result.elapsed_ms)} inferensi
            </span>
          )}
        </div>
        <ul className="m-0 p-0 list-none max-h-48 overflow-y-auto space-y-1">
          {(result.blocks ?? []).map((b, i) => (
            <li key={i} className="flex items-center gap-2 text-xs bg-muted/40 p-1.5 rounded font-mono">
              <span className="text-foreground font-mono font-medium">{b.confidence.toFixed(2)}</span>
              <span className="text-foreground">{b.text}</span>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  if (stage === 'STRUCTURING') {
    return (
      <div className="mt-2.5 overflow-x-auto">
        <table className="w-full text-xs">
          <tbody className="divide-y divide-border/40">
            {Object.entries(result.fields ?? {}).map(([name, f]) => {
              const val = f?.value
              const conf = f?.confidence ?? 0
              return (
                <tr key={name}>
                  <td className="py-1 font-mono text-muted-foreground">{name}</td>
                  <td className={`py-1 font-medium ${val == null ? 'text-amber-400' : 'text-foreground'}`}>
                    {val ?? '(tidak ditemukan)'}
                  </td>
                  <td className="py-1 text-right font-mono text-foreground font-medium w-16">
                    {(conf * 100).toFixed(0)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }
  if (stage === 'SCORING') {
    const s = result.score ?? {}
    const finalScore = s.final_score ?? result.confidence ?? 0
    const nameConf = s.name_match_confidence ?? 0
    return (
      <div className="mt-2.5 space-y-2 bg-muted/30 p-2.5 rounded-lg border border-border/50">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">Confidence Akhir:</span>
            <span className="font-mono font-bold text-emerald-400">{(finalScore * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${finalScore * 100}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">Name Match Confidence:</span>
            <span className="font-mono font-medium text-foreground">{(nameConf * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-primary rounded-full" style={{ width: `${nameConf * 100}%` }} />
          </div>
        </div>
      </div>
    )
  }
  return null
}

export default function PipelineView({ overview, focus }) {
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
  const [releasing, setReleasing] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [sidebarTab, setSidebarTab] = useState('submit') // 'submit' | 'simulation' | 'backlog'
  const [searchFilter, setSearchFilter] = useState('')
  const [activeStageCard, setActiveStageCard] = useState(null)
  const [copiedId, setCopiedId] = useState(false)

  const fileRef = useRef()
  const sourceRef = useRef()

  const [ocrResultsData, setOcrResultsData] = useState(null)
  const [loadingOcrResults, setLoadingOcrResults] = useState(false)

  const loadRequests = () => fetch('/api/requests').then((r) => r.json()).then(setRequests)
  const loadSim = () => fetch('/api/simulation').then((r) => r.json()).then(setSim)

  const loadOcrResults = async (id) => {
    if (!id) {
      setOcrResultsData(null)
      return
    }
    setLoadingOcrResults(true)
    try {
      const res = await fetch(`/api/requests/${encodeURIComponent(id)}/ocr-results`)
      const d = await res.json()
      setOcrResultsData(d)
    } catch (err) {
      setOcrResultsData({
        available: false,
        error: String(err?.message || err),
        rows: [],
      })
    } finally {
      setLoadingOcrResults(false)
    }
  }

  const handleDeleteRequest = async (rid) => {
    try {
      const res = await fetch(`/api/requests/${encodeURIComponent(rid)}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        if (current === rid) {
          setCurrent(null)
          setEvents([])
          setOcrResultsData(null)
        }
        await loadRequests()
      }
    } catch (err) {
      console.error('Failed to delete request:', err)
    }
  }

  const handleClearAllRequests = async () => {
    if (!window.confirm(`Hapus semua (${requests.length}) riwayat request dari daftar?`)) {
      return
    }
    try {
      const res = await fetch('/api/requests', {
        method: 'DELETE',
      })
      if (res.ok) {
        setCurrent(null)
        setEvents([])
        setOcrResultsData(null)
        await loadRequests()
      }
    } catch (err) {
      console.error('Failed to clear requests:', err)
    }
  }

  useEffect(() => {
    loadRequests()
    loadSim()
  }, [])

  useEffect(() => {
    if (focus?.rid) setCurrent(focus.rid)
  }, [focus])

  // Load nilam_ocr_results whenever current changes
  useEffect(() => {
    if (current) {
      loadOcrResults(current)
    } else {
      setOcrResultsData(null)
    }
  }, [current])

  // Real-time SSE
  useEffect(() => {
    if (!current) return
    setEvents([])
    sourceRef.current?.close()
    const sse = new EventSource(`/api/requests/${current}/events`)
    sourceRef.current = sse
    setLive(true)
    sse.addEventListener('stage', (msg) => {
      const ev = JSON.parse(msg.data)
      setEvents((prev) => [...prev, ev])
      if (ev.type === 'ocr_results' || ev.type === 'pipeline') {
        loadOcrResults(current)
      }
    })
    sse.addEventListener('end', () => {
      sse.close()
      setLive(false)
      loadRequests()
      loadOcrResults(current)
    })
    sse.onerror = () => {
      setLive(false)
    }
    return () => sse.close()
  }, [current])

  // Submit request
  async function submit(e) {
    e.preventDefault()
    const file = selectedFile || fileRef.current?.files[0]
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

  // Polling requests saat submit
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
        const id = current
        setCurrent(null)
        setTimeout(() => setCurrent(id), 0)
      }
    } finally {
      setReleasing(false)
    }
  }

  const handleCopyId = (id) => {
    navigator.clipboard.writeText(id)
    setCopiedId(true)
    setTimeout(() => setCopiedId(false), 2000)
  }

  const clientEvent = events.find((e) => e.type === 'client')
  const httpEvent = events.find((e) => e.type === 'http')
  const sequence = httpEvent?.errors === 'INVALID_PIPELINE_SEQUENCE' ? null : clientEvent?.sequence ?? null
  const lastStage = [...STAGES].reverse().find((stage) => !sequence || sequence.includes(SERVICE_OF_STAGE[stage]))
  const t0 = clientEvent?.ts ?? events[0]?.ts ?? 0
  const outboxRows = useMemo(() => outboxView(events), [events])
  const unavailable = events.find((e) => e.type === 'outbox' && e.status === 'UNAVAILABLE')
  const ended = events.find((e) => e.type === 'pipeline')
  const total = ended ? (ended.ts - t0) * 1000 : events.length > 1 ? (events[events.length - 1].ts - t0) * 1000 : null

  // Pre-calculate stage views
  const stageViews = useMemo(() => {
    const map = {}
    for (const stage of STAGES) {
      map[stage] = stageView(events, stage, sequence)
    }
    return map
  }, [events, sequence])

  const filteredRequests = requests.filter((r) => {
    if (!searchFilter) return true
    return (
      r.request_id?.toLowerCase().includes(searchFilter.toLowerCase()) ||
      r.filename?.toLowerCase().includes(searchFilter.toLowerCase()) ||
      String(r.http_status).includes(searchFilter)
    )
  })

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
      {/* =========================================================================
          LEFT COLUMN: Sidebar Controls, Simulation & Request History
          ========================================================================= */}
      <div className="space-y-4">
        {/* Studio Sub-Navigation Tabs */}
        <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-lg border border-border/50">
          <Button
            type="button"
            variant={sidebarTab === 'submit' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setSidebarTab('submit')}
            className={`flex-1 text-xs gap-1.5 ${sidebarTab === 'submit' ? 'bg-background shadow-xs font-semibold text-foreground border border-border/50' : 'text-muted-foreground'}`}
          >
            <Send size={13} />
            <span>Kirim Dokumen</span>
          </Button>

          <Button
            type="button"
            variant={sidebarTab === 'simulation' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setSidebarTab('simulation')}
            className={`flex-1 text-xs gap-1.5 ${sidebarTab === 'simulation' ? 'bg-background shadow-xs font-semibold text-foreground border border-border/50' : 'text-muted-foreground'}`}
          >
            <Radio size={13} />
            <span>Simulasi</span>
          </Button>

          <Button
            type="button"
            variant={sidebarTab === 'backlog' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setSidebarTab('backlog')}
            className={`flex-1 text-xs gap-1.5 ${sidebarTab === 'backlog' ? 'bg-background shadow-xs font-semibold text-foreground border border-border/50' : 'text-muted-foreground'}`}
          >
            <Database size={13} />
            <span>Backlog</span>
          </Button>
        </div>

        {/* Tab 1: Submit Form */}
        {sidebarTab === 'submit' && (
          <Card className="border-border bg-card border-t-2 border-t-sky-500 shadow-xs">
            <CardHeader className="pb-3 flex flex-row items-center gap-2.5">
              <div className="h-7 w-7 rounded-md bg-sky-500/10 border border-sky-500/25 flex items-center justify-center text-sky-600 dark:text-sky-400">
                <Send size={14} />
              </div>
              <div>
                <CardTitle className="text-sm">Kirim Dokumen NPWP</CardTitle>
                <CardDescription className="text-xs">Uji ekstraksi live &amp; verifikasi outbox</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={submit} className="space-y-3.5">
                {/* File Upload Dropzone */}
                <div>
                  <div className="flex justify-between items-center text-xs mb-1.5">
                    <span className="font-medium text-foreground">File Foto NPWP</span>
                    <span className="text-[11px] text-muted-foreground">JPG, PNG, PDF</span>
                  </div>
                  <label
                    className="flex flex-col items-center justify-center p-4 border border-dashed border-border/80 rounded-lg bg-background/40 hover:bg-muted/30 cursor-pointer transition-colors"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault()
                      if (e.dataTransfer.files?.[0]) setSelectedFile(e.dataTransfer.files[0])
                    }}
                  >
                    <input
                      ref={fileRef}
                      type="file"
                      className="hidden"
                      accept="image/jpeg,image/png,application/pdf"
                      onChange={(e) => {
                        if (e.target.files?.[0]) setSelectedFile(e.target.files[0])
                      }}
                    />
                    <UploadCloud size={24} className="text-muted-foreground mb-1" />
                    <span className="text-xs font-medium text-foreground text-center">
                      {selectedFile ? 'Ganti File Pilihan' : 'Klik atau seret foto dokumen'}
                    </span>
                  </label>

                  {selectedFile && (
                    <div className="flex items-center gap-2 mt-2 p-2 rounded-md bg-muted/40 border border-border/60 text-xs">
                      <FileText size={14} className="text-muted-foreground shrink-0" />
                      <span className="flex-1 truncate font-mono text-[11px]">
                        {selectedFile.name} ({(selectedFile.size / 1024).toFixed(0)} KB)
                      </span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="xs"
                        className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
                        onClick={() => {
                          setSelectedFile(null)
                          if (fileRef.current) fileRef.current.value = ''
                        }}
                      >
                        <XCircle size={13} />
                      </Button>
                    </div>
                  )}
                </div>

                {/* Pipeline Sequence Selector */}
                <SequenceToggles names={sendNames} onChange={setSendNames} allowInvalid />

                {/* Delay Simulation */}
                <div className="p-3 rounded-lg border border-border/70 bg-background/50 space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer text-xs font-medium text-foreground">
                    <input
                      type="checkbox"
                      checked={slow}
                      onChange={(e) => setSlow(e.target.checked)}
                      className="rounded border-border text-primary focus:ring-ring"
                    />
                    <span>Simulasi Pipeline Lambat (HTTP 202)</span>
                  </label>
                  {slow && (
                    <div className="flex items-center gap-2 pt-1 text-xs">
                      <span className="text-muted-foreground text-[11px]">Tunda OCR:</span>
                      <Input
                        type="number"
                        min="1"
                        max="120"
                        value={slowSeconds}
                        className="w-16 h-7 text-xs px-2"
                        onChange={(e) => setSlowSeconds(Number(e.target.value))}
                      />
                      <span className="text-muted-foreground text-[11px]">
                        dtk (wait: {sim.wait_seconds}s)
                      </span>
                    </div>
                  )}
                </div>

                {/* Advanced Options */}
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-foreground">Metode Pengiriman</label>
                  <select
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    className="flex h-9 w-full rounded-md border border-input bg-background/80 px-3 py-1.5 text-xs text-foreground shadow-sm transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    disabled={Boolean(sim.target) && sim.target !== 'local'}
                  >
                    {SOURCES.map((s) => (
                      <option key={s.value} value={s.value}>{s.label}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-foreground">Uji Idempotensi / Replay ID</span>
                    {current && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="xs"
                        className="h-5 px-1.5 text-[10px] text-muted-foreground hover:text-foreground"
                        onClick={() => setResendId(current)}
                      >
                        Pakai ID Aktif
                      </Button>
                    )}
                  </div>
                  <Input
                    value={resendId}
                    placeholder="REQ_... (kosong = buat baru)"
                    className="font-mono text-xs"
                    onChange={(e) => setResendId(e.target.value)}
                  />
                </div>

                <Button
                  type="submit"
                  variant="default"
                  disabled={busy || !selectedFile || sendNames.length === 0}
                  className="w-full text-xs font-semibold gap-1.5"
                >
                  {busy ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" />
                      <span>Memproses Pipeline…</span>
                    </>
                  ) : resendId.trim() ? (
                    <>
                      <Zap size={14} />
                      <span>Kirim Ulang Request ID</span>
                    </>
                  ) : (
                    <>
                      <Send size={14} />
                      <span>Kirim Dokumen ke Pipeline</span>
                    </>
                  )}
                </Button>
              </form>

              {error && (
                <Alert variant="destructive" className="mt-3 py-2 text-xs">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-xs">{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        )}

        {/* Tab 2: Simulation Mode Selector */}
        {sidebarTab === 'simulation' && (
          <Card className="border-border bg-card border-t-2 border-t-violet-500 shadow-xs">
            <CardHeader className="pb-3 flex flex-row items-center gap-2.5">
              <div className="h-7 w-7 rounded-md bg-violet-500/10 border border-violet-500/25 flex items-center justify-center text-violet-600 dark:text-violet-400">
                <Radio size={14} />
              </div>
              <div>
                <CardTitle className="text-sm">Simulasi Resiliensi Callback</CardTitle>
                <CardDescription className="text-xs">Uji cara tracker merespons callback relay outbox</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              {CALLBACK_MODES.map((m) => (
                <div
                  key={m.value}
                  onClick={() => setCallbackMode(m.value)}
                  className={`flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-all ${
                    sim.callback === m.value
                      ? 'bg-accent border-border text-foreground'
                      : 'bg-background/40 border-border/60 hover:bg-muted/30'
                  }`}
                >
                  <input
                    type="radio"
                    name="cb-mode"
                    checked={sim.callback === m.value}
                    onChange={() => setCallbackMode(m.value)}
                    className="mt-0.5 text-primary"
                  />
                  <div>
                    <div className="text-xs font-semibold text-foreground">{m.label}</div>
                    <div className="text-[11px] text-muted-foreground">{m.hint}</div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Tab 3: Service Outbox Backlog */}
        {sidebarTab === 'backlog' && (
          <Card className="border-border bg-card border-t-2 border-t-amber-500 shadow-xs">
            <CardHeader className="pb-3 flex flex-row items-center gap-2.5">
              <div className="h-7 w-7 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-600 dark:text-amber-400">
                <Database size={14} />
              </div>
              <div>
                <CardTitle className="text-sm">Backlog Outbox Tiap Service</CardTitle>
                <CardDescription className="text-xs">Monitor status relay antrean</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              {['OCR', 'STRUCTURING', 'SCORING'].map((stage) => {
                const s = overview?.[stage] ?? {}
                const hasDead = (s.dead_letters ?? 0) > 0
                return (
                  <div
                    key={stage}
                    className={`flex items-center justify-between p-3 rounded-lg border ${
                      hasDead ? 'border-rose-500/40 bg-rose-500/10' : 'border-border/60 bg-background/50'
                    }`}
                  >
                    <div>
                      <div className="text-xs font-semibold text-foreground">{stage}</div>
                      <div className="text-[11px] text-muted-foreground font-mono">
                        {s.pending ?? 0} pending · {s.retrying ?? 0} retry
                        {s.oldest_pending_seconds != null && ` · tertua ${fmtSec(s.oldest_pending_seconds)}`}
                      </div>
                    </div>
                    <div>
                      {hasDead ? (
                        <Badge variant="failed" className="text-[10px]">{s.dead_letters} DEAD</Badge>
                      ) : (
                        <Badge variant="done" className="text-[10px]">AMAN</Badge>
                      )}
                    </div>
                  </div>
                )
              })}
            </CardContent>
          </Card>
        )}

        {/* Recent Requests Section */}
        <Card className="border-border bg-card border-t-2 border-t-zinc-400 dark:border-t-zinc-600 shadow-xs">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded-md bg-muted border border-border flex items-center justify-center text-muted-foreground">
                <Clock size={13} />
              </div>
              <CardTitle className="text-xs font-semibold">
                Riwayat Request ({requests.length})
              </CardTitle>
            </div>
            {requests.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                size="xs"
                onClick={handleClearAllRequests}
                className="h-6 px-1.5 text-[10px] text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 gap-1 transition-colors"
                title="Hapus semua riwayat request"
              >
                <Trash2 size={11} />
                <span>Hapus Semua</span>
              </Button>
            )}
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="relative">
              <Input
                type="text"
                placeholder="Cari ID / status / file…"
                className="pl-8 text-xs h-8"
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
              />
              <Search size={13} className="absolute left-2.5 top-2.5 text-muted-foreground" />
            </div>

            <div className="space-y-1.5 max-h-[360px] overflow-y-auto pr-1">
              {filteredRequests.map((r) => {
                const active = r.request_id === current
                const httpVariant = badgeVariantOf(r.http_status)
                return (
                  <div
                    key={r.request_id}
                    onClick={() => setCurrent(r.request_id)}
                    className={`group p-2.5 rounded-lg border transition-all cursor-pointer text-left space-y-1 relative ${
                      active
                        ? 'bg-accent border-border text-foreground font-medium'
                        : 'bg-background/40 border-border/50 hover:bg-muted/40 hover:border-border'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="font-mono text-xs font-bold text-foreground truncate">
                        {r.request_id}
                      </span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {r.http_status ? (
                          <Badge variant={httpVariant} className="text-[10px] px-1.5 py-0">
                            HTTP {r.http_status}
                          </Badge>
                        ) : (
                          <Badge variant="processing" className="text-[10px] px-1.5 py-0">
                            JALAN
                          </Badge>
                        )}
                        <Button
                          type="button"
                          variant="ghost"
                          size="xs"
                          className="h-5 w-5 p-0 text-muted-foreground/50 hover:text-rose-500 hover:bg-rose-500/15 transition-all opacity-60 group-hover:opacity-100"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteRequest(r.request_id)
                          }}
                          title={`Hapus ${r.request_id} dari riwayat`}
                        >
                          <Trash2 size={11} />
                        </Button>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground font-mono truncate">
                      {r.filename && <span className="truncate">{r.filename}</span>}
                      {r.elapsed_ms != null && <span>· {fmtMs(r.elapsed_ms)}</span>}
                      {r.sequence && <span>· {sequenceText(r.sequence)}</span>}
                      {r.slow && <span className="text-amber-400">· {r.slow}s</span>}
                    </div>
                  </div>
                )
              })}
              {filteredRequests.length === 0 && (
                <div className="text-center py-6 text-xs text-muted-foreground">
                  Tidak ada request yang sesuai.
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* RIGHT COLUMN: Live Request Inspector & Stage Progress */}
      <div className="space-y-4">
        {!current && (
          <Card className="border-border bg-card p-12 text-center">
            <div className="max-w-md mx-auto space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border bg-muted text-muted-foreground mx-auto">
                <FileText size={18} />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-foreground">
                  Pilih atau Kirim Foto NPWP
                </h3>
                <p className="text-xs text-muted-foreground leading-normal">
                  Pilih salah satu riwayat request di panel samping atau unggah dokumen untuk memantau eksekusi pipeline dan transactional outbox.
                </p>
              </div>
              {requests.length > 0 && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrent(requests[0].request_id)}
                  className="text-xs"
                >
                  Lihat Request Terakhir ({requests[0].request_id})
                </Button>
              )}
            </div>
          </Card>
        )}

        {current && (
          <>
            {/* Top Pipeline Stepper */}
            <PipelineStepper
              stages={STAGES}
              stageViews={stageViews}
              activeStage={activeStageCard}
              onSelectStage={(stage) => {
                setActiveStageCard(stage)
                const el = document.getElementById(`stage-card-${stage}`)
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
              }}
            />

            {/* Active Request Header Card */}
            <Card className="border-border bg-card p-3.5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-bold text-foreground">
                    {current}
                  </span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                    onClick={() => handleCopyId(current)}
                    title="Salin Request ID"
                  >
                    {copiedId ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                  </Button>
                  {clientEvent && (
                    <span className="text-xs text-muted-foreground font-mono">
                      · {clientEvent.filename} ({(clientEvent.size / 1024).toFixed(0)} KB)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant={live ? 'processing' : 'done'} className="text-[10px] font-mono">
                    {live ? 'Live' : 'Selesai'}
                  </Badge>
                  {total != null && (
                    <Badge variant="outline" className="text-[10px] font-mono">
                      Total {fmtMs(total)}
                    </Badge>
                  )}
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    className="h-6 px-2 text-xs text-rose-500 hover:text-rose-600 hover:bg-rose-500/10 gap-1 transition-colors"
                    onClick={() => handleDeleteRequest(current)}
                    title="Hapus request ini dari riwayat"
                  >
                    <Trash2 size={12} />
                    <span className="hidden sm:inline">Hapus</span>
                  </Button>
                </div>
              </div>
            </Card>

            {/* Orchestrator HTTP Response Card */}
            {httpEvent && (
              <Card className="border-border bg-card border-t-2 border-t-blue-500 shadow-xs">
                <CardContent className="p-4 space-y-2.5">
                  <div className="flex items-center justify-between gap-2 flex-wrap pb-2 border-b border-border/50">
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="h-7 w-7 rounded-md bg-blue-500/10 border border-blue-500/25 flex items-center justify-center text-blue-600 dark:text-blue-400">
                        <Globe size={15} />
                      </div>
                      <Badge variant={badgeVariantOf(httpEvent.http_status)} className="text-[11px] font-mono">
                        HTTP {httpEvent.http_status}
                      </Badge>
                      <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-blue-600 dark:text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
                        API Gateway Response
                      </span>
                      <span className="text-xs font-semibold text-foreground">
                        job_status: {httpEvent.job_status ?? '-'}
                      </span>
                      {httpEvent.errors && (
                        <span className="text-xs font-mono text-rose-400">
                          [{httpEvent.errors}]
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] font-mono text-muted-foreground">
                      Dijawab setelah {fmtMs(httpEvent.elapsed_ms)} (wait {httpEvent.wait_seconds}s)
                    </div>
                  </div>

                  <div className="text-xs text-muted-foreground leading-relaxed">
                    {httpEvent.http_status === 200 && (
                      <span>
                        <b>200 Completed:</b> Pipeline tuntas sebelum batas tunggu habis ({httpEvent.wait_seconds}s). Hasil ekstraksi langsung dikembalikan pada respons HTTP ini.
                      </span>
                    )}
                    {httpEvent.http_status === 202 && (
                      <span>
                        <b>202 Accepted:</b> Waktu tunggu orchestrator habis sebelum pipeline tuntas. Hasil menyusul melalui callback outbox relay dan dapat dicek via GET status.
                      </span>
                    )}
                    {httpEvent.http_status === 400 && (
                      <span>
                        <b>400 Rejected:</b> Dokumen ditolak oleh model guardrails atau aturan penolakan ML structuring.
                      </span>
                    )}
                    {httpEvent.http_status === 422 && (
                      <span>
                        <b>422 Failed:</b> Eksekusi pipeline gagal atau urutan sequence tidak valid ({httpEvent.errors}).
                      </span>
                    )}
                  </div>

                  <div className="text-xs leading-relaxed rounded-md border border-border/60 bg-muted/30 p-2.5">
                    <span className="font-mono font-semibold text-foreground">
                      pipeline_last_stage: {httpEvent.pipeline_last_stage ?? 'null'}
                    </span>
                    <span className="text-muted-foreground">
                      {' · '}
                      {httpEvent.pipeline_last_stage
                        ? `error atau penolakan ini berasal dari service ${httpEvent.pipeline_last_stage}.`
                        : 'null karena tidak ada error.'}{' '}
                      Sesuai kontrak extract-ocr, field ini hanya diisi saat ada error atau penolakan, berisi nama
                      service asalnya. Pada 200 dan 202 nilainya selalu null; service yang dijalankan sudah
                      ditentukan pusat lewat <code>pipeline_name_sequence</code>.
                    </span>
                  </div>

                  {httpEvent.body && (
                    <div className="pt-1">
                      <JsonViewer value={httpEvent.body} label="Respons JSON extract-ocr" />
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Database Verification: nilam_ocr_results */}
            <OcrResultsCard
              requestId={current}
              data={ocrResultsData}
              loading={loadingOcrResults}
              onRefresh={() => loadOcrResults(current)}
            />

            {/* 4 Stages Detailed Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {STAGES.map((stage, i) => {
                const view = stageViews[stage]
                const isSkipped = view.status === 'SKIPPED'
                const rel = (ts) => `t+${fmtMs((ts - t0) * 1000)}`
                const rejectedCallbacks = view.callbacks.filter((c) => !c.accepted)
                const lastCb = view.callbacks[view.callbacks.length - 1]
                const isLast = Boolean(sequence) && !sequence.includes('scoring') && stage === lastStage
                const theme = STAGE_THEMES[stage] || {
                  border: 'border-t-primary',
                  icon: FileText,
                  tagClass: 'bg-muted text-muted-foreground border-border',
                  tag: stage,
                }
                const StageIcon = theme.icon

                return (
                  <Card
                    key={stage}
                    id={`stage-card-${stage}`}
                    className={`border-border bg-card border-t-2 ${theme.border} shadow-xs flex flex-col ${isSkipped ? 'opacity-60' : ''}`}
                  >
                    <CardHeader className="p-4 pb-3 flex flex-row items-center justify-between border-b border-border/40">
                      <div className="flex items-center gap-2.5">
                        <div className={`h-7 w-7 rounded-md border flex items-center justify-center shrink-0 ${theme.tagClass}`}>
                          <StageIcon size={14} />
                        </div>
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-[10px] text-muted-foreground font-semibold">0{i + 1}</span>
                            <CardTitle className="text-xs font-semibold">{LABELS[stage]}</CardTitle>
                          </div>
                          <span className="text-[9px] font-mono uppercase tracking-wider text-muted-foreground">
                            {theme.tag} {isLast ? '· (FINAL)' : ''}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <Badge variant={badgeVariantOf(view.status)} className="text-[10px]">
                          {view.status}
                        </Badge>
                        {view.elapsed != null && (
                          <span className="font-mono text-[11px] text-muted-foreground">
                            {fmtMs(view.elapsed)}
                          </span>
                        )}
                      </div>
                    </CardHeader>

                    <CardContent className="p-4 space-y-2.5 flex-1 flex flex-col">
                      {isSkipped ? (
                        <div className="text-xs text-muted-foreground">
                          Tahap ini dilewati karena tidak dimasukkan dalam <code>pipeline_name_sequence</code>.
                        </div>
                      ) : (
                        <>
                          {stage !== 'GUARDRAILS' && (
                            <div className="space-y-1 text-xs bg-background/50 p-2.5 rounded-lg border border-border/50">
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Hasil di DB:</span>
                                <span className="font-mono font-medium text-foreground">
                                  {view.db ? `${view.db.status} (${rel(view.db.ts)})` : 'Belum tercatat'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Orkestrasi Tahu:</span>
                                <span className="font-mono font-medium text-foreground">
                                  {view.acknowledged
                                    ? `Callback diterima (${rel(view.acknowledged.ts)})${lastCb ? ` (attempt #${lastCb.attempt})` : ''}`
                                    : view.callbackSkipped
                                    ? 'Tidak perlu: hanya tahap terakhir yang mengirim callback'
                                    : rejectedCallbacks.length
                                    ? `${rejectedCallbacks.length}x ditolak, retry`
                                    : 'Menunggu relay outbox'}
                                </span>
                              </div>
                            </div>
                          )}

                          {view.error && (
                            <Alert variant="destructive" className="py-2 text-xs">
                              <AlertDescription>{view.error}</AlertDescription>
                            </Alert>
                          )}

                          <Summary stage={stage} result={view.result} />

                          {view.result && (
                            <div className="mt-auto pt-2">
                              <JsonViewer value={view.result} label={`Data Mentah ${stage}`} />
                            </div>
                          )}
                        </>
                      )}
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Transactional Outbox Ledger */}
            <OutboxTable
              rows={outboxRows}
              t0={t0}
              onRelease={release}
              releasing={releasing}
              unavailable={unavailable}
            />

            {/* Event Timeline & Contract Verification */}
            <TimelineView events={events} t0={t0} />
          </>
        )}
      </div>
    </div>
  )
}
