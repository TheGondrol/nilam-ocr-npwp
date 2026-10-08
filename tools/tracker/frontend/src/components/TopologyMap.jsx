import { useState, useEffect } from 'react'
import {
  Server,
  Database,
  Flame,
  Power,
  Zap,
  ArrowRight,
  ShieldAlert,
  Cpu,
  FileText,
  ShieldCheck,
  CheckCheck,
  Activity,
  Layers,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

const SERVICE_CONFIGS = {
  orchestrator: {
    label: 'Orchestrator',
    role: 'API Gateway & Ingress',
    port: '8034',
    icon: Server,
    color: 'blue',
    accentBorder: 'border-blue-500/30 dark:border-blue-500/40',
    accentText: 'text-blue-600 dark:text-blue-400',
    accentBg: 'bg-blue-500/10',
  },
  guardrails: {
    label: 'Guardrails',
    role: 'Image Quality & Cropping',
    port: '8031',
    icon: ShieldCheck,
    color: 'amber',
    accentBorder: 'border-amber-500/30 dark:border-amber-500/40',
    accentText: 'text-amber-600 dark:text-amber-400',
    accentBg: 'bg-amber-500/10',
  },
  extraction: {
    label: 'Extraction',
    role: 'PaddleOCR & Outbox Relay',
    port: '8030',
    icon: FileText,
    color: 'sky',
    accentBorder: 'border-sky-500/30 dark:border-sky-500/40',
    accentText: 'text-sky-600 dark:text-sky-400',
    accentBg: 'bg-sky-500/10',
  },
  structuring: {
    label: 'Structuring',
    role: 'NPWP Parser & Outbox Relay',
    port: '8032',
    icon: Cpu,
    color: 'violet',
    accentBorder: 'border-violet-500/30 dark:border-violet-500/40',
    accentText: 'text-violet-600 dark:text-violet-400',
    accentBg: 'bg-violet-500/10',
  },
  scoring: {
    label: 'Scoring',
    role: 'Confidence & Outbox Relay',
    port: '8033',
    icon: CheckCheck,
    color: 'emerald',
    accentBorder: 'border-emerald-500/30 dark:border-emerald-500/40',
    accentText: 'text-emerald-600 dark:text-emerald-400',
    accentBg: 'bg-emerald-500/10',
  },
  postgres: {
    label: 'PostgreSQL DB',
    role: 'Atomic Outbox & Audit Table',
    port: '5433',
    icon: Database,
    color: 'teal',
    accentBorder: 'border-teal-500/30 dark:border-teal-500/40',
    accentText: 'text-teal-600 dark:text-teal-400',
    accentBg: 'bg-teal-500/10',
  },
}

const PIPELINE_ORDER = ['orchestrator', 'guardrails', 'extraction', 'structuring', 'scoring']

export default function TopologyMap({ chaos, onDisrupt, onStart, activeDisruptionService }) {
  const [countdown, setCountdown] = useState({})

  // Update live countdown timers for active disruptions
  useEffect(() => {
    if (!chaos?.disruptions?.length) {
      setCountdown({})
      return
    }

    const updateTimers = () => {
      const now = Date.now() / 1000
      const timers = {}
      chaos.disruptions.forEach((d) => {
        if (d.until) {
          const rem = Math.max(0, Math.ceil(d.until - now))
          timers[d.service] = rem
        } else if (d.started && d.seconds) {
          const rem = Math.max(0, Math.ceil(d.started + d.seconds - now))
          timers[d.service] = rem
        }
      })
      setCountdown(timers)
    }

    updateTimers()
    const timer = setInterval(updateTimers, 500)
    return () => clearInterval(timer)
  }, [chaos?.disruptions])

  const containerMap = {}
  ;(chaos?.containers ?? []).forEach((c) => {
    containerMap[c.service] = c
  })

  const disruptionMap = {}
  ;(chaos?.disruptions ?? []).forEach((d) => {
    disruptionMap[d.service] = d
  })

  const renderServiceNode = (serviceKey) => {
    const meta = SERVICE_CONFIGS[serviceKey]
    if (!meta) return null
    const IconComponent = meta.icon
    const container = containerMap[serviceKey] || { status: 'unknown' }
    const disruption = disruptionMap[serviceKey]
    const remSeconds = countdown[serviceKey]
    const isUp = container.status === 'running'
    const isDisrupted = Boolean(disruption) || activeDisruptionService === serviceKey

    return (
      <div
        key={serviceKey}
        className={`relative flex-1 min-w-[170px] max-w-[210px] p-3 rounded-xl border transition-all duration-300 ${
          isDisrupted
            ? 'border-rose-500/70 bg-rose-500/10 ring-2 ring-rose-500/40 shadow-md shadow-rose-500/10 scale-[1.02]'
            : isUp
            ? 'border-border/70 bg-card hover:border-border hover:shadow-sm'
            : 'border-rose-500/30 bg-muted/40 opacity-75'
        }`}
      >
        {/* Top Header */}
        <div className="flex items-center justify-between gap-1.5 mb-2">
          <div className="flex items-center gap-2 min-w-0">
            <div
              className={`h-7 w-7 rounded-lg ${meta.accentBg} ${meta.accentText} border ${meta.accentBorder} flex items-center justify-center shrink-0`}
            >
              <IconComponent size={14} />
            </div>
            <div className="min-w-0">
              <div className="font-semibold text-xs text-foreground truncate leading-tight">
                {meta.label}
              </div>
              <div className="text-[10px] font-mono text-muted-foreground truncate">
                Port {meta.port}
              </div>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="shrink-0 flex items-center">
            {isDisrupted ? (
              <Badge variant="destructive" className="text-[9px] px-1.5 py-0 h-4 font-mono animate-pulse">
                <Flame size={10} className="mr-0.5" />
                {disruption?.action === 'kill' ? 'SIGKILL' : 'SIGTERM'}
              </Badge>
            ) : isUp ? (
              <span className="flex items-center gap-1.5 text-[10px] text-emerald-500 font-mono">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span className="font-semibold text-[10px]">UP</span>
              </span>
            ) : (
              <Badge variant="outline" className="text-[9px] text-rose-500 border-rose-500/40 px-1 py-0 h-4">
                MATI
              </Badge>
            )}
          </div>
        </div>

        {/* Role description */}
        <div className="text-[10px] text-muted-foreground line-clamp-1 mb-2.5 font-sans">
          {meta.role}
        </div>

        {/* Live Disruption Countdown */}
        {isDisrupted && remSeconds != null && (
          <div className="mb-2 p-1 px-2 rounded-md bg-rose-500/15 border border-rose-500/30 text-[10px] text-rose-600 dark:text-rose-400 font-mono text-center animate-pulse">
            Mati: <b>{remSeconds}s</b> tersisa…
          </div>
        )}

        {/* Quick Actions */}
        <div className="pt-2 border-t border-border/40 flex items-center justify-between gap-1">
          {isUp ? (
            <div className="grid grid-cols-2 gap-1 w-full">
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => onDisrupt(serviceKey, 'stop', 20)}
                disabled={isDisrupted}
                title={`SIGTERM: Graceful shutdown ${meta.label}`}
                className="h-5 px-1 text-[9px] font-mono text-amber-600 dark:text-amber-400 border-amber-500/25 hover:bg-amber-500/10"
              >
                SIGTERM
              </Button>
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => onDisrupt(serviceKey, 'kill', 20)}
                disabled={isDisrupted}
                title={`SIGKILL: Crash mendadak ${meta.label}`}
                className="h-5 px-1 text-[9px] font-mono text-rose-600 dark:text-rose-400 border-rose-500/25 hover:bg-rose-500/10"
              >
                SIGKILL
              </Button>
            </div>
          ) : (
            <Button
              type="button"
              variant="default"
              size="xs"
              onClick={() => onStart(serviceKey)}
              className="h-5 px-2 text-[10px] w-full gap-1 bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              <Power size={10} />
              <span>Nyalakan</span>
            </Button>
          )}
        </div>
      </div>
    )
  }

  return (
    <Card className="border-border bg-card/95 border-t-2 border-t-rose-500 shadow-sm overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2.5">
            <div className="h-7 w-7 rounded-lg bg-rose-500/10 border border-rose-500/25 flex items-center justify-center text-rose-600 dark:text-rose-400">
              <Activity size={15} />
            </div>
            <div>
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <span>Peta Topologi Arsitektur &amp; Ketahanan Pod</span>
                <span className="text-[10px] font-mono font-normal text-muted-foreground hidden sm:inline">
                  (Live Docker Container)
                </span>
              </CardTitle>
              <CardDescription className="text-xs">
                Visualisasi rantai request HTTP dan outbox relay. Klik tombol untuk simulasi gangguan container secara instan.
              </CardDescription>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" /> Normal
            </span>
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-rose-500 inline-block animate-pulse" /> Gangguan
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 pt-1">
        {/* TOP ROW: Linear Processing Pipeline */}
        <div>
          <div className="text-[11px] font-semibold text-muted-foreground font-mono uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Zap size={12} className="text-amber-500" />
            <span>1. Alur Sinkron / Asinkron Permintaan (HTTP Pipeline)</span>
          </div>

          <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
            {PIPELINE_ORDER.map((s, idx) => (
              <div key={s} className="flex items-center gap-2 shrink-0">
                {renderServiceNode(s)}
                {idx < PIPELINE_ORDER.length - 1 && (
                  <div className="flex flex-col items-center justify-center text-muted-foreground/40 shrink-0 px-0.5">
                    <ArrowRight size={14} className="text-muted-foreground/60 animate-pulse" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* BOTTOM ROW: PostgreSQL Shared Atomic Storage & Outbox */}
        <div className="pt-2 border-t border-border/50">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
            <div className="text-[11px] font-semibold text-muted-foreground font-mono uppercase tracking-wider flex items-center gap-1.5">
              <Layers size={12} className="text-teal-500" />
              <span>2. Lapisan Penyimpanan Atomik &amp; Antrean Relay (PostgreSQL)</span>
            </div>
            <span className="text-[10px] text-muted-foreground font-mono">
              Tabel: <code>nilam_pipeline_outbox</code> &amp; <code>nilam_ocr_results</code>
            </span>
          </div>

          <div className="flex items-center gap-3 flex-wrap sm:flex-nowrap">
            {renderServiceNode('postgres')}

            <div className="flex-1 p-3 rounded-xl bg-muted/30 border border-border/60 text-xs text-muted-foreground leading-relaxed">
              <div className="font-semibold text-foreground text-xs flex items-center gap-1.5 mb-1">
                <Database size={13} className="text-teal-500" />
                <span>Jaminan Transaksional Dua Fase (Transactional Outbox Pattern)</span>
              </div>
              <p className="m-0 text-[11px]">
                Jika <b>Extraction</b>, <b>Structuring</b>, atau <b>Scoring</b> mengalami gangguan (crash / restart),
                transaksi yang belum di-commit akan rollback secara bersih. Pekerjaan yang sudah tersimpan memiliki
                status antrean di <code>nilam_pipeline_outbox</code> dan secara otomatis di-retry oleh relay begitu pod hidup kembali.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
