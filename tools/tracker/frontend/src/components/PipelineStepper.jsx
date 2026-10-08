import { ArrowRight, ShieldCheck, FileText, Cpu, CheckCheck } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

const STAGE_META = {
  GUARDRAILS: {
    label: 'Guardrails & Validasi',
    sub: 'Filter Dokumen & NPWP',
    icon: ShieldCheck,
    borderColor: 'border-t-amber-500',
    badgeClass: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/25',
    activeClass: 'border-amber-500/60 bg-amber-500/[0.05]',
  },
  OCR: {
    label: 'Extraction (OCR)',
    sub: 'Deteksi & Ekstraksi Teks',
    icon: FileText,
    borderColor: 'border-t-sky-500',
    badgeClass: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
    activeClass: 'border-sky-500/60 bg-sky-500/[0.05]',
  },
  STRUCTURING: {
    label: 'Structuring',
    sub: 'Parsing Entitas & Aturan RegEx',
    icon: Cpu,
    borderColor: 'border-t-violet-500',
    badgeClass: 'bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/25',
    activeClass: 'border-violet-500/60 bg-violet-500/[0.05]',
  },
  SCORING: {
    label: 'Scoring',
    sub: 'Confidence & Name Match',
    icon: CheckCheck,
    borderColor: 'border-t-emerald-500',
    badgeClass: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/25',
    activeClass: 'border-emerald-500/60 bg-emerald-500/[0.05]',
  },
}

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

function badgeVariantOf(status) {
  switch (status?.toLowerCase()) {
    case 'done': return 'done'
    case 'processing': return 'processing'
    case 'rejected': return 'rejected'
    case 'failed': return 'failed'
    default: return 'idle'
  }
}

export default function PipelineStepper({ stages, stageViews, onSelectStage, activeStage }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {stages.map((stage, i) => {
        const view = stageViews?.[stage] || { status: 'PENDING' }
        const status = (view.status || 'PENDING').toLowerCase()
        const isHighlight = activeStage === stage
        const meta = STAGE_META[stage] || {
          label: stage,
          sub: 'Tahap Pipeline',
          icon: FileText,
          borderColor: 'border-t-primary',
          badgeClass: 'bg-muted text-muted-foreground border-border',
          activeClass: 'border-primary/50 bg-accent',
        }
        const Icon = meta.icon

        return (
          <div
            key={stage}
            onClick={() => onSelectStage && onSelectStage(stage)}
            className={`relative flex items-center justify-between p-3.5 rounded-lg border border-border border-t-2 ${meta.borderColor} transition-all cursor-pointer select-none shadow-xs ${
              isHighlight
                ? `${meta.activeClass} ring-1 ring-border/80`
                : 'bg-card hover:bg-muted/40 hover:border-border'
            }`}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className={`h-8 w-8 rounded-md border flex items-center justify-center shrink-0 ${meta.badgeClass}`}>
                <Icon size={16} />
              </div>

              <div className="truncate">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] font-semibold text-muted-foreground">
                    0{i + 1}
                  </span>
                  <div className="text-xs font-semibold text-foreground truncate">
                    {meta.label}
                  </div>
                </div>

                <div className="flex items-center gap-1.5 mt-1">
                  <Badge variant={badgeVariantOf(status)} className="text-[10px] px-1.5 py-0 font-mono">
                    {view.status || 'PENDING'}
                  </Badge>
                  {view.elapsed != null && (
                    <span className="text-[10px] text-muted-foreground font-mono">
                      {fmtMs(view.elapsed)}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {i < stages.length - 1 && (
              <ArrowRight size={13} className="text-muted-foreground/30 shrink-0 ml-2 hidden lg:block" />
            )}
          </div>
        )
      })}
    </div>
  )
}
