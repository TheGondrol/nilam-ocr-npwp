import { Fragment } from 'react'
import { ArrowRight, Check, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'

export const PIPELINE_NAMES = ['guardrails', 'extraction', 'structuring', 'scoring']

export const SERVICE_LABELS = {
  guardrails: 'guardrails',
  extraction: 'extraction (OCR)',
  structuring: 'structuring',
  scoring: 'scoring',
}

export const SEQUENCE_PRESETS = [
  { label: 'Penuh (Semua)', value: PIPELINE_NAMES },
  { label: 'Tanpa Guardrails', value: ['extraction', 'structuring', 'scoring'] },
  { label: 'Guardrails Saja', value: ['guardrails'] },
  { label: 'Sampai Extraction', value: ['guardrails', 'extraction'] },
  { label: 'Sampai Structuring', value: ['guardrails', 'extraction', 'structuring'] },
  { label: 'Extraction Saja', value: ['extraction'] },
]

export const INVALID_SEQUENCE = 'INVALID_PIPELINE_SEQUENCE'

export function sequenceProblem(names) {
  if (!names.length) return 'Pilih minimal satu service'
  const start = PIPELINE_NAMES.indexOf(names[0])
  if (start > PIPELINE_NAMES.indexOf('extraction')) {
    return `${names[0]} tidak bisa di depan: butuh hasil ${PIPELINE_NAMES[start - 1]}`
  }
  if (names.some((name, i) => name !== PIPELINE_NAMES[start + i])) {
    return 'Tidak boleh melompati service di tengah rantai'
  }
  return null
}

export function sequenceToSend(names) {
  return names.length === PIPELINE_NAMES.length ? null : names
}

export function sequenceText(sequence) {
  return sequence ? sequence.join(' → ') : 'Penuh'
}

export default function SequenceToggles({ names, onChange, allowInvalid }) {
  const toggle = (name) => {
    onChange(PIPELINE_NAMES.filter((n) => (n === name ? !names.includes(n) : names.includes(n))))
  }

  const problem = sequenceProblem(names)
  const sent = sequenceToSend(names)

  return (
    <div className="rounded-lg border border-border/70 bg-background/50 p-3 space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-foreground font-mono">
          pipeline_name_sequence
        </span>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0 font-mono text-muted-foreground">
          {names.length}/4 aktif
        </Badge>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {PIPELINE_NAMES.map((name, i) => {
          const active = names.includes(name)
          return (
            <Fragment key={name}>
              {i > 0 && <ArrowRight size={11} className="text-muted-foreground/40 shrink-0" />}
              <Button
                type="button"
                variant={active ? 'secondary' : 'outline'}
                size="xs"
                onClick={() => toggle(name)}
                className={`gap-1 text-[11px] h-7 font-mono ${
                  active
                    ? 'bg-secondary text-secondary-foreground border-border font-medium'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                aria-pressed={active}
              >
                {active && <Check size={11} className="text-foreground" />}
                <span>{SERVICE_LABELS[name]}</span>
              </Button>
            </Fragment>
          )
        })}
      </div>

      <div className="flex flex-wrap items-center gap-1.5 pt-1">
        <span className="text-[11px] text-muted-foreground mr-1">Pintasan:</span>
        {SEQUENCE_PRESETS.map((p) => (
          <Button
            key={p.label}
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => onChange(p.value)}
            className="h-6 px-2 text-[10px] text-muted-foreground hover:text-foreground"
          >
            {p.label}
          </Button>
        ))}
      </div>

      <div className="text-[11px] text-muted-foreground font-mono">
        Dikirim ke API:{' '}
        {sent ? (
          <code className="text-foreground bg-muted px-1.5 py-0.5 rounded text-[10px]">
            {JSON.stringify(sent)}
          </code>
        ) : (
          <span className="italic text-muted-foreground">
            default null (pipeline penuh)
          </span>
        )}
      </div>

      {problem && (
        <Alert variant={allowInvalid && names.length ? 'warning' : 'destructive'} className="py-2 text-[11px]">
          <AlertCircle className="h-3.5 w-3.5" />
          <AlertDescription className="text-[11px] font-mono leading-tight">
            {allowInvalid && names.length
              ? `Akan ditolak orchestrator: 422 ${INVALID_SEQUENCE} (${problem})`
              : problem}
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
