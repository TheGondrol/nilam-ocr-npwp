import { useState } from 'react'
import { Copy, Check, ChevronDown, ChevronRight, FileCode } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function JsonViewer({ value, label = 'JSON Data', defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const [copied, setCopied] = useState(false)

  if (value == null) return null

  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  const length = text.length

  const handleCopy = (e) => {
    e.stopPropagation()
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-lg border border-border/70 bg-background/80 overflow-hidden shadow-xs transition-all">
      <div
        className="flex items-center justify-between px-3 py-2 bg-muted/40 cursor-pointer select-none hover:bg-muted/60 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown size={14} className="text-muted-foreground" /> : <ChevronRight size={14} className="text-muted-foreground" />}
          <FileCode size={14} className="text-muted-foreground" />
          <span className="text-xs font-medium text-foreground">{label}</span>
          <span className="text-[10px] text-muted-foreground font-mono">
            ({length.toLocaleString()} bytes)
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={handleCopy}
            className="h-6 px-2 text-[11px] gap-1 text-muted-foreground hover:text-foreground"
            title="Salin JSON"
          >
            {copied ? (
              <>
                <Check size={12} className="text-emerald-400" />
                <span className="text-emerald-400 font-medium">Tersalin</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Salin</span>
              </>
            )}
          </Button>
        </div>
      </div>
      {open && (
        <pre className="p-3 text-[11px] font-mono leading-relaxed bg-muted/50 text-foreground dark:bg-zinc-950 dark:text-zinc-300 overflow-x-auto max-h-[360px] border-t border-border select-text">
          {text}
        </pre>
      )}
    </div>
  )
}
