import { useState, useRef, useEffect } from 'react'
import {
  HelpCircle,
  X,
  ShieldAlert,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  ExternalLink,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { SCENARIO_EXPLANATIONS } from '../constants/scenarioExplanations'

export default function ScenarioExplanationPopover({ scenarioId, className = '' }) {
  const [isOpen, setIsOpen] = useState(false)
  const [isPinned, setIsPinned] = useState(false)
  const hoverTimeoutRef = useRef(null)
  const popoverRef = useRef(null)

  const exp = SCENARIO_EXPLANATIONS[scenarioId]
  if (!exp) return null

  // Handle outside click to close pinned popover
  useEffect(() => {
    if (!isOpen) return
    const handleClickOutside = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setIsOpen(false)
        setIsPinned(false)
      }
    }
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setIsOpen(false)
        setIsPinned(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen])

  const handleMouseEnter = () => {
    if (isPinned) return
    clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      setIsOpen(true)
    }, 180)
  }

  const handleMouseLeave = () => {
    if (isPinned) return
    clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      setIsOpen(false)
    }, 250)
  }

  const handleClick = (e) => {
    e.stopPropagation()
    if (isOpen && isPinned) {
      setIsOpen(false)
      setIsPinned(false)
    } else {
      setIsOpen(true)
      setIsPinned(true)
    }
  }

  const handleClose = (e) => {
    e.stopPropagation()
    setIsOpen(false)
    setIsPinned(false)
  }

  return (
    <div
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      ref={popoverRef}
    >
      {/* Trigger Button */}
      <button
        type="button"
        onClick={handleClick}
        className={`h-5 w-5 rounded-full inline-flex items-center justify-center transition-all ${
          isOpen
            ? 'text-sky-400 bg-sky-500/20 ring-2 ring-sky-500/30'
            : 'text-muted-foreground/70 hover:text-sky-400 hover:bg-sky-500/10'
        }`}
        title="Klik / Hover untuk melihat penjelasan arsitektur dan alasan status skenario ini"
        aria-label="Penjelasan Skenario"
      >
        <HelpCircle size={13} className="stroke-[2.2]" />
      </button>

      {/* Popover Card */}
      {isOpen && (
        <div
          className="absolute z-50 left-0 sm:left-auto sm:-translate-x-1/2 top-full mt-2 w-[340px] sm:w-[420px] max-w-[92vw] rounded-xl border border-sky-500/30 bg-card/98 backdrop-blur-xl shadow-2xl p-4 text-left animate-in fade-in zoom-in-95 duration-150"
          style={{
            maxHeight: '85vh',
            overflowY: 'auto',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start justify-between gap-3 border-b border-border/60 pb-3">
            <div className="space-y-1 min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <Badge variant="outline" className="text-[10px] font-mono text-sky-400 border-sky-400/30 py-0">
                  {exp.category}
                </Badge>
                <Badge
                  variant={
                    exp.expectedOutcome.includes('WARN')
                      ? 'rejected'
                      : exp.expectedOutcome.includes('FAIL')
                      ? 'failed'
                      : 'done'
                  }
                  className="text-[10px] font-mono py-0"
                >
                  {exp.expectedOutcome.split(' ')[0]}
                </Badge>
              </div>
              <h4 className="text-xs font-bold text-foreground leading-snug">
                {exp.title}
              </h4>
            </div>

            <button
              type="button"
              onClick={handleClose}
              className="text-muted-foreground hover:text-foreground p-1 rounded-md hover:bg-muted/60 transition-colors"
            >
              <X size={14} />
            </button>
          </div>

          {/* Body Content */}
          <div className="space-y-3.5 pt-3 text-xs leading-relaxed">
            {/* 1. Mekanisme di Balik Layar */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-sky-400 text-[11px] uppercase tracking-wide">
                <Cpu size={13} />
                <span>Mekanisme di Balik Layar</span>
              </div>
              <p className="text-muted-foreground text-xs leading-relaxed bg-muted/30 p-2 rounded-lg border border-border/40">
                {exp.mechanism}
              </p>
            </div>

            {/* 2. Mengapa Status Akhirnya Demikian */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-amber-400 text-[11px] uppercase tracking-wide">
                <AlertTriangle size={13} />
                <span>Alasan Status Akhir &amp; Auto-Recovery</span>
              </div>
              <p className="text-muted-foreground text-xs leading-relaxed bg-amber-500/5 p-2 rounded-lg border border-amber-500/20">
                {exp.whyStatus}
              </p>
            </div>

            {/* 3. Rekomendasi Arsitektur untuk BRI */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-emerald-400 text-[11px] uppercase tracking-wide">
                <Lightbulb size={13} />
                <span>Rekomendasi Arsitektur Produksi</span>
              </div>
              <p className="text-muted-foreground text-xs leading-relaxed bg-emerald-500/5 p-2 rounded-lg border border-emerald-500/20">
                {exp.recommendation}
              </p>
            </div>
          </div>

          {/* Footer Pin status */}
          <div className="mt-3.5 pt-2 border-t border-border/40 flex items-center justify-between text-[10px] text-muted-foreground font-mono">
            <span>ID: {scenarioId}</span>
            <span>{isPinned ? '📌 Tersemat (Klik X untuk tutup)' : 'Hover / Klik untuk pin'}</span>
          </div>
        </div>
      )}
    </div>
  )
}
