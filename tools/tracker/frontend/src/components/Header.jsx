import {
  Activity,
  BarChart3,
  ShieldAlert,
  Layers,
  Terminal,
  Database,
  CheckCircle2,
  AlertTriangle,
  Sun,
  Moon,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

export default function Header({ view, setView, overview, live, sim, theme, toggleTheme }) {
  let totalDeadLetters = 0
  if (overview) {
    Object.values(overview).forEach((s) => {
      if (s?.dead_letters) totalDeadLetters += s.dead_letters
    })
  }

  const isGke = sim?.target === 'gke'

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <div
          className="flex items-center gap-2.5 cursor-pointer select-none"
          onClick={() => setView('pipeline')}
        >
          <div className="h-7 w-7 rounded-md bg-secondary border border-border flex items-center justify-center text-foreground">
            <Terminal size={15} />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm tracking-tight text-foreground">
              Nilam OCR Tracker
            </span>
            <span className="text-muted-foreground/40 font-mono text-xs hidden sm:inline">/</span>
            <span className="text-xs text-muted-foreground hidden sm:inline">
              Pipeline &amp; Outbox
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-muted/50 p-1 rounded-md border border-border">
          <Button
            variant={view === 'pipeline' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setView('pipeline')}
            className={`gap-1.5 text-xs h-7 px-3 ${view === 'pipeline' ? 'bg-background shadow-xs font-medium text-foreground' : 'text-muted-foreground'}`}
          >
            <Activity size={13} />
            <span>Pipeline &amp; Outbox</span>
          </Button>

          <Button
            variant={view === 'loadtest' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setView('loadtest')}
            className={`gap-1.5 text-xs h-7 px-3 ${view === 'loadtest' ? 'bg-background shadow-xs font-medium text-foreground' : 'text-muted-foreground'}`}
          >
            <BarChart3 size={13} />
            <span className="hidden md:inline">Load Testing</span>
          </Button>

          <Button
            variant={view === 'scenarios' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setView('scenarios')}
            className={`gap-1.5 text-xs h-7 px-3 ${view === 'scenarios' ? 'bg-background shadow-xs font-medium text-foreground' : 'text-muted-foreground'}`}
          >
            <ShieldAlert size={13} />
            <span className="hidden md:inline">Skenario Gangguan</span>
          </Button>

          <Button
            variant={view === 'outbox' ? 'secondary' : 'ghost'}
            size="xs"
            onClick={() => setView('outbox')}
            className={`gap-1.5 text-xs h-7 px-3 ${view === 'outbox' ? 'bg-background shadow-xs font-medium text-foreground' : 'text-muted-foreground'}`}
          >
            <Layers size={13} />
            <span className="hidden md:inline">Panduan Outbox</span>
          </Button>
        </nav>

        {/* Status Indicators */}
        <div className="flex items-center gap-2">
          {totalDeadLetters > 0 ? (
            <Badge variant="rejected" className="gap-1 text-[11px] font-mono">
              <AlertTriangle size={11} className="text-amber-400" />
              <span>{totalDeadLetters} Dead Letter</span>
            </Badge>
          ) : (
            <Badge variant="done" className="gap-1 text-[11px] font-mono">
              <CheckCircle2 size={11} className="text-emerald-400" />
              <span>Outbox Bersih</span>
            </Badge>
          )}

          <Badge
            variant={live ? 'done' : 'outline'}
            className="gap-1.5 text-[11px] font-mono"
            title={live ? 'SSE terhubung secara live' : 'SSE idle'}
          >
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${live ? 'bg-emerald-500' : 'bg-zinc-500'}`} />
            <span>{live ? 'Live' : 'Idle'}</span>
          </Badge>

          <Badge
            variant="outline"
            className="gap-1 text-[11px] font-mono hidden lg:flex text-muted-foreground"
          >
            <Database size={11} />
            <span>{isGke ? 'GKE' : 'Docker Lokal'}</span>
          </Badge>

          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={toggleTheme}
            className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
            title={theme === 'dark' ? 'Ganti ke Mode Terang' : 'Ganti ke Mode Gelap'}
            aria-label="Toggle dark/light mode"
          >
            {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
          </Button>
        </div>
      </div>
    </header>
  )
}
