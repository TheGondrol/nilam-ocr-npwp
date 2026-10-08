import { useState, useEffect } from 'react'
import Header from './components/Header'
import PipelineView from './views/PipelineView'
import LoadTestView from './views/LoadTestView'
import ScenariosView from './views/ScenariosView'
import OutboxGuideView from './views/OutboxGuideView'
import { useTheme } from './lib/useTheme'
import './App.css'

export default function App() {
  const { theme, toggleTheme } = useTheme()
  const [view, setView] = useState(() => {
    try {
      const hash = window.location.hash.slice(1)
      const stored = hash || localStorage.getItem('tracker.view')
      return ['loadtest', 'scenarios', 'outbox'].includes(stored) ? stored : 'pipeline'
    } catch (_) {
      return 'pipeline'
    }
  })

  const [overview, setOverview] = useState(null)
  const [sim, setSim] = useState(null)
  const [focus, setFocus] = useState(null)

  useEffect(() => {
    try {
      localStorage.setItem('tracker.view', view)
      window.location.hash = view === 'pipeline' ? '' : view
    } catch (_) {
      // abaikan
    }
  }, [view])

  useEffect(() => {
    const loadOverview = () => {
      fetch('/api/outbox')
        .then((r) => r.json())
        .then(setOverview)
        .catch(() => {})
    }
    const loadSim = () => {
      fetch('/api/simulation')
        .then((r) => r.json())
        .then(setSim)
        .catch(() => {})
    }

    loadOverview()
    loadSim()

    const timer = setInterval(() => {
      loadOverview()
      loadSim()
    }, 2000)

    return () => clearInterval(timer)
  }, [])

  const openRequest = (rid) => {
    setFocus({ rid, at: Date.now() })
    setView('pipeline')
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans antialiased selection:bg-primary selection:text-primary-foreground">
      <Header
        view={view}
        setView={setView}
        overview={overview}
        sim={sim}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6">
        {view === 'pipeline' && (
          <PipelineView overview={overview} focus={focus} />
        )}

        {view === 'loadtest' && (
          <LoadTestView overview={overview} />
        )}

        {view === 'scenarios' && (
          <ScenariosView openRequest={openRequest} />
        )}

        {view === 'outbox' && (
          <OutboxGuideView openRequest={openRequest} />
        )}
      </main>
    </div>
  )
}
