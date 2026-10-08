import { useState, useEffect, useRef, useMemo } from 'react'
import {
  Play,
  Pause,
  RotateCcw,
  FastForward,
  ChevronLeft,
  ChevronRight,
  X,
  Flame,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Database,
  Cpu,
  Layers,
  Send,
  Webhook,
  Activity,
  Clock,
  Sparkles,
  Info,
  Lock,
  Unlock,
  Zap,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import ScenarioExplanationPopover from './ScenarioExplanationPopover'

// Helper format seconds
function fmtSec(sec) {
  if (sec == null || isNaN(sec)) return '0.0s'
  return `${sec.toFixed(1)}s`
}

export default function ScenarioReplayModal({ run, onClose, openRequest }) {
  if (!run) return null

  // Calculate duration and steps
  const steps = run.steps || []
  const checks = run.checks || []
  const maxTime = useMemo(() => {
    let t = 0
    if (run.ended_at && run.started_at && run.ended_at > run.started_at) {
      t = run.ended_at - run.started_at
    }
    for (const s of steps) {
      if (s.t > t) t = s.t
    }
    for (const c of checks) {
      if (c.t > t) t = c.t
    }
    // Tambahkan buffer visual 2.5 detik agar animasi pergerakan akhir menuju Callback Webhook tuntas dan mulus
    return Math.max(t + 2.5, 6.0)
  }, [run, steps, checks])

  // Playback state
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.5) // Default slightly fast for better experience
  const [activeTab, setActiveTab] = useState('steps') // 'steps' | 'checks'
  const animRef = useRef(null)
  const lastTickRef = useRef(null)
  const stepsContainerRef = useRef(null)

  // Auto-play loop using requestAnimationFrame
  useEffect(() => {
    if (!isPlaying) {
      lastTickRef.current = null
      return
    }

    const tick = (now) => {
      if (lastTickRef.current == null) {
        lastTickRef.current = now
      }
      const delta = (now - lastTickRef.current) / 1000
      lastTickRef.current = now

      setCurrentTime((prev) => {
        const next = prev + delta * speed
        if (next >= maxTime) {
          setIsPlaying(false)
          return maxTime
        }
        return next
      })

      animRef.current = requestAnimationFrame(tick)
    }

    animRef.current = requestAnimationFrame(tick)
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [isPlaying, speed, maxTime])

  // Keyboard shortcut: Space for Play/Pause, Left/Right for Seek
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault()
        setIsPlaying((p) => !p)
      } else if (e.key === 'ArrowRight') {
        setCurrentTime((t) => Math.min(maxTime, t + 1))
      } else if (e.key === 'ArrowLeft') {
        setCurrentTime((t) => Math.max(0, t - 1))
      } else if (e.key === 'Escape') {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [maxTime, onClose])

  // Active step index at currentTime
  const activeStepIdx = useMemo(() => {
    let lastIdx = 0
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].t <= currentTime) {
        lastIdx = i
      } else {
        break
      }
    }
    return lastIdx
  }, [steps, currentTime])

  const activeStep = steps[activeStepIdx]

  // Auto-scroll active step in list
  useEffect(() => {
    if (!stepsContainerRef.current) return
    const activeEl = stepsContainerRef.current.querySelector(`[data-step-idx="${activeStepIdx}"]`)
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [activeStepIdx])

  // Derive core timeline milestones from steps and checks
  const milestones = useMemo(() => {
    const stepSubmit = steps.find(
      (s) => s.text.toLowerCase().includes('kirim') || s.text.toLowerCase().includes('submit')
    )
    const tSubmit = stepSubmit ? stepSubmit.t : 0.2

    const stepOcr = steps.find(
      (s) => s.text.toLowerCase().includes('ocr processing') || s.text.toLowerCase().includes('job ocr')
    )
    const tOcrStart = stepOcr ? Math.max(stepOcr.t, tSubmit + 0.8) : tSubmit + 1.2

    const stepDisrupt = steps.find(
      (s) =>
        s.text.toLowerCase().includes('sigkill') ||
        s.text.toLowerCase().includes('sigterm') ||
        s.text.toLowerCase().includes('stop')
    )
    const tDisrupt = stepDisrupt ? stepDisrupt.t : null

    const checkReaper = checks.find(
      (c) =>
        c.label.toLowerCase().includes('reaper') ||
        c.label.toLowerCase().includes('mengambil alih') ||
        c.label.toLowerCase().includes('auto-recovery')
    )
    const stepReclaimed = steps.find(
      (s) =>
        !s.text.toLowerCase().startsWith('tunggu') &&
        (s.text.toLowerCase().includes('reaper') ||
          s.text.toLowerCase().includes('mengambil alih') ||
          s.text.toLowerCase().includes('auto-recovery'))
    )
    const tReaperReclaim = checkReaper
      ? checkReaper.t
      : stepReclaimed
      ? stepReclaimed.t
      : tDisrupt != null
      ? tDisrupt + 26.0
      : null

    const stepRestart = steps.find(
      (s) => s.text.toLowerCase().includes('sehat lagi') || s.text.toLowerCase().includes('nyalakan')
    )
    const tRestart = stepRestart ? stepRestart.t : null

    const stepCallback = steps.find(
      (s) =>
        s.text.toLowerCase().includes('callback done') ||
        s.text.toLowerCase().includes('callback failed') ||
        s.text.toLowerCase().includes('callback 200') ||
        s.text.toLowerCase().includes('callback 202')
    )
    const tCallback = stepCallback ? stepCallback.t : maxTime - 2.0

    const isInlineCrash =
      run.scenario === 'crash-inline' ||
      run.scenario === 'kill-during-ocr-inline' ||
      steps.some((s) => s.text.toLowerCase().includes('inline') && s.text.toLowerCase().includes('failed')) ||
      checks.some((c) => c.label.toLowerCase().includes('upload inline') && c.level === 'warn')

    const isStageDown =
      run.scenario === 'stage-down' || steps.some((s) => s.text.toLowerCase().includes('structuring mati'))

    const isPostgresDown =
      run.scenario === 'postgres-down' || steps.some((s) => s.text.toLowerCase().includes('database mati'))

    const isDrained =
      run.scenario === 'sigterm-drained' ||
      steps.some((s) => s.text.toLowerCase().includes('diselesaikan dulu') || s.text.toLowerCase().includes('drain'))

    return {
      tSubmit,
      tOcrStart,
      tDisrupt,
      tReaperReclaim,
      tRestart,
      tCallback,
      isInlineCrash,
      isStageDown,
      isPostgresDown,
      isDrained,
    }
  }, [steps, checks, run, maxTime])

  // Derive the Single Synchronized Flow Data Packet at currentTime
  const flowPacket = useMemo(() => {
    // Easing cubic untuk pergerakan mulus dan visual yang nyaman (tanpa loncatan mendadak)
    const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
    const smoothLerp = (a, b, t) => {
      const clamped = Math.max(0, Math.min(1, isNaN(t) ? 0 : t))
      return a + (b - a) * easeInOutCubic(clamped)
    }

    if (currentTime < 0.05) {
      return {
        x: 77.5,
        y: 120,
        visible: true,
        color: '#38bdf8',
        label: 'Siap Kirim',
        sublabel: 'Client POST /extract',
        stage: 'client',
        isLocked: false,
      }
    }

    const {
      tSubmit,
      tOcrStart,
      tDisrupt,
      tReaperReclaim,
      tRestart,
      tCallback,
      isInlineCrash,
      isStageDown,
      isPostgresDown,
      isDrained,
    } = milestones

    // 1. Ingress Phase: Client -> Guardrails (t=0 to tSubmit + 0.9)
    const tGuardrailsArrival = tSubmit + 0.9
    if (currentTime < tGuardrailsArrival) {
      const p = (currentTime - tSubmit) / 0.9
      if (p <= 0) {
        return {
          x: 77.5,
          y: 120,
          visible: true,
          color: '#38bdf8',
          label: 'Client Ingress',
          sublabel: 'POST /extract',
          stage: 'client',
          isLocked: false,
        }
      }
      return {
        x: smoothLerp(77.5, 260, p),
        y: 120,
        visible: true,
        color: '#38bdf8',
        label: 'Upload Dokumen',
        sublabel: 'Mengalir ke Guardrails',
        stage: 'wire-client-guardrails',
        isLocked: false,
      }
    }

    // 2. Guardrails -> Extraction (from tGuardrailsArrival to tOcrStart)
    if (currentTime < tOcrStart) {
      const dur = Math.max(0.6, tOcrStart - tGuardrailsArrival)
      const p = (currentTime - tGuardrailsArrival) / dur
      return {
        x: smoothLerp(260, 450, p),
        y: 120,
        visible: true,
        color: '#38bdf8',
        label: 'Validasi Guardrails OK',
        sublabel: 'Menuju Engine PaddleOCR',
        stage: 'wire-guardrails-extraction',
        isLocked: false,
      }
    }

    // 3. DISRUPTION SCENARIOS
    if (tDisrupt != null) {
      // 3A. Pre-disruption OCR processing (always at Extraction center 450, 120)
      if (currentTime < tDisrupt) {
        return {
          x: 450,
          y: 120,
          visible: true,
          color: '#0ea5e9',
          label: 'Memproses OCR',
          sublabel: 'PaddleOCR Ekstraksi Teks',
          stage: 'extraction',
          isLocked: false,
        }
      }

      // 3B. CRASH INLINE: Pod mati -> Turun ke DB -> Terkunci (Lease) -> Outbox Relay ke Webhook FAILED
      if (isInlineCrash) {
        const dropDur = 0.9
        const tDropEnd = tDisrupt + dropDur
        if (currentTime < tDropEnd) {
          const p = (currentTime - tDisrupt) / dropDur
          return {
            x: 450,
            y: smoothLerp(120, 305, p),
            visible: true,
            color: '#f43f5e',
            label: '💥 Pod Terputus (SIGKILL)',
            sublabel: 'Koneksi putus; job tertahan di DB',
            stage: 'dropping-to-db',
            isLocked: false,
          }
        }

        const tRecoveryPoint = tReaperReclaim != null ? tReaperReclaim : tCallback
        if (currentTime < tRecoveryPoint) {
          const remainingLease = Math.max(0, tRecoveryPoint - currentTime).toFixed(0)
          return {
            x: 450,
            y: 305,
            visible: true,
            color: '#f59e0b',
            label: '🔒 Menunggu Pemulihan (Lease)',
            sublabel: `Sisa Batas Waktu ~${remainingLease}s`,
            stage: 'db-orphan-locked',
            isLocked: true,
          }
        }

        // Outbox relay meluncur mulus dari titik diam (450, 305) ke Webhook (840, 305)
        const relayDur = 2.0
        if (currentTime < tRecoveryPoint + relayDur) {
          const p = (currentTime - tRecoveryPoint) / relayDur
          return {
            x: smoothLerp(450, 840, p),
            y: 305,
            visible: true,
            color: '#f59e0b',
            label: '📤 Outbox Relay: Callback FAILED',
            sublabel: 'Mitigasi Crash Dikirimkan Sesuai Kontrak',
            stage: 'outbox-relay',
            isLocked: false,
          }
        }

        return {
          x: 840,
          y: 305,
          visible: true,
          color: '#f59e0b',
          label: '✅ Webhook: Callback FAILED Diterima',
          sublabel: 'Mitigasi crash selesai sesuai kontrak',
          stage: 'callback-done',
          isLocked: false,
        }
      }

      // 3C. STAGE DOWN (Structuring down): Handoff tertahan di Outbox DB sampai structuring pulih
      if (isStageDown) {
        const tWaitRestart = tRestart != null ? tRestart : tDisrupt + 15.0
        const dropDur = 0.9
        if (currentTime < tDisrupt + dropDur) {
          const p = (currentTime - tDisrupt) / dropDur
          return {
            x: smoothLerp(450, 500, p),
            y: smoothLerp(120, 305, p),
            visible: true,
            color: '#f59e0b',
            label: 'Structuring Down',
            sublabel: 'Handoff tertahan ke Outbox DB',
            stage: 'dropping-to-db',
            isLocked: false,
          }
        }

        if (currentTime < tWaitRestart) {
          return {
            x: 500,
            y: 305,
            visible: true,
            color: '#f59e0b',
            label: '⏳ Outbox Retry Backoff',
            sublabel: 'Menunggu Structuring pulih…',
            stage: 'db-orphan-locked',
            isLocked: true,
          }
        }

        const tRecoverEnd = tWaitRestart + 1.6
        if (currentTime < tRecoverEnd) {
          const p = (currentTime - tWaitRestart) / 1.6
          return {
            x: smoothLerp(500, 640, p),
            y: smoothLerp(305, 120, p),
            visible: true,
            color: '#818cf8',
            label: '🚀 Structuring Pulih',
            sublabel: 'Handoff terkirim sukses',
            stage: 'structuring',
            isLocked: false,
          }
        }

        const tScoreEnd = tRecoverEnd + 1.4
        if (currentTime < tScoreEnd) {
          const p = (currentTime - tRecoverEnd) / 1.4
          return {
            x: smoothLerp(640, 830, p),
            y: 120,
            visible: true,
            color: '#c084fc',
            label: 'Scoring Quality',
            sublabel: 'Menghitung confidence score',
            stage: 'scoring',
            isLocked: false,
          }
        }

        const tCbEnd = tScoreEnd + 1.4
        if (currentTime < tCbEnd) {
          const p = (currentTime - tScoreEnd) / 1.4
          return {
            x: smoothLerp(830, 840, p),
            y: smoothLerp(120, 305, p),
            visible: true,
            color: '#10b981',
            label: 'Menuju Webhook',
            sublabel: 'Callback DONE dikirim',
            stage: 'callback-dispatch',
            isLocked: false,
          }
        }

        return {
          x: 840,
          y: 305,
          visible: true,
          color: '#10b981',
          label: '✅ Callback DONE Selesai',
          sublabel: 'Pipeline sukses setelah self-healing',
          stage: 'callback-done',
          isLocked: false,
        }
      }

      // 3D. POSTGRES DOWN: Database mati, job tertahan di DB
      if (isPostgresDown) {
        const tDbRestart = tRestart != null ? tRestart : tDisrupt + 20.0
        const dropDur = 0.9
        if (currentTime < tDisrupt + dropDur) {
          const p = (currentTime - tDisrupt) / dropDur
          return {
            x: smoothLerp(450, 500, p),
            y: smoothLerp(120, 305, p),
            visible: true,
            color: '#ef4444',
            label: '🔴 PostgreSQL Mati',
            sublabel: 'Koneksi terputus',
            stage: 'dropping-to-db',
            isLocked: false,
          }
        }

        if (currentTime < tDbRestart) {
          return {
            x: 500,
            y: 305,
            visible: true,
            color: '#ef4444',
            label: '🔴 PostgreSQL Terputus',
            sublabel: 'Request baru ditolak (5xx), job menunggu',
            stage: 'db-orphan-locked',
            isLocked: true,
          }
        }

        const p = (currentTime - tDbRestart) / Math.max(1.8, maxTime - 2.0 - tDbRestart)
        return {
          x: smoothLerp(500, 840, p),
          y: 305,
          visible: true,
          color: '#10b981',
          label: 'PostgreSQL Pulih → Selesai',
          sublabel: 'Koneksi tersambung kembali',
          stage: 'callback-dispatch',
          isLocked: false,
        }
      }

      // 3E. SIGTERM DRAINED (Rolling restart / eviction with graceful drain):
      // Pod is NOT crashed; job finishes in Extraction during drain, then continues normally!
      if (isDrained) {
        const tDrainFinish = tRestart != null ? tRestart : tDisrupt + 5.0
        if (currentTime < tDrainFinish) {
          return {
            x: 450,
            y: 120,
            visible: true,
            color: '#f59e0b',
            label: '⏳ Draining In-Flight Job',
            sublabel: 'Menyelesaikan job sebelum proses berhenti',
            stage: 'extraction',
            isLocked: false,
          }
        }

        const tStructEnd = tDrainFinish + 1.4
        if (currentTime < tStructEnd) {
          const p = (currentTime - tDrainFinish) / 1.4
          return {
            x: smoothLerp(450, 640, p),
            y: 120,
            visible: true,
            color: '#818cf8',
            label: 'Structuring Fields',
            sublabel: 'Parsing NIK, Nama, Alamat NPWP',
            stage: 'structuring',
            isLocked: false,
          }
        }

        const tScoreEnd = tStructEnd + 1.4
        if (currentTime < tScoreEnd) {
          const p = (currentTime - tStructEnd) / 1.4
          return {
            x: smoothLerp(640, 830, p),
            y: 120,
            visible: true,
            color: '#c084fc',
            label: 'Scoring Quality',
            sublabel: 'Validasi skor kualitas',
            stage: 'scoring',
            isLocked: false,
          }
        }

        const tCbEnd = tScoreEnd + 1.4
        if (currentTime < tCbEnd) {
          const p = (currentTime - tScoreEnd) / 1.4
          return {
            x: smoothLerp(830, 840, p),
            y: smoothLerp(120, 305, p),
            visible: true,
            color: '#10b981',
            label: 'Menuju Webhook',
            sublabel: 'Callback DONE dikirim',
            stage: 'callback-dispatch',
            isLocked: false,
          }
        }

        return {
          x: 840,
          y: 305,
          visible: true,
          color: '#10b981',
          label: '✅ Callback DONE Selesai',
          sublabel: 'Graceful shutdown sukses tanpa kehilangan data',
          stage: 'callback-done',
          isLocked: false,
        }
      }

      // 3F. RECOVERABLE POD CRASH / FILE URL (Auto-Recovery triggers retry -> Extraction -> Structuring -> Scoring -> Callback)
      const dropDur = 0.9
      const tDropEnd = tDisrupt + dropDur
      if (currentTime < tDropEnd) {
        const p = (currentTime - tDisrupt) / dropDur
        return {
          x: 450,
          y: smoothLerp(120, 305, p),
          visible: true,
          color: '#f43f5e',
          label: '💥 Pod Terputus (SIGKILL)',
          sublabel: 'Koneksi putus; job tertahan di DB',
          stage: 'dropping-to-db',
          isLocked: false,
        }
      }

      const tRecoveryPoint = tReaperReclaim != null ? tReaperReclaim : tCallback
      if (currentTime < tRecoveryPoint) {
        const remainingLease = Math.max(0, tRecoveryPoint - currentTime).toFixed(0)
        return {
          x: 450,
          y: 305,
          visible: true,
          color: '#f59e0b',
          label: '🔒 Menunggu Pemulihan (Lease)',
          sublabel: `Sisa Batas Waktu ~${remainingLease}s`,
          stage: 'db-orphan-locked',
          isLocked: true,
        }
      }

      if (currentTime < tRecoveryPoint + 1.4) {
        const p = (currentTime - tRecoveryPoint) / 1.4
        return {
          x: 450,
          y: smoothLerp(305, 120, p),
          visible: true,
          color: '#38bdf8',
          label: '🔄 Auto-Recovery Mengambil Alih',
          sublabel: 'Unduh ulang MinIO & Retry Pipeline',
          stage: 'reaper-retry',
          isLocked: false,
        }
      }

      if (currentTime < tRecoveryPoint + 2.8) {
        const p = (currentTime - (tRecoveryPoint + 1.4)) / 1.4
        return {
          x: smoothLerp(450, 640, p),
          y: 120,
          visible: true,
          color: '#818cf8',
          label: 'Structuring Fields',
          sublabel: 'Parsing NIK, Nama, Alamat NPWP',
          stage: 'structuring',
          isLocked: false,
        }
      }

      if (currentTime < tRecoveryPoint + 4.2) {
        const p = (currentTime - (tRecoveryPoint + 2.8)) / 1.4
        return {
          x: smoothLerp(640, 830, p),
          y: 120,
          visible: true,
          color: '#c084fc',
          label: 'Scoring Quality',
          sublabel: 'Validasi skor kualitas',
          stage: 'scoring',
          isLocked: false,
        }
      }

      if (currentTime < tRecoveryPoint + 5.5) {
        const p = (currentTime - (tRecoveryPoint + 4.2)) / 1.3
        return {
          x: smoothLerp(830, 840, p),
          y: smoothLerp(120, 305, p),
          visible: true,
          color: '#10b981',
          label: 'Menuju Webhook',
          sublabel: 'Callback DONE dikirim',
          stage: 'callback-dispatch',
          isLocked: false,
        }
      }

      return {
        x: 840,
        y: 305,
        visible: true,
        color: '#10b981',
        label: '✅ Callback DONE Selesai',
        sublabel: 'Pipeline sukses setelah auto-recovery mandiri',
        stage: 'callback-done',
        isLocked: false,
      }
    }

    // 4. STANDARD NORMAL PIPELINE (Extraction -> Structuring -> Scoring -> Callback Webhook)
    // Synchronized with real tCallback timestamp!
    const totalPipelineWindow = Math.max(3.5, tCallback - tOcrStart)
    const tExtractDone = tOcrStart + totalPipelineWindow * 0.35
    const tExtractToStruct = tOcrStart + totalPipelineWindow * 0.48
    const tStructDone = tOcrStart + totalPipelineWindow * 0.65
    const tStructToScore = tOcrStart + totalPipelineWindow * 0.78
    const tScoreDone = tOcrStart + totalPipelineWindow * 0.88

    if (currentTime < tExtractDone) {
      return {
        x: 450,
        y: 120,
        visible: true,
        color: '#38bdf8',
        label: 'Memproses OCR',
        sublabel: 'PaddleOCR Ekstraksi Karakter',
        stage: 'extraction',
        isLocked: false,
      }
    }

    if (currentTime < tExtractToStruct) {
      const p = (currentTime - tExtractDone) / Math.max(0.4, tExtractToStruct - tExtractDone)
      return {
        x: smoothLerp(450, 640, p),
        y: 120,
        visible: true,
        color: '#818cf8',
        label: 'Handoff OCR → Structuring',
        sublabel: 'Mengirim teks hasil ekstraksi',
        stage: 'structuring',
        isLocked: false,
      }
    }

    if (currentTime < tStructDone) {
      return {
        x: 640,
        y: 120,
        visible: true,
        color: '#818cf8',
        label: 'Structuring Fields',
        sublabel: 'Parsing NIK, Nama, Alamat NPWP',
        stage: 'structuring',
        isLocked: false,
      }
    }

    if (currentTime < tStructToScore) {
      const p = (currentTime - tStructDone) / Math.max(0.4, tStructToScore - tStructDone)
      return {
        x: smoothLerp(640, 830, p),
        y: 120,
        visible: true,
        color: '#c084fc',
        label: 'Handoff Structuring → Scoring',
        sublabel: 'Mengirim field terstruktur',
        stage: 'scoring',
        isLocked: false,
      }
    }

    if (currentTime < tScoreDone) {
      return {
        x: 830,
        y: 120,
        visible: true,
        color: '#c084fc',
        label: 'Scoring Quality',
        sublabel: 'Validasi skor & confidence level',
        stage: 'scoring',
        isLocked: false,
      }
    }

    if (currentTime < tCallback) {
      const p = (currentTime - tScoreDone) / Math.max(0.4, tCallback - tScoreDone)
      return {
        x: smoothLerp(830, 840, p),
        y: smoothLerp(120, 305, p),
        visible: true,
        color: '#10b981',
        label: 'Outbox Relay Dispatched',
        sublabel: 'Mengirim payload ke Webhook',
        stage: 'callback-dispatch',
        isLocked: false,
      }
    }

    return {
      x: 840,
      y: 305,
      visible: true,
      color: '#10b981',
      label: '✅ Callback Selesai',
      sublabel: 'Orkestrasi pusat menerima payload',
      stage: 'callback-done',
      isLocked: false,
    }
  }, [currentTime, steps, checks, run, maxTime, milestones])

  // Derive Service Pod States and Narratives synchronized with flowPacket
  const simulationState = useMemo(() => {
    const services = {
      client: { status: 'idle', label: 'Idle' },
      orchestrator: { status: 'healthy', label: 'Running' },
      guardrails: { status: 'healthy', label: 'Running' },
      extraction: { status: 'healthy', label: 'Running' },
      structuring: { status: 'healthy', label: 'Running' },
      scoring: { status: 'healthy', label: 'Running' },
      postgres: { status: 'healthy', label: 'Running' },
      outbox: { status: 'idle', count: 0, label: 'Outbox Relay' },
      callback: { status: 'idle', label: 'Menunggu' },
    }

    let activeStage = flowPacket ? flowPacket.stage : 'ingress'
    let flameService = null
    let flameType = null
    const leaseCountdown = flowPacket && flowPacket.isLocked ? flowPacket.sublabel : null
    const reaperActive =
      flowPacket && (flowPacket.isLocked || flowPacket.stage === 'db-orphan-locked' || flowPacket.stage === 'reaper-retry')
    let narrative = 'Memulai simulasi skenario…'

    const relevantSteps = steps.filter((s) => s.t <= currentTime)

    if (relevantSteps.length > 0) {
      services.client.status = 'active'
      services.client.label = 'Request Sent'
    }

    for (const s of relevantSteps) {
      const text = s.text.toLowerCase()

      // Ingress / Submission
      if (text.includes('kirim') || text.includes('menjawab callback') || text.includes('submit')) {
        narrative = `Request diinisiasi ke gerbang Orchestrator (${run.request_ids?.[0] || 'REQ_ID'}).`
      }

      // Disruptions
      if (text.includes('sigkill') || text.includes('kill')) {
        for (const svc of ['extraction', 'structuring', 'scoring', 'guardrails', 'postgres']) {
          if (text.includes(svc)) {
            services[svc].status = 'crashed'
            services[svc].label = '💥 SIGKILL (OOM)'
            flameService = svc
            flameType = 'sigkill'
            narrative = `💥 SIMULASI GANGGUAN: Sinyal SIGKILL mematikan pod ${svc} seketika (meniru OOM / node hilang)!`
          }
        }
      } else if (text.includes('sigterm') || text.includes('stop')) {
        for (const svc of ['extraction', 'structuring', 'scoring', 'guardrails', 'postgres']) {
          if (text.includes(svc)) {
            services[svc].status = 'draining'
            services[svc].label = '⚠️ SIGTERM (Draining)'
            flameService = svc
            flameType = 'sigterm'
            narrative = `⚠️ Rolling restart / eviction: SIGTERM dikirim ke pod ${svc}, proses melakukan drain tugas.`
          }
        }
      }

      // Restarts & Healed
      if (text.includes('sehat lagi') || text.includes('nyalakan') || text.includes('start')) {
        for (const svc of ['extraction', 'structuring', 'scoring', 'guardrails', 'postgres']) {
          if (text.includes(svc)) {
            services[svc].status = 'healthy'
            services[svc].label = 'Sehat Kembali'
            if (flameService === svc) flameService = null
            narrative = `Pod ${svc} berhasil restart dan lulus healthcheck /readyz.`
          }
        }
      }
    }

    // Synchronize Pod dynamic labels and statuses from current flowPacket stage
    if (flowPacket) {
      if (flowPacket.stage === 'wire-client-guardrails') {
        services.guardrails.label = 'Memvalidasi Ingress'
      } else if (flowPacket.stage === 'extraction' && services.extraction.status === 'healthy') {
        services.extraction.label = 'OCR Processing'
        narrative = 'Ekstraksi teks dokumen sedang diproses oleh PaddleOCR.'
      } else if (flowPacket.stage === 'structuring' && services.structuring.status === 'healthy') {
        services.structuring.label = 'Structuring Fields'
        narrative = 'Handoff terkirim: Structuring melakukan parsing dan normalisasi field.'
      } else if (flowPacket.stage === 'scoring' && services.scoring.status === 'healthy') {
        services.scoring.label = 'Scoring Quality'
        narrative = 'Structuring selesai: Scoring mengevaluasi confidence score dan validitas dokumen.'
      } else if (flowPacket.stage === 'db-orphan-locked') {
        services.outbox.status = 'reaping'
        narrative = 'Pekerjaan terputus di database, menunggu batas masa sewa (lease timeout) untuk diambil alih secara otomatis.'
      } else if (flowPacket.stage === 'reaper-retry') {
        services.outbox.status = 'reaping'
        narrative = '⚡ PEMULIHAN OTOMATIS: Masa sewa (lease) berakhir, sistem Auto-Recovery mengambil alih pekerjaan terputus.'
      } else if (flowPacket.stage === 'outbox-relay') {
        services.outbox.status = 'relaying'
        narrative = milestones.isInlineCrash
          ? 'Dokumen inline hilang saat crash; Transactional Outbox mengirimkan callback FAILED sesuai kontrak mitigasi.'
          : 'Transactional Outbox relay membersihkan antrean callback hasil akhir.'
      } else if (flowPacket.stage === 'callback-done') {
        services.outbox.status = 'drained'
        if (milestones.isInlineCrash) {
          services.callback.status = 'failed_handled'
          services.callback.label = 'Callback FAILED (Tertangani)'
          narrative = 'Callback FAILED diterima orkestrasi pusat sesuai kontrak mitigasi crash.'
        } else {
          services.callback.status = 'done'
          services.callback.label = 'Callback Terkirim'
          narrative = 'Hasil akhir pipeline berhasil dikirimkan ke webhook orkestrasi pusat!'
        }
      }
    }

    // End of scenario evaluation
    if (currentTime >= maxTime - 0.1) {
      if (run.status === 'pass') {
        services.callback.status = 'done'
        services.callback.label = 'PASS (Sesuai Kontrak)'
        narrative = 'Semua evaluasi dan kontrak terpenuhi dengan status LULUS (PASS).'
      } else if (run.status === 'warn') {
        services.callback.status = 'warn'
        services.callback.label = 'WARN (Tercatat)'
        narrative = 'Skenario selesai dengan catatan yang harus diketahui (WARN).'
      } else if (run.status === 'fail') {
        services.callback.status = 'fail'
        services.callback.label = 'FAIL (Pelanggaran)'
        narrative = 'Terjadi ketidaksesuaian kontrak pada skenario ini (FAIL).'
      }
    }

    return {
      services,
      activeStage,
      flameService,
      flameType,
      leaseCountdown,
      reaperActive,
      narrative,
    }
  }, [steps, currentTime, maxTime, run, flowPacket, milestones])

  // Scrub bar helper
  const handleSeek = (e) => {
    const val = parseFloat(e.target.value)
    setCurrentTime(val)
  }

  // Jump to specific step
  const jumpToStep = (idx) => {
    if (steps[idx]) {
      setCurrentTime(steps[idx].t)
      setIsPlaying(false)
    }
  }

  // Prev / Next Step buttons
  const handlePrevStep = () => {
    if (activeStepIdx > 0) {
      jumpToStep(activeStepIdx - 1)
    } else {
      setCurrentTime(0)
    }
  }

  const handleNextStep = () => {
    if (activeStepIdx < steps.length - 1) {
      jumpToStep(activeStepIdx + 1)
    } else {
      setCurrentTime(maxTime)
    }
  }

  // Reset to start
  const handleRestart = () => {
    setCurrentTime(0)
    setIsPlaying(true)
  }

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md flex items-center justify-center p-3 sm:p-6 animate-in fade-in duration-200">
      <div className="bg-card border border-border shadow-2xl rounded-2xl w-full max-w-6xl max-h-[94vh] flex flex-col overflow-hidden">
        {/* HEADER BAR */}
        <div className="px-5 py-4 border-b border-border/60 flex items-center justify-between gap-4 bg-muted/20">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <Sparkles size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <Badge
                  variant={
                    run.status === 'pass'
                      ? 'done'
                      : run.status === 'warn'
                      ? 'rejected'
                      : 'failed'
                  }
                  className="text-[10px] uppercase font-mono px-2 py-0.5"
                >
                  {run.status}
                </Badge>
                <h2 className="text-sm font-bold text-foreground truncate max-w-md sm:max-w-xl">
                  {run.title}
                </h2>
                <ScenarioExplanationPopover scenarioId={run.scenario} />
              </div>
              <div className="text-xs text-muted-foreground flex items-center gap-2 mt-0.5 font-mono">
                <span>ID: {run.scenario}</span>
                <span>·</span>
                <span>Durasi: {fmtSec(maxTime)}</span>
                {run.request_ids?.length > 0 && (
                  <>
                    <span>·</span>
                    <button
                      type="button"
                      className="text-primary hover:underline"
                      onClick={() => openRequest?.(run.request_ids[0])}
                    >
                      {run.request_ids[0]}
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="rounded-full h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <X size={16} />
          </Button>
        </div>

        {/* MAIN BODY: CANVAS (LEFT) + EVENT LOG (RIGHT) */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_320px] overflow-hidden min-h-[480px]">
          {/* LEFT: ANIMATED CANVAS & STORYBOARD */}
          <div className="flex flex-col bg-background/40 p-4 sm:p-5 overflow-y-auto">
            {/* SVG VISUALIZATION CANVAS */}
            <div className="relative w-full aspect-[16/9] min-h-[290px] max-h-[380px] bg-slate-950 rounded-xl border border-border/80 shadow-inner overflow-hidden flex items-center justify-center">
              {/* Dynamic Grid Background */}
              <div
                className="absolute inset-0 opacity-15 pointer-events-none"
                style={{
                  backgroundImage:
                    'radial-gradient(circle at 1px 1px, #94a3b8 1px, transparent 0)',
                  backgroundSize: '24px 24px',
                }}
              />

              {/* Status Header Overlay */}
              <div className="absolute top-3 left-4 z-20 flex items-center gap-2 text-xs">
                <span className="flex h-2 w-2 relative">
                  <span
                    className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                      isPlaying ? 'bg-emerald-400' : 'bg-amber-400'
                    }`}
                  />
                  <span
                    className={`relative inline-flex rounded-full h-2 w-2 ${
                      isPlaying ? 'bg-emerald-500' : 'bg-amber-500'
                    }`}
                  />
                </span>
                <span className="font-mono text-slate-300 font-semibold tracking-wide text-[11px]">
                  REPLAY TIME: {currentTime.toFixed(1)}s / {maxTime.toFixed(1)}s
                </span>
                {simulationState.flameService && (
                  <Badge variant="destructive" className="text-[10px] font-mono py-0 animate-pulse ml-2">
                    {simulationState.flameType === 'sigkill' ? '💥 OOM SIGKILL' : '⚠️ SIGTERM DRAIN'}
                  </Badge>
                )}
              </div>

              {/* SVG GRAPH PIPELINE */}
              <svg
                viewBox="0 0 960 400"
                className="w-full h-full select-none"
                preserveAspectRatio="xMidYMid meet"
              >
                <defs>
                  {/* Glowing line filters */}
                  <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                  <filter id="glow-rose" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="6" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                  <filter id="glow-amber" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>

                  {/* Gradient paths */}
                  <linearGradient id="pipe-cyan" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.8" />
                  </linearGradient>
                  <linearGradient id="pipe-down" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8" />
                  </linearGradient>
                </defs>

                {/* --- CONNECTING PIPELINES (PATHS) --- */}
                {/* 1. Client -> Guardrails */}
                <path
                  d="M 120 120 L 210 120"
                  stroke={flowPacket.stage === 'wire-client-guardrails' ? '#38bdf8' : '#334155'}
                  strokeWidth={flowPacket.stage === 'wire-client-guardrails' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'wire-client-guardrails' ? 'none' : '4 4'}
                  filter={flowPacket.stage === 'wire-client-guardrails' ? 'url(#glow-emerald)' : undefined}
                />
                {/* 2. Guardrails -> Extraction */}
                <path
                  d="M 310 120 L 400 120"
                  stroke={
                    flowPacket.stage === 'wire-guardrails-extraction'
                      ? '#38bdf8'
                      : simulationState.services.extraction.status === 'crashed'
                      ? '#f43f5e'
                      : '#334155'
                  }
                  strokeWidth={flowPacket.stage === 'wire-guardrails-extraction' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'wire-guardrails-extraction' ? 'none' : '4 4'}
                  filter={flowPacket.stage === 'wire-guardrails-extraction' ? 'url(#glow-emerald)' : undefined}
                />
                {/* 3. Extraction -> Structuring */}
                <path
                  d="M 500 120 L 590 120"
                  stroke={flowPacket.stage === 'structuring' ? '#818cf8' : '#334155'}
                  strokeWidth={flowPacket.stage === 'structuring' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'structuring' ? 'none' : '4 4'}
                  filter={flowPacket.stage === 'structuring' ? 'url(#glow-emerald)' : undefined}
                />
                {/* 4. Structuring -> Scoring */}
                <path
                  d="M 690 120 L 780 120"
                  stroke={flowPacket.stage === 'scoring' ? '#c084fc' : '#334155'}
                  strokeWidth={flowPacket.stage === 'scoring' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'scoring' ? 'none' : '4 4'}
                  filter={flowPacket.stage === 'scoring' ? 'url(#glow-emerald)' : undefined}
                />
                {/* 5. Scoring -> Callback Webhook */}
                <path
                  d="M 835 165 L 835 255"
                  stroke={flowPacket.stage === 'callback-dispatch' || simulationState.services.callback.status !== 'idle' ? '#10b981' : '#334155'}
                  strokeWidth={flowPacket.stage === 'callback-dispatch' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'callback-dispatch' ? 'none' : '4 4'}
                  filter={flowPacket.stage === 'callback-dispatch' ? 'url(#glow-emerald)' : undefined}
                />

                {/* Lower connections: Extraction down to PostgreSQL & Outbox */}
                <path
                  d="M 450 165 L 450 255"
                  stroke={
                    flowPacket.stage === 'dropping-to-db' || flowPacket.stage === 'db-orphan-locked'
                      ? '#f59e0b'
                      : flowPacket.stage === 'reaper-retry'
                      ? '#38bdf8'
                      : '#334155'
                  }
                  strokeWidth={flowPacket.stage === 'dropping-to-db' || flowPacket.stage === 'db-orphan-locked' ? '4' : '2.5'}
                  strokeDasharray={flowPacket.stage === 'dropping-to-db' ? 'none' : '3 3'}
                  fill="none"
                  filter={flowPacket.stage === 'dropping-to-db' || flowPacket.stage === 'db-orphan-locked' ? 'url(#glow-amber)' : undefined}
                />
                <path
                  d="M 640 165 C 640 215, 560 235, 540 255"
                  stroke="#334155"
                  strokeWidth="2.5"
                  strokeDasharray="3 3"
                  fill="none"
                />

                {/* Outbox Relay connection to Callback Webhook */}
                <path
                  d="M 600 305 L 780 305"
                  stroke={
                    flowPacket.stage === 'outbox-relay'
                      ? '#10b981'
                      : simulationState.services.outbox.status === 'drained' || simulationState.services.callback.status !== 'idle'
                      ? '#10b981'
                      : '#334155'
                  }
                  strokeWidth={flowPacket.stage === 'outbox-relay' ? '4' : '3'}
                  strokeDasharray={flowPacket.stage === 'outbox-relay' ? 'none' : '4 4'}
                  fill="none"
                  filter={flowPacket.stage === 'outbox-relay' ? 'url(#glow-emerald)' : undefined}
                />

                {/* --- NODE 1: CLIENT / INGRESS (x: 40, y: 75, w: 90, h: 90) --- */}
                <g transform="translate(35, 75)">
                  <rect
                    width="85"
                    height="90"
                    rx="12"
                    fill="#0f172a"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    className="transition-all duration-300"
                  />
                  <circle cx="42" cy="35" r="18" fill="#1e293b" />
                  <path
                    d="M 34 35 L 50 35 M 44 29 L 50 35 L 44 41"
                    stroke="#38bdf8"
                    strokeWidth="2.5"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <text x="42" y="66" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Client
                  </text>
                  <text x="42" y="80" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="monospace">
                    POST /extract
                  </text>
                </g>

                {/* --- NODE 2: GUARDRAILS (x: 210, y: 75) --- */}
                <g transform="translate(210, 75)">
                  <rect
                    width="100"
                    height="90"
                    rx="12"
                    fill="#0f172a"
                    stroke="#0ea5e9"
                    strokeWidth="2"
                  />
                  <circle cx="50" cy="35" r="18" fill="#1e293b" />
                  <path
                    d="M 42 30 L 50 25 L 58 30 C 58 40 50 46 50 46 C 50 46 42 40 42 30 Z"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2"
                  />
                  <text x="50" y="66" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Guardrails
                  </text>
                  <text x="50" y="80" textAnchor="middle" fill="#0ea5e9" fontSize="9" fontFamily="monospace">
                    :8031 (Validasi)
                  </text>
                </g>

                {/* --- NODE 3: EXTRACTION (PADDLEOCR) (x: 400, y: 75) --- */}
                <g transform="translate(400, 75)">
                  {simulationState.services.extraction.status === 'crashed' && (
                    <rect
                      x="-6"
                      y="-6"
                      width="112"
                      height="102"
                      rx="16"
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="3"
                      filter="url(#glow-rose)"
                      className="animate-pulse"
                    />
                  )}
                  <rect
                    width="100"
                    height="90"
                    rx="12"
                    fill={
                      simulationState.services.extraction.status === 'crashed'
                        ? '#450a0a'
                        : simulationState.services.extraction.status === 'draining'
                        ? '#451a03'
                        : '#0f172a'
                    }
                    stroke={
                      simulationState.services.extraction.status === 'crashed'
                        ? '#ef4444'
                        : simulationState.services.extraction.status === 'draining'
                        ? '#f59e0b'
                        : '#3b82f6'
                    }
                    strokeWidth="2"
                    className="transition-all duration-300"
                  />
                  <circle cx="50" cy="35" r="18" fill="#1e293b" />
                  {simulationState.services.extraction.status === 'crashed' ? (
                    <path
                      d="M 42 27 L 58 43 M 58 27 L 42 43"
                      stroke="#f43f5e"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                  ) : (
                    <path
                      d="M 40 32 L 60 32 M 40 38 L 55 38 M 40 44 L 50 44"
                      stroke="#38bdf8"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  )}
                  <text x="50" y="66" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Extraction
                  </text>
                  <text
                    x="50"
                    y="80"
                    textAnchor="middle"
                    fill={
                      simulationState.services.extraction.status === 'crashed'
                        ? '#f87171'
                        : '#38bdf8'
                    }
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {simulationState.services.extraction.label}
                  </text>
                </g>

                {/* --- NODE 4: STRUCTURING (x: 590, y: 75) --- */}
                <g transform="translate(590, 75)">
                  {simulationState.services.structuring.status === 'crashed' && (
                    <rect
                      x="-6"
                      y="-6"
                      width="112"
                      height="102"
                      rx="16"
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="3"
                      filter="url(#glow-rose)"
                      className="animate-pulse"
                    />
                  )}
                  <rect
                    width="100"
                    height="90"
                    rx="12"
                    fill="#0f172a"
                    stroke={
                      simulationState.services.structuring.status === 'crashed'
                        ? '#ef4444'
                        : '#6366f1'
                    }
                    strokeWidth="2"
                  />
                  <circle cx="50" cy="35" r="18" fill="#1e293b" />
                  <path
                    d="M 40 27 L 60 27 L 60 43 L 40 43 Z M 40 35 L 60 35 M 50 27 L 50 43"
                    stroke="#a5b4fc"
                    strokeWidth="2"
                    fill="none"
                  />
                  <text x="50" y="66" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Structuring
                  </text>
                  <text x="50" y="80" textAnchor="middle" fill="#818cf8" fontSize="9" fontFamily="monospace">
                    {simulationState.services.structuring.label}
                  </text>
                </g>

                {/* --- NODE 5: SCORING (x: 780, y: 75) --- */}
                <g transform="translate(780, 75)">
                  <rect
                    width="100"
                    height="90"
                    rx="12"
                    fill="#0f172a"
                    stroke="#8b5cf6"
                    strokeWidth="2"
                  />
                  <circle cx="50" cy="35" r="18" fill="#1e293b" />
                  <path
                    d="M 40 42 L 46 42 M 50 30 L 50 42 M 56 36 L 56 42 M 62 25 L 62 42"
                    stroke="#c084fc"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />
                  <text x="50" y="66" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Scoring
                  </text>
                  <text x="50" y="80" textAnchor="middle" fill="#c084fc" fontSize="9" fontFamily="monospace">
                    :8033 (Skor)
                  </text>
                </g>

                {/* --- NODE 6: POSTGRESQL & TRANSACTIONAL OUTBOX (x: 400, y: 260, w: 200, h: 100) --- */}
                <g transform="translate(400, 255)">
                  {simulationState.reaperActive && (
                    <rect
                      x="-6"
                      y="-6"
                      width="212"
                      height="112"
                      rx="18"
                      fill="none"
                      stroke="#f59e0b"
                      strokeWidth="2.5"
                      filter="url(#glow-amber)"
                      className="animate-pulse"
                    />
                  )}
                  <rect
                    width="200"
                    height="100"
                    rx="14"
                    fill="#0f172a"
                    stroke={simulationState.reaperActive ? '#f59e0b' : '#38bdf8'}
                    strokeWidth="2"
                  />
                  <circle cx="45" cy="45" r="22" fill="#1e293b" />
                  <path
                    d="M 33 37 C 33 34, 57 34, 57 37 C 57 40, 33 40, 33 37 Z M 33 44 C 33 47, 57 47, 57 44 M 33 51 C 33 54, 57 54, 57 51"
                    stroke={simulationState.reaperActive ? '#fbbf24' : '#38bdf8'}
                    strokeWidth="2"
                    fill="none"
                  />
                  <text x="80" y="38" fill="#f8fafc" fontSize="12" fontWeight="bold">
                    PostgreSQL DB
                  </text>
                  <text x="80" y="54" fill="#94a3b8" fontSize="10">
                    Outbox &amp; Auto-Recovery
                  </text>
                  <text
                    x="80"
                    y="72"
                    fill={simulationState.reaperActive ? '#fbbf24' : '#38bdf8'}
                    fontSize="9.5"
                    fontFamily="monospace"
                    fontWeight="semibold"
                  >
                    {simulationState.reaperActive ? '⚡ AUTO-RECOVERY AKTIF' : '✓ Outbox Buffer Bersih'}
                  </text>
                </g>

                {/* --- NODE 7: CALLBACK WEBHOOK (x: 790, y: 260) --- */}
                <g transform="translate(780, 255)">
                  <rect
                    width="120"
                    height="100"
                    rx="14"
                    fill={
                      simulationState.services.callback.status === 'done'
                        ? '#064e3b'
                        : simulationState.services.callback.status === 'failed_handled' ||
                          simulationState.services.callback.status === 'warn'
                        ? '#451a03'
                        : '#0f172a'
                    }
                    stroke={
                      simulationState.services.callback.status === 'done'
                        ? '#10b981'
                        : simulationState.services.callback.status === 'failed_handled' ||
                          simulationState.services.callback.status === 'warn'
                        ? '#f59e0b'
                        : '#475569'
                    }
                    strokeWidth="2"
                  />
                  <circle cx="60" cy="40" r="20" fill="#1e293b" />
                  <path
                    d="M 50 40 L 57 47 L 70 33"
                    stroke={
                      simulationState.services.callback.status === 'done'
                        ? '#10b981'
                        : simulationState.services.callback.status === 'failed_handled'
                        ? '#f59e0b'
                        : '#94a3b8'
                    }
                    strokeWidth="2.5"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <text x="60" y="74" textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="bold">
                    Callback Webhook
                  </text>
                  <text
                    x="60"
                    y="88"
                    textAnchor="middle"
                    fill={
                      simulationState.services.callback.status === 'done'
                        ? '#34d399'
                        : simulationState.services.callback.status === 'failed_handled'
                        ? '#fbbf24'
                        : '#94a3b8'
                    }
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {simulationState.services.callback.label}
                  </text>
                </g>

                {/* --- SYNCHRONIZED DATA PACKET ORB (SINGLE CLEAR FLOW) --- */}
                {flowPacket && flowPacket.visible && (
                  <g
                    transform={`translate(${flowPacket.x}, ${flowPacket.y})`}
                    className="pointer-events-none"
                  >
                    {/* 1. Pulsing Outer Aura */}
                    <circle
                      r={flowPacket.isLocked ? 22 : 18}
                      fill={flowPacket.color}
                      opacity={flowPacket.isLocked ? 0.35 : 0.22}
                      className={flowPacket.isLocked ? 'animate-pulse' : 'animate-ping'}
                      style={{ animationDuration: flowPacket.isLocked ? '1.5s' : '1.8s' }}
                    />

                    {/* 2. Soft Glow Halo */}
                    <circle
                      r="14"
                      fill={flowPacket.color}
                      opacity="0.45"
                      filter="url(#glow-emerald)"
                    />

                    {/* 3. Status Ring */}
                    <circle
                      r="9"
                      fill="none"
                      stroke={flowPacket.color}
                      strokeWidth="2.5"
                      strokeDasharray={flowPacket.isLocked ? '3 3' : 'none'}
                      className={flowPacket.isLocked ? 'animate-spin' : ''}
                      style={{ animationDuration: '3s' }}
                    />

                    {/* 4. Core Luminous Orb */}
                    <circle
                      r="5.5"
                      fill="#ffffff"
                      stroke={flowPacket.color}
                      strokeWidth="3"
                    />

                    {/* 5. Floating Badge right above packet */}
                    <g transform="translate(0, -25)">
                      <rect
                        x={-Math.max(52, (flowPacket.label.length * 6.5) / 2 + 12)}
                        y="-12"
                        width={Math.max(104, flowPacket.label.length * 6.5 + 24)}
                        height="24"
                        rx="12"
                        fill="#090d16"
                        stroke={flowPacket.color}
                        strokeWidth="1.8"
                      />
                      <circle
                        cx={-Math.max(52, (flowPacket.label.length * 6.5) / 2 + 12) + 10}
                        cy="0"
                        r="3.5"
                        fill={flowPacket.color}
                      />
                      <text
                        x="5"
                        y="3.5"
                        textAnchor="middle"
                        fill="#f8fafc"
                        fontSize="10"
                        fontWeight="bold"
                        fontFamily="system-ui, sans-serif"
                      >
                        {flowPacket.label}
                      </text>
                    </g>

                    {/* 6. Locked Sublabel below packet */}
                    {flowPacket.isLocked && (
                      <g transform="translate(0, 26)">
                        <rect
                          x="-85"
                          y="-9"
                          width="170"
                          height="19"
                          rx="6"
                          fill="#18181b"
                          stroke="#f59e0b"
                          strokeWidth="1.2"
                          opacity="0.95"
                        />
                        <text
                          x="0"
                          y="4"
                          textAnchor="middle"
                          fill="#fbbf24"
                          fontSize="9"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          {flowPacket.sublabel}
                        </text>
                      </g>
                    )}
                  </g>
                )}
              </svg>
            </div>

            {/* LIVE STORYBOARD / NARRATIVE BANNER */}
            <div className="mt-3.5 p-3.5 rounded-xl border border-border/80 bg-card shadow-xs flex items-start justify-between gap-3 flex-wrap sm:flex-nowrap">
              <div className="flex items-start gap-3 min-w-0">
                <div className="h-7 w-7 rounded-lg bg-sky-500/10 border border-sky-500/25 flex items-center justify-center text-sky-500 shrink-0 mt-0.5">
                  <Info size={15} />
                </div>
                <div className="space-y-0.5 min-w-0">
                  <div className="text-[11px] font-semibold text-foreground uppercase tracking-wide flex items-center gap-2">
                    <span>Alur Kejadian (Narrative):</span>
                    <span className="font-mono text-muted-foreground lowercase">
                      [t+{currentTime.toFixed(1)}s]
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground leading-relaxed">
                    {simulationState.narrative}
                  </div>
                </div>
              </div>

              {/* Status packet badge widget */}
              {flowPacket && (
                <div className="shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/50 border border-border/80 text-xs font-mono">
                  <span className="text-muted-foreground text-[10px] uppercase">Alur Aktif:</span>
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full animate-pulse shadow-sm"
                    style={{ backgroundColor: flowPacket.color }}
                  />
                  <span className="font-bold text-foreground text-[11px]">{flowPacket.label}</span>
                </div>
              )}
            </div>

            {/* PLAYBACK SCRUBBER & CONTROLS */}
            <div className="mt-4 p-4 rounded-xl border border-border/70 bg-card space-y-3">
              {/* Scrub timeline slider with keyframe ticks */}
              <div className="space-y-1.5 relative">
                <div className="flex justify-between text-[11px] font-mono text-muted-foreground">
                  <span>0.0s</span>
                  <span className="text-foreground font-bold">{currentTime.toFixed(1)}s</span>
                  <span>{maxTime.toFixed(1)}s</span>
                </div>

                <div className="relative flex items-center">
                  <input
                    type="range"
                    min="0"
                    max={maxTime}
                    step="0.1"
                    value={currentTime}
                    onChange={handleSeek}
                    className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                  />

                  {/* Event Milestone markers along the slider */}
                  <div className="absolute inset-x-0 top-0 h-2 pointer-events-none flex">
                    {steps.map((s, i) => {
                      const pct = Math.min(100, Math.max(0, (s.t / maxTime) * 100))
                      const isDanger = s.text.toLowerCase().includes('sigkill') || s.text.toLowerCase().includes('stop')
                      const isReaper = s.text.toLowerCase().includes('reaper') || s.text.toLowerCase().includes('lease')
                      return (
                        <div
                          key={i}
                          style={{ left: `${pct}%` }}
                          className={`absolute w-1.5 h-2 -translate-x-1/2 rounded-full ${
                            isDanger
                              ? 'bg-rose-500 ring-2 ring-rose-500/30'
                              : isReaper
                              ? 'bg-amber-400 ring-2 ring-amber-400/30'
                              : 'bg-primary/40'
                          }`}
                          title={`t+${s.t}s: ${s.text}`}
                        />
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Control Buttons Toolbar */}
              <div className="flex items-center justify-between gap-2 flex-wrap pt-1">
                {/* Play / Pause / Restart / Step navigation */}
                <div className="flex items-center gap-1.5">
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={handleRestart}
                    title="Ulangi dari awal"
                    className="h-8 w-8 rounded-lg"
                  >
                    <RotateCcw size={13} />
                  </Button>

                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={handlePrevStep}
                    disabled={activeStepIdx <= 0}
                    title="Langkah Sebelumnya"
                    className="h-8 w-8 rounded-lg"
                  >
                    <ChevronLeft size={15} />
                  </Button>

                  <Button
                    type="button"
                    variant={isPlaying ? 'secondary' : 'default'}
                    size="sm"
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="h-8 px-3.5 text-xs font-semibold gap-1.5 rounded-lg shadow-xs"
                  >
                    {isPlaying ? <Pause size={13} /> : <Play size={13} />}
                    <span>{isPlaying ? 'Jeda' : 'Putar'}</span>
                  </Button>

                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={handleNextStep}
                    disabled={activeStepIdx >= steps.length - 1}
                    title="Langkah Berikutnya"
                    className="h-8 w-8 rounded-lg"
                  >
                    <ChevronRight size={15} />
                  </Button>
                </div>

                {/* Speed toggle pills */}
                <div className="flex items-center gap-1 bg-muted/60 p-0.5 rounded-lg border border-border/50">
                  <span className="text-[10px] text-muted-foreground font-mono px-1.5">
                    Speed:
                  </span>
                  {[0.5, 1, 1.5, 2, 4].map((spd) => (
                    <button
                      key={spd}
                      type="button"
                      onClick={() => setSpeed(spd)}
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-md transition-all ${
                        speed === spd
                          ? 'bg-background text-foreground font-bold shadow-xs'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: SYNCHRONIZED EVENTS & CONTRACT CHECKS */}
          <div className="border-t lg:border-t-0 lg:border-l border-border/70 flex flex-col bg-muted/10 h-full overflow-hidden">
            {/* Tab switch: Steps vs Checks */}
            <div className="p-3 border-b border-border/60 flex items-center justify-between gap-2 bg-muted/20">
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant={activeTab === 'steps' ? 'secondary' : 'ghost'}
                  size="xs"
                  onClick={() => setActiveTab('steps')}
                  className="h-7 text-xs font-medium"
                >
                  <span>Langkah ({steps.length})</span>
                </Button>
                <Button
                  type="button"
                  variant={activeTab === 'checks' ? 'secondary' : 'ghost'}
                  size="xs"
                  onClick={() => setActiveTab('checks')}
                  className="h-7 text-xs font-medium"
                >
                  <span>Cek Kontrak ({checks.length})</span>
                </Button>
              </div>

              <div className="text-[10px] font-mono text-muted-foreground">
                Langkah {activeStepIdx + 1}/{steps.length || 1}
              </div>
            </div>

            {/* List content container */}
            <div
              ref={stepsContainerRef}
              className="flex-1 p-3 overflow-y-auto space-y-2 text-xs font-mono scrollbar-thin"
            >
              {activeTab === 'steps' ? (
                steps.map((s, idx) => {
                  const isActive = idx === activeStepIdx
                  const isPast = s.t <= currentTime
                  const isDisrupt =
                    s.text.toLowerCase().includes('sigkill') ||
                    s.text.toLowerCase().includes('stop') ||
                    s.text.toLowerCase().includes('mati')

                  return (
                    <div
                      key={idx}
                      data-step-idx={idx}
                      onClick={() => jumpToStep(idx)}
                      className={`p-2.5 rounded-xl border transition-all cursor-pointer select-none ${
                        isActive
                          ? 'border-primary bg-primary/10 shadow-sm ring-1 ring-primary/30 text-foreground'
                          : isPast
                          ? 'border-border/60 bg-card/60 text-foreground/90 hover:bg-card'
                          : 'border-transparent text-muted-foreground/60 hover:text-muted-foreground'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span
                          className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                            isDisrupt
                              ? 'bg-rose-500/20 text-rose-400 font-bold'
                              : isActive
                              ? 'bg-primary/20 text-primary font-bold'
                              : 'bg-muted text-muted-foreground'
                          }`}
                        >
                          t+{s.t}s
                        </span>
                        {isActive && (
                          <span className="flex h-1.5 w-1.5 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary" />
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] font-sans leading-snug">
                        {s.text}
                      </div>
                    </div>
                  )
                })
              ) : (
                checks.map((c, idx) => {
                  const isEvaluated = c.t <= currentTime
                  return (
                    <div
                      key={idx}
                      className={`p-2.5 rounded-xl border transition-all ${
                        isEvaluated
                          ? c.level === 'pass'
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : c.level === 'warn'
                            ? 'border-amber-500/30 bg-amber-500/5'
                            : 'border-rose-500/30 bg-rose-500/5'
                          : 'border-border/40 opacity-40'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <Badge
                          variant={
                            !isEvaluated
                              ? 'idle'
                              : c.level === 'pass'
                              ? 'done'
                              : c.level === 'warn'
                              ? 'rejected'
                              : 'failed'
                          }
                          className="text-[9px] font-mono px-1.5 py-0"
                        >
                          {!isEvaluated ? 'MENUNGGU' : c.level.toUpperCase()}
                        </Badge>
                        <span className="text-[10px] font-mono text-muted-foreground">
                          t+{c.t}s
                        </span>
                      </div>
                      <div className="text-[11px] font-sans font-semibold text-foreground">
                        {c.label}
                      </div>
                      {c.detail && (
                        <div className="text-[10px] font-mono text-muted-foreground mt-1 break-all">
                          {c.detail}
                        </div>
                      )}
                    </div>
                  )
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
