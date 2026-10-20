// Skenario k6 untuk performance test di GKE (tracker menu "Performance Test"): dijalankan sebagai Job di namespace
// pipeline, menembak Service orchestrator langsung (bukan lewat port-forward), endpoint -test supaya data
// Orkestrasi pusat tidak tercemar (tabel nilam_testing_*, tanpa callback).
//
// Satu baris log per request (`PT|<json>`), dibaca tracker lewat `kubectl logs -f`, karena pod tidak bisa
// menjangkau tracker di laptop. Ringkasan akhir k6 di stdout sebagai `PTSUMMARY|<json>`. Jalankan dengan
// `--log-format=raw` supaya console.log keluar apa adanya.
//
// Env (diisi tracker):
//   RUN_ID        id run; endpoint -test membuat request_id TEST_<RUN_ID>_<uuid>
//   TARGET        base URL orchestrator, mis. http://nilam-ocr-npwp-orchestrator:8034 (Service yang dipanggil Orkestrasi pusat)
//   ENDPOINT      default /v1/extract-ocr-test
//   STAGES        JSON [[target, detik], ...]: laju kedatangan per 10 detik (resolusi 0.1 rps) tiap tahap,
//                 dijalankan berurutan oleh ramping-arrival-rate (naik/turun linear di dalam satu tahap)
//   START_RATE    laju awal per 10 detik (default 0)
//   PRE_VUS / MAX_VUS  jumlah VU; koneksi ditahan orchestrator sampai WAIT_SECONDS, jadi VU ~ laju x (WAIT + 10)
//   IMAGES        nama file di /perf (ConfigMap), dipisah koma, dipakai bergiliran
//   WAIT_SECONDS  PIPELINE_WAIT_SECONDS orchestrator (default 15), untuk timeout request
//   API_KEY       header X-API-Key (dari Secret pipeline)
//   PIPELINE_NAME_SEQUENCE  JSON array service yang dijalankan (kosong = pipeline penuh)
//   SLO_ERROR_RATE, SLO_ENTRY_P95_MS  threshold k6 (hanya penanda lulus/gagal di ringkasan, tidak menghentikan run:
//                 penghentian otomatis diputuskan tracker per jendela waktu)
import http from 'k6/http'
import { Counter, Rate, Trend } from 'k6/metrics'

const RUN = __ENV.RUN_ID || `manual${Date.now().toString(36)}`
const TARGET = (__ENV.TARGET || 'http://nilam-ocr-npwp-orchestrator:8034').replace(/\/$/, '')
const ENDPOINT = __ENV.ENDPOINT || '/v1/extract-ocr-test'
const WAIT = Number(__ENV.WAIT_SECONDS || 15)
const API_KEY = __ENV.API_KEY || ''
const SEQUENCE = __ENV.PIPELINE_NAME_SEQUENCE || ''
const STAGES = JSON.parse(__ENV.STAGES || '[[2, 60]]')
const PRE_VUS = Number(__ENV.PRE_VUS || 20)
const MAX_VUS = Number(__ENV.MAX_VUS || 100)
const SLO_ERROR_RATE = Number(__ENV.SLO_ERROR_RATE || 0.01)
const SLO_ENTRY_P95_MS = Number(__ENV.SLO_ENTRY_P95_MS || WAIT * 1000)

const images = (__ENV.IMAGES || 'npwp1.jpg')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)
  .map((name) => ({ name, data: open(`/perf/${name}`, 'b') }))

export const options = {
  scenarios: {
    submit: {
      executor: 'ramping-arrival-rate',
      startRate: Number(__ENV.START_RATE || 0),
      timeUnit: '10s',
      preAllocatedVUs: PRE_VUS,
      maxVUs: MAX_VUS,
      stages: STAGES.map(([target, seconds]) => ({ target, duration: `${seconds}s` })),
    },
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    server_errors: [`rate<${SLO_ERROR_RATE}`],
    extract_duration: [`p(95)<${SLO_ENTRY_P95_MS}`],
  },
}

const statusCount = {
  ok: new Counter('extract_200'),
  rejected: new Counter('extract_rejected'),
  accepted: new Counter('extract_202'),
  client: new Counter('extract_4xx'),
  server: new Counter('extract_5xx'),
  timeout: new Counter('extract_timeout'),
}
// 5xx dan timeout: kegagalan yang dihitung SLO. 4xx dan dokumen ditolak bukan tanda beban.
const serverErrors = new Rate('server_errors')
const extractDuration = new Trend('extract_duration', true)

export default function () {
  const image = images[(__VU + __ITER) % images.length]
  const localId = `PT_${RUN}_${__VU}-${__ITER}`
  const form = {
    run_id: RUN,
    document_type: 'npwp',
    file: http.file(image.data, image.name, contentType(image.name)),
  }
  if (SEQUENCE) form.pipeline_name_sequence = SEQUENCE
  const startedAt = Date.now()
  const res = http.post(`${TARGET}${ENDPOINT}`, form, {
    timeout: `${WAIT + 30}s`,
    tags: { name: 'extract-ocr' },
    headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
  })
  const elapsedMs = Date.now() - startedAt
  extractDuration.add(elapsedMs)

  let body = null
  try {
    body = res.json()
  } catch (_) {
    body = null
  }
  const rejected = res.status === 400 && body && body.errors === 'DOWNSTREAM_VALIDATION_ERROR'
  if (rejected) statusCount.rejected.add(1)
  else if (res.status === 200) statusCount.ok.add(1)
  else if (res.status === 202) statusCount.accepted.add(1)
  else if (res.status === 0) statusCount.timeout.add(1)
  else if (res.status >= 500) statusCount.server.add(1)
  else if (res.status >= 400) statusCount.client.add(1)
  serverErrors.add(res.status === 0 || res.status >= 500)

  const message = body && body.message ? String(body.message) : res.error || null
  console.log(
    'PT|' +
      JSON.stringify({
        // Endpoint -test menjawab dengan request_id buatannya sendiri (TEST_<RUN>_<uuid>); tanpa jawaban: id lokal.
        r: body && body.request_id ? body.request_id : localId,
        s: res.status,
        t: startedAt / 1000,
        ms: elapsedMs,
        e: body && body.errors ? body.errors : null,
        m: message ? message.slice(0, 160) : null,
        g: body && body.pipeline_last_stage ? body.pipeline_last_stage : null,
      })
  )
}

export function handleSummary(data) {
  const metric = (name) => (data.metrics[name] || {}).values || {}
  const thresholds = {}
  for (const [name, m] of Object.entries(data.metrics)) {
    if (m.thresholds) thresholds[name] = Object.values(m.thresholds).every((t) => t.ok)
  }
  const compact = {
    iterations: metric('iterations').count || 0,
    dropped_iterations: metric('dropped_iterations').count || 0,
    vus_max: metric('vus_max').value || metric('vus_max').max || null,
    http_reqs: metric('http_reqs').count || 0,
    data_sent: metric('data_sent').count || 0,
    data_received: metric('data_received').count || 0,
    extract_duration: metric('extract_duration'),
    server_errors_rate: metric('server_errors').rate,
    thresholds,
  }
  return { stdout: `PTSUMMARY|${JSON.stringify(compact)}\n` }
}

function contentType(name) {
  const lower = name.toLowerCase()
  if (lower.endsWith('.png')) return 'image/png'
  if (lower.endsWith('.pdf')) return 'application/pdf'
  return 'image/jpeg'
}
