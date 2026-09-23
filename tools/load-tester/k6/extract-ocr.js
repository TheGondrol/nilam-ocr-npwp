// Skenario k6: kirim NPWP ke guardrails /v1/extract-ocr dengan laju kedatangan tetap
// (open model), catat status HTTP-nya, dan laporkan tiap sampel ke tracker supaya
// jumlah 200/202/4xx/5xx dan waktu end-to-end (sampai callback SCORING) bisa dihitung.
//
// Env (diisi tracker, atau manual lewat `k6 run -e ...`):
//   RUN_ID        id run, dipakai sebagai prefiks request_id: LT_<RUN_ID>_<vu>-<iter>
//   TARGET        base URL guardrails, mis. http://guardrails:8031
//   TRACKER       base URL tracker, mis. http://host.docker.internal:8090 (kosong = tanpa lapor)
//   RATE          request per detik (default 1)
//   DURATION      lama pengiriman, format k6 (default 60s)
//   MODE          constant | ramp (ramp: naik dari 0 ke RATE selama separuh DURATION)
//   IMAGES        daftar nama file di /images, dipisah koma; dipakai bergiliran
//   WAIT_SECONDS  PIPELINE_WAIT_SECONDS guardrails (default 15), untuk timeout dan jumlah VU
//   API_KEY       diisi kalau guardrails tidak memakai AUTH_DISABLED
import http from 'k6/http'
import { Counter, Trend } from 'k6/metrics'

const RUN = __ENV.RUN_ID || `manual${Date.now().toString(36)}`
const TARGET = (__ENV.TARGET || 'http://127.0.0.1:8031').replace(/\/$/, '')
const TRACKER = (__ENV.TRACKER || '').replace(/\/$/, '')
const RATE = Number(__ENV.RATE || 1)
const DURATION = __ENV.DURATION || '60s'
const MODE = __ENV.MODE || 'constant'
const WAIT = Number(__ENV.WAIT_SECONDS || 15)
const API_KEY = __ENV.API_KEY || ''

const names = (__ENV.IMAGES || 'npwp1.jpg')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean)
const images = names.map((name) => ({ name, data: open(`/images/${name}`, 'b') }))

// Koneksi ditahan guardrails sampai WAIT detik, jadi VU yang sibuk bersamaan ~ RATE x (WAIT + jeda).
const durationSeconds = parseDuration(DURATION)
const vus = Math.max(10, Math.ceil(RATE * (WAIT + 10)))

// k6 hanya menerima laju bulat, jadi laju pecahan (mis. 0.5 rps) dinyatakan per 10 detik (5 per 10s).
const scale = Number.isInteger(RATE) ? 1 : 10
const rateInt = Math.max(1, Math.round(RATE * scale))
const timeUnit = `${scale}s`

const scenario =
  MODE === 'ramp'
    ? {
        executor: 'ramping-arrival-rate',
        startRate: 0,
        timeUnit,
        preAllocatedVUs: vus,
        maxVUs: vus * 2,
        stages: [
          { target: rateInt, duration: `${Math.max(1, Math.floor(durationSeconds / 2))}s` },
          { target: rateInt, duration: `${Math.max(1, Math.ceil(durationSeconds / 2))}s` },
        ],
      }
    : {
        executor: 'constant-arrival-rate',
        rate: rateInt,
        timeUnit,
        duration: DURATION,
        preAllocatedVUs: vus,
        maxVUs: vus * 2,
      }

export const options = {
  scenarios: { submit: scenario },
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'max'],
  thresholds: {
    // Hanya supaya sub-metrik extract-ocr (tanpa laporan ke tracker) muncul di ringkasan;
    // bukan untuk menggagalkan run.
    'http_req_duration{name:extract-ocr}': ['p(95)<3600000'],
    'http_req_failed{name:extract-ocr}': ['rate<=1'],
  },
}

const status200 = new Counter('extract_200')
const status202 = new Counter('extract_202')
const status4xx = new Counter('extract_4xx')
const status5xx = new Counter('extract_5xx')
const statusTimeout = new Counter('extract_timeout')
const extractDuration = new Trend('extract_duration', true)

export default function () {
  const image = images[(__VU + __ITER) % images.length]
  const requestId = `LT_${RUN}_${__VU}-${__ITER}`
  const startedAt = Date.now()
  const res = http.post(
    `${TARGET}/v1/extract-ocr`,
    {
      request_id: requestId,
      document_type: 'npwp',
      file: http.file(image.data, image.name, contentType(image.name)),
    },
    {
      timeout: `${WAIT + 30}s`,
      tags: { name: 'extract-ocr' },
      headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
    }
  )
  const elapsedMs = Date.now() - startedAt
  extractDuration.add(elapsedMs)

  let body = null
  try {
    body = res.json()
  } catch (_) {
    body = null
  }
  if (res.status === 200) status200.add(1)
  else if (res.status === 202) status202.add(1)
  else if (res.status === 0) statusTimeout.add(1)
  else if (res.status >= 500) status5xx.add(1)
  else if (res.status >= 400) status4xx.add(1)

  if (TRACKER) {
    http.post(
      `${TRACKER}/api/loadtest/${RUN}/samples`,
      JSON.stringify({
        request_id: requestId,
        image: image.name,
        status: res.status,
        job_status: body && body.job_status ? body.job_status : null,
        errors: body && body.errors ? body.errors : null,
        message: body && body.message ? body.message : res.error || null,
        started_at: startedAt / 1000,
        elapsed_ms: elapsedMs,
      }),
      { headers: { 'Content-Type': 'application/json' }, tags: { name: 'tracker-sample' }, timeout: '5s' }
    )
  }
}

export function handleSummary(data) {
  return { [`/out/${RUN}.json`]: JSON.stringify(data, null, 2) }
}

function contentType(name) {
  const lower = name.toLowerCase()
  if (lower.endsWith('.png')) return 'image/png'
  if (lower.endsWith('.pdf')) return 'application/pdf'
  return 'image/jpeg'
}

function parseDuration(text) {
  const m = /^(\d+)(ms|s|m|h)?$/.exec(text.trim())
  if (!m) return 60
  const n = Number(m[1])
  return { ms: n / 1000, s: n, m: n * 60, h: n * 3600 }[m[2] || 's']
}
