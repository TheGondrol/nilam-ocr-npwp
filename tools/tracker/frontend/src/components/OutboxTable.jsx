import { useState } from 'react'
import { Database, AlertTriangle, RefreshCw, HelpCircle, Layers } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from '@/components/ui/table'

function badgeVariantOf(state) {
  switch (state) {
    case 'DELIVERED': return 'done'
    case 'SKIPPED': return 'idle'
    case 'QUEUED':
    case 'CLAIMED':
    case 'RELEASED': return 'processing'
    case 'RETRY': return 'rejected'
    case 'DEAD': return 'failed'
    default: return 'idle'
  }
}

function fmtMs(ms) {
  if (ms == null) return ''
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

export default function OutboxTable({ rows = [], t0 = 0, onRelease, releasing, unavailable }) {
  const [showLegend, setShowLegend] = useState(false)
  const dead = rows.filter((r) => r.state === 'DEAD').length
  const rel = (ts) => `t+${fmtMs((ts - t0) * 1000)}`

  return (
    <Card className="border-border bg-card border-t-2 border-t-amber-500 shadow-xs">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="h-7 w-7 rounded-md bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-600 dark:text-amber-400">
              <Layers size={15} />
            </div>
            <CardTitle className="text-sm font-semibold">
              Tabel nilam_pipeline_outbox
            </CardTitle>
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
              Transactional Relay
            </span>
            {dead > 0 && (
              <Badge variant="failed" className="text-[10px] font-mono">
                {dead} Dead Letter
              </Badge>
            )}
          </div>
          <CardDescription className="text-xs">
            Pesan keluar tiap tahap (handoff ke tahap berikutnya, callback ke Orkestrasi), disimpan dalam transaksi
            yang sama dengan hasil tahapnya lalu dikirim relay
          </CardDescription>
        </div>

        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => setShowLegend(!showLegend)}
            className="gap-1.5 text-xs text-muted-foreground"
          >
            <HelpCircle size={13} />
            <span>{showLegend ? 'Tutup Panduan' : 'Cara Kerja Outbox'}</span>
          </Button>

          {dead > 0 && (
            <Button
              type="button"
              variant="destructive"
              size="xs"
              disabled={releasing}
              onClick={onRelease}
              className="gap-1.5"
            >
              <RefreshCw size={12} className={releasing ? 'animate-spin' : ''} />
              <span>{releasing ? 'Melepaskan…' : `Lepaskan ${dead} Dead Letter`}</span>
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3 pt-1">
        {showLegend && (
          <div className="rounded-lg border border-border/80 bg-background/60 p-4 space-y-3 text-xs leading-relaxed">
            <div className="font-semibold text-foreground flex items-center gap-2">
              <Database size={14} className="text-muted-foreground" />
              <span>Siklus Hidup Baris Transactional Outbox:</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
              <div className="p-3 rounded-md bg-muted/40 border border-border/50">
                <Badge variant="processing" className="text-[10px] mb-1.5">QUEUED</Badge>
                <p className="text-[11px] text-muted-foreground">
                  Ditulis dalam transaksi ACID yang sama dengan hasil tahap. Jika pod mati detik itu juga, data tidak akan hilang.
                </p>
              </div>
              <div className="p-3 rounded-md bg-muted/40 border border-border/50">
                <Badge variant="processing" className="text-[10px] mb-1.5">CLAIMED</Badge>
                <p className="text-[11px] text-muted-foreground">
                  Relay mengklaim baris dengan <code>FOR UPDATE SKIP LOCKED</code> (lease 30 dtk) dan mengirimkan HTTP.
                </p>
              </div>
              <div className="p-3 rounded-md bg-muted/40 border border-border/50">
                <Badge variant="done" className="text-[10px] mb-1.5">DELIVERED</Badge>
                <p className="text-[11px] text-muted-foreground">
                  Penerima menjawab 2xx. Pesan sukses disampaikan dan baris outbox dibersihkan otomatis dari database.
                </p>
              </div>
              <div className="p-3 rounded-md bg-muted/40 border border-border/50">
                <Badge variant="idle" className="text-[10px] mb-1.5">SKIPPED</Badge>
                <p className="text-[11px] text-muted-foreground">
                  Callback tahap tengah pada format result: tidak dikirim (hanya tahap terakhir yang mengirim), baris dihapus.
                </p>
              </div>
              <div className="p-3 rounded-md bg-muted/40 border border-border/50">
                <Badge variant="failed" className="text-[10px] mb-1.5">DEAD / RETRY</Badge>
                <p className="text-[11px] text-muted-foreground">
                  <b>RETRY:</b> Jika 5xx/timeout, diulang backoff. <b>DEAD:</b> Jika 4xx, menetap hingga dilepaskan.
                </p>
              </div>
            </div>
            <p className="text-[11px] text-muted-foreground m-0">
              Di jalur normal baris hanya hidup beberapa milidetik, lebih cepat dari pembacaan tabel tiap 0,25 dtk. Baris
              bertanda <b>log</b> diketahui dari log relay service-nya, bukan dari tabel; id barisnya tidak tercatat di log.
            </p>
          </div>
        )}

        {unavailable && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-xs font-mono">{unavailable}</AlertDescription>
          </Alert>
        )}

        {!unavailable && rows.length === 0 && (
          <div className="p-8 text-center text-xs text-muted-foreground rounded-lg border border-dashed border-border/70 bg-background/30">
            Belum ada pesan outbox untuk request ini. Pesan muncul saat sebuah tahap meng-commit hasilnya; di jalur
            normal pesan itu langsung terkirim dan dihapus, dan tercatat di sini dari log relay.
          </div>
        )}

        {rows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[70px]">ID</TableHead>
                <TableHead>DITULIS OLEH</TableHead>
                <TableHead>PESAN &amp; TUJUAN</TableHead>
                <TableHead>STATUS</TableHead>
                <TableHead className="w-[80px] text-center">ATTEMPT</TableHead>
                <TableHead>RIWAYAT WAKTU</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => {
                return (
                  <TableRow key={r.message.id ?? `log-${i}`}>
                    <TableCell className="font-mono text-muted-foreground font-semibold">
                      {r.message.id != null ? `#${r.message.id}` : 'log'}
                    </TableCell>
                    <TableCell className="font-medium text-foreground">
                      {r.message.owner}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="outline" className="text-[10px] font-mono border-border/60">
                          {r.message.kind}
                        </Badge>
                        <span className="text-muted-foreground">
                          {r.message.kind === 'handoff' ? `→ ${r.message.target}` : r.message.message ?? '→ ORKESTRASI'}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={badgeVariantOf(r.state)} className="text-[10px]">
                        {r.state}
                      </Badge>
                      {r.message.last_error && r.state !== 'DELIVERED' && (
                        <div className="text-[10px] text-rose-400 mt-1 max-w-[280px] break-words font-mono">
                          {r.message.last_error}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-center font-mono">
                      <span className="px-2 py-0.5 rounded bg-muted/60 text-foreground text-xs">
                        {r.message.attempts}
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {r.history.map((h, i) => (
                          <span
                            key={i}
                            className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground border border-border/40"
                          >
                            {h.status.toLowerCase()} {rel(h.ts)}
                          </span>
                        ))}
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
