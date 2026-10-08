import { Database, AlertTriangle, RefreshCw, Terminal, CheckCircle2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import JsonViewer from './JsonViewer'

function badgeVariantOf(status) {
  if (status === 200) return 'done'
  if (status === 202) return 'processing'
  if (status === 400) return 'rejected'
  return 'failed'
}

export default function OcrResultsCard({ requestId, data, loading, onRefresh }) {
  const isAvailable = data?.available
  const rows = data?.rows || []
  const count = rows.length
  const dbHost = data?.db_host || '127.0.0.1:5433 (local)'
  const target = data?.target || 'local'

  return (
    <Card className="border-border bg-card border-t-2 border-t-emerald-500 shadow-xs">
      <CardHeader className="flex flex-row items-start justify-between pb-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="h-7 w-7 rounded-md bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
              <Database size={15} />
            </div>
            <CardTitle className="text-sm font-semibold">
              Tabel Database: nilam_ocr_npwp.nilam_ocr_results
            </CardTitle>
            <span className="text-[10px] font-mono uppercase tracking-wider font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              PostgreSQL Storage
            </span>
            {count > 0 ? (
              <Badge variant="done" className="text-[10px] font-mono">
                {count} Baris SQL Ditemukan
              </Badge>
            ) : isAvailable ? (
              <Badge variant="idle" className="text-[10px] font-mono">
                0 Baris (Kosong)
              </Badge>
            ) : (
              <Badge variant="failed" className="text-[10px] font-mono">
                PostgreSQL Offline
              </Badge>
            )}
          </div>
          <CardDescription className="text-xs">
            Audit log jawaban final untuk orkestrasi pusat di PostgreSQL (append-only)
          </CardDescription>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-muted-foreground hidden sm:inline">
            {dbHost} [{target.toUpperCase()}]
          </span>

          <Button
            variant="outline"
            size="xs"
            onClick={onRefresh}
            disabled={loading}
            className="h-7 px-2.5 text-xs gap-1.5"
            title="Muat ulang data dari tabel nilam_ocr_results"
          >
            <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
            <span>{loading ? 'Memeriksa…' : 'Cek Tabel'}</span>
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 pt-0">
        {/* SQL Query Preview Banner */}
        <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2 overflow-x-auto text-[11px] text-muted-foreground">
            <Terminal size={12} className="shrink-0" />
            <span className="shrink-0 text-foreground font-semibold">Query:</span>
            <code className="text-foreground">
              SELECT * FROM nilam_ocr_npwp.nilam_ocr_results WHERE request_id = &apos;{requestId || '...'}&apos;
            </code>
          </div>
          <span className="text-[10px] text-muted-foreground shrink-0 hidden md:inline ml-2">
            Target: {target === 'gke' ? 'GKE Cluster' : 'Container Docker Lokal (Port 5433)'}
          </span>
        </div>

        {!isAvailable && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle className="text-xs font-semibold">Koneksi Database Gagal</AlertTitle>
            <AlertDescription className="text-xs space-y-1">
              <div>{data?.error || 'Tidak dapat terhubung ke PostgreSQL.'}</div>
              <div className="text-[11px] opacity-80">
                Pastikan container <code>nilam-ocr-postgres</code> menyala di port 5433 (<code>docker compose up -d postgres</code>).
              </div>
            </AlertDescription>
          </Alert>
        )}

        {isAvailable && count === 0 && (
          <div className="p-6 text-center rounded-md border border-dashed border-border bg-muted/20">
            <p className="text-xs font-medium text-foreground mb-1">
              Belum ada baris tersimpan untuk request_id &quot;{requestId || '-'}&quot;
            </p>
            <p className="text-[11px] text-muted-foreground max-w-lg mx-auto">
              Baris dicatat otomatis oleh orchestrator tepat sebelum membalas <code>POST /v1/extract-ocr</code>, atau oleh tahap terakhir saat callback sukses terkirim ke pusat.
            </p>
          </div>
        )}

        {count > 0 && (
          <div className="space-y-4">
            {rows.map((row, idx) => {
              const gr = row.guardrails
              const grText = gr === 0 ? '0 (Lolos / Pass)' : gr === 1 ? '1 (Ditolak / Rejected)' : 'NULL (Dilewati)'

              return (
                <div
                  key={row.id ?? row.request_id ?? idx}
                  className="rounded-lg border border-border bg-card p-4 space-y-3.5 shadow-xs"
                >
                  {/* Row Header */}
                  <div className="flex items-center justify-between flex-wrap gap-2 pb-2.5 border-b border-border">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-muted text-foreground">
                        Baris #{idx + 1}
                      </span>
                      <Badge variant={badgeVariantOf(row.status_code)} className="text-[10px] font-mono">
                        HTTP {row.status_code} {row.status_desc}
                      </Badge>
                      {row.pipeline_last_stage && (
                        <Badge variant="outline" className="text-[10px] font-mono">
                          last_stage: {row.pipeline_last_stage}
                        </Badge>
                      )}
                      <span className="text-[11px] font-mono text-muted-foreground">
                        guardrails: {grText}
                      </span>
                    </div>

                    <div className="text-[11px] font-mono text-muted-foreground">
                      created_at: {row.created_at || row.update_at || '-'}
                    </div>
                  </div>

                  {/* Message Banner if available */}
                  {row.message && (
                    <div className="text-xs font-medium text-foreground bg-muted/40 p-2.5 rounded-md border border-border/50">
                      <span className="text-muted-foreground font-mono mr-1.5">[message]:</span>
                      {row.message}
                    </div>
                  )}

                  {row.errors && (
                    <div className="text-xs text-rose-400 font-mono bg-rose-500/10 p-2.5 rounded-md border border-rose-500/20">
                      <span className="font-semibold mr-1.5">[errors]:</span>
                      {row.errors}
                    </div>
                  )}

                  {/* SQL Column Values Table */}
                  <div className="space-y-1.5">
                    <div className="text-[11px] font-semibold text-foreground flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-emerald-500" />
                      <span>Rincian Kolom Tabel PostgreSQL (SELECT *):</span>
                    </div>

                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[180px]">NAMA KOLOM (FIELD)</TableHead>
                          <TableHead className="w-[120px]">TIPE DATA</TableHead>
                          <TableHead>NILAI DI DATABASE (VALUE)</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">request_id</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TEXT (PK)</TableCell>
                          <TableCell className="font-mono font-bold text-foreground">{row.request_id}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">status_code</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">INTEGER</TableCell>
                          <TableCell className="font-mono">
                            <Badge variant={badgeVariantOf(row.status_code)} className="text-[10px]">
                              {row.status_code} ({row.status_desc})
                            </Badge>
                          </TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">status_desc</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TEXT</TableCell>
                          <TableCell className="font-medium text-foreground">{row.status_desc}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">guardrails</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">INTEGER</TableCell>
                          <TableCell className="font-mono text-foreground">{grText}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">message</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TEXT</TableCell>
                          <TableCell className="text-foreground">{row.message || '-'}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">errors</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TEXT</TableCell>
                          <TableCell className="font-mono text-muted-foreground">{row.errors || '(NULL)'}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">created_at</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TIMESTAMPTZ</TableCell>
                          <TableCell className="font-mono text-muted-foreground">{row.created_at || '-'}</TableCell>
                        </TableRow>

                        <TableRow>
                          <TableCell className="font-mono font-medium text-foreground">update_at</TableCell>
                          <TableCell className="font-mono text-muted-foreground text-[10px]">TIMESTAMPTZ</TableCell>
                          <TableCell className="font-mono text-muted-foreground">{row.update_at || '-'}</TableCell>
                        </TableRow>

                        {row.pipeline_last_stage && (
                          <TableRow>
                            <TableCell className="font-mono font-medium text-foreground">pipeline_last_stage</TableCell>
                            <TableCell className="font-mono text-muted-foreground text-[10px]">TEXT</TableCell>
                            <TableCell className="font-mono text-foreground">{row.pipeline_last_stage}</TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  </div>

                  {/* JSON Column Viewers */}
                  <div className="space-y-2 pt-1">
                    {row.data && (
                      <JsonViewer
                        value={row.data}
                        label="Kolom 'data' (JSONB) — Hasil Terstruktur (Nama & NPWP)"
                        defaultOpen={true}
                      />
                    )}

                    <JsonViewer
                      value={row}
                      label="Raw Row Record (Semua Kolom SQL dalam Format JSON)"
                      defaultOpen={false}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
