import { useState, useRef } from 'react'
import {
  UploadCloud,
  FileText,
  Trash2,
  Check,
  Info,
  RefreshCw,
  Plus,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

function fmtBytes(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function ScenarioFileManager({
  images = [],
  imageFiles = [],
  selectedImage,
  onSelectImage,
  onImagesChanged,
}) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  const uploadFiles = async (files) => {
    if (!files?.length) return
    setUploading(true)
    setError(null)
    const formData = new FormData()
    for (const f of files) {
      formData.append('files', f)
    }

    try {
      const res = await fetch('/api/scenarios/images', {
        method: 'POST',
        body: formData,
      })
      const body = await res.json()
      if (!res.ok) throw new Error(body.detail || res.statusText)
      await onImagesChanged()
      if (body.saved?.length > 0) {
        onSelectImage(body.saved[0])
      }
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const deleteFile = async (name, e) => {
    e.stopPropagation()
    if (!window.confirm(`Hapus file uji "${name}"?`)) return
    try {
      const res = await fetch(`/api/scenarios/images/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || res.statusText)
      }
      await onImagesChanged()
      if (selectedImage === name) {
        const remaining = images.filter((n) => n !== name)
        onSelectImage(remaining[0] || '')
      }
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  // Create unified file list with metadata
  const fileMap = {}
  imageFiles.forEach((f) => {
    fileMap[f.name] = f
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-foreground flex items-center gap-1.5">
          <FileText size={13} className="text-primary" />
          <span>Dokumen Uji Skenario</span>
        </label>
        <Badge variant="outline" className="text-[10px] font-mono">
          {images.length} File Tersedia
        </Badge>
      </div>

      {/* Explanatory Note */}
      <div className="p-2.5 rounded-lg bg-muted/40 border border-border/60 text-[11px] text-muted-foreground flex items-start gap-2">
        <Info size={14} className="text-sky-500 shrink-0 mt-0.5" />
        <p className="m-0 leading-relaxed">
          File <code>npwp1.jpg</code> dan <code>npwp2.jpg</code> adalah berkas contoh default di repositori.
          Anda dapat mengunggah berkas NPWP atau KTP baru di bawah ini untuk menguji skenario dengan gambar kustom.
        </p>
      </div>

      {/* File List */}
      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
        {images.map((name) => {
          const isSelected = selectedImage === name
          const fileMeta = fileMap[name] || {}
          const isSample = fileMeta.sample ?? ['npwp1.jpg', 'npwp2.jpg'].includes(name)

          return (
            <div
              key={name}
              onClick={() => onSelectImage(name)}
              className={`flex items-center justify-between p-2 rounded-lg border text-xs cursor-pointer transition-all ${
                isSelected
                  ? 'bg-accent border-primary/50 text-foreground ring-1 ring-primary/30'
                  : 'bg-background/50 border-border/60 hover:bg-muted/40'
              }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <div
                  className={`h-5 w-5 rounded-md flex items-center justify-center shrink-0 ${
                    isSelected ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {isSelected ? <Check size={11} /> : <FileText size={11} />}
                </div>
                <div className="min-w-0">
                  <span className="font-mono text-[11px] font-medium truncate block">
                    {name}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                {fileMeta.size && (
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {fmtBytes(fileMeta.size)}
                  </span>
                )}
                {isSample ? (
                  <Badge variant="secondary" className="text-[9px] px-1 py-0 font-mono">
                    Sample
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-[9px] px-1 py-0 font-mono text-primary border-primary/30">
                    Upload
                  </Badge>
                )}
                {!isSample && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    onClick={(e) => deleteFile(name, e)}
                    className="h-5 w-5 p-0 text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 ml-0.5"
                    title={`Hapus ${name}`}
                  >
                    <Trash2 size={11} />
                  </Button>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Upload Dropzone */}
      <label
        className={`flex items-center justify-center gap-2 p-3 border border-dashed rounded-lg bg-background/40 cursor-pointer transition-all ${
          dragging ? 'border-primary bg-accent/40' : 'border-border/70 hover:bg-muted/30 hover:border-primary/50'
        } ${uploading ? 'opacity-60 pointer-events-none' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (!uploading) uploadFiles(e.dataTransfer.files)
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.pdf"
          disabled={uploading}
          onChange={(e) => uploadFiles(e.target.files)}
          className="hidden"
        />
        {uploading ? (
          <>
            <RefreshCw size={14} className="animate-spin text-primary" />
            <span className="text-[11px] text-muted-foreground">Mengunggah berkas…</span>
          </>
        ) : (
          <>
            <UploadCloud size={15} className="text-muted-foreground" />
            <span className="text-[11px] text-muted-foreground">
              Unggah Dokumen Baru (JPG, PNG, PDF)
            </span>
          </>
        )}
      </label>

      {error && (
        <div className="p-2 rounded bg-rose-500/10 border border-rose-500/30 text-[11px] text-rose-500 font-mono">
          {error}
        </div>
      )}
    </div>
  )
}
