"""
Satu cara menerima gambar untuk semua endpoint yang butuh file: kirim `file`
(multipart) ATAU `file_url` (service ini yang mengunduh, mis. presigned MinIO
GET), tepat satu dari keduanya. Sama dengan extract-ocr di ocr-* dan
ocr-orchestration.
"""

from fastapi import File, Form, HTTPException, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.core.fetch_url import FetchUrlError, fetch

# Union[..., str, ...] BUKAN sekadar UploadFile: `-F 'file='` (persis yang
# dikirim Swagger UI saat kolom file dikosongkan) datang sebagai string
# kosong, dan pydantic menolaknya 422 SEBELUM normalisasi sempat jalan.
# Akibatnya jalur file_url mustahil dipakai dari /docs.
FileField = File(None, description="Document image (JPEG/PNG). Omit when sending file_url.")
FileUrlField = Form(
    None,
    description=(
        "URL this service fetches the document image from (e.g. a presigned MinIO GET). Omit when uploading file."
    ),
)


async def read_image(
    file: UploadFile | str | None, file_url: str | None, *, limit: int
) -> tuple[bytes, str, str | None]:
    """-> (content, filename, content_type). Raise HTTPException 400 kalau intake tidak valid."""
    # Form kosong datang sebagai string kosong, dan UploadFile tanpa berkas
    # datang sebagai objek tanpa filename: keduanya harus dibaca sebagai
    # "tidak dikirim", kalau tidak pengecekan di bawah menganggap dua-duanya ada.
    file_url = file_url or None
    # Kelas Starlette, bukan fastapi.UploadFile: pada anotasi union FastAPI
    # menyerahkan objek Starlette langsung, dan kelas FastAPI adalah turunannya.
    upload = file if isinstance(file, StarletteUploadFile) and file.filename else None
    if (upload is None) == (file_url is None):
        raise HTTPException(status_code=400, detail="Send exactly one of file or file_url")

    if file_url is not None:
        try:
            return await fetch(file_url, limit=limit)
        except FetchUrlError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    assert upload is not None  # dijamin oleh pengecekan tepat-satu di atas
    return await upload.read(), upload.filename or "", upload.content_type
