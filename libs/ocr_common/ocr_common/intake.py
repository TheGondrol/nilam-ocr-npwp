"""
Satu cara menerima gambar untuk semua endpoint yang butuh file: kirim `file`
(multipart) ATAU `file_url` (service yang mengunduh, mis. presigned MinIO
GET), tepat satu dari keduanya. Sama dengan extract-ocr di ocr-* dan
ocr-orchestration.
"""

from fastapi import File, Form, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from ocr_common.fetch_url import FetchUrlError, fetch

# Union[..., str, ...] BUKAN sekadar UploadFile: `-F 'file='` (persis yang
# dikirim Swagger UI saat kolom file dikosongkan) datang sebagai string
# kosong, dan pydantic menolaknya 422 SEBELUM normalisasi sempat jalan.
# Akibatnya jalur file_url mustahil dipakai dari /docs.
FileField = File(None, description="Document image (JPEG/PNG/PDF). Omit when sending file_url.")
FileUrlField = Form(
    None,
    description=(
        "URL this service fetches the document image from (e.g. a presigned MinIO GET). Omit when uploading file."
    ),
)


def resolve_intake(
    file: UploadFile | str | None, file_url: str | None
) -> tuple[StarletteUploadFile | None, str | None]:
    """-> (upload, file_url), tepat satu yang terisi. Raise HTTPException 400 kalau tidak."""
    # Form kosong datang sebagai string kosong, dan UploadFile tanpa berkas
    # datang sebagai objek tanpa filename: keduanya harus dibaca sebagai
    # "tidak dikirim", kalau tidak pengecekan di bawah menganggap dua-duanya ada.
    file_url = file_url or None
    # Kelas Starlette, bukan fastapi.UploadFile: pada anotasi union FastAPI
    # menyerahkan objek Starlette langsung, dan kelas FastAPI adalah turunannya.
    upload = file if isinstance(file, StarletteUploadFile) and file.filename else None
    if (upload is None) == (file_url is None):
        raise HTTPException(status_code=400, detail="Send exactly one of file or file_url")
    return upload, file_url


async def read_image(
    request: Request, file: UploadFile | str | None, file_url: str | None
) -> tuple[bytes, str, str | None]:
    """-> (content, filename, content_type). Raise HTTPException 400 kalau intake tidak valid."""
    upload, file_url = resolve_intake(file, file_url)

    if file_url is not None:
        try:
            return await fetch(file_url, limit=request.app.state.settings.max_upload_bytes)
        except FetchUrlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    assert upload is not None  # dijamin oleh pengecekan tepat-satu di atas
    return await upload.read(), upload.filename or "", upload.content_type
