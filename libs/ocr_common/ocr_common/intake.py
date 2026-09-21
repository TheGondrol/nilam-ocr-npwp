from fastapi import File, Form, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from ocr_common.fetch_url import FetchUrlError, fetch

FileField = File(None, description="Document image (JPEG/PNG/PDF). Omit when sending file_url.")
FileUrlField = Form(
    None,
    description=(
        "URL this service fetches the document image from (e.g. a presigned MinIO GET). Omit when uploading file. "
        "The host must be listed in the service's `FILE_URL_ALLOWED_HOSTS`, or resolve to a public address when "
        "that is empty; redirects are not followed."
    ),
)


def resolve_intake(
    file: UploadFile | str | None, file_url: str | None
) -> tuple[StarletteUploadFile | None, str | None]:
    file_url = file_url or None
    upload = file if isinstance(file, StarletteUploadFile) and file.filename else None
    if (upload is None) == (file_url is None):
        raise HTTPException(status_code=400, detail="Send exactly one of file or file_url")
    return upload, file_url


async def read_image(
    request: Request, file: UploadFile | str | None, file_url: str | None
) -> tuple[bytes, str, str | None]:
    upload, file_url = resolve_intake(file, file_url)

    if file_url is not None:
        try:
            settings = request.app.state.settings
            return await fetch(file_url, limit=settings.max_upload_bytes, policy=settings.file_url_policy)
        except FetchUrlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    assert upload is not None
    return await upload.read(), upload.filename or "", upload.content_type
