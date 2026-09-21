import io

from PIL import Image, UnidentifiedImageError

from ocr_common.errors import ServiceError


def render_pages(content_type: str | None, content: bytes, *, dpi: int, max_pages: int) -> list[Image.Image]:
    if (content_type or "").lower() == "application/pdf":
        return _render_pdf(content, dpi=dpi, max_pages=max_pages)
    return [_open_image(content)]


def _open_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ServiceError(400, "Uploaded file is not a readable image") from exc
    return image.convert("RGB")


def _render_pdf(content: bytes, *, dpi: int, max_pages: int) -> list[Image.Image]:
    import fitz

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ServiceError(400, "Uploaded file is not a readable PDF") from exc

    pages: list[Image.Image] = []
    with document:
        if document.page_count == 0:
            raise ServiceError(400, "PDF has no pages")
        zoom = dpi / 72.0
        for index in range(min(document.page_count, max_pages)):
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    return pages
