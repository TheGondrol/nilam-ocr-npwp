import io

from PIL import Image, UnidentifiedImageError

from ocr_common.errors import BadRequest

# Asked for by the ML team (23 Sep 2026): a genuine NPWP upload is at most 2 pages (front and back, or
# husband and wife); more is almost always another document bundled in. Refused here, before any model
# or OCR runs, with a message the client can show as is.
TOO_MANY_PAGES_MESSAGE = "Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP"


def check_page_count(n_pages: int, max_document_pages: int) -> None:
    """Raises `BadRequest` (400) when the document has more pages than `GUARDRAILS_MAX_DOCUMENT_PAGES`."""
    if n_pages > max_document_pages:
        raise BadRequest(TOO_MANY_PAGES_MESSAGE)


def render_pages(
    content_type: str | None, content: bytes, *, dpi: int, max_pages: int, max_document_pages: int | None = None
) -> list[Image.Image]:
    """An image is one page; a PDF is rendered page by page (at most `max_pages`). With
    `max_document_pages`, a PDF with more pages than that is refused before anything is rendered."""
    if (content_type or "").lower() == "application/pdf":
        return _render_pdf(content, dpi=dpi, max_pages=max_pages, max_document_pages=max_document_pages)
    return [_open_image(content)]


def _open_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise BadRequest("Uploaded file is not a readable image") from exc
    return image.convert("RGB")


def _render_pdf(
    content: bytes, *, dpi: int, max_pages: int, max_document_pages: int | None = None
) -> list[Image.Image]:
    import fitz

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise BadRequest("Uploaded file is not a readable PDF") from exc

    pages: list[Image.Image] = []
    with document:
        if document.page_count == 0:
            raise BadRequest("PDF has no pages")
        if max_document_pages is not None:
            check_page_count(document.page_count, max_document_pages)
        zoom = dpi / 72.0
        for index in range(min(document.page_count, max_pages)):
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    return pages
