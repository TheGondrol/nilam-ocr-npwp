"""The checks a document goes through here, before the guardrails model or any stage sees it, so the
services behind this one only judge what they are for."""

import fitz

from ocr_common.errors import BadRequest
from ocr_common.image_validation import validate_image

from app.config import Settings

PDF_CONTENT_TYPE = "application/pdf"

# Asked for by the ML team (23 Sep 2026): a genuine NPWP upload is at most 2 pages (front and back, or
# husband and wife); more is almost always another document bundled in. The client can show it as is.
TOO_MANY_PAGES_MESSAGE = "Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP"


def check_document(content_type: str | None, content: bytes, settings: Settings) -> None:
    """Type and empty file (400), size above `MAX_UPLOAD_BYTES` (413), then for a PDF the page count
    above `MAX_DOCUMENT_PAGES` (400) and an unreadable PDF (400)."""
    validate_image(content_type, content, settings)
    if (content_type or "").lower() == PDF_CONTENT_TYPE and count_pdf_pages(content) > settings.max_document_pages:
        raise BadRequest(TOO_MANY_PAGES_MESSAGE)


def count_pdf_pages(content: bytes) -> int:
    """The page count from the PDF's page tree, without rendering anything; 400 when it is not a PDF
    PyMuPDF can read (the parser guardrails renders it with) or has no pages."""
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise BadRequest("Uploaded file is not a readable PDF") from exc
    with document:
        if document.page_count == 0:
            raise BadRequest("PDF has no pages")
        return document.page_count
