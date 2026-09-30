"""The type of a document, as the checks read it: from what the caller declared, else from the file itself.

A JPG sent by a client that does not name its type properly (`application/octet-stream`, no type at all, a
type with parameters, the old `image/pjpeg`, a bare extension such as `jpg`, or any other type we do not
support) is still a JPG; only a file that is none of the supported kinds is refused."""

GENERIC_TYPES = ("", "application/octet-stream", "binary/octet-stream")
# `image/pjpeg`: the JPEG type some Windows clients still send.
_ALIASES = {"image/pjpeg": "image/jpeg"}
# File signatures of the supported kinds.
SIGNATURES = ((b"%PDF-", "application/pdf"), (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))
# Declared types taken as they are: the supported kinds and `image/jpg`, the common misspelling of `image/jpeg`.
DECLARED_TYPES = {content_type for _, content_type in SIGNATURES} | {"image/jpg"}


def sniff_content_type(content: bytes) -> str | None:
    """The type the file's signature names, or None when it is none of the supported kinds."""
    for signature, content_type in SIGNATURES:
        if content.startswith(signature):
            return content_type
    return None


def upload_content_type(declared: str | None, content: bytes) -> str:
    """The type of an uploaded file: the declared type without parameters (`; charset=...`), an alias as its
    standard name, and any type that is not a supported one (generic, missing, a bare `jpg`, `text/plain`, ...)
    from the file's signature. A file no signature matches keeps its declared type, so the upload check refuses
    it as an unsupported type."""
    content_type = (declared or "").split(";", 1)[0].strip().lower()
    content_type = _ALIASES.get(content_type, content_type)
    if content_type not in DECLARED_TYPES:
        return sniff_content_type(content) or content_type
    return content_type
