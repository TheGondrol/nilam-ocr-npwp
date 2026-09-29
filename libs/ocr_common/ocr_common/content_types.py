"""The type of a document, as the checks read it: from what the caller declared, else from the file itself.

A JPG sent by a client that does not name its type (`application/octet-stream`, no type at all, a type with
parameters, the old `image/pjpeg`) is still a JPG; only a file that is none of the supported kinds is refused."""

GENERIC_TYPES = ("", "application/octet-stream", "binary/octet-stream")
# `image/pjpeg`: the JPEG type some Windows clients still send.
_ALIASES = {"image/pjpeg": "image/jpeg"}
# File signatures of the supported kinds.
SIGNATURES = ((b"%PDF-", "application/pdf"), (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))


def sniff_content_type(content: bytes) -> str | None:
    """The type the file's signature names, or None when it is none of the supported kinds."""
    for signature, content_type in SIGNATURES:
        if content.startswith(signature):
            return content_type
    return None


def upload_content_type(declared: str | None, content: bytes) -> str:
    """The type of an uploaded file: the declared type without parameters (`; charset=...`), an alias as its
    standard name, and a generic or missing type from the file's signature. A generic type that no signature
    matches is kept, so the upload check refuses it as an unsupported type."""
    content_type = (declared or "").split(";", 1)[0].strip().lower()
    content_type = _ALIASES.get(content_type, content_type)
    if content_type in GENERIC_TYPES:
        return sniff_content_type(content) or content_type
    return content_type
