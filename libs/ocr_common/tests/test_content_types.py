import pytest

from ocr_common.content_types import upload_content_type

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
PDF = b"%PDF-1.7\n"


@pytest.mark.parametrize(
    ("declared", "content", "expected"),
    [
        ("image/jpeg", JPEG, "image/jpeg"),
        ("image/jpg", JPEG, "image/jpg"),
        ("IMAGE/JPEG", JPEG, "image/jpeg"),
        ("image/jpeg; charset=binary", JPEG, "image/jpeg"),
        ("image/pjpeg", JPEG, "image/jpeg"),
        ("application/octet-stream", JPEG, "image/jpeg"),
        (None, JPEG, "image/jpeg"),
        ("", PNG, "image/png"),
        ("binary/octet-stream", PDF, "application/pdf"),
        ("jpg", JPEG, "image/jpeg"),
        ("JPEG", JPEG, "image/jpeg"),
        ("png", PNG, "image/png"),
        ("text/plain", JPEG, "image/jpeg"),
    ],
)
def test_a_supported_file_gets_its_type_whatever_the_client_declares(declared, content, expected):
    assert upload_content_type(declared, content) == expected


@pytest.mark.parametrize("declared", ["application/octet-stream", "text/plain", "jpg"])
def test_a_file_no_signature_matches_keeps_its_declared_type_for_the_check_to_refuse(declared):
    assert upload_content_type(declared, b"hello") == declared


def test_a_supported_declared_type_is_not_second_guessed():
    assert upload_content_type("image/png", JPEG) == "image/png"
