"""openapi.yaml adalah turunan kode; kalau berbeda, jalankan `python -m ocr_common.openapi` di folder ini."""

from ocr_common.testing import assert_error_responses_have_examples, assert_openapi_up_to_date
from src.main import app


def test_openapi_yaml_is_up_to_date():
    assert_openapi_up_to_date(app)


def test_error_responses_have_their_own_examples():
    assert_error_responses_have_examples(app)
