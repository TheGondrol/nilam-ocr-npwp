"""Pemeriksa kontrak callback: apakah body yang dikirim pipeline ke Orkestrasi pusat sesuai kesepakatan.

Acuannya kontrak "Callback Hasil OCR" Orkestrasi pusat (2 Okt 2026) dengan jawaban mereka 5 Okt 2026, seperti
yang dibangun `result_callback_body` (libs/ocr_common/ocr_common/pipeline/callbacks.py). Bukan bagian C di
docs/confluence/spesifikasi-api-ocr-npwp.xml: bagian itu masih versi sebelum kesepakatan.

Format result (/v1/ocr-callback, yang dipakai di dev), satu per request saat request berakhir:

    selesai   {"request_id", "status": "completed", "result": {...}, "guardrails": 0}
    ditolak   {"request_id", "status": "completed", "result": null, "guardrails": 1, "message", "error_code"?}
    gagal     {"request_id", "status": "failed", "error_code": "<STAGE>_FAILED", "message"}

`result` yang selesai sama persis dengan `data` jawaban 200 extract-ocr request yang sama; kalau pipeline berakhir
di scoring: tepat `nomor_npwp` dan `nama`, masing-masing `{value: string | null, confidence: 0 | 1}`.

Format stage (/v1/callbacks/stage, default lokal) bukan kontrak pusat; yang diperiksa hanya bentuknya dan bahwa
data internal (`answer`, `traceparent`) tidak ikut terkirim.

Setiap cek: `pass`, `warn` (bukan pelanggaran kontrak, tapi perlu diketahui), atau `fail`. Fungsi di sini murni
(tanpa Redis / HTTP) supaya bisa diuji sendiri: `python -m pytest tools/tracker/backend`.
"""

from typing import Any

PASS, WARN, FAIL = "pass", "warn", "fail"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

REJECTED_CODE = "DOWNSTREAM_VALIDATION_ERROR"
FAILED_CODES = ("OCR_FAILED", "STRUCTURING_FAILED", "SCORING_FAILED")
CONTRACT_FIELDS = ("nomor_npwp", "nama")
# Disimpan pipeline untuk dirinya sendiri di body outbox; tidak boleh ikut terkirim.
INTERNAL_KEYS = ("answer", "traceparent")

COMPLETED_KEYS = {"request_id", "status", "result", "guardrails"}
REJECTED_KEYS = COMPLETED_KEYS | {"message", "error_code"}
FAILED_KEYS = {"request_id", "status", "error_code", "message", "result", "guardrails"}
STAGE_KEYS = {"request_id", "stage", "status", "result", "error_message", "error_code", "final"}


def _check(rule: str, ok: bool, detail: str = "", *, level: str = FAIL) -> dict[str, str]:
    """Satu cek: `pass` kalau ok, selain itu `level` (fail, atau warn untuk yang bukan pelanggaran)."""
    return {"rule": rule, "level": PASS if ok else level, "detail": "" if ok else detail}


def _is_int(value: Any, *allowed: int) -> bool:
    """Integer JSON (bukan boolean, bukan 0.0) dengan salah satu nilai `allowed`."""
    return type(value) is int and value in allowed


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def verdict(checks: list[dict[str, str]]) -> str:
    """Level terburuk dari semua cek."""
    return max((check["level"] for check in checks), key=_RANK.__getitem__, default=PASS)


def _report(kind: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    return {"kind": kind, "verdict": verdict(checks), "checks": checks}


def _no_internal_keys(body: dict[str, Any]) -> dict[str, str]:
    leaked = [key for key in INTERNAL_KEYS if key in body]
    return _check("Tidak membawa data internal pipeline", not leaked, f"ikut terkirim: {', '.join(leaked)}")


def _only_keys(body: dict[str, Any], allowed: set[str]) -> dict[str, str]:
    extra = sorted(set(body) - allowed - set(INTERNAL_KEYS))
    return _check("Tidak ada field di luar kontrak", not extra, f"field tambahan: {', '.join(extra)}", level=WARN)


def kind_of_result(body: Any) -> str:
    """completed / rejected / failed / invalid, menurut bentuk body result callback."""
    if not isinstance(body, dict):
        return "invalid"
    if body.get("status") == "completed":
        return "rejected" if body.get("result") is None else "completed"
    if body.get("status") == "failed":
        return "failed"
    return "invalid"


def check_result_callback(
    body: Any,
    *,
    key_ok: bool = True,
    ends_at_scoring: bool = True,
    answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cek body result callback. `ends_at_scoring`: pipeline_name_sequence request ini berakhir di scoring (bentuk
    `result` diperiksa per field). `answer`: jawaban GET /v1/extract-ocr/{request_id} orchestrator saat callback
    tiba, `{"http_status", "body"}`; None = tidak bisa dibandingkan (orchestrator tidak terjangkau / load test)."""
    checks = [_check("X-Callback-Key benar", key_ok, "header X-Callback-Key salah atau tidak ada")]
    if not isinstance(body, dict):
        checks.append(_check("Body berupa JSON object", False, f"body: {_show(body)}"))
        return _report("invalid", checks)
    checks.append(
        _check("request_id berupa string tidak kosong", _text(body.get("request_id")), _show(body.get("request_id")))
    )
    kind = kind_of_result(body)
    checks.append(_check("status completed atau failed", kind != "invalid", f"status: {_show(body.get('status'))}"))
    checks.append(_no_internal_keys(body))
    if kind == "completed":
        checks += _completed(body, ends_at_scoring)
    elif kind == "rejected":
        checks += _rejected(body)
    elif kind == "failed":
        checks += _failed(body)
    if kind != "invalid":
        checks += compare_with_answer(kind, body, answer)
    return _report(kind, checks)


def _completed(body: dict[str, Any], ends_at_scoring: bool) -> list[dict[str, str]]:
    result = body.get("result")
    checks = [
        _check(
            "guardrails = 0 (integer)",
            _is_int(body.get("guardrails"), 0),
            f"guardrails: {_show(body.get('guardrails'))}",
        ),
        _check("result berupa object", isinstance(result, dict), f"result: {_show(result)}"),
        _only_keys(body, COMPLETED_KEYS),
    ]
    if ends_at_scoring and isinstance(result, dict):
        fields = sorted(result)
        checks.append(
            _check(
                "result berisi tepat nomor_npwp dan nama",
                fields == sorted(CONTRACT_FIELDS),
                f"field result: {', '.join(fields) or '(kosong)'}",
            )
        )
        for name in CONTRACT_FIELDS:
            if name in result:
                checks.append(_contract_field(name, result[name]))
    return checks


def _contract_field(name: str, field: Any) -> dict[str, str]:
    rule = f"result.{name} = {{value: string | null, confidence: 0 | 1}}"
    if not isinstance(field, dict) or sorted(field) != ["confidence", "value"]:
        return _check(rule, False, _show(field))
    value, confidence = field["value"], field["confidence"]
    problems = []
    if value is not None and not isinstance(value, str):
        problems.append(f"value {_show(value)}")
    if not _is_int(confidence, 0, 1):
        problems.append(f"confidence {_show(confidence)} (harus integer 0 atau 1, bukan probabilitas)")
    return _check(rule, not problems, "; ".join(problems))


def _rejected(body: dict[str, Any]) -> list[dict[str, str]]:
    code = body.get("error_code")
    return [
        _check(
            "guardrails = 1 (integer)",
            _is_int(body.get("guardrails"), 1),
            f"guardrails: {_show(body.get('guardrails'))}",
        ),
        _check("message berisi alasan penolakan", _text(body.get("message")), f"message: {_show(body.get('message'))}"),
        _check(
            f"error_code (di luar kontrak) {REJECTED_CODE}",
            code in (None, REJECTED_CODE),
            f"error_code: {_show(code)}",
            level=WARN,
        ),
        _only_keys(body, REJECTED_KEYS),
    ]


def _failed(body: dict[str, Any]) -> list[dict[str, str]]:
    code = body.get("error_code")
    return [
        _check("message berisi alasan", _text(body.get("message")), f"message: {_show(body.get('message'))}"),
        _check(
            "error_code <STAGE>_FAILED",
            code in FAILED_CODES,
            f"error_code: {_show(code)} (harus salah satu dari {', '.join(FAILED_CODES)})",
            level=WARN,
        ),
        _check("result kosong", body.get("result") is None, f"result: {_show(body.get('result'))}"),
        _only_keys(body, FAILED_KEYS),
    ]


def compare_with_answer(kind: str, body: dict[str, Any], answer: dict[str, Any] | None) -> list[dict[str, str]]:
    """Body callback dibandingkan dengan jawaban GET /v1/extract-ocr/{request_id}: keduanya harus menyatakan hal
    yang sama, karena pusat bisa menerima hasil lewat salah satunya."""
    rule = "Sama dengan jawaban GET /v1/extract-ocr/{request_id}"
    if answer is None:
        return [_check(rule, False, "tidak dibandingkan: jawaban orchestrator tidak terbaca", level=WARN)]
    status = answer.get("http_status")
    got = answer.get("body") if isinstance(answer.get("body"), dict) else {}
    if kind == "completed":
        if status == 202:
            return [_check(rule, False, "orchestrator masih menjawab 202 padahal callback completed", level=WARN)]
        if status != 200:
            return [_check(rule, False, f"callback completed, orchestrator menjawab {status} {got.get('errors')}")]
        checks = [
            _check(
                f"{rule}: result = data",
                body.get("result") == got.get("data"),
                _diff(body.get("result"), got.get("data")),
            )
        ]
        if "guardrails" in got:
            checks.append(
                _check(
                    f"{rule}: guardrails",
                    got.get("guardrails") == body.get("guardrails"),
                    f"GET guardrails: {got.get('guardrails')}",
                )
            )
        return checks
    if kind == "rejected":
        if status != 400 or got.get("errors") != REJECTED_CODE:
            return [_check(rule, False, f"callback ditolak, orchestrator menjawab {status} {got.get('errors')}")]
        return [
            _check(f"{rule}: ditolak", True),
            _check(
                f"{rule}: message",
                got.get("message") == body.get("message"),
                f"GET message: {_show(got.get('message'))}",
                level=WARN,
            ),
        ]
    # failed
    if status == 202:
        # Batasan GET yang terdokumentasi: hand-off yang mati meninggalkan tahap berikutnya tanpa job.
        return [
            _check(rule, False, "orchestrator masih menjawab 202 (hand-off gagal permanen: batasan GET)", level=WARN)
        ]
    if status != 422:
        return [_check(rule, False, f"callback failed, orchestrator menjawab {status} {got.get('errors')}")]
    return [
        _check(
            f"{rule}: error_code",
            got.get("errors") == body.get("error_code"),
            f"GET errors: {got.get('errors')}, callback error_code: {body.get('error_code')}",
            level=WARN,
        )
    ]


def _diff(sent: Any, expected: Any) -> str:
    """Field result yang berbeda dari data jawaban GET."""
    if not isinstance(sent, dict) or not isinstance(expected, dict):
        return f"callback {_show(sent)} ≠ GET {_show(expected)}"
    keys = sorted(set(sent) | set(expected))
    differ = [
        f"{key}: callback {_show(sent.get(key))} ≠ GET {_show(expected.get(key))}"
        for key in keys
        if sent.get(key) != expected.get(key)
    ]
    return "; ".join(differ)


def check_stage_callback(body: Any, *, key_ok: bool = True) -> dict[str, Any]:
    """Cek bentuk callback format stage (lokal; bukan kontrak pusat)."""
    checks = [_check("X-Callback-Key benar", key_ok, "header X-Callback-Key salah")]
    if not isinstance(body, dict):
        checks.append(_check("Body berupa JSON object", False, f"body: {_show(body)}"))
        return _report("invalid", checks)
    status = body.get("status")
    checks += [
        _check("request_id berupa string tidak kosong", _text(body.get("request_id")), _show(body.get("request_id"))),
        _check(
            "stage OCR, STRUCTURING, atau SCORING",
            body.get("stage") in ("OCR", "STRUCTURING", "SCORING"),
            _show(body.get("stage")),
        ),
        _check("status DONE atau FAILED", status in ("DONE", "FAILED"), _show(status)),
        _no_internal_keys(body),
        _check("final hanya true", body.get("final", True) is True, f"final: {_show(body.get('final'))}"),
        _only_keys(body, STAGE_KEYS),
    ]
    if status == "DONE":
        checks.append(
            _check("result berupa object", isinstance(body.get("result"), dict), _show(body.get("result")), level=WARN)
        )
    elif status == "FAILED":
        checks.append(
            _check("error_message berisi alasan", _text(body.get("error_message")), _show(body.get("error_message")))
        )
    return _report("stage", checks)
