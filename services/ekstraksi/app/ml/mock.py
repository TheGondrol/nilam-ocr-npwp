"""The `mock` backend for local development and tests: a plausible NPWP card derived from the file
bytes, so the same upload always reads the same."""

import hashlib
import random
from typing import Any

from ocr_common.errors import ServiceError

NAMA_POOL = [
    "BUDI SANTOSO",
    "SITI AMINAH",
    "AGUS WIJAYA",
    "DEWI LESTARI",
    "RUDI HARTONO",
    "ANI SURYANI",
    "EKO PRASETYO",
    "RINA MARLINA",
    "JOKO SUSILO",
    "WATI RAHAYU",
]
NAMA_BADAN_POOL = [
    "PT SINAR ABADI SEJAHTERA",
    "PT CIPTA KARYA MANDIRI",
    "PT NUSANTARA DIGITAL TEKNOLOGI",
    "PT BUMI MAKMUR SENTOSA",
    "PT GLOBAL MITRA INDUSTRI",
]
LINE_HEIGHT = 40


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(count))


def _format_npwp(d: str) -> str:
    return f"{d[0:2]}.{d[2:5]}.{d[5:8]}.{d[8]}-{d[9:12]}.{d[12:15]}"


class MockOcrEngine:
    name = "mock"

    async def extract(self, filename: str, content: bytes, content_type: str | None = None) -> dict[str, Any]:
        if "servererror" in (filename or "").lower():
            raise ServiceError(500, "Internal server error while processing OCR")

        rng = random.Random(hashlib.sha256(content).hexdigest())
        lines = [
            "KEMENTERIAN KEUANGAN REPUBLIK INDONESIA",
            "DIREKTORAT JENDERAL PAJAK",
            f"NPWP : {_format_npwp(_digits(rng, 15))}",
            f"NAMA : {rng.choice(NAMA_POOL)}",
            f"NAMA BADAN : {rng.choice(NAMA_BADAN_POOL)}",
        ]
        blocks = [
            {
                "text": text,
                "confidence": round(rng.uniform(0.85, 0.99), 3),
                "bbox": {"x1": 20, "y1": 20 + i * LINE_HEIGHT, "x2": 20 + 12 * len(text), "y2": 50 + i * LINE_HEIGHT},
                "page": 0,
            }
            for i, text in enumerate(lines)
        ]
        return {"blocks": blocks, "model": None}
