"""
Pembungkus OCR engine. extract() mengembalikan list blok teks terurut:
    [{"text": str, "confidence": 0..1, "bbox": {"x1","y1","x2","y2"} | None}, ...]

Mock menghasilkan baris ala kartu NPWP yang deterministik (seed dari hash isi
file) berbentuk "LABEL : NILAI" supaya bisa langsung dimakan structurer.
Skenario nama file mengikuti mock ocr-npwp: 'servererror' -> 500.
"""

import hashlib
import random
from functools import lru_cache

from src.core.config import get_settings
from src.core.errors import ServiceError
from src.models.registry import Factory, build_backend

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

    def extract(self, filename: str, content: bytes) -> list[dict]:
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
        return [
            {
                "text": text,
                "confidence": round(rng.uniform(0.85, 0.99), 3),
                "bbox": {"x1": 20, "y1": 20 + i * LINE_HEIGHT, "x2": 20 + 12 * len(text), "y2": 50 + i * LINE_HEIGHT},
            }
            for i, text in enumerate(lines)
        ]


OCR_BACKENDS: dict[str, Factory] = {
    "mock": lambda settings: MockOcrEngine(),
}


@lru_cache
def get_ocr_engine():
    return build_backend(OCR_BACKENDS, get_settings().ekstraksi_backend, "ekstraksi OCR")
