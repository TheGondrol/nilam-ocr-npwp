"""
Pemilihan implementasi model berdasarkan nama backend di Settings.

Tiap modul src/models/<app>.py punya dict registry {nama: factory(settings)}.
Nama yang tidak dikenal langsung gagal saat pertama dipakai dengan pesan jelas.
"""

from collections.abc import Callable, Mapping
from typing import TypeVar

from src.core.config import Settings, get_settings

T = TypeVar("T")
Factory = Callable[[Settings], T]


def build_backend(registry: Mapping[str, Factory[T]], name: str, kind: str) -> T:
    if name not in registry:
        raise RuntimeError(f"Unknown {kind} backend {name!r}. Available: {sorted(registry)}")
    return registry[name](get_settings())
