"""
Pemilihan implementasi model berdasarkan nama backend di Settings.

Tiap modul services/<nama>/src/models/<app>.py punya dict registry
{nama: factory(settings)}. Nama yang tidak dikenal langsung gagal dengan
pesan jelas; service memanggilnya saat startup supaya gagalnya saat boot.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")
Factory = Callable[[Any], T]


def build_backend(registry: Mapping[str, Factory[T]], name: str, settings: Any, kind: str) -> T:
    if name not in registry:
        raise RuntimeError(f"Unknown {kind} backend {name!r}. Available: {sorted(registry)}")
    return registry[name](settings)
