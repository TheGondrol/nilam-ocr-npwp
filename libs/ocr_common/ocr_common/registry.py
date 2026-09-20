from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")
Factory = Callable[[Any], T]


def build_backend(registry: Mapping[str, Factory[T]], name: str, settings: Any, kind: str) -> T:
    if name not in registry:
        raise RuntimeError(f"Unknown {kind} backend {name!r}. Available: {sorted(registry)}")
    return registry[name](settings)
