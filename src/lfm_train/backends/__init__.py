"""Backend selection.

Selection order: explicit ``name`` arg -> ``LFM_TRAIN_BACKEND`` env var ->
platform default (``mlx`` on macOS, ``cuda`` elsewhere). The install-time uv
extra (``--extra cuda`` / ``--extra mlx``) and this runtime default both key off
the platform, so they line up without configuration; the env var / ``--backend``
flag is the override for testing one backend on the other's host.

The concrete backend modules are imported lazily so that resolving the selector
never imports the wrong heavy stack.
"""
from __future__ import annotations

import os
import sys

from .base import Backend

VALID = ("cuda", "mlx")


def default_backend() -> str:
    return "mlx" if sys.platform == "darwin" else "cuda"


def resolve_name(name: str | None = None) -> str:
    """Resolve a backend name without importing it. ``auto``/None/"" -> default."""
    name = name or os.environ.get("LFM_TRAIN_BACKEND") or ""
    if name in ("", "auto"):
        name = default_backend()
    if name not in VALID:
        raise ValueError(f"unknown backend {name!r}; expected one of {VALID} or 'auto'")
    return name


def get_backend(name: str | None = None) -> Backend:
    name = resolve_name(name)
    if name == "cuda":
        from .cuda import CudaBackend
        return CudaBackend()
    from .mlx import MlxBackend
    return MlxBackend()


__all__ = ["Backend", "get_backend", "resolve_name", "default_backend", "VALID"]
