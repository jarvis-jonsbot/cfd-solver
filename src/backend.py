"""Array backend abstraction — MLX (Apple GPU), CuPy (NVIDIA GPU), or NumPy (CPU).

Set environment variable CFD_BACKEND to select:
    mlx     — Apple Silicon GPU via MLX (float32 only)
    cupy    — NVIDIA GPU via CuPy
    numpy   — CPU (default)
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

_BACKEND_NAME: str = os.environ.get("CFD_BACKEND", "numpy").lower()

# Precision-aware floor values: float32 needs much larger epsilons
# than float64 to avoid catastrophic cancellation.
EPS_TINY: float = 1e-30 if _BACKEND_NAME != "mlx" else 1e-7
"""Epsilon for preventing division by zero in safe divisions."""

EPS_SLOPE: float = 1e-12 if _BACKEND_NAME != "mlx" else 1e-5
"""Epsilon for slope limiter ratio denominators."""


class _MLXShim:
    """Thin wrapper around mlx.core that patches NumPy-compat gaps."""

    def __init__(self, mx: ModuleType):
        self._mx = mx
        self.__name__ = "mlx.core"
        mx.set_default_device(mx.gpu)

    def __getattr__(self, name: str):
        return getattr(self._mx, name)

    def linspace(self, start, stop, num=50, *, endpoint=True, **kw):
        """mlx.core.linspace doesn't support endpoint=False."""
        mx = self._mx
        if endpoint:
            return mx.linspace(start, stop, num, **kw)
        # Compute with one extra point, then drop the last
        return mx.linspace(start, stop, num + 1, **kw)[:-1]


def _load_mlx() -> _MLXShim:
    """Load MLX and configure for GPU compute."""
    import mlx.core as mx

    return _MLXShim(mx)


def get_backend():
    """Return the array module (mlx shim, cupy, or numpy)."""
    if _BACKEND_NAME == "mlx":
        try:
            return _load_mlx()
        except ImportError:
            import warnings

            warnings.warn("MLX not installed — falling back to NumPy", stacklevel=2)
            return importlib.import_module("numpy")
    if _BACKEND_NAME == "cupy":
        try:
            return importlib.import_module("cupy")
        except ImportError:
            import warnings

            warnings.warn("CuPy not installed — falling back to NumPy", stacklevel=2)
            return importlib.import_module("numpy")
    return importlib.import_module("numpy")


xp = get_backend()


def to_numpy(arr):
    """Convert array to NumPy (no-op if already NumPy)."""
    if hasattr(arr, "get"):
        # CuPy
        return arr.get()
    try:
        import numpy as np

        return np.array(arr)
    except Exception:
        return arr
