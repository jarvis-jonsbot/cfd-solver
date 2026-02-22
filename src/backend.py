"""Array backend abstraction — CuPy (GPU) or NumPy (CPU).

Set environment variable CFD_BACKEND=cupy to use GPU acceleration.
Defaults to numpy.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

_BACKEND_NAME: str = os.environ.get("CFD_BACKEND", "numpy").lower()


def get_backend() -> ModuleType:
    """Return the array module (cupy or numpy)."""
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
        return arr.get()
    return arr
