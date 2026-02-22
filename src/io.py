"""Solution I/O and checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backend import to_numpy, xp


def save_solution(Q, grid, filepath: str, t: float = 0.0, step: int = 0) -> None:
    """Save solution and grid to compressed NumPy archive.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object
        filepath: output file path (.npz)
        t: current simulation time
        step: current time step number
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        Q=to_numpy(Q),
        x=to_numpy(grid.x),
        y=to_numpy(grid.y),
        t=np.array([t]),
        step=np.array([step]),
    )
    print(f"Saved solution to {filepath}")


def load_solution(filepath: str):
    """Load solution from NumPy archive.

    Returns:
        dict with keys: Q, x, y, t, step
    """
    data = np.load(filepath)
    return {
        "Q": xp.array(data["Q"]),
        "x": xp.array(data["x"]),
        "y": xp.array(data["y"]),
        "t": float(data["t"][0]),
        "step": int(data["step"][0]),
    }
