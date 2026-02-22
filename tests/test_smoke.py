"""Smoke test: run the solver briefly and verify no NaN/Inf in output."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.backend import xp
from src.boundary import freestream_state
from src.grid import generate_cylinder_grid
from src.solver import SolverConfig, solve


def test_smoke_subsonic():
    """Run 100 steps at Mach 0.3 on a small grid — must not produce NaN."""
    grid = generate_cylinder_grid(ni=32, nj=16)
    Q_inf = freestream_state(0.3)
    Q0 = xp.zeros((4, grid.ni, grid.nj))
    for eq in range(4):
        Q0[eq, :, :] = Q_inf[eq]

    config = SolverConfig(mach=0.3, cfl=0.3, max_steps=100, print_interval=50)
    Q = solve(Q0, grid, config)

    Q_np = np.asarray(Q)
    assert not np.any(np.isnan(Q_np)), "NaN detected in solution"
    assert not np.any(np.isinf(Q_np)), "Inf detected in solution"
    assert np.all(Q_np[0] > 0), "Negative density detected"


def test_smoke_supersonic():
    """Run 100 steps at Mach 2.0 on a small grid — must not produce NaN."""
    grid = generate_cylinder_grid(ni=32, nj=16)
    Q_inf = freestream_state(2.0)
    Q0 = xp.zeros((4, grid.ni, grid.nj))
    for eq in range(4):
        Q0[eq, :, :] = Q_inf[eq]

    config = SolverConfig(mach=2.0, cfl=0.2, max_steps=100, print_interval=50)
    Q = solve(Q0, grid, config)

    Q_np = np.asarray(Q)
    assert not np.any(np.isnan(Q_np)), "NaN detected in solution"
    assert not np.any(np.isinf(Q_np)), "Inf detected in solution"
