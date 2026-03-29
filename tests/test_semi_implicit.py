"""Tests for semi-implicit pressure solver (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.backend import xp
from src.gas import GAMMA, pressure, primitive_to_conservative
from src.grid import Grid
from src.solver import step_semi_implicit
from src.splitting import split_flux_x, split_flux_y


def test_flux_split_sums_to_original():
    """Verify that F_advect + F_acoustic == F_euler for random valid states."""
    # Create a random valid state
    np.random.seed(42)
    ni, nj = 10, 5

    rho = np.random.uniform(0.5, 2.0, (ni, nj))
    u = np.random.uniform(-1.0, 1.0, (ni, nj))
    v = np.random.uniform(-1.0, 1.0, (ni, nj))
    p = np.random.uniform(0.5, 2.0, (ni, nj))

    W = np.stack([rho, u, v, p], axis=0)
    Q = primitive_to_conservative(W)

    # Compute Euler fluxes manually
    F_euler_x = xp.stack([
        rho * u,
        rho * u**2 + p,
        rho * u * v,
        (Q[3] + p) * u,
    ], axis=0)

    G_euler_y = xp.stack([
        rho * v,
        rho * u * v,
        rho * v**2 + p,
        (Q[3] + p) * v,
    ], axis=0)

    # Split fluxes
    F_advect, F_acoustic = split_flux_x(Q, None)
    G_advect, G_acoustic = split_flux_y(Q, None)

    # Verify sum
    F_sum = F_advect + F_acoustic
    G_sum = G_advect + G_acoustic

    np.testing.assert_allclose(F_sum, F_euler_x, rtol=1e-12)
    np.testing.assert_allclose(G_sum, G_euler_y, rtol=1e-12)


def test_sl_advect_constant():
    """Advecting a constant field should return the same constant."""
    from src.advection import sl_advect

    ni, nj = 20, 10
    f = xp.ones((ni, nj)) * 2.5
    u = xp.ones((ni, nj)) * 0.3
    v = xp.ones((ni, nj)) * -0.2

    dx, dy = 0.1, 0.1
    dt = 0.01

    f_new = sl_advect(f, u, v, dx, dy, dt)

    # Should be constant (within floating-point precision)
    np.testing.assert_allclose(f_new, 2.5, rtol=1e-10)


def test_semi_implicit_no_nan():
    """Run 50 steps on a simple 1D-like setup, check for NaN/Inf."""
    # Create a simple uniform grid (not body-fitted, just for testing)
    ni, nj = 50, 10

    # Create a minimal grid with uniform metrics
    x = np.linspace(0, 1, ni)
    y = np.linspace(0, 0.2, nj)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Simple uniform grid metrics
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Create a mock grid object
    grid = Grid.__new__(Grid)
    grid.ni = ni
    grid.nj = nj
    grid.x = xp.array(X)
    grid.y = xp.array(Y)

    # Uniform grid metrics (Jacobian = dx*dy, area-weighted normals are just dy and dx)
    grid.jacobian = xp.ones((ni, nj)) * (-dx * dy)  # Negative for consistency with O-grid
    grid.xi_x_area = xp.ones((ni, nj)) * dy
    grid.xi_y_area = xp.zeros((ni, nj))
    grid.eta_x_area = xp.zeros((ni, nj))
    grid.eta_y_area = xp.ones((ni, nj)) * dx

    # Initialize with a Sod-like shock tube in x-direction
    rho = xp.ones((ni, nj))
    u = xp.zeros((ni, nj))
    v = xp.zeros((ni, nj))
    p = xp.ones((ni, nj))

    # Left state: high pressure
    rho[:ni // 2, :] = 1.0
    p[:ni // 2, :] = 1.0

    # Right state: low pressure
    rho[ni // 2:, :] = 0.125
    p[ni // 2:, :] = 0.1

    W = xp.stack([rho, u, v, p], axis=0)
    Q = primitive_to_conservative(W)

    # Run 50 steps
    dt = 0.0001  # Small time step for stability
    for step in range(50):
        Q = step_semi_implicit(Q, dt, grid)

        # Check for NaN/Inf
        assert xp.all(xp.isfinite(Q)), f"NaN/Inf detected at step {step}"

    # Final check
    assert xp.all(Q[0] > 0), "Negative density detected"
    p_final = pressure(Q)
    assert xp.all(p_final > 0), "Negative pressure detected"


def test_mass_conservation():
    """Total mass should be conserved over 50 steps."""
    ni, nj = 30, 8

    # Create a minimal uniform grid
    x = np.linspace(0, 1, ni)
    y = np.linspace(0, 0.2, nj)
    X, Y = np.meshgrid(x, y, indexing='ij')

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    grid = Grid.__new__(Grid)
    grid.ni = ni
    grid.nj = nj
    grid.x = xp.array(X)
    grid.y = xp.array(Y)
    grid.jacobian = xp.ones((ni, nj)) * (-dx * dy)
    grid.xi_x_area = xp.ones((ni, nj)) * dy
    grid.xi_y_area = xp.zeros((ni, nj))
    grid.eta_x_area = xp.zeros((ni, nj))
    grid.eta_y_area = xp.ones((ni, nj)) * dx

    # Initialize with smooth variation
    rho = xp.ones((ni, nj)) * 1.0 + 0.1 * xp.sin(2 * np.pi * xp.arange(ni)[:, None] / ni)
    u = xp.zeros((ni, nj))
    v = xp.zeros((ni, nj))
    p = xp.ones((ni, nj))

    W = xp.stack([rho, u, v, p], axis=0)
    Q = primitive_to_conservative(W)

    # Compute initial total mass
    cell_volume = xp.abs(grid.jacobian)
    mass_initial = float(xp.sum(Q[0] * cell_volume))

    # Run 50 steps
    dt = 0.0001
    for step in range(50):
        Q = step_semi_implicit(Q, dt, grid)

    # Compute final mass
    mass_final = float(xp.sum(Q[0] * cell_volume))

    # Check conservation (relative error < 1e-5)
    rel_error = abs(mass_final - mass_initial) / mass_initial
    assert rel_error < 1e-5, f"Mass not conserved: relative error = {rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
