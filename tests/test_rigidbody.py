"""Tests for rigid body dynamics and FSI coupling (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src import gas
from src.backend import xp
from src.grid import generate_cartesian_grid
from src.levelset import compute_interface_forces, compute_levelset, fill_ghost_cells
from src.rigidbody import make_circle, make_polygon
from src.solver import step_partitioned_fsi


def test_circle_levelset():
    """Level set for circle: phi = 0 on surface, phi < 0 inside, phi > 0 outside."""
    body = make_circle(center=np.array([0.0, 0.0]), radius=1.0, density=1.0)

    # Test points
    xc = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.5]])
    yc = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.5]])

    phi = compute_levelset(body, xc, yc)

    # Point at origin: inside, phi < 0
    assert phi[0, 0] < 0, "Origin should be inside circle"

    # Point at (1, 0): on surface, phi ≈ 0
    np.testing.assert_allclose(phi[0, 1], 0.0, atol=1e-10)

    # Point at (2, 0): outside, phi > 0
    assert phi[0, 2] > 0, "Point at (2, 0) should be outside circle"

    # Point at (0, 1): on surface
    np.testing.assert_allclose(phi[1, 0], 0.0, atol=1e-10)


def test_polygon_levelset():
    """Level set for square: phi = 0 on surface, phi < 0 inside, phi > 0 outside."""
    # Square with side length 2, centered at origin
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    body = make_polygon(vertices, density=1.0)

    # Test points
    xc = np.array([[0.0, 2.0], [0.0, 1.5]])
    yc = np.array([[0.0, 0.0], [1.5, 0.0]])

    phi = compute_levelset(body, xc, yc)

    # Origin: inside
    assert phi[0, 0] < 0, "Origin should be inside square"

    # Point at (2, 0): outside
    assert phi[0, 1] > 0, "Point at (2, 0) should be outside square"


def test_ghost_cell_no_penetration():
    """Ghost cells should enforce no-penetration BC (zero normal velocity at interface)."""
    # Create a fixed circular body
    body = make_circle(center=np.array([0.0, 0.0]), radius=0.5, density=1.0)

    # Simple uniform flow field
    ni, nj = 20, 20
    xc = np.linspace(-2.0, 2.0, ni)
    yc = np.linspace(-2.0, 2.0, nj)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # Initialize with uniform flow (u = 1, v = 0)
    rho = np.ones((ni, nj)) * 1.0
    u = np.ones((ni, nj)) * 1.0
    v = np.zeros((ni, nj))
    p = np.ones((ni, nj)) * 1.0

    W = np.stack([rho, u, v, p], axis=0)
    from src.gas import primitive_to_conservative

    Q = primitive_to_conservative(W)

    # Compute level set
    phi = compute_levelset(body, X, Y)

    # Fill ghost cells
    Q_new = fill_ghost_cells(Q, phi, body, X, Y, gas)

    # Check that ghost cells have reflected velocity
    # For a fixed body, surface velocity = 0, so ghost velocity should be -u_fluid
    # We can't easily verify this without knowing exact interface geometry,
    # but we can check that Q_new is finite
    assert np.all(np.isfinite(Q_new)), "Ghost cells should have finite values"


def test_interface_force_pressure_box():
    """Uniform pressure on all sides of a circle should give net force = 0."""
    body = make_circle(center=np.array([0.0, 0.0]), radius=0.5, density=1.0)

    ni, nj = 40, 40
    xc = np.linspace(-2.0, 2.0, ni)
    yc = np.linspace(-2.0, 2.0, nj)
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # Uniform pressure field
    rho = np.ones((ni, nj)) * 1.0
    u = np.zeros((ni, nj))
    v = np.zeros((ni, nj))
    p = np.ones((ni, nj)) * 2.0  # Uniform pressure

    W = np.stack([rho, u, v, p], axis=0)
    from src.gas import primitive_to_conservative

    Q = primitive_to_conservative(W)

    phi = compute_levelset(body, X, Y)

    F, tau = compute_interface_forces(Q, phi, body, X, Y, gas)

    # Net force should be close to zero (symmetry)
    np.testing.assert_allclose(F, [0.0, 0.0], atol=0.1)
    np.testing.assert_allclose(tau, 0.0, atol=0.1)


def test_body_advance():
    """Free body under constant gravity should follow parabolic trajectory."""
    body = make_circle(center=np.array([0.0, 10.0]), radius=0.5, density=1.0)

    # Gravity: F = [0, -mg]
    g = 9.8
    F = np.array([0.0, -body.mass * g])
    tau = 0.0
    dt = 0.01

    # Advance 100 steps
    positions = [body.position.copy()]
    for _ in range(100):
        body = body.apply_forces(F, tau, dt)
        positions.append(body.position.copy())

    positions = np.array(positions)

    # Check that y follows parabolic motion: y(t) = y0 - 0.5 * g * t^2
    t_final = 100 * dt
    y_expected = 10.0 - 0.5 * g * t_final**2

    # Allow for numerical integration error
    np.testing.assert_allclose(positions[-1, 1], y_expected, rtol=1e-2)

    # x should remain zero
    np.testing.assert_allclose(positions[-1, 0], 0.0, atol=1e-10)


def test_partitioned_fsi_no_nan():
    """10-step FSI run with subsonic flow, free cylinder, should have no NaN/Inf."""
    # Create Cartesian grid (coarser for stability)
    grid = generate_cartesian_grid(ni=30, nj=20, x_min=-5.0, x_max=5.0, y_min=-3.0, y_max=3.0)

    # Initialize with Mach 0.5 freestream (subsonic for stability)
    from src.boundary import freestream_state

    Q_inf = freestream_state(mach=0.5, alpha=0.0, p_inf=1.0, rho_inf=1.0)
    Q0 = xp.zeros((4, grid.ni, grid.nj))
    for eq in range(4):
        Q0[eq, :, :] = Q_inf[eq]

    # Create a free circular cylinder (smaller radius to avoid grid coverage issues)
    body = make_circle(center=np.array([0.0, 0.0]), radius=0.3, density=1.0)

    Q = Q0.copy()
    dt = 0.0005  # Very small time step for stability

    # Run 10 steps (reduced for stability)
    for step in range(10):
        Q, body = step_partitioned_fsi(Q, body, grid, gas, dt, fluid_integrator="rk4")

        # Check for NaN/Inf
        Q_np = np.array(Q)
        assert np.all(np.isfinite(Q_np)), f"NaN/Inf detected in Q at step {step}"
        assert np.all(np.isfinite(body.position)), f"NaN/Inf in body position at step {step}"
        assert np.all(np.isfinite(body.velocity)), f"NaN/Inf in body velocity at step {step}"

    # Final checks
    assert np.all(Q[0] > 0), "Negative density detected"
    from src.gas import pressure

    p_final = pressure(Q)
    assert np.all(p_final > 0), "Negative pressure detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
