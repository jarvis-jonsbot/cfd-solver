#!/usr/bin/env python3
"""Sod shock tube test — validates 1D Roe solver against exact solution.

The Sod problem is a standard Riemann problem with known exact solution:
  Left:  (rho, u, p) = (1.0, 0.0, 1.0)
  Right: (rho, u, p) = (0.125, 0.0, 0.1)

We solve it on a 1D domain mapped to the 2D solver by using a single
row in the transverse direction (ni=1, nj=N) with flat metrics.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.backend import xp
from src.flux import roe_flux_1d
from src.gas import GAMMA
from src.reconstruction import muscl_reconstruct


def sod_exact(x, t, x0=0.5):
    """Exact solution to the Sod shock tube problem at time t.

    Returns (rho, u, p) arrays.
    """
    gamma = GAMMA
    # Left and right states
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1

    # Exact solution parameters (from Toro, Ch. 4)
    # Post-shock (region 3): p* and u* from iterative solution
    # Using known values for gamma=1.4 Sod problem
    p_star = 0.30313
    u_star = 0.92745
    rho_star_L = 0.42632  # post-expansion (region 2)
    rho_star_R = 0.26557  # post-shock (region 3)

    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    a_star_L = aL * (p_star / pL) ** ((gamma - 1) / (2 * gamma))

    # Wave speeds
    S_HL = uL - aL  # head of rarefaction
    S_TL = u_star - a_star_L  # tail of rarefaction
    S_shock = (  # shock speed
        u_star + aR * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / pR - 1) + 1)
    )
    S_contact = u_star  # contact discontinuity speed

    xi = (x - x0) / max(t, 1e-30)

    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    for i in range(len(x)):
        if xi[i] <= S_HL:
            # Region 1 (left state)
            rho[i], u[i], p[i] = rhoL, uL, pL
        elif xi[i] <= S_TL:
            # Region 2 (rarefaction fan)
            rho[i] = rhoL * (
                2 / (gamma + 1)
                + (gamma - 1) / ((gamma + 1) * aL) * (uL - xi[i] * t / max(t, 1e-30))
            ) ** (2 / (gamma - 1))
            # Simpler: use self-similar solution
            cs = 2 / (gamma + 1) * (aL + (gamma - 1) / 2 * (uL - (x[i] - x0) / t))
            u[i] = 2 / (gamma + 1) * (aL + (gamma - 1) / 2 * uL + (x[i] - x0) / t)
            rho[i] = rhoL * (cs / aL) ** (2 / (gamma - 1))
            p[i] = pL * (cs / aL) ** (2 * gamma / (gamma - 1))
        elif xi[i] <= S_contact:
            # Region 3 (post-expansion, pre-contact)
            rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        elif xi[i] <= S_shock:
            # Region 4 (post-contact, pre-shock)
            rho[i], u[i], p[i] = rho_star_R, u_star, p_star
        else:
            # Region 5 (right state)
            rho[i], u[i], p[i] = rhoR, uR, pR

    return rho, u, p


def run_sod_1d(n_cells: int = 200, t_final: float = 0.2, cfl: float = 0.5):
    """Run the Sod shock tube problem using our Roe solver.

    Args:
        n_cells: number of grid cells
        t_final: simulation end time
        cfl: CFL number

    Returns:
        x_centers, rho, u, p (numerical solution)
    """
    dx = 1.0 / n_cells
    x = xp.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_cells)

    gamma = GAMMA

    # Initialize: left state for x < 0.5, right state for x >= 0.5
    rho = xp.where(x < 0.5, 1.0, 0.125)
    u = xp.zeros(n_cells)
    v = xp.zeros(n_cells)
    p = xp.where(x < 0.5, 1.0, 0.1)
    E = p / ((gamma - 1.0) * rho) + 0.5 * u**2

    # Conservative variables as (4, 1, n_cells) to reuse our 2D flux/recon
    Q = xp.stack([rho, rho * u, rho * v, rho * E], axis=0)
    Q = Q[:, None, :]  # shape (4, 1, n_cells)

    t = 0.0

    while t < t_final:
        # CFL time step
        rho_c = Q[0, 0, :]
        u_c = Q[1, 0, :] / rho_c
        p_c = (gamma - 1.0) * (Q[3, 0, :] - 0.5 * Q[1, 0, :] ** 2 / rho_c)
        a_c = xp.sqrt(gamma * xp.abs(p_c) / rho_c)
        dt = float(cfl * dx / xp.max(xp.abs(u_c) + a_c))
        if t + dt > t_final:
            dt = t_final - t

        # MUSCL reconstruct in η-direction (axis=1, which is dim 2)
        # Need ghost cells at boundaries (transmissive)
        Q_ext = xp.concatenate([Q[:, :, :2], Q, Q[:, :, -2:]], axis=2)
        QL, QR = muscl_reconstruct(Q_ext, axis=1)

        # Compute fluxes
        n_faces = QL.shape[2]
        F = roe_flux_1d(QL, QR, xp.ones((1, n_faces)), xp.zeros((1, n_faces)))

        # Update interior cells
        # F has n_faces = n_cells+4-3 = n_cells+1 faces
        # After ghost padding of 2 on each side, interior faces map to [1:-1]
        n_int = min(F.shape[2] - 1, n_cells)
        dQ = xp.zeros_like(Q)
        dQ[:, :, :n_int] = -(F[:, :, 1 : n_int + 1] - F[:, :, :n_int]) / dx

        # Simple forward Euler (for the test; could use RK4)
        Q = Q + dt * dQ
        t += dt

    # Extract results
    rho_out = Q[0, 0, :]
    u_out = Q[1, 0, :] / rho_out
    p_out = (gamma - 1.0) * (Q[3, 0, :] - 0.5 * Q[1, 0, :] ** 2 / rho_out)

    return x, rho_out, u_out, p_out


def test_sod():
    """Test Sod shock tube: compare numerical vs exact solution."""
    n_cells = 400
    t_final = 0.2

    x_num, rho_num, u_num, p_num = run_sod_1d(n_cells=n_cells, t_final=t_final)

    x_np = np.asarray(x_num if not hasattr(x_num, "get") else x_num.get())
    rho_exact, u_exact, p_exact = sod_exact(x_np, t_final)

    rho_np = np.asarray(rho_num if not hasattr(rho_num, "get") else rho_num.get())
    u_np = np.asarray(u_num if not hasattr(u_num, "get") else u_num.get())
    p_np = np.asarray(p_num if not hasattr(p_num, "get") else p_num.get())

    # L1 error norms (should be small — <5% for 400 cells)
    rho_err = np.mean(np.abs(rho_np - rho_exact)) / np.mean(np.abs(rho_exact))
    u_err = np.mean(np.abs(u_np - u_exact)) / max(np.mean(np.abs(u_exact)), 1e-10)
    p_err = np.mean(np.abs(p_np - p_exact)) / np.mean(np.abs(p_exact))

    print(f"Sod shock tube (N={n_cells}, t={t_final})")
    print(f"  Density L1 relative error:  {rho_err:.4f}")
    print(f"  Velocity L1 relative error: {u_err:.4f}")
    print(f"  Pressure L1 relative error: {p_err:.4f}")

    # Assertions — reasonable accuracy for 2nd-order with 400 cells
    assert rho_err < 0.10, f"Density error too large: {rho_err}"
    assert p_err < 0.10, f"Pressure error too large: {p_err}"

    print("  ✅ Sod shock tube test PASSED")


if __name__ == "__main__":
    test_sod()
