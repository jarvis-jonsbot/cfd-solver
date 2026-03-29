"""Implicit pressure solver for semi-implicit time integration.

Solves the variable-coefficient elliptic equation arising from
implicit treatment of acoustic waves:
    (I + rho * c^2 * dt^2 * div(1/rho * grad)) p^{n+1} = p^*

Uses finite differences on a uniform grid (Phase 1 simplification).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.backend import EPS_TINY, to_numpy


def solve_pressure(rho, rho_u, rho_v, c2, dx, dy, dt, p_wall_neumann=True, xp=None):
    """Solve implicit pressure equation using conjugate gradient.

    Assembles the sparse matrix for:
        (I + rho * c^2 * dt^2 * div(1/rho * grad)) p^{n+1} = rhs

    where rhs is computed from the current pressure estimate p^*.

    Args:
        rho: density field, shape (ni, nj)
        rho_u: x-momentum field (for initial pressure guess), shape (ni, nj)
        rho_v: y-momentum field (for initial pressure guess), shape (ni, nj)
        c2: sound speed squared (gamma * p / rho), shape (ni, nj)
        dx: grid spacing in x
        dy: grid spacing in y
        dt: time step
        p_wall_neumann: if True, apply Neumann BC (dp/dn=0) at j=0 wall
        xp: array module (for converting back to backend arrays)

    Returns:
        p_new: updated pressure field, shape (ni, nj)

    Note:
        This implementation assumes uniform grid spacing (dx, dy) for simplicity.
        Boundary conditions: periodic in i (circumferential), Neumann at j=0 wall,
        extrapolation at j=nj-1 freestream.
    """
    # Convert to NumPy for scipy sparse solver
    rho_np = to_numpy(rho)
    c2_np = to_numpy(c2)

    ni, nj = rho_np.shape
    n_cells = ni * nj

    # Initial pressure guess from equation of state
    from src.gas import GAMMA
    u = to_numpy(rho_u) / (rho_np + EPS_TINY)
    v = to_numpy(rho_v) / (rho_np + EPS_TINY)
    # Use a simple estimate: p ~ rho * c^2 / gamma
    p_guess = rho_np * c2_np / GAMMA

    # Flatten arrays for matrix assembly: index = i * nj + j
    rho_flat = rho_np.ravel()
    c2_flat = c2_np.ravel()
    p_flat = p_guess.ravel()

    # Build sparse matrix A for (I + L) p = p_rhs
    # where L = rho * c^2 * dt^2 * div(1/rho * grad)

    # Use list-of-lists format for efficient assembly
    data = []
    rows = []
    cols = []

    def idx(i, j):
        """Convert (i, j) to flat index with periodic wrapping in i."""
        i = i % ni
        j = max(0, min(j, nj - 1))  # Clamp j
        return i * nj + j

    # Finite difference stencil for variable-coefficient Laplacian
    # d/dx(1/rho * dp/dx) ≈ 1/dx^2 * [(1/rho_{i+1/2}) * (p_{i+1} - p_i)
    #                                  - (1/rho_{i-1/2}) * (p_i - p_{i-1})]
    # where 1/rho_{i+1/2} = 2/(rho_i + rho_{i+1})  (harmonic average)

    for i in range(ni):
        for j in range(nj):
            k = idx(i, j)

            # Diagonal: identity + negative sum of off-diagonals
            diag = 1.0

            # Coefficient for this cell
            coef = rho_np[i, j] * c2_np[i, j] * dt * dt

            # --- x-direction (periodic) ---
            # Forward difference at i+1/2
            i_p = (i + 1) % ni
            rho_half_p = 2.0 / (rho_np[i, j] + rho_np[i_p, j] + EPS_TINY)
            a_p = coef * rho_half_p / (dx * dx)

            rows.append(k)
            cols.append(idx(i_p, j))
            data.append(a_p)
            diag -= a_p

            # Backward difference at i-1/2
            i_m = (i - 1) % ni
            rho_half_m = 2.0 / (rho_np[i, j] + rho_np[i_m, j] + EPS_TINY)
            a_m = coef * rho_half_m / (dx * dx)

            rows.append(k)
            cols.append(idx(i_m, j))
            data.append(a_m)
            diag -= a_m

            # --- y-direction ---
            # Forward difference at j+1/2
            if j < nj - 1:
                rho_half_p = 2.0 / (rho_np[i, j] + rho_np[i, j + 1] + EPS_TINY)
                b_p = coef * rho_half_p / (dy * dy)

                rows.append(k)
                cols.append(idx(i, j + 1))
                data.append(b_p)
                diag -= b_p

            # Backward difference at j-1/2
            if j > 0:
                rho_half_m = 2.0 / (rho_np[i, j] + rho_np[i, j - 1] + EPS_TINY)
                b_m = coef * rho_half_m / (dy * dy)

                rows.append(k)
                cols.append(idx(i, j - 1))
                data.append(b_m)
                diag -= b_m
            elif p_wall_neumann:
                # At j=0 wall: Neumann BC dp/dn = 0
                # Implement as one-sided difference: p[i,0] = p[i,1]
                # This is already handled by not adding the j-1 term
                pass

            # Add diagonal entry
            rows.append(k)
            cols.append(k)
            data.append(diag)

    # Assemble sparse matrix
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    A = A.tocsr()

    # Right-hand side: current pressure estimate
    rhs = p_flat.copy()

    # Solve using conjugate gradient
    # Use current pressure as initial guess for faster convergence
    p_new_flat, info = spla.cg(A, rhs, x0=p_flat, rtol=1e-6, maxiter=1000)

    if info != 0:
        import warnings
        warnings.warn(f"Pressure solve CG did not converge: info={info}", stacklevel=2)

    # Reshape to 2D
    p_new = p_new_flat.reshape((ni, nj))

    # Convert back to backend array if xp provided
    if xp is not None:
        p_new = xp.array(p_new)

    return p_new
