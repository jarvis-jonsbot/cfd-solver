"""Implicit pressure solver for semi-implicit time integration.

Solves the variable-coefficient elliptic equation arising from
implicit treatment of acoustic waves in curvilinear coordinates:

    (I - dt² · ρc² · ∇·(1/ρ ∇)) p^{n+1} = p^*

The Laplacian is expressed in index (ξ, η) coordinates using the
area-weighted face normals already stored on the Grid:

    ξ-face normal magnitude  |n_ξ|  = sqrt(ξ_x_area² + ξ_y_area²)  = dr  (radial spacing)
    η-face normal magnitude  |n_η|  = sqrt(η_x_area² + η_y_area²)  = r·dθ (arc spacing)

These are the *physical* face areas per unit depth. The finite-difference
coefficient for the ξ-direction Laplacian term is:

    a_{i±1/2} = ρc²·dt² · (1/ρ)_{i±1/2} · |n_ξ|²_{i±1/2} / |J|

which after dividing by |J| (cell volume) gives units of 1/time², consistent
with the identity term. Using area-weighted normals (not divided by J) ensures
the stencil coefficients are O(1) across the grid, keeping the matrix
well-conditioned for CG.

A Jacobi (diagonal) preconditioner is applied to handle the residual
variation across the stretched grid.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]

from src.backend import EPS_TINY, to_numpy


def solve_pressure(
    rho,
    rho_u,
    rho_v,
    c2,
    xi_x_area,
    xi_y_area,
    eta_x_area,
    eta_y_area,
    jacobian,
    dt,
    p_wall_neumann=True,
    xp=None,
):
    """Solve implicit pressure equation using preconditioned conjugate gradient.

    Assembles the sparse matrix for:
        A p^{n+1} = p^*

    where A = I + L, with L the variable-coefficient curvilinear Laplacian
    discretised using area-weighted face normals divided by the cell Jacobian.

    Args:
        rho: density field, shape (ni, nj)
        rho_u, rho_v: momentum fields, shape (ni, nj)
        c2: sound speed squared (gamma * p / rho), shape (ni, nj)
        xi_x_area, xi_y_area:   ξ-face area-weighted normals, shape (ni, nj)
        eta_x_area, eta_y_area: η-face area-weighted normals, shape (ni, nj)
        jacobian: cell Jacobian (signed area element), shape (ni, nj)
        dt: time step
        p_wall_neumann: apply dp/dn=0 at j=0
        xp: array module for output conversion

    Returns:
        p_new: updated pressure field, shape (ni, nj)
    """
    rho_np = to_numpy(rho)
    c2_np = to_numpy(c2)
    xi_xa_np = to_numpy(xi_x_area)
    xi_ya_np = to_numpy(xi_y_area)
    eta_xa_np = to_numpy(eta_x_area)
    eta_ya_np = to_numpy(eta_y_area)
    jac_np = to_numpy(jacobian)

    ni, nj = rho_np.shape
    n_cells = ni * nj
    J_abs = np.abs(jac_np) + EPS_TINY  # cell volume (area per unit depth)

    # Face area magnitudes: |n_ξ| = dr, |n_η| = r·dθ
    nxi_sq = xi_xa_np**2 + xi_ya_np**2  # shape (ni, nj)
    neta_sq = eta_xa_np**2 + eta_ya_np**2

    from src.gas import GAMMA

    p_guess = rho_np * c2_np / GAMMA
    p_flat = p_guess.ravel()

    data, rows, cols = [], [], []

    def idx(i, j):
        return (i % ni) * nj + max(0, min(j, nj - 1))

    for i in range(ni):
        i_p = (i + 1) % ni
        i_m = (i - 1) % ni
        for j in range(nj):
            k = i * nj + j
            diag = 1.0
            Jk = J_abs[i, j]
            coef = c2_np[i, j] * dt * dt / Jk  # ρc²dt²/J; ρ cancels with 1/ρ harmonic mean

            # ξ-direction: face i+1/2 uses average face area and harmonic mean density
            # fmt: off
            nxi_p  = 0.5 * (nxi_sq[i, j]   + nxi_sq[i_p, j])
            rho_hp = 2.0 / (rho_np[i, j] + rho_np[i_p, j] + EPS_TINY)
            a_p    = coef * rho_np[i, j] * rho_hp * nxi_p
            # fmt: on

            rows.append(k)
            cols.append(idx(i_p, j))
            data.append(-a_p)
            diag += a_p

            # fmt: off
            nxi_m  = 0.5 * (nxi_sq[i, j]   + nxi_sq[i_m, j])
            rho_hm = 2.0 / (rho_np[i, j] + rho_np[i_m, j] + EPS_TINY)
            a_m    = coef * rho_np[i, j] * rho_hm * nxi_m
            # fmt: on

            rows.append(k)
            cols.append(idx(i_m, j))
            data.append(-a_m)
            diag += a_m

            # η-direction
            if j < nj - 1:
                # fmt: off
                neta_p = 0.5 * (neta_sq[i, j] + neta_sq[i, j + 1])
                rho_hp = 2.0 / (rho_np[i, j] + rho_np[i, j + 1] + EPS_TINY)
                b_p    = coef * rho_np[i, j] * rho_hp * neta_p
                # fmt: on

                rows.append(k)
                cols.append(idx(i, j + 1))
                data.append(-b_p)
                diag += b_p

            if j > 0:
                # fmt: off
                neta_m = 0.5 * (neta_sq[i, j] + neta_sq[i, j - 1])
                rho_hm = 2.0 / (rho_np[i, j] + rho_np[i, j - 1] + EPS_TINY)
                b_m    = coef * rho_np[i, j] * rho_hm * neta_m
                # fmt: on

                rows.append(k)
                cols.append(idx(i, j - 1))
                data.append(-b_m)
                diag += b_m
            # j=0 Neumann: no j-1 term (dp/dn = 0 enforced by omission)

            rows.append(k)
            cols.append(k)
            data.append(diag)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    rhs = p_flat.copy()

    # Jacobi preconditioner: M^{-1} = diag(A)^{-1}
    # Critical for convergence on stretched grids where diagonal varies by ~100x
    diag_A = np.array(A.diagonal())
    diag_A[diag_A == 0] = 1.0
    M = sp.diags(1.0 / diag_A)

    p_new_flat, info = spla.cg(A, rhs, x0=p_flat, M=M, rtol=1e-6, maxiter=2000)

    if info != 0:
        import warnings

        warnings.warn(f"Pressure solve CG did not converge: info={info}", stacklevel=2)

    p_new = p_new_flat.reshape((ni, nj))
    if xp is not None:
        p_new = xp.array(p_new)
    return p_new

