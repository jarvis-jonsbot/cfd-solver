"""Implicit pressure solver for semi-implicit time integration.

Solves the variable-coefficient elliptic equation arising from
implicit treatment of acoustic waves in curvilinear coordinates:

    (I - dt² · ρc² · ∇·(1/ρ ∇)) p^{n+1} = p^*

The Laplacian is expressed in index (ξ, η) coordinates using the
local metric magnitudes |∇ξ|² = ξ_x² + ξ_y² and |∇η|² = η_x² + η_y²
so that the pressure operator is consistent with the curvilinear
gradient used in the momentum correction step.

Uses finite differences on the curvilinear grid (Phase 1: diagonal metric
terms only, off-diagonal ξ·η cross-terms deferred to Phase 2).
"""

from __future__ import annotations

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.backend import EPS_TINY, to_numpy


def solve_pressure(rho, rho_u, rho_v, c2, xi_x, xi_y, eta_x, eta_y,
                   dt, p_wall_neumann=True, xp=None):
    """Solve implicit pressure equation using conjugate gradient.

    Assembles the sparse matrix for:
        (I - dt² · ρc² · ∇·(1/ρ ∇)) p^{n+1} = p^*

    The spatial operator is discretised in index (ξ, η) coordinates
    using local metric magnitudes:
        - ξ-direction: 1/dξ² → |∇ξ|²_i  (= ξ_x² + ξ_y²  at cell i,j)
        - η-direction: 1/dη² → |∇η|²_i

    This ensures the pressure Laplacian and the momentum-correction
    gradient (which uses the same contravariant metrics) are discretised
    identically — preventing the coordinate mismatch that caused the
    pressure fluctuations when a uniform dx_avg was used here while
    exact chain-rule differences were used in the momentum step.

    Args:
        rho: density field, shape (ni, nj)
        rho_u: x-momentum field, shape (ni, nj)
        rho_v: y-momentum field, shape (ni, nj)
        c2: sound speed squared (gamma * p / rho), shape (ni, nj)
        xi_x:  ∂ξ/∂x contravariant metric, shape (ni, nj)
        xi_y:  ∂ξ/∂y contravariant metric, shape (ni, nj)
        eta_x: ∂η/∂x contravariant metric, shape (ni, nj)
        eta_y: ∂η/∂y contravariant metric, shape (ni, nj)
        dt: time step
        p_wall_neumann: apply Neumann BC (dp/dn=0) at j=0 wall
        xp: array module (for converting back to backend arrays)

    Returns:
        p_new: updated pressure field, shape (ni, nj)
    """
    # Convert to NumPy for scipy sparse solver
    rho_np  = to_numpy(rho)
    c2_np   = to_numpy(c2)
    xi_x_np  = to_numpy(xi_x)
    xi_y_np  = to_numpy(xi_y)
    eta_x_np = to_numpy(eta_x)
    eta_y_np = to_numpy(eta_y)

    ni, nj = rho_np.shape
    n_cells = ni * nj

    # Metric magnitudes squared: |∇ξ|² and |∇η|²
    # These replace 1/dx² and 1/dy² in the stencil, giving a
    # curvilinear-consistent discrete Laplacian.
    grad_xi_sq  = xi_x_np**2  + xi_y_np**2    # shape (ni, nj)
    grad_eta_sq = eta_x_np**2 + eta_y_np**2

    # Initial pressure guess: p ~ rho * c^2 / gamma
    from src.gas import GAMMA
    p_guess = rho_np * c2_np / GAMMA
    p_flat  = p_guess.ravel()

    data, rows, cols = [], [], []

    def idx(i, j):
        i = i % ni
        j = max(0, min(j, nj - 1))
        return i * nj + j

    for i in range(ni):
        for j in range(nj):
            k    = idx(i, j)
            diag = 1.0
            coef = rho_np[i, j] * c2_np[i, j] * dt * dt

            # --- ξ-direction (periodic in i) ---
            i_p = (i + 1) % ni
            i_m = (i - 1) % ni

            # Metric magnitude at i+1/2 interface (average of neighbours)
            gxi_p = 0.5 * (grad_xi_sq[i, j] + grad_xi_sq[i_p, j])
            rho_h_p = 2.0 / (rho_np[i, j] + rho_np[i_p, j] + EPS_TINY)
            a_p = coef * rho_h_p * gxi_p

            rows.append(k); cols.append(idx(i_p, j)); data.append(-a_p)
            diag += a_p

            gxi_m = 0.5 * (grad_xi_sq[i, j] + grad_xi_sq[i_m, j])
            rho_h_m = 2.0 / (rho_np[i, j] + rho_np[i_m, j] + EPS_TINY)
            a_m = coef * rho_h_m * gxi_m

            rows.append(k); cols.append(idx(i_m, j)); data.append(-a_m)
            diag += a_m

            # --- η-direction ---
            if j < nj - 1:
                geta_p = 0.5 * (grad_eta_sq[i, j] + grad_eta_sq[i, j + 1])
                rho_h_p = 2.0 / (rho_np[i, j] + rho_np[i, j + 1] + EPS_TINY)
                b_p = coef * rho_h_p * geta_p

                rows.append(k); cols.append(idx(i, j + 1)); data.append(-b_p)
                diag += b_p

            if j > 0:
                geta_m = 0.5 * (grad_eta_sq[i, j] + grad_eta_sq[i, j - 1])
                rho_h_m = 2.0 / (rho_np[i, j] + rho_np[i, j - 1] + EPS_TINY)
                b_m = coef * rho_h_m * geta_m

                rows.append(k); cols.append(idx(i, j - 1)); data.append(-b_m)
                diag += b_m
            # j=0: Neumann — no j-1 term added (dp/dη = 0 at wall)

            rows.append(k); cols.append(k); data.append(diag)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    rhs = p_flat.copy()

    p_new_flat, info = spla.cg(A, rhs, x0=p_flat, rtol=1e-6, maxiter=1000)

    if info != 0:
        import warnings
        warnings.warn(f"Pressure solve CG did not converge: info={info}", stacklevel=2)

    p_new = p_new_flat.reshape((ni, nj))
    if xp is not None:
        p_new = xp.array(p_new)
    return p_new
