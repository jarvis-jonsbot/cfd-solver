"""Main solver: RK4 time integration for 2D compressible Euler equations.

Implements the semi-discrete finite volume method on a structured
curvilinear grid with explicit Runge-Kutta time stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.backend import EPS_TINY, xp
from src.boundary import apply_freestream, apply_wall
from src.flux import roe_flux_1d
from src.gas import GAMMA, pressure, sound_speed
from src.grid import Grid
from src.numba_kernels import HAS_NUMBA, compute_dt_numba, compute_residual_numba
from src.pressure import solve_pressure
from src.reconstruction import muscl_reconstruct


@dataclass
class SolverConfig:
    """Configuration for the flow solver."""

    mach: float = 0.3
    alpha: float = 0.0  # angle of attack (radians)
    cfl: float = 0.5
    max_steps: int = 10000
    p_inf: float = 1.0
    rho_inf: float = 1.0
    print_interval: int = 100
    output_interval: int = 1000
    output_dir: str = "output"


def compute_dt(Q, grid: Grid, cfl: float) -> float:
    """Compute stable time step from CFL condition (acoustic + advective).

    dt = CFL * min(|J| / (|U| + a * |n|))

    In curvilinear coordinates, we use the spectral radius of the
    flux Jacobian in each direction. Includes both advective and acoustic
    wave speeds — appropriate for the explicit RK4 integrator.
    """
    if HAS_NUMBA:
        import numpy as np

        Qnp = np.asarray(Q)
        return compute_dt_numba(
            Qnp,
            np.asarray(grid.xi_x_area),
            np.asarray(grid.xi_y_area),
            np.asarray(grid.eta_x_area),
            np.asarray(grid.eta_y_area),
            np.asarray(grid.jacobian),
            grid.ni,
            grid.nj,
            cfl,
        )
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    p = pressure(Q)
    a = sound_speed(rho, p)

    # Spectral radii in ξ and η directions using area-weighted normals
    # Contravariant velocities (use area normals / |J| for proper scaling)
    J_abs = xp.abs(grid.jacobian) + EPS_TINY
    U_xi = u * grid.xi_x_area + v * grid.xi_y_area  # contravariant vel * |J|
    U_eta = u * grid.eta_x_area + v * grid.eta_y_area

    # Face normal magnitudes (area-weighted)
    xi_mag = xp.sqrt(grid.xi_x_area**2 + grid.xi_y_area**2)
    eta_mag = xp.sqrt(grid.eta_x_area**2 + grid.eta_y_area**2)

    sr_xi = xp.abs(U_xi) + a * xi_mag
    sr_eta = xp.abs(U_eta) + a * eta_mag

    # dt = CFL * |J| / (sr_xi + sr_eta)
    dt_local = cfl * J_abs / (sr_xi + sr_eta + EPS_TINY)
    return float(xp.min(dt_local))


def compute_dt_advective(Q, grid: Grid, cfl: float) -> float:
    """Compute stable time step using advective CFL only (no sound speed).

    dt = CFL * min(|J| / |U|)

    Omits the acoustic wave speed from the spectral radius. This is the
    correct CFL criterion for the semi-implicit integrator, which treats
    acoustic waves implicitly and is only constrained by the advective CFL.
    Using the full acoustic CFL here would defeat the purpose of the
    semi-implicit scheme (no large-dt benefit at low Mach numbers).

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object
        cfl: advective CFL number (can be > 1 for semi-implicit)

    Returns:
        dt: stable advective time step
    """
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho

    J_abs = xp.abs(grid.jacobian) + EPS_TINY
    U_xi = u * grid.xi_x_area + v * grid.xi_y_area
    U_eta = u * grid.eta_x_area + v * grid.eta_y_area

    sr_xi = xp.abs(U_xi)
    sr_eta = xp.abs(U_eta)

    dt_local = cfl * J_abs / (sr_xi + sr_eta + EPS_TINY)
    return float(xp.min(dt_local))


def compute_residual(Q, grid: Grid) -> object:
    """Compute the spatial residual R(Q) = -(1/J)(dF_hat/dξ + dG_hat/dη).

    Uses Numba-accelerated fused kernel when available, otherwise falls
    back to vectorized NumPy/CuPy implementation.

    Uses MUSCL reconstruction + Roe flux in each coordinate direction.
    Face normals are area-weighted (not divided by J) so the Roe flux
    returns the physical flux through each face. We then divide by J
    (cell volume) at the end.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object

    Returns:
        R: residual, shape (4, ni, nj)
    """
    if HAS_NUMBA:
        import numpy as np

        return compute_residual_numba(
            np.asarray(Q),
            np.asarray(grid.xi_x_area),
            np.asarray(grid.xi_y_area),
            np.asarray(grid.eta_x_area),
            np.asarray(grid.eta_y_area),
            np.asarray(grid.jacobian),
            grid.ni,
            grid.nj,
        )

    ni = grid.ni
    R = xp.zeros_like(Q)

    # --- ξ-direction fluxes (periodic) ---
    # Pad Q periodically: 2 ghost cells on each side
    Q_padded = xp.concatenate([Q[:, -2:, :], Q, Q[:, :2, :]], axis=1)
    QL_xi, QR_xi = muscl_reconstruct(Q_padded, axis=0)  # along dim 1

    # MUSCL on (ni+4) points produces (ni+4-3) = ni+1 interfaces — perfect for ni cells
    # Pad area-weighted normals the same way and average at interfaces
    sx_pad = xp.concatenate([grid.xi_x_area[-2:, :], grid.xi_x_area, grid.xi_x_area[:2, :]], axis=0)
    sy_pad = xp.concatenate([grid.xi_y_area[-2:, :], grid.xi_y_area, grid.xi_y_area[:2, :]], axis=0)

    # Interface i+1/2 normal = average of cell i and i+1 normals
    # After MUSCL trimming: interface k corresponds to padded cells k+1 and k+2
    n_ifaces = QL_xi.shape[1]
    nx_xi = 0.5 * (sx_pad[1 : 1 + n_ifaces, :] + sx_pad[2 : 2 + n_ifaces, :])
    ny_xi = 0.5 * (sy_pad[1 : 1 + n_ifaces, :] + sy_pad[2 : 2 + n_ifaces, :])

    F_xi = roe_flux_1d(QL_xi, QR_xi, nx_xi, ny_xi)

    # Accumulate: R -= F_{i+1/2} - F_{i-1/2} for each cell
    # F_xi has ni+1 interfaces. Cell i uses faces i and i+1.
    R -= F_xi[:, 1 : ni + 1, :] - F_xi[:, :ni, :]

    # --- η-direction fluxes (non-periodic) ---
    QL_eta, QR_eta = muscl_reconstruct(Q, axis=1)  # along dim 2

    # MUSCL on nj points produces nj-3 interfaces
    # Interface k is between cells j=k+1 and j=k+2 (0-indexed)
    n_efaces = QL_eta.shape[2]
    # Average area-weighted normals at η interfaces
    nx_eta = 0.5 * (grid.eta_x_area[:, 1 : 1 + n_efaces] + grid.eta_x_area[:, 2 : 2 + n_efaces])
    ny_eta = 0.5 * (grid.eta_y_area[:, 1 : 1 + n_efaces] + grid.eta_y_area[:, 2 : 2 + n_efaces])

    G_eta = roe_flux_1d(QL_eta, QR_eta, nx_eta, ny_eta)

    # Accumulate: cell j gets G_{j+1/2} - G_{j-1/2}
    # G_eta[:,:,k] is interface between j=k+1 and j=k+2
    # So cell j=k+2 has left face at k and right face at k+1
    # Cells updated: j=2 through j=2+(n_efaces-2) = j=n_efaces
    if n_efaces >= 2:
        R[:, :, 2 : 2 + n_efaces - 1] -= G_eta[:, :, 1:] - G_eta[:, :, :-1]

    # Also handle first-order flux at wall-adjacent cells (j=1) using
    # direct first-order Roe flux between j=0 (ghost) and j=1
    # This ensures the wall BC propagates into the domain
    nx_w = grid.eta_x_area[:, 0]
    ny_w = grid.eta_y_area[:, 0]
    F_wall = roe_flux_1d(Q[:, :, 0:1], Q[:, :, 1:2], nx_w[:, None], ny_w[:, None])
    # Cell j=1: left face is wall face, right face is G_eta[:,:,0] (if it exists)
    if n_efaces >= 1:
        R[:, :, 1:2] -= G_eta[:, :, 0:1] - F_wall[:, :, 0:1]

    # Divide by cell volume (|J|)
    R /= xp.abs(grid.jacobian[None, :, :]) + EPS_TINY

    return R


def step_semi_implicit(Q, dt, grid: Grid, bcs=None):
    """One semi-implicit time step using flux splitting.

    Algorithm:
    1. Extract primitive variables (rho, u, v, p)
    2. Compute explicit advective update for (rho, rho_u, rho_v) using first-order upwind
    3. Compute c^2 = gamma * p / rho
    4. Solve implicit pressure equation -> p^{n+1}
    5. Apply pressure gradient correction to momentum
    6. Recompute energy from updated pressure and velocities
    7. Apply boundary conditions
    8. Return Q^{n+1}

    Args:
        Q: conservative state, shape (4, ni, nj)
        dt: time step (should be based on advective CFL via compute_dt_advective)
        grid: Grid object (used for dx, dy estimates)
        bcs: boundary condition dict (unused in Phase 1)

    Returns:
        Q_new: updated conservative state, shape (4, ni, nj)

    Note:
        Phase 1 assumes uniform grid spacing. Uses simple first-order upwind
        for advective fluxes. Boundary conditions are periodic for now.
    """
    ni, nj = Q.shape[1], Q.shape[2]

    # Extract primitive variables
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    p = pressure(Q)

    J_abs = xp.abs(grid.jacobian) + EPS_TINY   # cell volume (area per unit depth)

    # Area-weighted contravariant volume fluxes (physical velocity · face area).
    # U_xi_area  = u·ξ_x_area + v·ξ_y_area  ≡  (physical flux through ξ-face) * sign
    # These are exactly what compute_dt_advective uses, so CFL is consistent.
    # Dividing by J gives the volume-specific rate; the divergence then is
    #   (1/J) · (F_{i+1/2} - F_{i-1/2})  with F = U_xi_area · q
    U_xi_area  = u * grid.xi_x_area  + v * grid.xi_y_area    # shape (ni, nj)
    U_eta_area = u * grid.eta_x_area + v * grid.eta_y_area

    # --- Step 1: Explicit advective update for mass and momentum ---
    # Finite-volume upwind divergence in curvilinear coordinates:
    #   Δq/Δt = -(1/J) · [F_{i+1/2} - F_{i-1/2}]
    # where F_{i+1/2} = upwind(U_xi_area_{i+1/2}) · q
    # This is the correct FV form on a curvilinear grid; no dx/h factors needed
    # because the area normals are already metric-weighted.

    rho_new   = rho.copy()
    rho_u_new = Q[1].copy()
    rho_v_new = Q[2].copy()

    # ξ-direction sweep (periodic in i)
    for i in range(ni):
        i_m = (i - 1) % ni
        i_p = (i + 1) % ni
        for j in range(nj):
            # Face velocity at i+1/2: average of neighbours (first-order)
            Uxi_p = 0.5 * (float(U_xi_area[i, j]) + float(U_xi_area[i_p, j]))
            Uxi_m = 0.5 * (float(U_xi_area[i_m, j]) + float(U_xi_area[i, j]))
            Jk    = float(J_abs[i, j])

            # Upwind selection at each face
            Frho_p = Uxi_p * float(rho[i, j])   if Uxi_p > 0 else Uxi_p * float(rho[i_p, j])
            Frho_m = Uxi_m * float(rho[i_m, j]) if Uxi_m > 0 else Uxi_m * float(rho[i, j])
            rho_new[i, j]   -= dt / Jk * (Frho_p - Frho_m)

            Fru_p = Uxi_p * float(Q[1, i, j])   if Uxi_p > 0 else Uxi_p * float(Q[1, i_p, j])
            Fru_m = Uxi_m * float(Q[1, i_m, j]) if Uxi_m > 0 else Uxi_m * float(Q[1, i, j])
            rho_u_new[i, j] -= dt / Jk * (Fru_p - Fru_m)

            Frv_p = Uxi_p * float(Q[2, i, j])   if Uxi_p > 0 else Uxi_p * float(Q[2, i_p, j])
            Frv_m = Uxi_m * float(Q[2, i_m, j]) if Uxi_m > 0 else Uxi_m * float(Q[2, i, j])
            rho_v_new[i, j] -= dt / Jk * (Frv_p - Frv_m)

    # η-direction sweep (clamped at boundaries)
    for i in range(ni):
        for j in range(nj):
            j_m = max(0, j - 1)
            j_p = min(nj - 1, j + 1)
            Ueta_p = 0.5 * (float(U_eta_area[i, j]) + float(U_eta_area[i, j_p]))
            Ueta_m = 0.5 * (float(U_eta_area[i, j_m]) + float(U_eta_area[i, j]))
            Jk     = float(J_abs[i, j])

            Frho_p = Ueta_p * float(rho[i, j])   if Ueta_p > 0 else Ueta_p * float(rho[i, j_p])
            Frho_m = Ueta_m * float(rho[i, j_m]) if Ueta_m > 0 else Ueta_m * float(rho[i, j])
            rho_new[i, j]   -= dt / Jk * (Frho_p - Frho_m)

            Fru_p = Ueta_p * float(Q[1, i, j])   if Ueta_p > 0 else Ueta_p * float(Q[1, i, j_p])
            Fru_m = Ueta_m * float(Q[1, i, j_m]) if Ueta_m > 0 else Ueta_m * float(Q[1, i, j])
            rho_u_new[i, j] -= dt / Jk * (Fru_p - Fru_m)

            Frv_p = Ueta_p * float(Q[2, i, j])   if Ueta_p > 0 else Ueta_p * float(Q[2, i, j_p])
            Frv_m = Ueta_m * float(Q[2, i, j_m]) if Ueta_m > 0 else Ueta_m * float(Q[2, i, j])
            rho_v_new[i, j] -= dt / Jk * (Frv_p - Frv_m)

    # --- Step 2: Compute c^2 for pressure solve ---
    c2 = GAMMA * p / xp.maximum(rho, EPS_TINY)

    # --- Step 3: Solve implicit pressure equation ---
    # Pass area-weighted face normals and Jacobian — the pressure operator
    # uses the same metric quantities as the gradient correction and
    # advective update, ensuring full coordinate consistency.
    p_new = solve_pressure(
        rho_new, rho_u_new, rho_v_new, c2,
        grid.xi_x_area, grid.xi_y_area, grid.eta_x_area, grid.eta_y_area,
        grid.jacobian, dt, p_wall_neumann=True, xp=xp
    )

    # --- Step 4: Apply pressure gradient correction to momentum ---
    # d(rho*u)/dt = -dp/dx * dt  (implicit pressure gradient)
    # d(rho*v)/dt = -dp/dy * dt
    #
    # IMPORTANT: i is the circumferential (ξ) index, j is the radial (η) index.
    # On an O-grid these are NOT aligned with x/y — we must use the chain rule:
    #
    #   ∂p/∂x = ∂p/∂ξ · ξ_x + ∂p/∂η · η_x
    #   ∂p/∂y = ∂p/∂ξ · ξ_y + ∂p/∂η · η_y
    #
    # Using index-space differences (dimensionless Δξ = Δη = 1) and the
    # contravariant metrics already stored on the Grid object.
    # Without this transform, pressure gradients are rotated by the local
    # grid angle — producing the SW/NE pressure inversion artifact.
    for i in range(ni):
        i_p = (i + 1) % ni
        i_m = (i - 1) % ni
        for j in range(nj):
            j_p = min(nj - 1, j + 1)
            j_m = max(0, j - 1)

            # Central differences in index space (Δξ = Δη = 1 by convention)
            dp_dxi  = (p_new[i_p, j] - p_new[i_m, j]) / 2.0
            dp_deta = (p_new[i, j_p] - p_new[i, j_m]) / 2.0

            # Chain rule: project onto physical (x, y) directions.
            # ξ_x = ξ_x_area / |J|  (contravariant = area-normal / cell-volume)
            # Using area normals + Jacobian avoids requiring xi_x/eta_x
            # attributes on the Grid (which may not be set in test fixtures).
            Jk = float(J_abs[i, j])
            xi_x_ij  =  float(grid.xi_x_area[i, j])  / Jk
            xi_y_ij  =  float(grid.xi_y_area[i, j])  / Jk
            eta_x_ij =  float(grid.eta_x_area[i, j]) / Jk
            eta_y_ij =  float(grid.eta_y_area[i, j]) / Jk
            dp_dx = dp_dxi * xi_x_ij  + dp_deta * eta_x_ij
            dp_dy = dp_dxi * xi_y_ij  + dp_deta * eta_y_ij

            rho_u_new[i, j] -= dt * dp_dx
            rho_v_new[i, j] -= dt * dp_dy

    # --- Step 5: Recompute energy from updated pressure and velocities ---
    u_new = rho_u_new / xp.maximum(rho_new, EPS_TINY)
    v_new = rho_v_new / xp.maximum(rho_new, EPS_TINY)
    rho_E_new = p_new / (GAMMA - 1.0) + 0.5 * rho_new * (u_new**2 + v_new**2)

    # Assemble updated state
    Q_new = xp.stack([rho_new, rho_u_new, rho_v_new, rho_E_new], axis=0)

    return Q_new


def solve(Q0, grid: Grid, config: SolverConfig, callback: Callable | None = None) -> object:
    """Run the solver with RK4 time integration.

    Args:
        Q0: initial conservative state, shape (4, ni, nj)
        grid: Grid object
        config: solver configuration
        callback: optional function called each output_interval with (step, t, Q)

    Returns:
        Q: final conservative state
    """
    Q = xp.array(Q0)
    t = 0.0

    for step in range(1, config.max_steps + 1):
        # Apply boundary conditions
        apply_wall(Q, grid)
        apply_freestream(Q, grid, config.mach, config.alpha, config.p_inf, config.rho_inf)

        # Compute stable time step
        dt = compute_dt(Q, grid, config.cfl)

        # RK4 stages
        k1 = compute_residual(Q, grid)
        Q1 = Q + 0.5 * dt * k1
        apply_wall(Q1, grid)
        apply_freestream(Q1, grid, config.mach, config.alpha, config.p_inf, config.rho_inf)

        k2 = compute_residual(Q1, grid)
        Q2 = Q + 0.5 * dt * k2
        apply_wall(Q2, grid)
        apply_freestream(Q2, grid, config.mach, config.alpha, config.p_inf, config.rho_inf)

        k3 = compute_residual(Q2, grid)
        Q3 = Q + dt * k3
        apply_wall(Q3, grid)
        apply_freestream(Q3, grid, config.mach, config.alpha, config.p_inf, config.rho_inf)

        k4 = compute_residual(Q3, grid)

        Q = Q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt

        if step % config.print_interval == 0:
            p = pressure(Q)
            rho_min = float(xp.min(Q[0, :, 1:-1]))
            rho_max = float(xp.max(Q[0, :, 1:-1]))
            p_min = float(xp.min(p[:, 1:-1]))
            p_max = float(xp.max(p[:, 1:-1]))
            print(
                f"Step {step:6d}  t={t:.6f}  dt={dt:.2e}  "
                f"rho=[{rho_min:.4f}, {rho_max:.4f}]  "
                f"p=[{p_min:.4f}, {p_max:.4f}]"
            )

        if callback and step % config.output_interval == 0:
            callback(step, t, Q)

    return Q
