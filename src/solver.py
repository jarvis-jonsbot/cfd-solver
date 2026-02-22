"""Main solver: RK4 time integration for 2D compressible Euler equations.

Implements the semi-discrete finite volume method on a structured
curvilinear grid with explicit Runge-Kutta time stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.backend import xp
from src.boundary import apply_freestream, apply_wall
from src.flux import roe_flux_1d
from src.gas import pressure, sound_speed
from src.grid import Grid
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
    """Compute stable time step from CFL condition.

    dt = CFL * min(dx / (|u| + a))

    In curvilinear coordinates, we use the spectral radius of the
    flux Jacobian in each direction.
    """
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    p = pressure(Q)
    a = sound_speed(rho, p)

    # Spectral radii in ξ and η directions using area-weighted normals
    # Contravariant velocities (use area normals / |J| for proper scaling)
    J_abs = xp.abs(grid.jacobian) + 1e-30
    U_xi = u * grid.xi_x_area + v * grid.xi_y_area  # contravariant vel * |J|
    U_eta = u * grid.eta_x_area + v * grid.eta_y_area

    # Face normal magnitudes (area-weighted)
    xi_mag = xp.sqrt(grid.xi_x_area**2 + grid.xi_y_area**2)
    eta_mag = xp.sqrt(grid.eta_x_area**2 + grid.eta_y_area**2)

    sr_xi = xp.abs(U_xi) + a * xi_mag
    sr_eta = xp.abs(U_eta) + a * eta_mag

    # dt = CFL * |J| / (sr_xi + sr_eta)
    dt_local = cfl * J_abs / (sr_xi + sr_eta + 1e-30)
    return float(xp.min(dt_local))


def compute_residual(Q, grid: Grid) -> object:
    """Compute the spatial residual R(Q) = -(1/J)(dF_hat/dξ + dG_hat/dη).

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
    R /= xp.abs(grid.jacobian[None, :, :]) + 1e-30

    return R


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
    Q = Q0.copy()
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
