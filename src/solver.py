"""Main solver: RK4 time integration for 2D compressible Euler equations.

Implements the semi-discrete finite volume method on a structured
curvilinear grid with explicit Runge-Kutta time stepping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

from src.backend import xp
from src.grid import Grid
from src.gas import GAMMA, pressure, sound_speed
from src.flux import roe_flux_1d
from src.reconstruction import muscl_reconstruct
from src.boundary import apply_freestream, apply_wall


@dataclass
class SolverConfig:
    """Configuration for the flow solver."""
    mach: float = 0.3
    alpha: float = 0.0          # angle of attack (radians)
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

    # Spectral radii in ξ and η directions
    # Contravariant velocities
    U_xi = u * grid.xi_x + v * grid.xi_y    # contravariant velocity in ξ
    U_eta = u * grid.eta_x + v * grid.eta_y  # contravariant velocity in η

    # Face normal magnitudes
    xi_mag = xp.sqrt(grid.xi_x**2 + grid.xi_y**2)
    eta_mag = xp.sqrt(grid.eta_x**2 + grid.eta_y**2)

    sr_xi = xp.abs(U_xi) + a * xi_mag
    sr_eta = xp.abs(U_eta) + a * eta_mag

    # dt from each direction
    dt_local = cfl / (sr_xi + sr_eta + 1e-30)
    return float(xp.min(dt_local))


def compute_residual(Q, grid: Grid) -> object:
    """Compute the spatial residual R(Q) = -dF/dξ - dG/dη.

    Uses MUSCL reconstruction + Roe flux in each coordinate direction.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object

    Returns:
        R: residual, shape (4, ni, nj)
    """
    ni, nj = grid.ni, grid.nj
    R = xp.zeros_like(Q)

    # --- ξ-direction fluxes ---
    # Reconstruct in ξ (periodic direction)
    # Pad Q periodically for reconstruction stencil
    Q_padded = xp.concatenate([Q[:, -2:, :], Q, Q[:, :2, :]], axis=1)
    QL_xi, QR_xi = muscl_reconstruct(Q_padded, axis=0)  # along ni+4 dim

    # Face normals at ξ interfaces (averaged from neighboring cells)
    # Interface i+1/2 is between cells i and i+1
    # After padding of 2, interior starts at index 2
    # QL_xi, QR_xi have shape (4, ni+4-3, nj) = (4, ni+1, nj)
    # We need ni interfaces for the periodic domain

    xi_x_pad = xp.concatenate([grid.xi_x[-2:, :], grid.xi_x, grid.xi_x[:2, :]], axis=0)
    xi_y_pad = xp.concatenate([grid.xi_y[-2:, :], grid.xi_y, grid.xi_y[:2, :]], axis=0)

    # Average normals at interfaces
    nx_xi = 0.5 * (xi_x_pad[1:-1, :] + xi_x_pad[2:, :])  # shape (ni+2, nj)
    ny_xi = 0.5 * (xi_y_pad[1:-1, :] + xi_y_pad[2:, :])

    # Trim to match QL_xi size
    n_interfaces = QL_xi.shape[1]
    nx_xi = nx_xi[:n_interfaces, :]
    ny_xi = ny_xi[:n_interfaces, :]

    F_xi = roe_flux_1d(QL_xi, QR_xi, nx_xi, ny_xi)

    # Accumulate: R -= (F_{i+1/2} - F_{i-1/2}) for each cell
    # F_xi has n_interfaces = ni+1 faces. After trimming for the periodic domain:
    # We need exactly ni+1 interfaces to get ni cells
    # Map back from padded to original indices
    for eq in range(4):
        if F_xi.shape[1] >= ni + 1:
            R[eq, :, :] -= (F_xi[eq, 1:ni+1, :] - F_xi[eq, :ni, :])
        else:
            # Fallback: first-order in ξ
            for i in range(ni):
                ip1 = (i + 1) % ni
                im1 = (i - 1) % ni
                nx = 0.5 * (grid.xi_x[i, :] + grid.xi_x[ip1, :])
                ny = 0.5 * (grid.xi_y[i, :] + grid.xi_y[ip1, :])
                F_p = roe_flux_1d(Q[:, i:i+1, :], Q[:, ip1:ip1+1, :], nx[None, :], ny[None, :])
                nx_m = 0.5 * (grid.xi_x[im1, :] + grid.xi_x[i, :])
                ny_m = 0.5 * (grid.xi_y[im1, :] + grid.xi_y[i, :])
                F_m = roe_flux_1d(Q[:, im1:im1+1, :], Q[:, i:i+1, :], nx_m[None, :], ny_m[None, :])
                R[eq, i, :] -= (F_p[eq, 0, :] - F_m[eq, 0, :])

    # --- η-direction fluxes ---
    QL_eta, QR_eta = muscl_reconstruct(Q, axis=1)  # along nj dim

    # Face normals for η interfaces
    n_eta_faces = QL_eta.shape[2]
    eta_x_avg = 0.5 * (grid.eta_x[:, 1:n_eta_faces+1] + grid.eta_x[:, 2:n_eta_faces+2]) \
        if grid.eta_x.shape[1] > n_eta_faces + 1 else grid.eta_x[:, :n_eta_faces]
    eta_y_avg = 0.5 * (grid.eta_y[:, 1:n_eta_faces+1] + grid.eta_y[:, 2:n_eta_faces+2]) \
        if grid.eta_y.shape[1] > n_eta_faces + 1 else grid.eta_y[:, :n_eta_faces]

    G_eta = roe_flux_1d(QL_eta, QR_eta, eta_x_avg, eta_y_avg)

    # Accumulate η fluxes for interior cells
    nf = G_eta.shape[2]
    # G_eta faces correspond to interfaces j+1/2 for j in [1..nj-3] (from MUSCL trimming)
    # Map: G_eta[:,:,k] is face between cell j=k+1 and j=k+2
    for eq in range(4):
        if nf >= 2:
            R[eq, :, 2:2+nf-1] -= (G_eta[eq, :, 1:] - G_eta[eq, :, :-1])

    # Divide by cell volume (Jacobian)
    R /= xp.abs(grid.jacobian[None, :, :]) + 1e-30

    return R


def solve(Q0, grid: Grid, config: SolverConfig,
          callback: Optional[Callable] = None) -> object:
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
            print(f"Step {step:6d}  t={t:.6f}  dt={dt:.2e}  "
                  f"rho=[{rho_min:.4f}, {rho_max:.4f}]  "
                  f"p=[{p_min:.4f}, {p_max:.4f}]")

        if callback and step % config.output_interval == 0:
            callback(step, t, Q)

    return Q
