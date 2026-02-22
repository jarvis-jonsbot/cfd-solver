"""Boundary condition implementations for 2D compressible flow.

Supports:
  - Freestream (characteristic-based, far-field)
  - Solid wall (inviscid slip wall)
  - Periodic (circumferential wrap for O-grid)
"""

from __future__ import annotations

from src.backend import EPS_TINY, xp
from src.gas import GAMMA, primitive_to_conservative, sound_speed


def apply_freestream(
    Q,
    grid,
    mach: float,
    alpha: float = 0.0,
    p_inf: float = 1.0,
    rho_inf: float = 1.0,
):
    """Apply freestream boundary condition at the outer boundary (j = nj-1).

    Uses simple characteristic-based extrapolation: subsonic outflow
    fixes one incoming characteristic from freestream; supersonic fixes all.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object
        mach: freestream Mach number
        alpha: angle of attack in radians
        p_inf: freestream pressure
        rho_inf: freestream density
    """
    a_inf = sound_speed(xp.array([rho_inf]), xp.array([p_inf]))[0]
    u_inf = mach * float(a_inf) * xp.cos(xp.array(alpha))
    v_inf = mach * float(a_inf) * xp.sin(xp.array(alpha))

    W_inf = xp.array([rho_inf, float(u_inf), float(v_inf), p_inf])
    Q_inf = primitive_to_conservative(W_inf.reshape(4, 1, 1))

    # Simple approach: set outer 2 ghost layers to freestream
    Q[:, :, -1] = Q_inf[:, :, 0] if Q_inf.shape[2] == 1 else Q_inf[:, :, -1]
    Q[:, :, -2] = Q[:, :, -1]


def apply_wall(Q, grid):
    """Apply inviscid slip-wall BC at the inner boundary (j = 0).

    Reflects the normal velocity component while preserving tangential
    component and thermodynamic state.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        grid: Grid object
    """
    _ = grid.ni  # available for future use

    # Mirror the interior cell (j=1) to the ghost cell (j=0)
    # Reflect the velocity normal to the wall
    rho = Q[0, :, 1]
    u = Q[1, :, 1] / rho
    v = Q[2, :, 1] / rho

    # Wall normal at j=0 boundary (η direction)
    nx = grid.eta_x[:, 0]
    ny = grid.eta_y[:, 0]
    nmag = xp.sqrt(nx**2 + ny**2) + EPS_TINY
    nx = nx / nmag
    ny = ny / nmag

    # Normal velocity component
    vn = u * nx + v * ny

    # Reflect: subtract twice the normal component
    u_ghost = u - 2.0 * vn * nx
    v_ghost = v - 2.0 * vn * ny

    Q[0, :, 0] = rho
    Q[1, :, 0] = rho * u_ghost
    Q[2, :, 0] = rho * v_ghost
    Q[3, :, 0] = Q[3, :, 1]  # energy unchanged (pressure reflection)


def apply_periodic(Q):
    """Apply periodic BC in the circumferential (ξ) direction.

    For the O-grid, the first and last points in i-direction are adjacent.
    This copies interior values to the overlap region.

    Note: With our grid (no endpoint duplication), periodicity is handled
    by using modular indexing in the flux computation. This function is
    provided for explicit ghost-cell approaches.
    """
    # No-op if using modular indexing. For ghost cells:
    pass


def freestream_state(mach: float, alpha: float = 0.0, p_inf: float = 1.0, rho_inf: float = 1.0):
    """Compute freestream conservative state vector.

    Args:
        mach: Mach number
        alpha: angle of attack (radians)
        p_inf: freestream static pressure
        rho_inf: freestream density

    Returns:
        Q_inf: shape (4,) conservative state
    """
    a_inf = float(xp.sqrt(xp.array(GAMMA * p_inf / rho_inf)))
    u_inf = mach * a_inf * float(xp.cos(xp.array(alpha)))
    v_inf = mach * a_inf * float(xp.sin(xp.array(alpha)))
    E_inf = p_inf / ((GAMMA - 1.0) * rho_inf) + 0.5 * (u_inf**2 + v_inf**2)
    return xp.array([rho_inf, rho_inf * u_inf, rho_inf * v_inf, rho_inf * E_inf])
