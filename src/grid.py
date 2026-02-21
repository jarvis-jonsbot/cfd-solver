"""Structured O-grid generator for cylinder flow.

Generates a body-fitted curvilinear grid around a circular cylinder.
The grid wraps 360° around the cylinder and extends radially to a far-field boundary.

Coordinate system:
  - ξ (xi): circumferential direction (index i, 0..ni-1), periodic
  - η (eta): radial direction (index j, 0..nj-1), wall to far-field

Grid stretching uses geometric progression in the radial direction
to cluster points near the cylinder surface.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.backend import xp


@dataclass
class Grid:
    """Structured 2D grid with precomputed metrics."""

    x: object          # physical x-coordinates, shape (ni, nj)
    y: object          # physical y-coordinates, shape (ni, nj)
    ni: int            # number of points in ξ (circumferential)
    nj: int            # number of points in η (radial)
    # Metric terms for flux computation in curvilinear coordinates
    # ξ-direction face normals (area-weighted)
    xi_x: object       # ∂ξ/∂x * J,  shape (ni, nj)
    xi_y: object       # ∂ξ/∂y * J
    # η-direction face normals (area-weighted)
    eta_x: object      # ∂η/∂x * J
    eta_y: object      # ∂η/∂y * J
    jacobian: object   # cell Jacobian (area element), shape (ni, nj)


def generate_cylinder_grid(
    ni: int = 128,
    nj: int = 64,
    r_cylinder: float = 0.5,
    r_outer: float = 20.0,
    stretch: float = 1.05,
) -> Grid:
    """Generate an O-grid around a cylinder.

    Args:
        ni: circumferential points (should be even for symmetry)
        nj: radial points (wall to far-field)
        r_cylinder: cylinder radius
        r_outer: outer boundary radius (in cylinder radii from center)
        stretch: geometric stretching ratio for radial spacing

    Returns:
        Grid object with coordinates and precomputed metrics.
    """
    # Circumferential angles — uniform spacing, periodic
    theta = xp.linspace(0, 2 * xp.pi, ni, endpoint=False)

    # Radial distribution — geometric stretching
    if abs(stretch - 1.0) < 1e-10:
        r = xp.linspace(r_cylinder, r_outer, nj)
    else:
        # Geometric series: dr_j = dr_0 * stretch^j
        s = xp.array([(stretch**j - 1.0) / (stretch**nj - 1.0) for j in range(nj)])
        r = r_cylinder + (r_outer - r_cylinder) * s

    # Meshgrid: theta[i] x r[j]
    THETA, R = xp.meshgrid(theta, r, indexing="ij")  # shape (ni, nj)

    x = R * xp.cos(THETA)
    y = R * xp.sin(THETA)

    # Compute metrics using finite differences
    grid = Grid(
        x=x, y=y, ni=ni, nj=nj,
        xi_x=None, xi_y=None,  # type: ignore
        eta_x=None, eta_y=None,  # type: ignore
        jacobian=None,  # type: ignore
    )
    _compute_metrics(grid)
    return grid


def _compute_metrics(grid: Grid) -> None:
    """Compute grid metric terms using central finite differences.

    For a mapping (ξ, η) → (x, y):
        x_ξ = ∂x/∂ξ,  x_η = ∂x/∂η, etc.
        J = x_ξ * y_η - x_η * y_ξ   (Jacobian determinant)

    The contravariant basis vectors (used for flux projection):
        ξ_x =  y_η / J,   ξ_y = -x_η / J
        η_x = -y_ξ / J,   η_y =  x_ξ / J
    """
    ni, nj = grid.ni, grid.nj
    x, y = grid.x, grid.y

    # ξ derivatives (periodic in i-direction)
    x_xi = xp.zeros_like(x)
    y_xi = xp.zeros_like(y)
    x_xi[1:-1, :] = (x[2:, :] - x[:-2, :]) * 0.5
    y_xi[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    # Periodic wrap
    x_xi[0, :] = (x[1, :] - x[-1, :]) * 0.5
    y_xi[0, :] = (y[1, :] - y[-1, :]) * 0.5
    x_xi[-1, :] = (x[0, :] - x[-2, :]) * 0.5
    y_xi[-1, :] = (y[0, :] - y[-2, :]) * 0.5

    # η derivatives (one-sided at boundaries)
    x_eta = xp.zeros_like(x)
    y_eta = xp.zeros_like(y)
    x_eta[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
    y_eta[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    # One-sided at wall (j=0) and far-field (j=nj-1)
    x_eta[:, 0] = x[:, 1] - x[:, 0]
    y_eta[:, 0] = y[:, 1] - y[:, 0]
    x_eta[:, -1] = x[:, -1] - x[:, -2]
    y_eta[:, -1] = y[:, -1] - y[:, -2]

    # Jacobian
    J = x_xi * y_eta - x_eta * y_xi
    J = xp.where(xp.abs(J) < 1e-30, 1e-30, J)  # prevent division by zero

    # Contravariant metrics (area-weighted normals)
    grid.xi_x = y_eta / J
    grid.xi_y = -x_eta / J
    grid.eta_x = -y_xi / J
    grid.eta_y = x_xi / J
    grid.jacobian = J
