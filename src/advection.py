"""Semi-Lagrangian advection for scalar fields.

Implements backward characteristic tracing with bilinear interpolation.
Simple uniform-grid implementation suitable for Phase 1.
"""

from __future__ import annotations

from src.backend import xp


def sl_advect(f, u, v, dx, dy, dt):
    """Advect scalar field f by velocity (u, v) using semi-Lagrangian method.

    Traces characteristics backward in time, then bilinearly interpolates
    the field value at the departure point.

    Args:
        f: scalar field to advect, shape (ni, nj)
        u: x-velocity, shape (ni, nj)
        v: y-velocity, shape (ni, nj)
        dx: grid spacing in x (assumed uniform)
        dy: grid spacing in y (assumed uniform)
        dt: time step

    Returns:
        f_new: advected field, shape (ni, nj)

    Note:
        Uses periodic boundary conditions in both directions (Phase 1 simplification).
        Grid is assumed uniform with spacing dx, dy.
    """
    ni, nj = f.shape

    # Create grid indices
    i_grid = xp.arange(ni)[:, None]
    j_grid = xp.arange(nj)[None, :]

    # Compute departure points by tracing backward: x_d = x - u*dt
    # Convert to fractional grid indices
    i_depart = i_grid - u * dt / dx
    j_depart = j_grid - v * dt / dy

    # Periodic wrap (modulo grid size)
    i_depart = i_depart % ni
    j_depart = j_depart % nj

    # Bilinear interpolation
    # Floor to get lower-left cell indices
    i0 = xp.floor(i_depart).astype(int) % ni
    j0 = xp.floor(j_depart).astype(int) % nj
    i1 = (i0 + 1) % ni
    j1 = (j0 + 1) % nj

    # Fractional parts (weights)
    fi = i_depart - xp.floor(i_depart)
    fj = j_depart - xp.floor(j_depart)

    # Pad field periodically for easy indexing
    f_pad = xp.concatenate([f, f[:1, :]], axis=0)
    f_pad = xp.concatenate([f_pad, f_pad[:, :1]], axis=1)

    # Gather corner values
    f00 = f_pad[i0, j0]
    f10 = f_pad[i1, j0]
    f01 = f_pad[i0, j1]
    f11 = f_pad[i1, j1]

    # Bilinear interpolation weights
    f_new = f00 * (1 - fi) * (1 - fj) + f10 * fi * (1 - fj) + f01 * (1 - fi) * fj + f11 * fi * fj

    return f_new
