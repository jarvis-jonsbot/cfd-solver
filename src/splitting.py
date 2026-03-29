"""Zha-Bilgen flux splitting for semi-implicit pressure solver.

Decomposes Euler fluxes into advective (convective) and acoustic (pressure)
components for time-split integration.
"""

from __future__ import annotations

from src.backend import xp
from src.gas import pressure


def split_flux_x(Q, gas):
    """Split x-direction Euler flux into advective and acoustic parts.

    Zha-Bilgen splitting:
        F_advect = u * [rho, rho*u, rho*v, rho*E]   (pressure excluded)
        F_acoustic = [0, p, 0, p*u]                  (pressure contribution)

    Args:
        Q: conservative variables, shape (4, ...)
        gas: gas model (unused, for API consistency)

    Returns:
        (F_advect, F_acoustic): tuple of flux arrays, each shape (4, ...)
    """
    rho = Q[0]
    u = Q[1] / rho
    p = pressure(Q)

    # Advective part: mass transport without pressure
    F_advect = xp.stack([
        rho * u,        # rho*u
        Q[1] * u,       # rho*u^2
        Q[2] * u,       # rho*u*v
        Q[3] * u,       # rho*E*u
    ], axis=0)

    # Acoustic part: pressure contribution only
    F_acoustic = xp.stack([
        xp.zeros_like(rho),  # no mass flux from pressure
        p,                    # pressure force in x-momentum
        xp.zeros_like(p),    # no pressure force in y-momentum
        p * u,                # pressure work in energy
    ], axis=0)

    return F_advect, F_acoustic


def split_flux_y(Q, gas):
    """Split y-direction Euler flux into advective and acoustic parts.

    Zha-Bilgen splitting:
        G_advect = v * [rho, rho*u, rho*v, rho*E]   (pressure excluded)
        G_acoustic = [0, 0, p, p*v]                  (pressure contribution)

    Args:
        Q: conservative variables, shape (4, ...)
        gas: gas model (unused, for API consistency)

    Returns:
        (G_advect, G_acoustic): tuple of flux arrays, each shape (4, ...)
    """
    rho = Q[0]
    v = Q[2] / rho
    p = pressure(Q)

    # Advective part: mass transport without pressure
    G_advect = xp.stack([
        rho * v,        # rho*v
        Q[1] * v,       # rho*u*v
        Q[2] * v,       # rho*v^2
        Q[3] * v,       # rho*E*v
    ], axis=0)

    # Acoustic part: pressure contribution only
    G_acoustic = xp.stack([
        xp.zeros_like(rho),  # no mass flux from pressure
        xp.zeros_like(p),    # no pressure force in x-momentum
        p,                    # pressure force in y-momentum
        p * v,                # pressure work in energy
    ], axis=0)

    return G_advect, G_acoustic
