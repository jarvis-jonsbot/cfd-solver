"""Ideal gas thermodynamic relations.

Convention: Q = [rho, rho*u, rho*v, rho*E]  (conservative variables)
Primitive:  W = [rho, u, v, p]
"""

from __future__ import annotations

from src.backend import EPS_TINY, xp

GAMMA: float = 1.4  # ratio of specific heats for air


def conservative_to_primitive(Q):
    """Convert conservative variables to primitive.

    Args:
        Q: array of shape (4, ...) — conservative variables

    Returns:
        W: array of shape (4, ...) — [rho, u, v, p]
    """
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    E = Q[3] / rho
    p = (GAMMA - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    return xp.stack([rho, u, v, p], axis=0)


def primitive_to_conservative(W):
    """Convert primitive variables to conservative.

    Args:
        W: array of shape (4, ...) — [rho, u, v, p]

    Returns:
        Q: array of shape (4, ...) — conservative variables
    """
    rho, u, v, p = W[0], W[1], W[2], W[3]
    E = p / ((GAMMA - 1.0) * rho) + 0.5 * (u**2 + v**2)
    return xp.stack([rho, rho * u, rho * v, rho * E], axis=0)


def pressure(Q):
    """Compute pressure from conservative variables."""
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    E = Q[3] / rho
    return (GAMMA - 1.0) * rho * (E - 0.5 * (u**2 + v**2))


def sound_speed(rho, p):
    """Compute speed of sound."""
    return xp.sqrt(GAMMA * xp.abs(p) / xp.maximum(rho, EPS_TINY))


def enthalpy(Q):
    """Compute specific total enthalpy H = (E + p) / rho."""
    p = pressure(Q)
    return (Q[3] + p) / Q[0]
