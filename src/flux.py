"""Roe's approximate Riemann solver for 2D compressible Euler equations.

Computes the numerical flux at cell interfaces using Roe-averaged states
with Harten's entropy fix.
"""

from __future__ import annotations

from src.backend import EPS_TINY, xp
from src.gas import GAMMA, enthalpy, pressure, sound_speed


def roe_flux_1d(QL, QR, nx, ny):
    """Compute Roe flux across an interface with normal (nx, ny).

    Args:
        QL: left state, shape (4, ...)
        QR: right state, shape (4, ...)
        nx: x-component of face normal, broadcastable to QL[0] shape
        ny: y-component of face normal, broadcastable to QL[0] shape

    Returns:
        F: numerical flux, shape (4, ...)
    """
    # Left state primitives
    rhoL = QL[0]
    uL = QL[1] / rhoL
    vL = QL[2] / rhoL
    pL = pressure(QL)
    aL = sound_speed(rhoL, pL)  # noqa: F841 — kept for clarity
    HL = enthalpy(QL)
    vnL = uL * nx + vL * ny

    # Right state primitives
    rhoR = QR[0]
    uR = QR[1] / rhoR
    vR = QR[2] / rhoR
    pR = pressure(QR)
    aR = sound_speed(rhoR, pR)  # noqa: F841 — kept for clarity
    HR = enthalpy(QR)
    vnR = uR * nx + vR * ny

    # Physical fluxes projected onto face normal
    FL = _euler_flux(QL, uL, vL, pL, vnL, nx, ny)
    FR = _euler_flux(QR, uR, vR, pR, vnR, nx, ny)

    # Roe averages
    sqrtL = xp.sqrt(xp.maximum(rhoL, EPS_TINY))
    sqrtR = xp.sqrt(xp.maximum(rhoR, EPS_TINY))
    denom = sqrtL + sqrtR

    u_roe = (sqrtL * uL + sqrtR * uR) / denom
    v_roe = (sqrtL * vL + sqrtR * vR) / denom
    H_roe = (sqrtL * HL + sqrtR * HR) / denom
    q2 = u_roe**2 + v_roe**2
    a_roe = xp.sqrt(xp.maximum((GAMMA - 1.0) * (H_roe - 0.5 * q2), EPS_TINY))
    vn_roe = u_roe * nx + v_roe * ny
    rho_roe = sqrtL * sqrtR

    # Jump in primitive variables
    dvn = vnR - vnL
    dp = pR - pL
    drho = rhoR - rhoL

    # Eigenvalues with entropy fix
    eps = 0.1 * a_roe
    lam1 = _entropy_fix(vn_roe - a_roe, eps)
    lam2 = _entropy_fix(vn_roe, eps)
    lam3 = _entropy_fix(vn_roe + a_roe, eps)

    # Wave strengths (characteristic decomposition)
    alpha1 = 0.5 * (dp - rho_roe * a_roe * dvn) / (a_roe**2)
    alpha2 = drho - dp / (a_roe**2)
    alpha3 = 0.5 * (dp + rho_roe * a_roe * dvn) / (a_roe**2)

    # Tangential velocity jump
    du = uR - uL
    dv = vR - vL
    dvt_x = du - dvn * nx
    dvt_y = dv - dvn * ny

    # Build dissipation term: sum of |lambda_k| * alpha_k * r_k
    # Use list-of-arrays approach to avoid shape issues
    d0 = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    d1 = (
        lam1 * alpha1 * (u_roe - a_roe * nx)
        + lam2 * (alpha2 * u_roe + rho_roe * dvt_x)
        + lam3 * alpha3 * (u_roe + a_roe * nx)
    )
    d2 = (
        lam1 * alpha1 * (v_roe - a_roe * ny)
        + lam2 * (alpha2 * v_roe + rho_roe * dvt_y)
        + lam3 * alpha3 * (v_roe + a_roe * ny)
    )
    d3 = (
        lam1 * alpha1 * (H_roe - a_roe * vn_roe)
        + lam2 * (alpha2 * 0.5 * q2 + rho_roe * (u_roe * dvt_x + v_roe * dvt_y))
        + lam3 * alpha3 * (H_roe + a_roe * vn_roe)
    )

    diss = xp.stack([d0, d1, d2, d3], axis=0)

    return 0.5 * (FL + FR) - 0.5 * diss


def _euler_flux(Q, u, v, p, vn, nx, ny):
    """Compute Euler flux projected onto face normal (nx, ny)."""
    F0 = Q[0] * vn
    F1 = Q[1] * vn + p * nx
    F2 = Q[2] * vn + p * ny
    F3 = (Q[3] + p) * vn
    return xp.stack([F0, F1, F2, F3], axis=0)


def _entropy_fix(lam, eps):
    """Harten's entropy fix for eigenvalues."""
    return xp.where(xp.abs(lam) < eps, (lam**2 + eps**2) / (2.0 * eps), xp.abs(lam))
