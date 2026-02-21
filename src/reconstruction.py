"""MUSCL reconstruction with slope limiters for 2nd-order accuracy.

Reconstructs left and right states at cell interfaces from cell-centered
values using piecewise-linear reconstruction with TVD limiters.
"""
from __future__ import annotations

from src.backend import xp


def van_leer_limiter(r):
    """Van Leer slope limiter: φ(r) = (r + |r|) / (1 + |r|)."""
    return (r + xp.abs(r)) / (1.0 + xp.abs(r))


def minmod_limiter(r):
    """Minmod slope limiter."""
    return xp.maximum(0.0, xp.minimum(1.0, r))


def muscl_reconstruct(Q, axis: int, limiter=van_leer_limiter):
    """MUSCL reconstruction along the given axis.

    Args:
        Q: conservative variables, shape (4, ni, nj)
        axis: spatial axis for reconstruction (0=ξ, 1=η in the ni/nj dims,
              which is axis 1 or 2 of the Q array)
        limiter: slope limiter function

    Returns:
        QL: left-biased state at i+1/2 interfaces, shape (4, n-1, ...)
        QR: right-biased state at i+1/2 interfaces
    """
    # Map axis: Q has shape (4, ni, nj), axis 0=xi means dim 1, axis 1=eta means dim 2
    dim = axis + 1  # offset for the 4-variable leading dimension

    n = Q.shape[dim]
    if n < 4:
        # Not enough points for MUSCL — fall back to first order
        sl_l = [slice(None)] * Q.ndim
        sl_r = [slice(None)] * Q.ndim
        sl_l[dim] = slice(0, n - 1)
        sl_r[dim] = slice(1, n)
        return Q[tuple(sl_l)].copy(), Q[tuple(sl_r)].copy()

    # Build slices for Q[..., i-1, ...], Q[..., i, ...], Q[..., i+1, ...], Q[..., i+2, ...]
    def _sl(start, stop):
        s = [slice(None)] * Q.ndim
        s[dim] = slice(start, stop)
        return tuple(s)

    Qim1 = Q[_sl(0, n - 3)]    # i-1
    Qi = Q[_sl(1, n - 2)]      # i
    Qip1 = Q[_sl(2, n - 1)]    # i+1
    Qip2 = Q[_sl(3, n)]        # i+2

    # Forward and backward differences
    dQf = Qip1 - Qi        # forward diff at i
    dQb = Qi - Qim1        # backward diff at i
    dQf2 = Qip2 - Qip1     # forward diff at i+1
    dQb2 = Qip1 - Qi       # backward diff at i+1

    # Slope ratios (regularized to avoid division by zero)
    eps = 1e-12
    r_L = xp.where(xp.abs(dQf) > eps, dQb / (dQf + xp.sign(dQf) * eps), 0.0)
    r_R = xp.where(xp.abs(dQb2) > eps, dQf2 / (dQb2 + xp.sign(dQb2) * eps), 0.0)

    phi_L = limiter(r_L)
    phi_R = limiter(r_R)

    # Reconstructed states at i+1/2 interface (between Qi and Qip1)
    QL = Qi + 0.5 * phi_L * dQf       # left state
    QR = Qip1 - 0.5 * phi_R * dQb2    # right state

    return QL, QR
