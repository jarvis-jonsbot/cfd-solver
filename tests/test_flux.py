#!/usr/bin/env python3
"""Flux computation tests."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.backend import xp, to_numpy
from src.gas import GAMMA, primitive_to_conservative
from src.flux import roe_flux_1d


def test_flux_symmetry():
    """Roe flux should return physical flux for identical left/right states."""
    W = xp.array([1.0, 1.0, 0.0, 1.0]).reshape(4, 1)
    Q = primitive_to_conservative(W)

    nx = xp.array([[1.0]])
    ny = xp.array([[0.0]])

    F = roe_flux_1d(Q, Q, nx, ny)
    F_np = to_numpy(F).flatten()

    # Physical flux: F = [rho*u, rho*u^2+p, rho*u*v, (rho*E+p)*u]
    rho, u, v, p = 1.0, 1.0, 0.0, 1.0
    E = p / ((GAMMA - 1.0) * rho) + 0.5 * u**2
    F_exact = np.array([rho * u, rho * u**2 + p, rho * u * v, (rho * E + p) * u])

    np.testing.assert_allclose(F_np, F_exact, atol=1e-10)
    print("  ✅ Flux symmetry test PASSED")


def test_flux_conservation():
    """Flux should satisfy conservation: F(L,R) = F(R,L) with reversed normal."""
    WL = xp.array([1.0, 0.5, 0.1, 1.0]).reshape(4, 1)
    WR = xp.array([0.5, -0.3, 0.2, 0.5]).reshape(4, 1)
    QL = primitive_to_conservative(WL)
    QR = primitive_to_conservative(WR)

    nx = xp.array([[0.6]])
    ny = xp.array([[0.8]])

    F_lr = roe_flux_1d(QL, QR, nx, ny)
    F_rl = roe_flux_1d(QR, QL, -nx, -ny)

    # F(L,R,n) = -F(R,L,-n) (consistency)
    np.testing.assert_allclose(to_numpy(F_lr), -to_numpy(F_rl), atol=1e-10)
    print("  ✅ Flux conservation test PASSED")


if __name__ == "__main__":
    test_flux_symmetry()
    test_flux_conservation()
