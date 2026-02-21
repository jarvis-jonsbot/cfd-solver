#!/usr/bin/env python3
"""Grid quality tests."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.backend import xp, to_numpy
from src.grid import generate_cylinder_grid


def test_grid_basic():
    """Test grid generation produces valid output."""
    grid = generate_cylinder_grid(ni=64, nj=32)

    assert grid.ni == 64
    assert grid.nj == 32
    assert grid.x.shape == (64, 32)
    assert grid.y.shape == (64, 32)

    # Inner boundary should be on the cylinder (r = 0.5)
    r_inner = to_numpy(xp.sqrt(grid.x[:, 0]**2 + grid.y[:, 0]**2))
    np.testing.assert_allclose(r_inner, 0.5, atol=1e-10)

    # Jacobian should be positive everywhere
    J = to_numpy(grid.jacobian)
    assert np.all(J > 0) or np.all(J < 0), "Jacobian should not change sign"

    print("  ✅ Grid basic test PASSED")


def test_grid_orthogonality():
    """Check near-orthogonality at the cylinder surface."""
    grid = generate_cylinder_grid(ni=128, nj=64)

    # At the wall (j=0), ξ-lines should be tangent and η-lines should be normal
    # Check that grid lines are reasonably orthogonal
    # Dot product of ξ and η tangent vectors at wall
    dx_xi = to_numpy(grid.x[1, 0] - grid.x[0, 0])
    dy_xi = to_numpy(grid.y[1, 0] - grid.y[0, 0])
    dx_eta = to_numpy(grid.x[0, 1] - grid.x[0, 0])
    dy_eta = to_numpy(grid.y[0, 1] - grid.y[0, 0])

    dot = dx_xi * dx_eta + dy_xi * dy_eta
    mag = np.sqrt(dx_xi**2 + dy_xi**2) * np.sqrt(dx_eta**2 + dy_eta**2)
    cos_angle = dot / max(mag, 1e-30)

    # Should be close to 0 (orthogonal)
    assert abs(cos_angle) < 0.3, f"Grid not orthogonal at wall: cos(angle)={cos_angle}"

    print("  ✅ Grid orthogonality test PASSED")


if __name__ == "__main__":
    test_grid_basic()
    test_grid_orthogonality()
