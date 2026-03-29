#!/usr/bin/env python3
"""Post-processing and visualization for CFD solutions.

Usage:
    python scripts/visualize.py --input output/solution_final.npz
    python scripts/visualize.py --input output/solution_final.npz --field mach --save
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Circle

from src.gas import GAMMA


def _build_triangulation(x: np.ndarray, y: np.ndarray) -> mtri.Triangulation:
    """Build an unstructured Triangulation from an O-grid (ni x nj) array.

    The O-grid is periodic in the circumferential (i) direction: the last
    column (i = ni-1) is spatially adjacent to the first (i = 0).  Naively
    passing the 2-D x/y arrays to ``contourf`` causes matplotlib to draw a
    spurious degenerate triangle across the periodic seam.

    Instead we:
      1. Flatten the (ni, nj) node arrays to 1-D (row-major: node = i*nj + j).
      2. Build quads for every (i, j) cell — including the wrap-around quads
         that connect column ni-1 back to column 0.
      3. Split each quad into two triangles and pass them to Triangulation.

    Returns a Triangulation that correctly captures the O-grid topology with
    no missing region or seam artifact.
    """
    ni, nj = x.shape

    x_flat = x.ravel()  # length ni*nj
    y_flat = y.ravel()

    def node(i: int, j: int) -> int:
        """Global node index, with periodic wrap in i."""
        return (i % ni) * nj + j

    triangles = []
    for i in range(ni):  # periodic: i+1 wraps via modulo
        for j in range(nj - 1):  # no wrap in radial direction
            # Quad corners (counter-clockwise)
            n00 = node(i, j)
            n10 = node(i + 1, j)  # wraps to 0 when i == ni-1
            n11 = node(i + 1, j + 1)
            n01 = node(i, j + 1)
            # Split into two triangles
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))

    triangles = np.array(triangles, dtype=np.int32)
    return mtri.Triangulation(x_flat, y_flat, triangles)


def load_and_plot(filepath: str, field: str = "pressure", save: bool = False):
    """Load solution and create contour plot using unstructured triangulation.

    Using ``matplotlib.tri.Triangulation`` instead of the raw structured arrays
    correctly handles the O-grid's periodic seam in the circumferential
    direction, eliminating the spurious no-data triangle behind the cylinder.
    """
    data = np.load(filepath)
    Q = data["Q"]
    x = data["x"]
    y = data["y"]
    t = float(data["t"][0])

    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    E = Q[3] / rho
    p = (GAMMA - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    a = np.sqrt(GAMMA * np.abs(p) / np.maximum(rho, 1e-30))
    mach = np.sqrt(u**2 + v**2) / a

    fields = {
        "pressure": (p, "Pressure", "RdBu_r"),
        "density": (rho, "Density", "viridis"),
        "mach": (mach, "Mach Number", "hot"),
        "velocity": (np.sqrt(u**2 + v**2), "Velocity Magnitude", "coolwarm"),
    }

    if field not in fields:
        print(f"Unknown field '{field}'. Available: {list(fields.keys())}")
        return

    val, title, cmap = fields[field]

    # Build topology-aware triangulation (handles periodic O-grid seam)
    triang = _build_triangulation(x, y)
    val_flat = val.ravel()

    # Mask triangles whose centroid falls inside the cylinder (r < r_cyl).
    # These cells are inside the body and should not be rendered.
    r_cylinder = 0.5
    xm = x.ravel()[triang.triangles].mean(axis=1)
    ym = y.ravel()[triang.triangles].mean(axis=1)
    triang.set_mask(xm**2 + ym**2 < r_cylinder**2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    levels = 40
    cf = ax.tricontourf(triang, val_flat, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=title)
    ax.tricontour(triang, val_flat, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

    # Draw cylinder
    cylinder = Circle((0, 0), r_cylinder, fill=True, color="gray", ec="black", lw=2)
    ax.add_patch(cylinder)

    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} — t = {t:.4f}")

    if save:
        outpath = filepath.replace(".npz", f"_{field}.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="CFD Solution Visualization")
    parser.add_argument("--input", required=True, help="Solution .npz file")
    parser.add_argument(
        "--field", default="pressure", choices=["pressure", "density", "mach", "velocity"]
    )
    parser.add_argument("--save", action="store_true", help="Save to PNG instead of showing")
    args = parser.parse_args()

    load_and_plot(args.input, args.field, args.save)


if __name__ == "__main__":
    main()
