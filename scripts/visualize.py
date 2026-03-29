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


def _build_triangulation(x: np.ndarray, y: np.ndarray) -> tuple[mtri.Triangulation, np.ndarray]:
    """Build an unstructured Triangulation from an O-grid (ni x nj) array.

    The O-grid is periodic in the circumferential (i) direction: the last
    column (i = ni-1) is spatially adjacent to the first (i = 0).  Two
    approaches fail here:

    1. Naive: passing the raw (ni, nj) arrays leaves a missing wedge at the
       seam because matplotlib has no way to know i=0 and i=ni-1 are adjacent.

    2. Modular triangles (previous approach): building quads with ``node(i+1)
       = node(0)`` for the last ring produces *degenerate* triangles in
       physical (x, y) space — one vertex is at theta=0 and the other at
       theta≈2π, which are the same physical point but distinct nodes.
       matplotlib renders these as a hair-thin strip that appears as a gap.

    Correct approach: append a *phantom closing column* (i = ni) that is a
    copy of column i = 0 in both coordinates and scalar values.  The seam
    quad (i = ni-1, i = ni) then has proper finite extent in (x, y), giving
    tricontourf a well-formed triangle to interpolate across.

    Returns:
        (triang, idx_map) where idx_map is an index array of length
        (ni+1)*nj that maps flat phantom-grid indices back to the original
        (ni, nj) scalar array.  Use it as:  val_closed = val.ravel()[idx_map]
    """
    ni, nj = x.shape

    # Close the grid: append column i=0 as a phantom column i=ni
    xc = np.concatenate([x, x[:1, :]], axis=0)   # (ni+1, nj)
    yc = np.concatenate([y, y[:1, :]], axis=0)

    # Index map: phantom node (i, j) → original node (i % ni, j)
    orig_i = np.arange(ni + 1) % ni              # (ni+1,)
    idx_map = (orig_i[:, None] * nj + np.arange(nj)[None, :]).ravel()  # (ni+1)*nj

    x_flat = xc.ravel()
    y_flat = yc.ravel()

    ni1 = ni + 1  # number of rows in extended grid
    triangles = []
    for i in range(ni):           # ni quads; the last one bridges the seam
        for j in range(nj - 1):
            n00 = i       * nj + j
            n10 = (i + 1) * nj + j       # phantom column when i == ni-1
            n11 = (i + 1) * nj + j + 1
            n01 = i       * nj + j + 1
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))

    triangles = np.array(triangles, dtype=np.int32)
    triang = mtri.Triangulation(x_flat, y_flat, triangles)
    return triang, idx_map


def load_and_plot(filepath: str, field: str = "pressure", save: bool = False):
    """Load solution and create contour plot using unstructured triangulation.

    Using a phantom-closing column approach with ``matplotlib.tri.Triangulation``
    correctly handles the O-grid's periodic seam in the circumferential
    direction, eliminating both the spurious no-data gap and the degenerate
    near-zero-area seam triangle that caused rendering artifacts.
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

    # Build topology-aware triangulation with phantom closing column.
    # idx_map remaps phantom grid flat indices → original (ni*nj) flat indices.
    triang, idx_map = _build_triangulation(x, y)

    # Apply idx_map to get scalar values at phantom-grid nodes
    val_closed = val.ravel()[idx_map]

    # Mask triangles whose centroid falls inside the cylinder (r < r_cyl).
    # These cells are inside the body and should not be rendered.
    r_cylinder = 0.5
    xm = triang.x[triang.triangles].mean(axis=1)
    ym = triang.y[triang.triangles].mean(axis=1)
    triang.set_mask(xm**2 + ym**2 < r_cylinder**2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    levels = 40
    cf = ax.tricontourf(triang, val_closed, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=title)
    ax.tricontour(triang, val_closed, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

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
