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
import numpy as np
from matplotlib.patches import Circle

from src.gas import GAMMA


def load_and_plot(filepath: str, field: str = "pressure", save: bool = False):
    """Load solution and create contour plot."""
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Contour plot
    levels = 40
    cf = ax.contourf(x, y, val, levels=levels, cmap=cmap)
    plt.colorbar(cf, ax=ax, label=title)
    ax.contour(x, y, val, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

    # Draw cylinder
    cylinder = Circle((0, 0), 0.5, fill=True, color="gray", ec="black", lw=2)
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
