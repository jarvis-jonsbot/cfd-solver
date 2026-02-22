#!/usr/bin/env python3
"""Animate CFD solution snapshots into a GIF or MP4.

Usage:
    python scripts/animate.py --input output/ --field pressure --output flow.gif
    python scripts/animate.py --input output/ --field mach --output flow.mp4 --fps 15
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from src.gas import GAMMA


def _compute_field(Q, name: str):
    """Compute a derived field from conservative variables."""
    rho = Q[0]
    u = Q[1] / rho
    v = Q[2] / rho
    E = Q[3] / rho
    p = (GAMMA - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    a = np.sqrt(GAMMA * np.abs(p) / np.maximum(rho, 1e-30))
    fields = {
        "pressure": (p, "Pressure", "RdBu_r"),
        "density": (rho, "Density", "viridis"),
        "mach": (np.sqrt(u**2 + v**2) / a, "Mach Number", "hot"),
        "velocity": (np.sqrt(u**2 + v**2), "Velocity Magnitude", "coolwarm"),
    }
    return fields[name]


def animate(
    input_dir: str,
    field: str = "pressure",
    output: str = "flow.gif",
    fps: int = 10,
    xlim: tuple[float, float] = (-3, 6),
    ylim: tuple[float, float] = (-3, 3),
):
    """Create animation from solution snapshots."""
    # Find and sort snapshot files (exclude final)
    pattern = os.path.join(input_dir, "solution_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No solution files found in {input_dir}")
        return

    # Load first frame to set up plot
    data0 = np.load(files[0])
    x, y = data0["x"], data0["y"]
    val0, title, cmap = _compute_field(data0["Q"], field)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    levels = np.linspace(np.nanpercentile(val0, 2), np.nanpercentile(val0, 98), 40)

    # Initial contour
    cf = ax.contourf(x, y, val0, levels=levels, cmap=cmap, extend="both")
    plt.colorbar(cf, ax=ax, label=title)
    cylinder = Circle((0, 0), 0.5, fill=True, color="gray", ec="black", lw=2, zorder=10)
    ax.add_patch(cylinder)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} — t = {float(data0['t'][0]):.4f}")

    def update(frame_idx):
        ax.clear()
        data = np.load(files[frame_idx])
        val, _, _ = _compute_field(data["Q"], field)
        t = float(data["t"][0])

        ax.contourf(x, y, val, levels=levels, cmap=cmap, extend="both")
        ax.contour(x, y, val, levels=levels, colors="k", linewidths=0.3, alpha=0.3)
        cyl = Circle((0, 0), 0.5, fill=True, color="gray", ec="black", lw=2, zorder=10)
        ax.add_patch(cyl)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title} — t = {t:.4f}")
        return []

    anim = FuncAnimation(fig, update, frames=len(files), interval=1000 // fps, blit=False)

    ext = os.path.splitext(output)[1].lower()
    if ext == ".gif":
        anim.save(output, writer=PillowWriter(fps=fps))
    elif ext in (".mp4", ".mov"):
        anim.save(output, writer="ffmpeg", fps=fps)
    else:
        anim.save(output, fps=fps)

    plt.close(fig)
    print(f"Saved animation: {output} ({len(files)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Animate CFD solution snapshots")
    parser.add_argument("--input", default="output", help="Directory with solution_*.npz files")
    parser.add_argument(
        "--field", default="pressure", choices=["pressure", "density", "mach", "velocity"]
    )
    parser.add_argument("--output", default="flow.gif", help="Output file (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-3, 6], help="X-axis limits")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-3, 3], help="Y-axis limits")
    args = parser.parse_args()

    animate(args.input, args.field, args.output, args.fps, tuple(args.xlim), tuple(args.ylim))


if __name__ == "__main__":
    main()
