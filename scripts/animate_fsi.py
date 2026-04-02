#!/usr/bin/env python3
"""Animate the FSI (rigid body + compressible flow) simulation results.

Usage:
    python scripts/animate_fsi.py --input /tmp/cfd-fsi-output --output fsi.gif
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from src.gas import GAMMA


def _pressure(Q: np.ndarray) -> np.ndarray:
    rho = np.maximum(Q[0], 1e-8)
    u = Q[1] / rho
    v = Q[2] / rho
    E = Q[3] / rho
    return np.maximum((GAMMA - 1.0) * rho * (E - 0.5 * (u**2 + v**2)), 0.0)


def _mach(Q: np.ndarray) -> np.ndarray:
    rho = np.maximum(Q[0], 1e-8)
    u = Q[1] / rho
    v = Q[2] / rho
    p = _pressure(Q)
    a = np.sqrt(GAMMA * np.maximum(p, 1e-8) / rho)
    return np.sqrt(u**2 + v**2) / np.maximum(a, 1e-8)


def animate_fsi(input_dir: str, output: str, field: str = "pressure", fps: int = 12):
    pattern = os.path.join(input_dir, "solution_*.npz")
    files = sorted(glob.glob(pattern))
    traj_path = os.path.join(input_dir, "body_trajectory.npz")

    if not files:
        print(f"No solution files in {input_dir}")
        return

    # Load trajectory for body position per frame
    traj = None
    if os.path.exists(traj_path):
        traj = np.load(traj_path)

    # Load first frame to set up grid
    d0 = np.load(files[0])
    x, y = d0["x"], d0["y"]   # (ni, nj) cell centers
    Q0 = d0["Q"]               # (4, ni, nj)

    # Grid bounds
    xlim = (float(x.min()), float(x.max()))
    ylim = (float(y.min()), float(y.max()))

    # Field function
    compute_field = _pressure if field == "pressure" else _mach
    field_label = "Pressure" if field == "pressure" else "Mach Number"
    cmap = "RdBu_r" if field == "pressure" else "hot"

    # Compute colour scale from all frames (sample every 5th)
    sample_vals = []
    for f in files[::max(1, len(files)//20)]:
        Q = np.load(f)["Q"]
        val = compute_field(Q)
        sample_vals.append(val.ravel())
    all_vals = np.concatenate(sample_vals)
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    # Build figure
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.tight_layout(pad=1.5)

    val0 = compute_field(Q0)
    im = ax.pcolormesh(x, y, val0, vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    plt.colorbar(im, ax=ax, label=field_label, fraction=0.046, pad=0.04)

    t0 = float(d0["t"][0]) if "t" in d0 else 0.0
    body_x0, body_y0 = (0.0, 0.0)
    if traj is not None and len(traj["x"]) > 0:
        body_x0 = float(traj["x"][0])
        body_y0 = float(traj["y"][0])

    cyl = Circle((body_x0, body_y0), 0.5, fill=True, color="silver",
                 ec="black", lw=1.5, zorder=10)
    ax.add_patch(cyl)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"{field_label} — t = {t0:.3f}")

    def update(frame_idx):
        data = np.load(files[frame_idx])
        Q = data["Q"]
        t = float(data["t"][0]) if "t" in data else frame_idx * 0.01

        val = compute_field(Q)
        im.set_array(val.ravel())

        # Update body position from trajectory
        if traj is not None:
            # Find closest trajectory time
            times = traj["time"]
            idx = np.argmin(np.abs(times - t))
            bx = float(traj["x"][idx])
            by = float(traj["y"][idx])
        else:
            bx, by = 0.0, 0.0

        cyl.center = (bx, by)
        title.set_text(f"{field_label} — t = {t:.3f}  |  body x = {bx:.3f}")
        return [im, cyl, title]

    anim = FuncAnimation(fig, update, frames=len(files), interval=1000 // fps, blit=True)

    ext = os.path.splitext(output)[1].lower()
    print(f"Rendering {len(files)} frames → {output}")
    if ext == ".gif":
        anim.save(output, writer=PillowWriter(fps=fps), dpi=110)
    else:
        anim.save(output, writer="ffmpeg", fps=fps, dpi=110)

    plt.close(fig)
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="output")
    p.add_argument("--output", default="fsi.gif")
    p.add_argument("--field", choices=["pressure", "mach"], default="pressure")
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()
    animate_fsi(args.input, args.output, args.field, args.fps)


if __name__ == "__main__":
    main()
