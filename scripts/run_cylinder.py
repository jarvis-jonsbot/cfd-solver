#!/usr/bin/env python3
"""Driver script for 2D compressible flow over a cylinder.

Usage:
    python scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 10000
    python scripts/run_cylinder.py --mach 2.0 --ni 256 --nj 128 --steps 20000
"""
from __future__ import annotations

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend import xp
from src.grid import generate_cylinder_grid
from src.gas import primitive_to_conservative
from src.boundary import freestream_state
from src.solver import SolverConfig, solve
from src.io import save_solution


def main():
    parser = argparse.ArgumentParser(description="2D Compressible Cylinder Flow Solver")
    parser.add_argument("--mach", type=float, default=0.3, help="Freestream Mach number")
    parser.add_argument("--alpha", type=float, default=0.0, help="Angle of attack (degrees)")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL number")
    parser.add_argument("--steps", type=int, default=5000, help="Max time steps")
    parser.add_argument("--ni", type=int, default=128, help="Circumferential grid points")
    parser.add_argument("--nj", type=int, default=64, help="Radial grid points")
    parser.add_argument("--r-outer", type=float, default=20.0, help="Outer boundary radius")
    parser.add_argument("--stretch", type=float, default=1.05, help="Radial grid stretching")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--print-every", type=int, default=100, help="Print interval")
    parser.add_argument("--save-every", type=int, default=1000, help="Save interval")
    args = parser.parse_args()

    alpha_rad = args.alpha * xp.pi / 180.0

    print(f"=== 2D Compressible Cylinder Flow ===")
    print(f"Mach = {args.mach}, AoA = {args.alpha}°, CFL = {args.cfl}")
    print(f"Grid: {args.ni} x {args.nj}, R_outer = {args.r_outer}")
    print(f"Max steps: {args.steps}")
    print()

    # Generate grid
    print("Generating grid...")
    grid = generate_cylinder_grid(
        ni=args.ni, nj=args.nj,
        r_outer=args.r_outer, stretch=args.stretch,
    )
    print(f"Grid: {grid.ni} x {grid.nj} = {grid.ni * grid.nj} cells")

    # Initialize with freestream
    Q_inf = freestream_state(args.mach, float(alpha_rad))
    Q0 = xp.zeros((4, grid.ni, grid.nj))
    for eq in range(4):
        Q0[eq, :, :] = Q_inf[eq]

    # Solver config
    config = SolverConfig(
        mach=args.mach,
        alpha=float(alpha_rad),
        cfl=args.cfl,
        max_steps=args.steps,
        print_interval=args.print_every,
        output_interval=args.save_every,
        output_dir=args.output,
    )

    # Callback for periodic saves
    def save_callback(step, t, Q):
        save_solution(Q, grid, f"{args.output}/solution_{step:06d}.npz", t, step)

    # Run solver
    print("Starting solver...")
    Q_final = solve(Q0, grid, config, callback=save_callback)

    # Save final solution
    save_solution(Q_final, grid, f"{args.output}/solution_final.npz")
    print("Done!")


if __name__ == "__main__":
    main()
