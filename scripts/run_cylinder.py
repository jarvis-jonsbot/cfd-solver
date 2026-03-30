#!/usr/bin/env python3
"""Driver script for 2D compressible flow over a cylinder.

Usage:
    python scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 10000
    python scripts/run_cylinder.py --mach 2.0 --ni 256 --nj 128 --steps 20000
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend import xp
from src.boundary import apply_freestream, apply_wall, freestream_state
from src.gas import pressure
from src.grid import generate_cylinder_grid
from src.io import save_solution
from src.solver import SolverConfig, compute_dt_advective, solve, step_semi_implicit


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
    parser.add_argument(
        "--semi-implicit",
        action="store_true",
        help="Use semi-implicit pressure solver",
    )
    args = parser.parse_args()

    alpha_rad = args.alpha * xp.pi / 180.0

    # Auto-reduce CFL for float32 backends (MLX) — float32 needs more conservative stepping
    from src.backend import _BACKEND_NAME

    if _BACKEND_NAME == "mlx" and args.cfl > 0.08:
        print(f"[MLX] Reducing CFL from {args.cfl} to 0.08 (float32 stability limit)")
        args.cfl = 0.08

    print("=== 2D Compressible Cylinder Flow ===")
    print(f"Mach = {args.mach}, AoA = {args.alpha}°, CFL = {args.cfl}")
    print(f"Grid: {args.ni} x {args.nj}, R_outer = {args.r_outer}")
    print(f"Max steps: {args.steps}")
    print()

    # Generate grid
    print("Generating grid...")
    grid = generate_cylinder_grid(
        ni=args.ni,
        nj=args.nj,
        r_outer=args.r_outer,
        stretch=args.stretch,
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
    if args.semi_implicit:
        print("Using semi-implicit pressure solver (Phase 1)")
        print("  Time step computed from advective CFL only (acoustic waves treated implicitly)")
        # Manual time loop for semi-implicit
        Q = Q0.copy()
        t = 0.0
        for step in range(1, args.steps + 1):
            # Apply boundary conditions
            apply_wall(Q, grid)
            apply_freestream(Q, grid, args.mach, float(alpha_rad), 1.0, 1.0)

            # Compute time step using ADVECTIVE CFL only — not acoustic.
            # The semi-implicit scheme treats acoustic waves implicitly, so
            # stability only requires dt ~ dx/|u|, not dt ~ dx/(|u|+c).
            dt = compute_dt_advective(Q, grid, args.cfl)

            # Semi-implicit step
            Q = step_semi_implicit(Q, dt, grid)
            t += dt

            if step % args.print_every == 0:
                p = pressure(Q)
                rho_min = float(xp.min(Q[0, :, 1:-1]))
                rho_max = float(xp.max(Q[0, :, 1:-1]))
                p_min = float(xp.min(p[:, 1:-1]))
                p_max = float(xp.max(p[:, 1:-1]))
                print(
                    f"Step {step:6d}  t={t:.6f}  dt={dt:.2e}  "
                    f"rho=[{rho_min:.4f}, {rho_max:.4f}]  "
                    f"p=[{p_min:.4f}, {p_max:.4f}]"
                )

            if step % args.save_every == 0:
                save_callback(step, t, Q)

        Q_final = Q
    else:
        Q_final = solve(Q0, grid, config, callback=save_callback)

    # Save final solution
    save_solution(Q_final, grid, f"{args.output}/solution_final.npz")
    print("Done!")


if __name__ == "__main__":
    main()
