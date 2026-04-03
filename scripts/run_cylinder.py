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
from src.grid import generate_cartesian_grid, generate_cylinder_grid
from src.io import save_solution
from src.solver import (
    SolverConfig,
    compute_dt_advective,
    solve,
    step_partitioned_fsi,
    step_semi_implicit,
)


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
    parser.add_argument(
        "--rigid-body",
        action="store_true",
        help="Use rigid body FSI mode (Mach 3 shock hit, free cylinder)",
    )
    parser.add_argument(
        "--csl",
        action="store_true",
        help="Use Conservative Semi-Lagrangian advection (stable at high CFL)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid CSL/MUSCL advection with shock detection",
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
    if args.rigid_body:
        print("[Rigid Body FSI Mode] Free cylinder, shock hit at Mach 3")
    if args.hybrid:
        print("[Hybrid CSL/MUSCL] Shock-adaptive advection with high-CFL stability")
    elif args.csl:
        print("[CSL Advection] Conservative Semi-Lagrangian (stable at high CFL)")
    print()

    # Generate grid
    if args.rigid_body:
        print("Generating Cartesian grid for rigid body mode...")
        grid = generate_cartesian_grid(
            ni=args.ni,
            nj=args.nj,
            x_min=-10.0,
            x_max=10.0,
            y_min=-5.0,
            y_max=5.0,
        )
        # Override Mach number for shock-hit scenario
        args.mach = 3.0
        print(f"  (Cartesian grid for free body, Mach = {args.mach})")
    else:
        print("Generating O-grid...")
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
        use_csl=args.csl,
        use_hybrid=args.hybrid,
    )

    # Callback for periodic saves
    def save_callback(step, t, Q):
        save_solution(Q, grid, f"{args.output}/solution_{step:06d}.npz", t, step)

    # Run solver
    print("Starting solver...")
    if args.rigid_body:
        print("Using partitioned FSI coupling (Phase 2)")
        # Import rigid body module
        import numpy as np

        from src import gas
        from src.rigidbody import make_circle

        # Create a free circular cylinder at origin.
        # Density must be much larger than the fluid (rho_fluid ~ 1–4 in non-dim units)
        # so the body isn't launched at unphysical acceleration.
        # A solid aluminium-like cylinder in air at Mach 3 has density ratio ~2700.
        # We use density=100 here (100× freestream) as a physically reasonable
        # intermediate value that keeps the simulation well-behaved while still
        # showing visible body motion during the simulation window.
        body = make_circle(center=np.array([0.0, 0.0]), radius=0.5, density=100.0)

        # Manual time loop for FSI
        Q = Q0.copy()
        t = 0.0
        trajectory = []  # Store body position/velocity history

        # Initialize with shock wave at x = -3.0
        # Rankine-Hugoniot relations for M=3, gamma=1.4, p_inf=1, rho_inf=1
        # Non-dim: a_inf = sqrt(gamma*p/rho) = 1.183, u_inf = M*a = 3.55
        # Pre-shock (x >= -3.0): rho=1.0, u=3.55, v=0.0, p=1.0
        # Post-shock (x < -3.0): rho=3.857, u=0.920, v=0.0, p=10.333
        print("Initializing shock wave at x = -3.0 (Mach 3 normal shock)")
        import numpy as np

        xc_np = np.array(grid.x)
        gamma = gas.GAMMA

        # Post-shock state (downstream, x < -3.0) from Rankine-Hugoniot
        rho_post = 3.857143
        u_post = 0.920279
        v_post = 0.0
        p_post = 10.333333
        E_post = p_post / (gamma - 1.0) + 0.5 * rho_post * (u_post**2 + v_post**2)

        # Pre-shock state (upstream, x >= -3.0) is already set to freestream above
        # Just overwrite the post-shock region
        shock_x = -3.0
        post_shock_mask = xc_np < shock_x
        Q_np = np.array(Q)
        Q_np[0, post_shock_mask] = rho_post
        Q_np[1, post_shock_mask] = rho_post * u_post
        Q_np[2, post_shock_mask] = rho_post * v_post
        Q_np[3, post_shock_mask] = E_post
        Q = xp.array(Q_np)

        a_inf = float((gamma * 1.0 / 1.0) ** 0.5)
        u_pre = float(args.mach * a_inf)
        print(f"  Pre-shock:  rho={1.0:.4f}, u={u_pre:.4f}, p={1.0:.4f}")
        print(f"  Post-shock: rho={rho_post:.4f}, u={u_post:.4f}, p={p_post:.4f}")

        # Initial ghost cell fill BEFORE first dt computation
        # (otherwise interior cells have stale freestream state → bad CFL)
        import numpy as _np_init

        from src.levelset import compute_levelset as _cl_init
        from src.levelset import fill_ghost_cells as _fgc_init

        _phi0 = _cl_init(body, _np_init.array(grid.x), _np_init.array(grid.y))
        Q = _fgc_init(Q, _phi0, body, _np_init.array(grid.x), _np_init.array(grid.y), gas)

        for step in range(1, args.steps + 1):
            # Compute time step from fluid CFL
            # Note: Q here is the OUTPUT of the previous step_partitioned_fsi, which
            # already has ghost cells filled at the END of that step.
            import numpy as np

            from src.levelset import compute_levelset
            from src.solver import compute_dt

            # Recompute phi at current body position to exclude ghost cells from CFL.
            # Ghost cells have rho=1e-6 + reflected velocity → dt collapses without this.
            _phi_cfl = compute_levelset(body, np.array(grid.x), np.array(grid.y))
            dt = compute_dt(Q, grid, args.cfl, phi=_phi_cfl)

            # Additional CFL constraint from body velocity
            dx_min = float(abs(grid.x[1, 0] - grid.x[0, 0]))
            v_body_mag = float(np.linalg.norm(body.velocity))
            if v_body_mag > 1e-10:
                dt_body = args.cfl * dx_min / (v_body_mag + 1e-10)
                dt = min(dt, dt_body)

            # Partitioned FSI step
            Q, body = step_partitioned_fsi(
                Q,
                body,
                grid,
                gas,
                dt,
                fluid_integrator="rk4",
                use_csl=args.csl,
                use_hybrid=args.hybrid,
            )
            t += dt

            # Record trajectory
            trajectory.append(
                {
                    "t": t,
                    "x": float(body.position[0]),
                    "y": float(body.position[1]),
                    "vx": float(body.velocity[0]),
                    "vy": float(body.velocity[1]),
                    "angle": float(body.angle),
                    "omega": float(body.angular_velocity),
                }
            )

            if step % args.print_every == 0:
                p = pressure(Q)
                rho_min = float(xp.min(Q[0]))
                rho_max = float(xp.max(Q[0]))
                p_min = float(xp.min(p))
                p_max = float(xp.max(p))
                print(
                    f"Step {step:6d}  t={t:.6f}  dt={dt:.2e}  "
                    f"rho=[{rho_min:.4f}, {rho_max:.4f}]  p=[{p_min:.4f}, {p_max:.4f}]"
                )
                print(
                    f"  Body: x=[{body.position[0]:.4f}, {body.position[1]:.4f}]  "
                    f"v=[{body.velocity[0]:.4f}, {body.velocity[1]:.4f}]"
                )

            if step % args.save_every == 0:
                save_callback(step, t, Q)

        Q_final = Q

        # Save trajectory
        os.makedirs(args.output, exist_ok=True)
        import numpy as np

        np.savez(
            f"{args.output}/body_trajectory.npz",
            time=[rec["t"] for rec in trajectory],
            x=[rec["x"] for rec in trajectory],
            y=[rec["y"] for rec in trajectory],
            vx=[rec["vx"] for rec in trajectory],
            vy=[rec["vy"] for rec in trajectory],
            angle=[rec["angle"] for rec in trajectory],
            omega=[rec["omega"] for rec in trajectory],
        )
        print(f"Body trajectory saved to {args.output}/body_trajectory.npz")
    elif args.semi_implicit:
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
