# 2D Compressible Flow Solver

## Architecture

- **Python** for orchestration, grid generation, boundary conditions, I/O, visualization
- **MLX** for Apple Silicon GPU acceleration (float32 only — CFL auto-capped at 0.08)
- **CuPy** for NVIDIA GPU-accelerated computation (flux calculation, reconstruction, time stepping)
- **Fallback**: NumPy backend when no GPU available (same interface via array module abstraction)

## Numerical Methods

- **Equations**: 2D compressible Euler equations in conservation form (density, x-momentum, y-momentum, energy)
- **Grid**: Structured body-fitted O-grid around a cylinder (or Cartesian for FSI)
- **Flux scheme**: Roe's approximate Riemann solver with Harten's entropy fix
- **Reconstruction**: MUSCL with van Leer limiter (2nd order)
- **Time integration**: Explicit 4-stage Runge-Kutta (RK4) or semi-implicit pressure solver
- **Boundary conditions**: Freestream (characteristic-based), solid wall (slip/no-slip), periodic (circumferential), immersed boundary (level set)
- **FSI coupling**: Partitioned rigid body dynamics with level set ghost cells (Phase 2)

## Project Structure

```text
cfd-solver/
├── CLAUDE.md
├── README.md
├── pyproject.toml         # PEP 621 packaging (replaces bare requirements.txt)
├── requirements.txt       # Kept for compatibility
├── Makefile               # make setup / test / run / clean
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── grid.py            # O-grid and Cartesian grid generators; stores both contravariant metrics AND area-weighted face normals
│   ├── solver.py          # Main solver loop, RK4 time stepping, semi-implicit pressure, partitioned FSI
│   ├── flux.py            # Roe flux computation
│   ├── reconstruction.py  # MUSCL reconstruction with limiters
│   ├── boundary.py        # Boundary condition implementations
│   ├── gas.py             # Equation of state, thermodynamic relations
│   ├── backend.py         # Array backend abstraction (MLX/CuPy/NumPy)
│   ├── pressure.py        # Implicit pressure solver (Phase 1)
│   ├── rigidbody.py       # Rigid body dynamics (Phase 2)
│   ├── levelset.py        # Level set and ghost cells for immersed boundaries (Phase 2)
│   └── io.py              # Solution I/O, checkpointing
├── scripts/
│   ├── run_cylinder.py    # Main driver script for cylinder flow
│   └── visualize.py       # Post-processing and visualization (matplotlib)
└── tests/
    ├── test_flux.py          # Validate Roe flux against exact solutions
    ├── test_grid.py          # Grid quality checks
    ├── test_sod.py           # Sod shock tube validation (1D subset)
    ├── test_smoke.py         # End-to-end solver smoke tests (NaN/Inf checks)
    ├── test_semi_implicit.py # Semi-implicit pressure solver tests (Phase 1)
    └── test_rigidbody.py     # Rigid body dynamics and FSI tests (Phase 2)
```

## Key Design Decisions

1. **Backend abstraction**: `backend.py` provides `xp` module that's MLX, CuPy, or NumPy. All compute code uses `xp.array()`, `xp.zeros()`, etc. Switch with env var `CFD_BACKEND=mlx|cupy|numpy`. MLX uses a shim class to patch API gaps (e.g. `linspace(endpoint=False)`).
2. **Conservative variables**: Store as 4D array `Q[4, ni, nj]` — density, rho*u, rho*v, rho*E
3. **Area-weighted face normals**: Grid stores both contravariant metrics (`xi_x`, etc. — divided by J) and area-weighted normals (`xi_x_area = y_eta`, `xi_y_area = -x_eta`, etc. — NOT divided by J). The flux computation uses the area-weighted normals so the Roe solver returns physical flux through each face, which is then divided by cell volume (|J|) to get the residual.
4. **CFL-based time stepping**: Compute stable dt from CFL condition each step using area-weighted spectral radii.
5. **Partitioned FSI** (Phase 2): Rigid body dynamics with level set immersed boundary. Fluid forces → body motion → ghost cell update → fluid step. No inner iteration (explicit sequential coupling).

## Critical Numerical Notes

- **Metric scaling**: The face normals passed to the Roe flux MUST be area-weighted (not divided by J). Using contravariant metrics (divided by J) and then dividing by J again produces 1/J² scaling → immediate overflow/NaN.
- **Jacobian sign**: Area-weighted normals are multiplied by `sign(J)` to ensure correct orientation regardless of coordinate handedness. The O-grid has J < 0; without this correction, normals point inward and the Roe dissipation becomes anti-dissipation.
- **Float32 precision**: MLX only supports float32. The solver uses `EPS_TINY` and `EPS_SLOPE` constants from `backend.py` that auto-adjust based on backend (1e-30/1e-12 for float64, 1e-7/1e-5 for float32). CFL must be ≤ 0.08 for float32 stability.
- **MUSCL stencil trimming**: MUSCL reconstruction on N points produces N-3 interfaces. The face-to-cell index mapping must account for this offset.
- **Wall boundary**: Ghost cell at j=0, first interior cell at j=1. The wall-adjacent cell gets a first-order flux from the wall BC.

## Build & Run (Recommended: use venv)

```bash
# First time setup
make setup          # creates .venv, installs deps

# Run tests
make test           # pytest with all tests including smoke tests

# Run solver
make run            # Mach 0.3 cylinder flow, 5000 steps

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 10000
python scripts/visualize.py --input output/solution.npz

# Apple Silicon GPU acceleration:
CFD_BACKEND=mlx python scripts/run_cylinder.py --mach 0.3 --steps 5000
# or: pip install -e ".[gpu-apple]"

# Rigid body FSI mode (Phase 2):
python scripts/run_cylinder.py --rigid-body --steps 1000
# Automatically uses Cartesian grid, Mach 3 shock hit, saves body trajectory
```

## Validation Targets

- **Sod shock tube** (1D): exact solution comparison (test_sod.py)
- **Smoke tests**: subsonic (M=0.3) and supersonic (M=2.0) runs must produce no NaN/Inf (test_smoke.py)
- **Cylinder flow** (2D): drag coefficient, pressure distribution vs published data

## CI

GitHub Actions runs on push/PR to main. Lint (Ruff + mypy) on 3.12, tests on 3.9/3.11/3.12, markdown lint.

## Code Quality

```bash
make lint          # ruff check + format check
make lint-fix      # auto-fix lint issues + format
make typecheck     # mypy strict mode
```

## ⚠️ Definition of Done — Required Before Every Commit

**All four of these must pass before you consider any task complete:**

```bash
make lint          # must exit 0 — no ruff errors, no formatting issues
make test          # must exit 0 — all tests passing
make typecheck     # must exit 0 — no mypy errors
make animate       # generate animation for PR (see below)
```

If `make lint` fails:

1. Run `make lint-fix` to auto-fix what ruff can fix automatically
2. Manually fix any remaining issues (E501 line-too-long, SIM/B rules that need manual rewrites)
3. Re-run `make lint` until it passes
4. Then run `make test` to confirm nothing broke

**Never commit code that fails `make lint`.** CI will block the PR and it wastes review cycles.

Common lint mistakes to avoid proactively:

- Remove unused imports immediately (F401)
- Remove unused local variables (F841) — use `_name` convention if a variable must exist but is unused
- Keep lines ≤ 100 characters (E501) — break long `parser.add_argument(...)` calls across lines
- Use ternary operator for simple if/else (SIM108): `x = a if cond else b`
- Don't use bare `except:` (E722) — always specify exception type
- Rename unused loop vars to `_` or `_name` (B007)

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature (flux scheme, BC, limiter)
- `fix:` bug fix
- `perf:` performance improvement
- `refactor:` code restructure without behavior change
- `test:` add/update tests
- `docs:` documentation only
- `chore:` build, CI, tooling changes

Scope is optional: `feat(flux): add HLLC scheme`

## 🎬 PR Animation (Required)

Every PR that touches numerical code (flux, advection, pressure, FSI coupling) **must** include a flow animation in the PR description. This is a best practice — we have caught multiple bugs visually that passed all tests.

### How to generate

```bash
make animate       # runs a short simulation and produces /tmp/cfd-anim.gif
```

Or manually:

```bash
# Run a representative simulation and save frames
python scripts/run_cylinder.py --mach 0.5 --steps 400 --save-every 10 --output /tmp/anim-frames

# Produce the GIF (requires pillow)
python scripts/animate.py --input /tmp/anim-frames --output /tmp/cfd-anim.gif --fps 12
```

### Upload to PR

1. Upload the GIF to a GitHub release (use `gh release create` or drag-and-drop on GitHub)
2. Add to the PR description as:

```markdown
## 🎬 Visual Verification

> Mach X cylinder flow, N steps, [method name] active. Density (left) and pressure (right).

![Flow animation](https://github.com/jarvis-jonsbot/cfd-solver/releases/download/<tag>/cfd-anim.gif)
```

**What to look for:**
- Smooth, symmetric pressure distribution (asymmetry = bug in boundary conditions or flux)
- No checkerboard patterns (pressure-velocity decoupling)
- No unphysical density values (ρ < 0 = major bug)
- Shock structures that don't smear unexpectedly over time
- For FSI: body motion that follows physical intuition (shock push = body moves downstream)

## Don't

- Don't commit without running `make lint && make test` (see Definition of Done above)
- Don't open a PR on numerical code without a flow animation in the description
- Don't use bare `numpy` imports — always go through `backend.xp`
- Don't divide by Jacobian in flux computation (use area-weighted normals directly)
- Don't hardcode gamma=1.4 — use `gas.gamma`
- Don't skip MUSCL stencil offset accounting (N points → N-3 interfaces)
- Don't commit output files (`.npz`, images) — they're gitignored
- Don't merge to main without CI green

## Skills

See `.claude/skills/` for workflow guides (adding flux schemes, BCs, limiters).
