# 2D Compressible Flow Solver

## Architecture

- **Python** for orchestration, grid generation, boundary conditions, I/O, visualization
- **CuPy** for GPU-accelerated computation (flux calculation, reconstruction, time stepping)
- **Fallback**: NumPy backend when no GPU available (same interface via array module abstraction)

## Numerical Methods

- **Equations**: 2D compressible Euler equations in conservation form (density, x-momentum, y-momentum, energy)
- **Grid**: Structured body-fitted O-grid around a cylinder
- **Flux scheme**: Roe's approximate Riemann solver with Harten's entropy fix
- **Reconstruction**: MUSCL with van Leer limiter (2nd order)
- **Time integration**: Explicit 4-stage Runge-Kutta (RK4)
- **Boundary conditions**: Freestream (characteristic-based), solid wall (slip/no-slip), periodic (circumferential)

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
│   ├── grid.py            # O-grid generator; stores both contravariant metrics AND area-weighted face normals
│   ├── solver.py          # Main solver loop, RK4 time stepping, residual computation
│   ├── flux.py            # Roe flux computation
│   ├── reconstruction.py  # MUSCL reconstruction with limiters
│   ├── boundary.py        # Boundary condition implementations
│   ├── gas.py             # Equation of state, thermodynamic relations
│   ├── backend.py         # Array backend abstraction (CuPy/NumPy)
│   └── io.py              # Solution I/O, checkpointing
├── scripts/
│   ├── run_cylinder.py    # Main driver script for cylinder flow
│   └── visualize.py       # Post-processing and visualization (matplotlib)
└── tests/
    ├── test_flux.py       # Validate Roe flux against exact solutions
    ├── test_grid.py       # Grid quality checks
    ├── test_sod.py        # Sod shock tube validation (1D subset)
    └── test_smoke.py      # End-to-end solver smoke tests (NaN/Inf checks)
```

## Key Design Decisions

1. **Backend abstraction**: `backend.py` provides `xp` module that's either CuPy or NumPy. All compute code uses `xp.array()`, `xp.zeros()`, etc. Switch with env var `CFD_BACKEND=cupy|numpy`.
2. **Conservative variables**: Store as 4D array `Q[4, ni, nj]` — density, rho*u, rho*v, rho*E
3. **Area-weighted face normals**: Grid stores both contravariant metrics (`xi_x`, etc. — divided by J) and area-weighted normals (`xi_x_area = y_eta`, `xi_y_area = -x_eta`, etc. — NOT divided by J). The flux computation uses the area-weighted normals so the Roe solver returns physical flux through each face, which is then divided by cell volume (|J|) to get the residual.
4. **CFL-based time stepping**: Compute stable dt from CFL condition each step using area-weighted spectral radii.

## Critical Numerical Notes

- **Metric scaling**: The face normals passed to the Roe flux MUST be area-weighted (not divided by J). Using contravariant metrics (divided by J) and then dividing by J again produces 1/J² scaling → immediate overflow/NaN.
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

## Don't

- Don't use bare `numpy` imports — always go through `backend.xp`
- Don't divide by Jacobian in flux computation (use area-weighted normals directly)
- Don't hardcode gamma=1.4 — use `gas.gamma`
- Don't skip MUSCL stencil offset accounting (N points → N-3 interfaces)
- Don't commit output files (`.npz`, images) — they're gitignored
- Don't merge to main without CI green

## Skills

See `.claude/skills/` for workflow guides (adding flux schemes, BCs, limiters).
