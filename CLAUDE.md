# 2D Compressible Flow Solver

## Architecture
- **Python** for orchestration, grid generation, boundary conditions, I/O, visualization
- **CuPy** for GPU-accelerated computation (flux calculation, reconstruction, time stepping)
- **Fallback**: NumPy backend when no GPU available (same interface via array module abstraction)

## Numerical Methods
- **Equations**: 2D compressible Euler equations in conservation form (density, x-momentum, y-momentum, energy)
- **Grid**: Structured body-fitted O-grid around a cylinder
- **Flux scheme**: Roe's approximate Riemann solver with entropy fix
- **Reconstruction**: MUSCL with van Leer limiter (2nd order)
- **Time integration**: Explicit 4-stage Runge-Kutta (RK4)
- **Boundary conditions**: Freestream (characteristic-based), solid wall (slip/no-slip), non-reflecting outflow

## Project Structure
```
cfd-solver/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── grid.py          # O-grid generator around cylinder
│   ├── solver.py        # Main solver loop, RK4 time stepping
│   ├── flux.py          # Roe flux computation (GPU-accelerated)
│   ├── reconstruction.py # MUSCL reconstruction with limiters
│   ├── boundary.py      # Boundary condition implementations
│   ├── gas.py           # Equation of state, thermodynamic relations
│   ├── backend.py       # Array backend abstraction (CuPy/NumPy)
│   └── io.py            # Solution I/O, checkpointing
├── scripts/
│   ├── run_cylinder.py  # Main driver script for cylinder flow
│   └── visualize.py     # Post-processing and visualization (matplotlib)
└── tests/
    ├── test_flux.py     # Validate Roe flux against exact solutions
    ├── test_grid.py     # Grid quality checks
    └── test_sod.py      # Sod shock tube validation (1D subset)
```

## Key Design Decisions
1. **Backend abstraction**: `backend.py` provides `xp` module that's either CuPy or NumPy. All compute code uses `xp.array()`, `xp.zeros()`, etc. Switch with env var `CFD_BACKEND=cupy|numpy`.
2. **Conservative variables**: Store as 4D array `Q[4, ni, nj]` — density, rho*u, rho*v, rho*E
3. **Metric terms**: Pre-compute and store grid metrics (Jacobian, contravariant basis vectors) for curvilinear coordinates
4. **CFL-based time stepping**: Compute stable dt from CFL condition each step

## Validation Targets
- **Sod shock tube** (1D): exact solution comparison
- **Cylinder flow** (2D): drag coefficient, pressure distribution, Strouhal number vs published data
- Mach 0.3 (subsonic, vortex shedding), Mach 2.0 (supersonic, bow shock)

## Build & Run
```bash
pip install -r requirements.txt
python scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 10000
python scripts/visualize.py --input output/solution.npz
```
