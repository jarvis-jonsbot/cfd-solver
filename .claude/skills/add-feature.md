# Adding Features to cfd-solver

## Adding a New Flux Scheme

1. Create a new function in `src/flux.py` following the signature of `roe_flux()`
2. Accept conservative state arrays `QL`, `QR` and area-weighted face normals
3. Return the numerical flux array `F[4, ...]`
4. Add unit tests in `tests/test_flux.py` comparing against exact/known solutions
5. Wire it into `solver.py` as a selectable option

## Adding a New Boundary Condition

1. Add a new function in `src/boundary.py`
2. It should fill ghost cells given interior state + boundary parameters
3. Add tests in a new or existing test file
4. Document the BC in CLAUDE.md under "Boundary conditions"

## Adding a New Limiter

1. Add the limiter function in `src/reconstruction.py`
2. It must accept ratio `r` and return the limiter value `phi(r)`
3. Test against known limiter properties (symmetry, TVD region)

## General Checklist

- [ ] All new code passes `make lint` and `make typecheck`
- [ ] Tests pass: `make test`
- [ ] Commit with conventional format: `feat: add <description>`
- [ ] Update CLAUDE.md if the change affects architecture or numerics
