"""Numba-accelerated compute kernels for the 2D Euler solver.

Fuses MUSCL reconstruction + Roe flux into tight loops, eliminating
temporary array allocations. Falls back gracefully if Numba is absent.

The η-direction logic exactly mirrors the vectorized implementation:
  - MUSCL interfaces between (j=1,j=2) through (j=nj-3,j=nj-2)
  - Cell j=1: wall first-order + MUSCL top face
  - Cells j=2..nj-3: MUSCL flux differences
  - j=0, j>=nj-2: no η flux (BCs overwrite)
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange  # type: ignore[import-not-found]

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _wrap

    def prange(*args):  # type: ignore[no-redef]
        return range(*args)


GAMMA = 1.4


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


@njit(inline="always")
def _pressure(rho, rhou, rhov, rhoE):
    u = rhou / rho
    v = rhov / rho
    return (GAMMA - 1.0) * (rhoE - 0.5 * rho * (u * u + v * v))


@njit(inline="always")
def _sound_speed(rho, p):
    return math.sqrt(GAMMA * abs(p) / max(rho, 1e-30))


@njit(inline="always")
def _enthalpy(rho, rhou, rhov, rhoE):
    p = _pressure(rho, rhou, rhov, rhoE)
    return (rhoE + p) / rho


@njit(inline="always")
def _entropy_fix(lam, eps):
    if abs(lam) < eps:
        return (lam * lam + eps * eps) / (2.0 * eps)
    return abs(lam)


@njit(inline="always")
def _van_leer(r):
    return (r + abs(r)) / (1.0 + abs(r))


@njit(inline="always")
def _roe_flux(rhoL, rhouL, rhovL, rhoEL, rhoR, rhouR, rhovR, rhoER, nx, ny):
    """Roe flux for a single interface."""
    uL = rhouL / rhoL
    vL = rhovL / rhoL
    pL = _pressure(rhoL, rhouL, rhovL, rhoEL)
    HL = _enthalpy(rhoL, rhouL, rhovL, rhoEL)
    vnL = uL * nx + vL * ny

    uR = rhouR / rhoR
    vR = rhovR / rhoR
    pR = _pressure(rhoR, rhouR, rhovR, rhoER)
    HR = _enthalpy(rhoR, rhouR, rhovR, rhoER)
    vnR = uR * nx + vR * ny

    FL0 = rhoL * vnL
    FL1 = rhouL * vnL + pL * nx
    FL2 = rhovL * vnL + pL * ny
    FL3 = (rhoEL + pL) * vnL
    FR0 = rhoR * vnR
    FR1 = rhouR * vnR + pR * nx
    FR2 = rhovR * vnR + pR * ny
    FR3 = (rhoER + pR) * vnR

    sqL = math.sqrt(max(rhoL, 1e-30))
    sqR = math.sqrt(max(rhoR, 1e-30))
    denom = sqL + sqR
    u_roe = (sqL * uL + sqR * uR) / denom
    v_roe = (sqL * vL + sqR * vR) / denom
    H_roe = (sqL * HL + sqR * HR) / denom
    q2 = u_roe * u_roe + v_roe * v_roe
    a_roe = math.sqrt(max((GAMMA - 1.0) * (H_roe - 0.5 * q2), 1e-30))
    vn_roe = u_roe * nx + v_roe * ny
    rho_roe = sqL * sqR

    dvn = vnR - vnL
    dp = pR - pL
    drho = rhoR - rhoL

    eps = 0.1 * a_roe
    lam1 = _entropy_fix(vn_roe - a_roe, eps)
    lam2 = _entropy_fix(vn_roe, eps)
    lam3 = _entropy_fix(vn_roe + a_roe, eps)

    a2 = a_roe * a_roe
    alpha1 = 0.5 * (dp - rho_roe * a_roe * dvn) / a2
    alpha2 = drho - dp / a2
    alpha3 = 0.5 * (dp + rho_roe * a_roe * dvn) / a2

    du = uR - uL
    dv = vR - vL
    dvt_x = du - dvn * nx
    dvt_y = dv - dvn * ny

    d0 = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    d1 = (
        lam1 * alpha1 * (u_roe - a_roe * nx)
        + lam2 * (alpha2 * u_roe + rho_roe * dvt_x)
        + lam3 * alpha3 * (u_roe + a_roe * nx)
    )
    d2 = (
        lam1 * alpha1 * (v_roe - a_roe * ny)
        + lam2 * (alpha2 * v_roe + rho_roe * dvt_y)
        + lam3 * alpha3 * (v_roe + a_roe * ny)
    )
    d3 = (
        lam1 * alpha1 * (H_roe - a_roe * vn_roe)
        + lam2 * (alpha2 * 0.5 * q2 + rho_roe * (u_roe * dvt_x + v_roe * dvt_y))
        + lam3 * alpha3 * (H_roe + a_roe * vn_roe)
    )

    return (
        0.5 * (FL0 + FR0) - 0.5 * d0,
        0.5 * (FL1 + FR1) - 0.5 * d1,
        0.5 * (FL2 + FR2) - 0.5 * d2,
        0.5 * (FL3 + FR3) - 0.5 * d3,
    )


@njit(inline="always")
def _muscl_lr(qim1, qi, qip1, qip2):
    """MUSCL reconstruct left and right states at i+1/2."""
    eps = 1e-12
    dQf = qip1 - qi
    dQb = qi - qim1
    dQf_s = dQf if abs(dQf) > eps else 1.0
    r_L = dQb / dQf_s if abs(dQf) > eps else 0.0
    qL = qi + 0.5 * _van_leer(r_L) * dQf

    dQf2 = qip2 - qip1
    dQb2 = qip1 - qi
    dQb2_s = dQb2 if abs(dQb2) > eps else 1.0
    r_R = dQf2 / dQb2_s if abs(dQb2) > eps else 0.0
    qR = qip1 - 0.5 * _van_leer(r_R) * dQb2

    return qL, qR


@njit(inline="always")
def _xi_flux_at(Q, xi_x_area, xi_y_area, i, ip1, j, ni):
    """ξ-direction MUSCL + Roe flux at interface i+1/2, j."""
    im1 = (i - 1) % ni
    ip2 = (ip1 + 1) % ni
    nx = 0.5 * (xi_x_area[i, j] + xi_x_area[ip1, j])
    ny = 0.5 * (xi_y_area[i, j] + xi_y_area[ip1, j])

    rL0, rR0 = _muscl_lr(Q[0, im1, j], Q[0, i, j], Q[0, ip1, j], Q[0, ip2, j])
    rL1, rR1 = _muscl_lr(Q[1, im1, j], Q[1, i, j], Q[1, ip1, j], Q[1, ip2, j])
    rL2, rR2 = _muscl_lr(Q[2, im1, j], Q[2, i, j], Q[2, ip1, j], Q[2, ip2, j])
    rL3, rR3 = _muscl_lr(Q[3, im1, j], Q[3, i, j], Q[3, ip1, j], Q[3, ip2, j])

    return _roe_flux(rL0, rL1, rL2, rL3, rR0, rR1, rR2, rR3, nx, ny)


@njit(inline="always")
def _eta_flux_muscl_at(Q, eta_x_area, eta_y_area, i, j):
    """η-direction MUSCL + Roe flux at interface j+1/2. Needs j-1..j+2."""
    nx = 0.5 * (eta_x_area[i, j] + eta_x_area[i, j + 1])
    ny = 0.5 * (eta_y_area[i, j] + eta_y_area[i, j + 1])

    rL0, rR0 = _muscl_lr(Q[0, i, j - 1], Q[0, i, j], Q[0, i, j + 1], Q[0, i, j + 2])
    rL1, rR1 = _muscl_lr(Q[1, i, j - 1], Q[1, i, j], Q[1, i, j + 1], Q[1, i, j + 2])
    rL2, rR2 = _muscl_lr(Q[2, i, j - 1], Q[2, i, j], Q[2, i, j + 1], Q[2, i, j + 2])
    rL3, rR3 = _muscl_lr(Q[3, i, j - 1], Q[3, i, j], Q[3, i, j + 1], Q[3, i, j + 2])

    return _roe_flux(rL0, rL1, rL2, rL3, rR0, rR1, rR2, rR3, nx, ny)


# ---------------------------------------------------------------------------
# Main residual kernel
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def compute_residual_numba(Q, xi_x_area, xi_y_area, eta_x_area, eta_y_area, jacobian, ni, nj):
    """Compute spatial residual matching the vectorized implementation exactly.

    ξ: periodic MUSCL + Roe for ALL cells.
    η: MUSCL interfaces between (j=1,j=2)..(j=nj-3,j=nj-2), i.e. nj-3 total.
       Cell j=1: bottom = first-order wall(j=0,j=1), top = interface 0
       Cells j=2..nj-3: R -= G[j-1] - G[j-2]
       Cells j=0, j≥nj-2: no η flux.
    """
    R = np.zeros((4, ni, nj))
    n_efaces = nj - 3  # number of MUSCL η-interfaces

    for i in prange(ni):
        ip1 = (i + 1) % ni

        # --- ξ fluxes: compute F_{i+1/2} once, accumulate to cells i and i+1 ---
        # Store interface fluxes in temp array, then do differences
        Fxi = np.zeros((4, nj))
        for j in range(nj):
            f0, f1, f2, f3 = _xi_flux_at(Q, xi_x_area, xi_y_area, i, ip1, j, ni)
            Fxi[0, j] = f0
            Fxi[1, j] = f1
            Fxi[2, j] = f2
            Fxi[3, j] = f3

        # Cell i: right face is Fxi[i+1/2], left face is Fxi[(i-1)+1/2] stored by prev i
        # But since we parallelize over i, we can't share between iterations.
        # Instead: R[i] -= F_{i+1/2} and R[i] += F_{i-1/2}
        # We compute F_{i+1/2} above. We need F_{i-1/2} too.
        im1 = (i - 1) % ni
        for j in range(nj):
            # Subtract right face
            R[0, i, j] -= Fxi[0, j]
            R[1, i, j] -= Fxi[1, j]
            R[2, i, j] -= Fxi[2, j]
            R[3, i, j] -= Fxi[3, j]
            # Add left face (computed fresh — this is the im1+1/2 interface)
            fl0, fl1, fl2, fl3 = _xi_flux_at(Q, xi_x_area, xi_y_area, im1, i, j, ni)
            R[0, i, j] += fl0
            R[1, i, j] += fl1
            R[2, i, j] += fl2
            R[3, i, j] += fl3

        # --- η fluxes ---
        # Compute all MUSCL interface fluxes G[k], k=0..n_efaces-1
        # Interface k is between j=k+1 and j=k+2
        G = np.zeros((4, n_efaces))
        for k in range(n_efaces):
            j = k + 1
            f0, f1, f2, f3 = _eta_flux_muscl_at(Q, eta_x_area, eta_y_area, i, j)
            G[0, k] = f0
            G[1, k] = f1
            G[2, k] = f2
            G[3, k] = f3

        # Cell j=1: bottom = wall flux, top = G[0]
        nx_w = eta_x_area[i, 0]
        ny_w = eta_y_area[i, 0]
        Fw0, Fw1, Fw2, Fw3 = _roe_flux(
            Q[0, i, 0],
            Q[1, i, 0],
            Q[2, i, 0],
            Q[3, i, 0],
            Q[0, i, 1],
            Q[1, i, 1],
            Q[2, i, 1],
            Q[3, i, 1],
            nx_w,
            ny_w,
        )
        if n_efaces >= 1:
            R[0, i, 1] -= G[0, 0] - Fw0
            R[1, i, 1] -= G[1, 0] - Fw1
            R[2, i, 1] -= G[2, 0] - Fw2
            R[3, i, 1] -= G[3, 0] - Fw3

        # Cells j=2..nj-3: R[j] -= G[j-1] - G[j-2]
        if n_efaces >= 2:
            for jj in range(2, 2 + n_efaces - 1):
                # jj ranges from 2 to nj-3
                # G index for top face of cell jj: k_top = jj-1-1 = jj-2... let me re-derive.
                # Interface k between j=k+1 and j=k+2.
                # Cell jj's bottom face: interface between j=jj-1 and j=jj → k = jj-2
                # Cell jj's top face: interface between j=jj and j=jj+1 → k = jj-1
                k_top = jj - 1
                k_bot = jj - 2
                R[0, i, jj] -= G[0, k_top] - G[0, k_bot]
                R[1, i, jj] -= G[1, k_top] - G[1, k_bot]
                R[2, i, jj] -= G[2, k_top] - G[2, k_bot]
                R[3, i, jj] -= G[3, k_top] - G[3, k_bot]

        # --- Divide by |J| ---
        for j in range(nj):
            inv_J = 1.0 / (abs(jacobian[i, j]) + 1e-30)
            R[0, i, j] *= inv_J
            R[1, i, j] *= inv_J
            R[2, i, j] *= inv_J
            R[3, i, j] *= inv_J

    return R


# ---------------------------------------------------------------------------
# CFL time step
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def compute_dt_numba(Q, xi_x_area, xi_y_area, eta_x_area, eta_y_area, jacobian, ni, nj, cfl):
    """Compute stable time step from CFL condition."""
    dt_per_row = np.full(ni, 1e30)
    for i in prange(ni):
        local_min = 1e30
        for j in range(nj):
            rho = Q[0, i, j]
            u = Q[1, i, j] / rho
            v = Q[2, i, j] / rho
            p = _pressure(rho, Q[1, i, j], Q[2, i, j], Q[3, i, j])
            a = _sound_speed(rho, p)

            J_abs = abs(jacobian[i, j]) + 1e-30
            U_xi = u * xi_x_area[i, j] + v * xi_y_area[i, j]
            U_eta = u * eta_x_area[i, j] + v * eta_y_area[i, j]

            xi_mag = math.sqrt(xi_x_area[i, j] ** 2 + xi_y_area[i, j] ** 2)
            eta_mag = math.sqrt(eta_x_area[i, j] ** 2 + eta_y_area[i, j] ** 2)

            sr = abs(U_xi) + a * xi_mag + abs(U_eta) + a * eta_mag + 1e-30
            dt_local = cfl * J_abs / sr
            if dt_local < local_min:
                local_min = dt_local
        dt_per_row[i] = local_min
    return np.min(dt_per_row)
