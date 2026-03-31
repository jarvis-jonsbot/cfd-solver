"""Level set and ghost cell treatment for immersed boundaries.

Implements signed distance computation and ghost cell filling for
rigid bodies immersed in the fluid domain.

Based on Grétarsson (2012), Chapter 3.
"""

from __future__ import annotations

import numpy as np

from src.backend import EPS_TINY, xp
from src.rigidbody import RigidBody


def compute_levelset(body: RigidBody, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Compute signed distance function phi at cell centers.

    Convention: phi < 0 inside body, phi > 0 outside.

    Args:
        body: RigidBody object
        xc: cell center x-coordinates, shape (ni, nj)
        yc: cell center y-coordinates, shape (ni, nj)

    Returns:
        phi: signed distance, shape (ni, nj)
    """
    if body.shape == "circle":
        return _levelset_circle(body, xc, yc)
    if body.shape == "polygon":
        return _levelset_polygon(body, xc, yc)
    raise ValueError(f"Unknown body shape: {body.shape}")


def _levelset_circle(body: RigidBody, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Signed distance to circle: phi = |x - center| - R."""
    dx = xc - body.position[0]
    dy = yc - body.position[1]
    dist = np.sqrt(dx**2 + dy**2)
    return dist - body.radius  # type: ignore[no-any-return]


def _levelset_polygon(body: RigidBody, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Signed distance to polygon.

    Uses point-in-polygon test + nearest edge distance.
    """
    if body.vertices_body is None:
        raise ValueError("Polygon body must have vertices_body set")

    vertices_world = body.vertices_world()
    ni, nj = xc.shape
    phi = np.zeros((ni, nj))

    for i in range(ni):
        for j in range(nj):
            x = np.array([xc[i, j], yc[i, j]])
            inside = _point_in_polygon(x, vertices_world)
            dist = _point_to_polygon_distance(x, vertices_world)
            phi[i, j] = -dist if inside else dist

    return phi  # type: ignore[no-any-return]


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Ray casting algorithm for point-in-polygon test.

    Args:
        point: [x, y]
        vertices: shape (N, 2)

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(vertices)
    inside = False

    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def _point_to_polygon_distance(point: np.ndarray, vertices: np.ndarray) -> float:
    """Minimum distance from point to polygon boundary.

    Args:
        point: [x, y]
        vertices: shape (N, 2)

    Returns:
        minimum distance to any edge
    """
    n = len(vertices)
    min_dist: float = float(np.inf)

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        dist = _point_to_segment_distance(point, v1, v2)
        min_dist = min(min_dist, dist)

    return min_dist


def _point_to_segment_distance(p: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Distance from point p to line segment [v1, v2]."""
    # Project p onto line through v1, v2
    edge = v2 - v1
    edge_len2 = np.dot(edge, edge)
    if edge_len2 < 1e-12:
        return float(np.linalg.norm(p - v1))

    t = np.dot(p - v1, edge) / edge_len2
    t = np.clip(t, 0.0, 1.0)
    proj = v1 + t * edge
    return float(np.linalg.norm(p - proj))


def fill_ghost_cells(
    Q: np.ndarray,
    phi: np.ndarray,
    body: RigidBody,
    xc: np.ndarray,
    yc: np.ndarray,
    gas,
) -> np.ndarray:
    """Fill ghost cells (phi < 0) with reflected state for no-penetration BC.

    For each ghost cell G at (i, j):
    1. Find mirror point M across the interface (project along grad(phi))
    2. Interpolate fluid state at M
    3. Set ghost pressure/density = interpolated (mirror)
    4. Set ghost velocity = 2*v_body - v_fluid_mirror (reflect normal, match tangential)

    Args:
        Q: conservative variables, shape (4, ni, nj)
        phi: signed distance, shape (ni, nj)
        body: RigidBody object
        xc, yc: cell centers, shape (ni, nj)
        gas: gas module (for EOS)

    Returns:
        Q_new: updated conservative variables with ghost cells filled
    """
    ni, nj = phi.shape

    # Convert to NumPy for easier indexing
    Q_np = np.array(Q)
    phi_np = np.array(phi)
    xc_np = np.array(xc)
    yc_np = np.array(yc)

    # Compute gradient of phi using central differences
    grad_phi_x = np.zeros_like(phi_np)
    grad_phi_y = np.zeros_like(phi_np)

    # Interior points: central difference
    grad_phi_x[1:-1, 1:-1] = (phi_np[2:, 1:-1] - phi_np[:-2, 1:-1]) / 2.0
    grad_phi_y[1:-1, 1:-1] = (phi_np[1:-1, 2:] - phi_np[1:-1, :-2]) / 2.0

    # Edges: one-sided differences
    grad_phi_x[0, :] = phi_np[1, :] - phi_np[0, :]
    grad_phi_x[-1, :] = phi_np[-1, :] - phi_np[-2, :]
    grad_phi_y[:, 0] = phi_np[:, 1] - phi_np[:, 0]
    grad_phi_y[:, -1] = phi_np[:, -1] - phi_np[:, -2]

    # Normalize gradient
    grad_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + EPS_TINY
    n_x = grad_phi_x / grad_mag
    n_y = grad_phi_y / grad_mag

    # Fill ghost cells (only near interface: -2 < phi < 0)
    for i in range(ni):
        for j in range(nj):
            if -2.0 < phi_np[i, j] < 0.0:
                # Ghost cell position
                xg = xc_np[i, j]
                yg = yc_np[i, j]

                # Mirror point: project across interface along -grad(phi)
                # distance to interface ≈ |phi|
                d = abs(phi_np[i, j])
                xm = xg + 2.0 * d * n_x[i, j]
                ym = yg + 2.0 * d * n_y[i, j]

                # Bilinear interpolation of Q at mirror point
                Q_mirror = _bilinear_interp(Q_np, xm, ym, xc_np, yc_np)

                # Extract primitive variables at mirror
                rho_m = Q_mirror[0]
                u_m = Q_mirror[1] / rho_m
                v_m = Q_mirror[2] / rho_m
                p_m = (gas.GAMMA - 1.0) * (Q_mirror[3] - 0.5 * rho_m * (u_m**2 + v_m**2))

                # Body surface velocity at ghost cell location
                v_body = body.surface_velocity(np.array([xg, yg]))

                # Reflect velocity: v_ghost = 2*v_body - v_mirror
                u_ghost = 2.0 * v_body[0] - u_m
                v_ghost = 2.0 * v_body[1] - v_m

                # Set ghost cell state (mirror pressure/density, reflected velocity)
                Q_np[0, i, j] = rho_m
                Q_np[1, i, j] = rho_m * u_ghost
                Q_np[2, i, j] = rho_m * v_ghost
                Q_np[3, i, j] = p_m / (gas.GAMMA - 1.0) + 0.5 * rho_m * (u_ghost**2 + v_ghost**2)

    # Convert back to backend array
    return xp.array(Q_np)  # type: ignore[no-any-return]


def _bilinear_interp(
    Q: np.ndarray, x: float, y: float, xc: np.ndarray, yc: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation of Q at (x, y).

    Args:
        Q: field to interpolate, shape (4, ni, nj)
        x, y: query point
        xc, yc: cell centers, shape (ni, nj)

    Returns:
        Q_interp: interpolated state, shape (4,)
    """
    ni, nj = xc.shape

    # Assume uniform grid spacing (for Cartesian grid)
    dx = xc[1, 0] - xc[0, 0] if ni > 1 else 1.0
    dy = yc[0, 1] - yc[0, 0] if nj > 1 else 1.0

    # Find cell containing (x, y)
    i = int((x - xc[0, 0]) / dx)
    j = int((y - yc[0, 0]) / dy)

    # Clamp to valid range
    i = max(0, min(i, ni - 2))
    j = max(0, min(j, nj - 2))

    # Bilinear weights
    x0 = xc[i, j]
    y0 = yc[i, j]
    x1 = xc[i + 1, j]
    y1 = yc[i, j + 1]

    wx = (x - x0) / (x1 - x0 + EPS_TINY)
    wy = (y - y0) / (y1 - y0 + EPS_TINY)
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)

    # Interpolate each component
    Q_interp = (
        (1 - wx) * (1 - wy) * Q[:, i, j]
        + wx * (1 - wy) * Q[:, i + 1, j]
        + (1 - wx) * wy * Q[:, i, j + 1]
        + wx * wy * Q[:, i + 1, j + 1]
    )

    return Q_interp  # type: ignore[no-any-return]


def compute_interface_forces(
    Q: np.ndarray,
    phi: np.ndarray,
    body: RigidBody,
    xc: np.ndarray,
    yc: np.ndarray,
    gas,
) -> tuple[np.ndarray, float]:
    """Compute pressure force and torque on rigid body.

    Integrates pressure over the interface:
        F = sum_f p * n_f * dA_f
        tau = sum_f (x_f - x_cm) × (p * n_f * dA_f)

    Args:
        Q: conservative variables, shape (4, ni, nj)
        phi: signed distance, shape (ni, nj)
        body: RigidBody object
        xc, yc: cell centers, shape (ni, nj)
        gas: gas module

    Returns:
        F: total force, shape (2,) [Fx, Fy]
        tau: total torque (scalar)
    """
    # Convert to NumPy
    Q_np = np.array(Q)
    phi_np = np.array(phi)
    xc_np = np.array(xc)
    yc_np = np.array(yc)

    ni, nj = phi_np.shape
    F = np.zeros(2)
    tau = 0.0

    # Assume uniform Cartesian grid
    dx = xc_np[1, 0] - xc_np[0, 0] if ni > 1 else 1.0
    dy = yc_np[0, 1] - yc_np[0, 0] if nj > 1 else 1.0

    # Compute pressure
    rho = Q_np[0]
    u = Q_np[1] / rho
    v = Q_np[2] / rho
    p = (gas.GAMMA - 1.0) * (Q_np[3] - 0.5 * rho * (u**2 + v**2))

    # Gradient of phi (interface normal)
    grad_phi_x = np.zeros_like(phi_np)
    grad_phi_y = np.zeros_like(phi_np)
    grad_phi_x[1:-1, :] = (phi_np[2:, :] - phi_np[:-2, :]) / (2.0 * dx)
    grad_phi_y[:, 1:-1] = (phi_np[:, 2:] - phi_np[:, :-2]) / (2.0 * dy)

    # Normalize to get outward normal (points into fluid, away from body)
    grad_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + EPS_TINY
    n_x = grad_phi_x / grad_mag
    n_y = grad_phi_y / grad_mag

    # Sum over interface cells: phi changes sign across a face
    for i in range(ni - 1):
        for j in range(nj - 1):
            # Check if interface crosses this cell
            # Interface cells: -1 < phi < 1 (within one cell of interface)
            if -1.0 < phi_np[i, j] < 1.0:
                # Face area element (2D: line segment, dA = dx or dy)
                # For simplicity, use cell area * |grad(phi)| as interface area proxy
                dA = dx * dy * grad_mag[i, j]

                # Pressure force: F = -p * n * dA
                # (minus sign: normal points outward from body)
                # n = grad(phi) points outward (into fluid),
                # so force on body is -p*n*dA
                F[0] -= p[i, j] * n_x[i, j] * dA
                F[1] -= p[i, j] * n_y[i, j] * dA

                # Torque: tau = (r × F)_z = r_x * F_y - r_y * F_x
                r_x = xc_np[i, j] - body.position[0]
                r_y = yc_np[i, j] - body.position[1]
                dF_x = -p[i, j] * n_x[i, j] * dA
                dF_y = -p[i, j] * n_y[i, j] * dA
                tau += r_x * dF_y - r_y * dF_x

    return F, tau
