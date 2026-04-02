"""Rigid body dynamics for partitioned FSI coupling.

Implements 2D rigid body with:
  - Circle or polygon geometry
  - Translation and rotation
  - Force/torque integration (Verlet/RK2)
  - Surface velocity computation for no-penetration BC

Based on Grétarsson (2012), Chapter 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigidBody:
    """2D rigid body with mass, geometry, and kinematics.

    Attributes:
        position: center of mass, shape (2,) [x, y]
        velocity: translational velocity, shape (2,) [vx, vy]
        angle: rotation angle in radians
        angular_velocity: omega (rad/s)
        mass: total mass
        moment_of_inertia: I about center of mass
        shape: "circle" or "polygon"
        radius: for circle geometry
        vertices_body: for polygon, shape (N, 2) in body frame
    """

    position: np.ndarray
    velocity: np.ndarray
    angle: float
    angular_velocity: float
    mass: float
    moment_of_inertia: float
    shape: str
    radius: float = 0.0
    vertices_body: np.ndarray | None = None

    def vertices_world(self) -> np.ndarray:
        """Transform polygon vertices from body frame to world frame.

        Returns:
            vertices in world frame, shape (N, 2)
        """
        if self.shape != "polygon" or self.vertices_body is None:
            raise ValueError("vertices_world() only valid for polygon bodies")

        # Rotation matrix
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])

        # Rotate and translate
        return self.vertices_body @ R.T + self.position[None, :]  # type: ignore[no-any-return]

    def surface_velocity(self, x: np.ndarray) -> np.ndarray:
        """Compute surface velocity at world point x.

        v_surface = v_cm + omega × (x - x_cm)

        In 2D: omega × r = omega * [-r_y, r_x]

        Args:
            x: world coordinates, shape (..., 2) or (2,)

        Returns:
            surface velocity, same shape as x
        """
        r = x - self.position
        # omega × r in 2D: omega * [-r_y, r_x]
        if r.ndim == 1:
            v_rot = self.angular_velocity * np.array([-r[1], r[0]])
        else:
            v_rot = self.angular_velocity * np.stack([-r[..., 1], r[..., 0]], axis=-1)
        return self.velocity + v_rot  # type: ignore[no-any-return]

    def apply_forces(self, F: np.ndarray, tau: float, dt: float) -> RigidBody:
        """Advance rigid body state by dt using explicit Verlet/RK2.

        m * a = F
        I * alpha = tau

        Uses semi-implicit (symplectic) Euler:
          v^{n+1} = v^n + (F/m) * dt
          x^{n+1} = x^n + v^{n+1} * dt
          omega^{n+1} = omega^n + (tau/I) * dt
          theta^{n+1} = theta^n + omega^{n+1} * dt

        Args:
            F: total force, shape (2,) [Fx, Fy]
            tau: total torque (scalar)
            dt: time step

        Returns:
            new RigidBody state
        """
        # Semi-implicit Euler (velocity Verlet)
        v_new = self.velocity + (F / self.mass) * dt
        x_new = self.position + v_new * dt

        omega_new = self.angular_velocity + (tau / self.moment_of_inertia) * dt
        theta_new = self.angle + omega_new * dt

        # Return updated body (immutable pattern)
        return RigidBody(
            position=x_new,
            velocity=v_new,
            angle=theta_new,
            angular_velocity=omega_new,
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            shape=self.shape,
            radius=self.radius,
            vertices_body=self.vertices_body,
        )


def make_circle(center: np.ndarray, radius: float, density: float) -> RigidBody:
    """Factory: create a circular rigid body.

    Args:
        center: initial position, shape (2,)
        radius: circle radius
        density: mass per unit area (mass = density * pi * r^2)

    Returns:
        RigidBody with circle geometry
    """
    area = np.pi * radius**2
    mass = density * area
    # Moment of inertia for uniform disk: I = 0.5 * m * r^2
    moment_of_inertia = 0.5 * mass * radius**2

    return RigidBody(
        position=np.array(center, dtype=float),
        velocity=np.zeros(2),
        angle=0.0,
        angular_velocity=0.0,
        mass=mass,
        moment_of_inertia=moment_of_inertia,
        shape="circle",
        radius=radius,
        vertices_body=None,
    )


def make_polygon(vertices: np.ndarray, density: float) -> RigidBody:
    """Factory: create a polygonal rigid body.

    Vertices are interpreted in body frame, centered at (0, 0).

    Args:
        vertices: polygon vertices in body frame, shape (N, 2)
        density: mass per unit area

    Returns:
        RigidBody with polygon geometry
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2)")

    # Compute area using shoelace formula
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    mass = density * area

    # Compute center of mass (should be near origin if vertices are centered)
    cx = np.mean(x)
    cy = np.mean(y)

    # Moment of inertia for uniform polygon (approximate as point masses at vertices)
    # I = sum_i m_i * r_i^2, where m_i = mass / N
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    moment_of_inertia = mass * np.mean(r2)

    return RigidBody(
        position=np.array([cx, cy], dtype=float),
        velocity=np.zeros(2),
        angle=0.0,
        angular_velocity=0.0,
        mass=mass,
        moment_of_inertia=moment_of_inertia,
        shape="polygon",
        radius=0.0,
        vertices_body=vertices - np.array([cx, cy]),  # Re-center to origin
    )
