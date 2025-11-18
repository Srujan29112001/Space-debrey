"""
Orbital Mechanics
Implements orbital propagation with perturbations
"""

from typing import Tuple

import numpy as np


class OrbitalMechanics:
    """
    Orbital mechanics calculations and propagation
    Includes perturbations: J2, drag, solar pressure
    """

    def __init__(self):
        """Initialize orbital mechanics"""
        # Earth parameters
        self.mu = 398600.4418  # km^3/s^2
        self.J2 = 1.08263e-3
        self.Re = 6378.137  # km
        self.omega_earth = 7.2921159e-5  # rad/s

        # Atmosphere model parameters
        self.rho0 = 1.225  # kg/m^3 at sea level
        self.H = 8.5  # km scale height

        # Solar radiation pressure
        self.P_solar = 4.56e-6  # N/m^2

    def propagate(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        dt: float,
        include_j2: bool = True,
        include_drag: bool = True,
        include_solar_pressure: bool = True,
        area_to_mass: float = 0.01,  # m^2/kg
        Cd: float = 2.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit one time step

        Args:
            position: Position vector (km)
            velocity: Velocity vector (km/s)
            dt: Time step (seconds)
            include_j2: Include J2 perturbation
            include_drag: Include atmospheric drag
            include_solar_pressure: Include solar radiation pressure
            area_to_mass: Area-to-mass ratio (m^2/kg)
            Cd: Drag coefficient

        Returns:
            (new_position, new_velocity)
        """
        # Convert dt to same time units as velocity (seconds)
        dt_sec = dt

        # Compute total acceleration
        acceleration = np.zeros(3)

        # Two-body gravity
        r = np.linalg.norm(position)
        a_gravity = -self.mu * position / (r**3)
        acceleration += a_gravity

        # J2 perturbation
        if include_j2:
            a_j2 = self.calculate_j2_acceleration(position)
            acceleration += a_j2

        # Atmospheric drag
        if include_drag:
            a_drag = self.calculate_drag_acceleration(position, velocity, area_to_mass, Cd)
            acceleration += a_drag

        # Solar radiation pressure
        if include_solar_pressure:
            a_solar = self.calculate_solar_pressure_acceleration(position, area_to_mass)
            acceleration += a_solar

        # Integrate using RK4
        new_position, new_velocity = self._rk4_step(position, velocity, acceleration, dt_sec)

        return new_position, new_velocity

    def calculate_j2_acceleration(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate J2 perturbation acceleration

        Args:
            position: Position vector (km)

        Returns:
            J2 acceleration (km/s^2)
        """
        x, y, z = position
        r = np.linalg.norm(position)

        factor = 1.5 * self.J2 * self.mu * (self.Re**2) / (r**5)

        ax = factor * x * (5 * (z**2) / (r**2) - 1)
        ay = factor * y * (5 * (z**2) / (r**2) - 1)
        az = factor * z * (5 * (z**2) / (r**2) - 3)

        return np.array([ax, ay, az])

    def calculate_drag_acceleration(
        self, position: np.ndarray, velocity: np.ndarray, area_to_mass: float, Cd: float
    ) -> np.ndarray:
        """
        Calculate atmospheric drag acceleration

        Args:
            position: Position vector (km)
            velocity: Velocity vector (km/s)
            area_to_mass: Area-to-mass ratio (m^2/kg)
            Cd: Drag coefficient

        Returns:
            Drag acceleration (km/s^2)
        """
        r = np.linalg.norm(position)
        altitude = r - self.Re

        if altitude > 1000:  # No drag above 1000 km
            return np.zeros(3)

        # Atmospheric density (exponential model)
        rho = self.rho0 * np.exp(-altitude / self.H)  # kg/m^3

        # Velocity relative to rotating atmosphere
        # Simplified: ignore rotation for now
        v_rel = velocity  # km/s

        # Drag acceleration
        v_mag = np.linalg.norm(v_rel)

        if v_mag < 1e-10:
            return np.zeros(3)

        # Convert to m/s for drag calculation
        v_rel_m = v_rel * 1000  # m/s

        # Drag force per unit mass (m/s^2)
        a_drag_m = -0.5 * Cd * area_to_mass * rho * np.linalg.norm(v_rel_m) * v_rel_m

        # Convert back to km/s^2
        a_drag = a_drag_m / 1000

        return a_drag

    def calculate_solar_pressure_acceleration(
        self, position: np.ndarray, area_to_mass: float
    ) -> np.ndarray:
        """
        Calculate solar radiation pressure acceleration

        Args:
            position: Position vector (km)
            area_to_mass: Area-to-mass ratio (m^2/kg)

        Returns:
            Solar pressure acceleration (km/s^2)
        """
        # Simplified: assume Sun direction
        # In production, use actual Sun position

        # Sun direction (simplified: along +X axis)
        sun_direction = np.array([1.0, 0.0, 0.0])

        # Check if in Earth's shadow (simplified)
        if self._in_shadow(position):
            return np.zeros(3)

        # Solar pressure acceleration (m/s^2)
        # Assuming reflectivity coefficient of 1.5
        CR = 1.5  # Reflectivity coefficient
        a_solar_m = CR * self.P_solar * area_to_mass * sun_direction

        # Convert to km/s^2
        a_solar = a_solar_m / 1000

        return a_solar

    def _in_shadow(self, position: np.ndarray) -> bool:
        """Check if position is in Earth's shadow"""
        # Simplified cylindrical shadow model
        # Sun direction (simplified)
        sun_dir = np.array([1.0, 0.0, 0.0])

        # Projection onto sun direction
        projection = np.dot(position, sun_dir)

        if projection < 0:  # Behind Earth
            # Perpendicular distance from shadow axis
            perp_dist = np.linalg.norm(position - projection * sun_dir)

            if perp_dist < self.Re:
                return True

        return False

    def _rk4_step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 integration step

        Args:
            position: Current position (km)
            velocity: Current velocity (km/s)
            acceleration: Current acceleration (km/s^2)
            dt: Time step (seconds)

        Returns:
            (new_position, new_velocity)
        """
        # Simple Euler integration for now
        # In production, implement full RK4

        new_velocity = velocity + acceleration * dt
        new_position = position + velocity * dt + 0.5 * acceleration * (dt**2)

        return new_position, new_velocity

    def kepler_elements(self, position: np.ndarray, velocity: np.ndarray) -> dict:
        """
        Convert state vector to Keplerian elements

        Args:
            position: Position vector (km)
            velocity: Velocity vector (km/s)

        Returns:
            Keplerian elements
        """
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)

        # Specific angular momentum
        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)

        # Eccentricity vector
        e_vec = np.cross(velocity, h_vec) / self.mu - position / r
        e = np.linalg.norm(e_vec)

        # Specific orbital energy
        E = v**2 / 2 - self.mu / r

        # Semi-major axis
        if abs(E) > 1e-10:
            a = -self.mu / (2 * E)
        else:
            a = float("inf")

        # Inclination
        i = np.arccos(h_vec[2] / h)

        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)

        # RAAN
        if n > 1e-10:
            omega = np.arccos(n_vec[0] / n)
            if n_vec[1] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0

        # Argument of periapsis
        if n > 1e-10 and e > 1e-10:
            w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0

        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0

        return {
            "a": a,  # Semi-major axis (km)
            "e": e,  # Eccentricity
            "i": np.degrees(i),  # Inclination (deg)
            "omega": np.degrees(omega),  # RAAN (deg)
            "w": np.degrees(w),  # Argument of periapsis (deg)
            "nu": np.degrees(nu),  # True anomaly (deg)
        }


if __name__ == "__main__":
    # Test orbital mechanics
    om = OrbitalMechanics()

    # ISS-like orbit
    position = np.array([6778.0, 0.0, 0.0])  # km
    velocity = np.array([0.0, 7.66, 0.0])  # km/s

    print("Initial state:")
    print(f"  Position: {position}")
    print(f"  Velocity: {velocity}")

    # Compute Keplerian elements
    elements = om.kepler_elements(position, velocity)
    print("\nKeplerian elements:")
    for key, value in elements.items():
        print(f"  {key}: {value:.2f}")

    # Propagate one orbit
    period = 2 * np.pi * np.sqrt(elements["a"] ** 3 / om.mu)
    print(f"\nOrbital period: {period / 60:.2f} minutes")

    # Propagate 10 minutes
    dt = 600  # seconds
    new_pos, new_vel = om.propagate(position, velocity, dt)

    print(f"\nAfter {dt} seconds:")
    print(f"  Position: {new_pos}")
    print(f"  Velocity: {new_vel}")
