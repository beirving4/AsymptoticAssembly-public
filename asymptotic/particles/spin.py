from __future__ import annotations

import numpy as np, pdb

from typing import Iterable
from astropy import units as u
from attrs import define, field


def get_linear_momenta(masses: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    return masses[:, np.newaxis] * velocities

def get_linear_momentum(masses: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    return np.sum(get_linear_momenta(masses, velocities), axis=0)

def get_specific_angular_momenta(
        positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
    return np.cross(
        (positions * u.Mpc).to("km").value, 
        (velocities * u.km / u.s).to("km/s").value
    )

def get_specific_angular_momentum(
        positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
    return np.sum(get_specific_angular_momenta(positions, velocities), axis=0)

def get_angular_momenta(
        positions: np.ndarray, 
        velocities: np.ndarray, 
        masses: np.ndarray
    ) -> np.ndarray:
    specific = get_specific_angular_momenta(positions, velocities)
    return masses[:, np.newaxis] * specific

def get_angular_momentum(
        positions: np.ndarray, 
        velocities: np.ndarray, 
        masses: np.ndarray
    ) -> np.ndarray:
    return np.sum(
        get_angular_momenta(positions, velocities, masses), axis=0
    )

def get_total_angular_momentum(
        positions: np.ndarray, 
        velocities: np.ndarray, 
        masses: np.ndarray
    ) -> float:
    return np.linalg.norm(get_angular_momentum(positions, velocities, masses))


def get_spin(
        enclosed_radii: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        enclosed_masses: np.ndarray,
        mass: float,
        radius: float,
        velocity: float
    ) -> float | np.ndarray:

    if not isinstance(radius, np.ndarray): radius = np.array([radius])

    if not isinstance(velocity, np.ndarray): velocity = np.array([velocity])

    # pdb.set_trace()

    mask = enclosed_radii <= radius
    L = get_angular_momentum(
        positions=positions[mask,:], 
        velocities=velocities[mask,:], 
        masses=enclosed_masses[mask]
    )    
    J = np.linalg.norm((np.sum(L, axis=0) * u.km * u.km * u.Msun / u.s))
    M = (mass * u.Msun)
    R = (radius * u.Mpc).to(u.km)
    V = (velocity * u.km/u.s)

    return (J / (np.sqrt(2) * M * V * R)).to(u.dimensionless_unscaled).value


