import numpy as np

from numba import jit, njit, prange

# from ..particles.orbital import (
#     get_angular_momentum,
#     get_radial_phase_space_properties
# )

# Use the radial.py file to help with

def get_halo_mass_distribution(
        radii: np.ndarray,   
        region_radius: float,
        particle_mass: float,
        num_radial_points: int,
    ) -> np.ndarray:     

    radii_mask = (radii < region_radius) 
    particle_radii = radii[radii_mask]
    sorted_r_idx = np.argsort(particle_radii)
    sorted_radii = particle_radii[sorted_r_idx]

    interped_radii = np.logspace(
        np.log10(sorted_radii[0]), 
        np.log10(sorted_radii[-1]), 
        num_radial_points
    )

    M_enclosed = sorted_r_idx * particle_mass
    rho_enclosed = (3.0 * M_enclosed) / (4.0  * np.pi * sorted_radii**3)

    interp_M_enclosed = 10.0**np.interp(
        np.log10(interped_radii), 
        np.log10(sorted_radii), 
        np.log10(M_enclosed)
    )
    
    interp_rho_enclosed = 10.0**np.interp(
        np.log10(interped_radii), 
        np.log10(sorted_radii), 
        np.log10(rho_enclosed)
    )

    dMdr = np.gradient(interp_M_enclosed, interped_radii)
    rho = dMdr / (4 * np.pi * interped_radii**2)

    return np.vstack([
        interped_radii, 
        interp_M_enclosed, 
        interp_rho_enclosed, 
        rho
    ])

def get_halo_phase_space_distribution() -> np.ndarray:
    return np.empty((4, 0))

def get_halo_spin(
        angular_momentum: np.ndarray,
        mass: float,
        radius: float,
        circular_velocity: float
    ) -> np.ndarray:

    J = np.linalg.norm(angular_momentum)
    return J / (np.sqrt(2) * mass * circular_velocity * radius)
