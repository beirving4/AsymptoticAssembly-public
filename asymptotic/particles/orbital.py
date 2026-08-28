import numpy as np, pdb 
import astropy.cosmology.units as cu

from astropy import units as u
# from numba import jit, njit, prange

# @jit
def make_periodic(
        coordinates: np.ndarray, box_size: float, center_of_mass: np.ndarray
    ) -> np.ndarray:

    distance_from_center = coordinates - center_of_mass
    too_far = distance_from_center > box_size / 2
    too_close = distance_from_center < -box_size / 2
    distance_from_center[too_far] -= box_size
    distance_from_center[too_close] += box_size
    return distance_from_center

# @jit
def get_radial_distances(   
        positions: np.ndarray,
        box_size: float,
        center_of_mass: np.ndarray,
    ) -> np.ndarray:

    distance_from_center = make_periodic(positions, box_size, center_of_mass)
    return np.linalg.norm(distance_from_center, axis=1)

# @jit 
def apply_hubble_velocity(
        distances: np.ndarray,
        velocities: np.ndarray,
        hubble: float | None,
        little_h: float | None = None,
    ) -> np.ndarray:

    # need to update this to use little h from astropy cosmology
    if hubble is None: return velocities
    if little_h is None:
        raise ValueError("Hubble constant or little h must be provided")
    return velocities + hubble * (distances * u.Mpc / little_h).to("km").value

# @jit
def get_peculiar_velocities(
        radial_distances: np.ndarray,
        velocities: np.ndarray,
        distance_from_center: np.ndarray,
    ) -> np.ndarray:

    non_center_points = np.where(radial_distances != 0)[0]
    non_center_distances = radial_distances[non_center_points]

    vr_dot_product = np.einsum(
        "ij,ij->i",
        distance_from_center[non_center_points],
        velocities[non_center_points],
    )

    radial_velocities = np.zeros_like(radial_distances)
    radial_velocities[non_center_points] = vr_dot_product / non_center_distances

    return radial_velocities

# @jit
def get_orbital_peculiar_velocities(
        radial_distances: np.ndarray,
        velocities: np.ndarray,
        distance_from_center: np.ndarray,
    ) -> dict[str, np.ndarray]:

    non_center_points = np.where(radial_distances != 0)[0]
    non_center_distances = radial_distances[non_center_points]

    r = distance_from_center[non_center_points]
    v = velocities[non_center_points]

    vr_dot_product = np.einsum("ij,ij->i", r, v)

    radial_velocities = np.zeros_like(radial_distances)
    radial_velocities[non_center_points] = vr_dot_product / non_center_distances

    angular_velocities = np.zeros_like(radial_distances)
    omega = np.cross(r, v) / np.linalg.norm(r, axis=1) ** 2
    angular_velocities[non_center_points] = omega

    return {"radial": radial_velocities, "angular": angular_velocities}
# @jit
def get_radial_velocities(
        radial_distances: np.ndarray,
        velocities: np.ndarray,
        distance_from_center: np.ndarray,
        hubble: float | None = None,
        little_h: float | None = None,
    ) -> np.ndarray:

    non_center_points = np.where(radial_distances != 0)[0]
    non_center_distances = radial_distances[non_center_points]

    vr_dot_product = np.einsum(
        "ij,ij->i",
        distance_from_center[non_center_points],
        velocities[non_center_points],
    )

    radial_velocities = np.zeros_like(radial_distances)
    radial_velocities[non_center_points] = vr_dot_product / non_center_distances

    return apply_hubble_velocity(
        distances=radial_distances,
        velocities=radial_velocities,
        hubble=hubble,
        little_h=little_h,
    )

# @jit
def get_radial_accelerations(
        radial_velocities: np.ndarray,
        radial_distances: np.ndarray,
    ) -> np.ndarray:
    r = (radial_distances * u.Mpc).to("km").value
    return np.dot(radial_velocities, radial_velocities) / r

# @jit
def get_angular_momentum(
        distances_from_halo_center: np.ndarray,
        particle_velocities: np.ndarray,
        particle_mass: float,
    ) -> np.ndarray:

    r_cross_v = np.cross(distances_from_halo_center, particle_velocities)
    return np.sum(r_cross_v, axis=0) * particle_mass

# @jit
def get_radial_phase_space_properties(
        coordinates: np.ndarray,
        velocities: np.ndarray,
        box_size: float, 
        center_of_mass: np.ndarray, 
        group_velocity: np.ndarray,
        hubble: float | None = None,
        little_h: float | None = None,
        precomputed_radial_distances: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:

    distance_from_center = make_periodic(coordinates, box_size, center_of_mass)
    peculiar_relative_velocities = velocities - group_velocity
    relative_velocities = apply_hubble_velocity(
        distances=distance_from_center,
        velocities=peculiar_relative_velocities,
        hubble=hubble,
        little_h=little_h
    )

    radial_distances = (
        precomputed_radial_distances
        if precomputed_radial_distances is not None
        else np.linalg.norm(distance_from_center, axis=1)
    )

    # may need to remove particles below softening length or have r=0. 

    peculiar_velocities = get_peculiar_velocities(
        radial_distances=radial_distances,
        velocities=peculiar_relative_velocities,
        distance_from_center=distance_from_center
    )

    radial_velocities = apply_hubble_velocity(
        distances=radial_distances,
        velocities=peculiar_velocities,
        hubble=hubble,
        little_h=little_h
    )

    return {
        "positions" : distance_from_center,
        "radial_distances": radial_distances,
        "peculiar_relative_velocities": peculiar_relative_velocities,
        "actual_relative_velocities": relative_velocities,
        "peculiar_radial_velocities": peculiar_velocities,
        "actual_radial_velocities": radial_velocities,
        "peculiar_radial_accelerations": get_radial_accelerations(
            radial_velocities=peculiar_velocities,
            radial_distances=radial_distances,
        ),
        "actual_radial_accelerations": get_radial_accelerations(
            radial_velocities=radial_velocities,
            radial_distances=radial_distances,
        ),
    }

# @jit
def get_phase_space_properties(
        coordinates: np.ndarray,
        velocities: np.ndarray,
        box_size: float, 
        center_of_mass: np.ndarray, 
        group_velocity: np.ndarray,
        hubble: float | None = None,
        little_h: float | None = None,
        precomputed_radial_distances: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:

    # Add a function that converts cartesian to spherical coordinates

    radial = get_radial_phase_space_properties(
        coordinates=coordinates,
        velocities=velocities,
        box_size=box_size, 
        center_of_mass=center_of_mass, 
        group_velocity=group_velocity,
        hubble=hubble,
        little_h=little_h,
        precomputed_radial_distances=precomputed_radial_distances,
    )

    # Make functions for these later ... 
    angular = {}
    tangential = {}

    return {**radial, **angular, **tangential}



def cartesioan_to_spherical_velocities(
        cartiesian_velocities: np.ndarray,
        spherical_coordinates: np.ndarray,
    ) -> np.ndarray:
    
    vx, vy, vz = cartiesian_velocities.T
    r, theta, phi = spherical_coordinates.T

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    v_r = (sin_t * cos_p) * vx + (sin_t * sin_p) * vy + cos_t * vz
    v_theta = (cos_t * cos_p) * vx + (cos_t * sin_p) * vy - sin_t * vz
    v_phi = (-sin_p) * vx + (cos_p) * vy

    return np.column_stack((v_r, v_theta, v_phi))


def cartesioan_to_spherical_accelerations(
        cartiesian_accelerations: np.ndarray,
        spherical_coordinates: np.ndarray,
    ) -> np.ndarray:

    ax, ay, az = cartiesian_accelerations.T
    _, theta, phi = spherical_coordinates.T

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    a_r = (sin_t * cos_p) * ax + (sin_t * sin_p) * ay + cos_t * az
    a_theta = (cos_t * cos_p) * ax + (cos_t * sin_p) * ay - sin_t * az
    a_phi = (-sin_p) * ax + (cos_p) * ay

    return np.column_stack((a_r, a_theta, a_phi))