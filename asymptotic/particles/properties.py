from __future__ import annotations

import numpy as np, pdb 

from time import time
from datetime import timedelta
from astropy import units as u
from attrs import define, field
from scipy.spatial import cKDTree
from abc import ABC, abstractmethod
from collections.abc import Collection
from astropy.coordinates import cartesian_to_spherical

from .spin import get_spin
from .identify import ParticleIdentifier
from .nbody import NBodyPhaseSpaceProperties, NBodyProperties
from .orbital import (
    get_phase_space_properties, 
    cartesioan_to_spherical_velocities,
    cartesioan_to_spherical_accelerations
)

does_exist = lambda arr: (arr is not None) and (arr.size > 0)
update_if_exist = lambda arr, idxs: arr[idxs] if does_exist(arr) else None
def in_place_update(arr: np.ndarray | None, idxs: Collection[int]) -> None:
    if arr is not None: arr[:] = update_if_exist(arr, idxs)

@define(slots=True)
class PhaseSpaceProperties:
    velocities: np.ndarray
    radial_velocities: np.ndarray | None = field(default=None)

    accelerations: np.ndarray | None = field(default=None)
    radial_accelerations: np.ndarray | None = field(default=None)

    def __getitem__(self, index: int) ->  NBodyPhaseSpaceProperties:
        return NBodyPhaseSpaceProperties(
            velocity=self.velocities[index],
            radial_velocity=update_if_exist(self.radial_velocities, index),
            acceleration=update_if_exist(self.accelerations, index),
            radial_acceleration=update_if_exist(self.radial_accelerations, index)
        )

    def get_subset(self, indices: Collection[int]) -> PhaseSpaceProperties:
        if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices)        
        
        return PhaseSpaceProperties(
            velocities=self.velocities[indices],
            radial_velocities=update_if_exist(self.radial_velocities, indices),
            accelerations=update_if_exist(self.accelerations, indices),
            radial_accelerations=update_if_exist(self.radial_accelerations, indices)
        )
    
    def reorder(self, indices: Collection[int]) -> None:
        if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices)
        self.velocities[:] = self.velocities[indices]
        in_place_update(self.radial_velocities, indices)
        in_place_update(self.accelerations, indices)
        in_place_update(self.radial_accelerations, indices)


    @classmethod
    def from_dictionary(
            cls, properties: dict[str, dict[str, np.ndarray]]
        ) -> PhaseSpaceProperties:
        return cls(
            velocities=properties["velocities"]["relative"],
            radial_velocities=properties["velocities"]["radial"],
            accelerations=properties["accelerations"].get("relative"),
            radial_accelerations=properties["accelerations"].get("radial")
        )

@define(slots=True)
class ParticleCollectionProperties(ABC): 
    mass_values: np.ndarray
    positions: np.ndarray | None

    peculiar: PhaseSpaceProperties
    actual: PhaseSpaceProperties | None

    in_comoving_units: bool
    softening_length: float

    @abstractmethod
    def __getitem__(self, index: int) -> NBodyProperties:
        ...

    @abstractmethod
    def get_subset(self, ids: Collection[int], *args, **kwargs) -> ParticleCollectionProperties:
        ...

    @property
    def masses(self) -> np.ndarray:
        return 1E10 * self.mass_values
    
    @property
    def num_particles(self) -> int:
        return len(self.mass_values)
    
    @property
    def enclosed_mass(self) -> float:
        return np.sum(self.masses)

    @property
    @abstractmethod
    def coordinates(self) -> np.ndarray:
        ...

    @property
    def velocities(self) -> np.ndarray:
        return self.peculiar.velocities
    
    @property
    def accelerations(self) -> np.ndarray | None:
        return (
            None 
            if self.peculiar.accelerations is None else 
            self.peculiar.accelerations
        )

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.coordinates[:, 0]
    
    @property
    def y_coordinates(self) -> np.ndarray:
        return self.coordinates[:, 1]
    
    @property
    def z_coordinates(self) -> np.ndarray:
        return self.coordinates[:, 2]
    
    @property
    def x_velocities(self) -> np.ndarray:
        return self.velocities[:, 0]
    
    @property
    def y_velocities(self) -> np.ndarray:
        return self.velocities[:, 1]
    
    @property
    def z_velocities(self) -> np.ndarray:
        return self.velocities[:, 2]
    
    @property
    def x_accelerations(self) -> np.ndarray | None:
        return None if self.accelerations is None else self.accelerations[:, 0]
    
    @property
    def y_accelerations(self) -> np.ndarray | None:
        return None if self.accelerations is None else self.accelerations[:, 1]
    
    @property
    def z_accelerations(self) -> np.ndarray | None:
        return None if self.accelerations is None else self.accelerations[:, 2]
    
    @property
    def linear_momenta(self) -> np.ndarray: # Msun * km/s
        return self.masses[:, np.newaxis] * self.velocities
    
    @property
    def particle_forces(self) -> np.ndarray: # Msun * km/s^2
        if self.accelerations is None:
            return np.zeros_like(self.masses)
        return self.masses[:, np.newaxis] * self.accelerations
    
    @property
    def linear_momentum(self) -> np.ndarray:
        return np.sum(self.linear_momenta, axis=0)
    
    @property
    def forces(self) -> np.ndarray:
        return np.sum(self.particle_forces, axis=0)
    
    @property
    def total_momentum(self) -> float:
        return np.linalg.norm(self.linear_momentum)
    
    @property
    def total_force(self) -> float:
        return np.linalg.norm(self.forces)
    
    def get_particles_attribute(
            self, 
            ids: Collection[int], 
            attribute: str, 
            identifier: ParticleIdentifier
        ) -> np.ndarray:
        return getattr(self, attribute)[identifier.get_index_locations(ids)]


    

@define(slots=True)
class BoundedParticleProperties(ParticleCollectionProperties):
    ''' The properties of a particles in a bounded region given moment in time. '''

    positions: np.ndarray
    center_of_mass: np.ndarray

    radial_distances: np.ndarray 

    peculiar: PhaseSpaceProperties
    actual: PhaseSpaceProperties

    in_comoving_units: bool = field(default=False)
    run_sort_by_radial_distance: bool = field(default=True)

    order: np.ndarray = field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.run_sort_by_radial_distance:
            self._sort_by_radial_distance()
        else:
            self.order = np.arange(len(self.positions))


    def __getitem__(self, index: int) -> NBodyProperties:
        return NBodyProperties(
            mass=self.masses[index],
            position=self.positions[index],
            peculiar=self.peculiar[index],
            actual=(
                self.actual[index] # type: ignore
                if self.actual is not None
                else None
            )
        )
    
    def get_subset(
            self, indices: Collection[int], 
            run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleProperties:
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return BoundedParticleProperties(
            mass_values=self.mass_values[indices],
            positions=self.positions[indices],
            peculiar=self.peculiar.get_subset(indices),
            actual=self.actual.get_subset(indices),
            radial_distances=self.radial_distances[indices],
            center_of_mass=self.center_of_mass,
            in_comoving_units=self.in_comoving_units,
            softening_length=self.softening_length,
            run_sort_by_radial_distance=run_sort_by_radial_distance
        )
    
    def get_radial_subset(
            self, radius: float, 
            run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleProperties:
        return self.get_subset(
            indices=(self.radial_distances <= radius),
            run_sort_by_radial_distance=run_sort_by_radial_distance
        )
    
    def _sort_by_radial_distance(self) -> None:
        self.update_order(np.argsort(self.radial_distances))

    def update_order(self, order: np.ndarray) -> None:
        self.order = order
        self.mass_values = self.mass_values[self.order]
        self.positions = self.positions[self.order]
        self.peculiar.reorder(self.order)
        self.actual.reorder(self.order)
        self.radial_distances = self.radial_distances[self.order]

    @classmethod
    def from_dictionary(cls, properties: dict) -> BoundedParticleProperties:
        return cls(
            mass_values=properties["mass_values"],
            positions=properties["positions"]["coordinates"],
            radial_distances=properties["positions"]["radial"],
            peculiar=PhaseSpaceProperties.from_dictionary(properties["peculiar"]),
            actual=PhaseSpaceProperties.from_dictionary(properties["actual"]),
            center_of_mass=properties["center_of_mass"],
            in_comoving_units=properties["in_comoving_units"],
            softening_length=properties["softening_length"],
            run_sort_by_radial_distance=properties["run_sort_by_radial_distance"]
        )
    
    @property
    def coordinates(self) -> np.ndarray:
        return self.positions
    
    @property
    def spherical_coordinates(self) -> np.ndarray:
        x, y, z = self.coordinates.T
        r, theta, phi = cartesian_to_spherical(x, y, z)
        return np.column_stack((r.value, theta.value, phi.value))
    
    @property
    def spherical_velocities(self) -> np.ndarray:
        return cartesioan_to_spherical_velocities(
            cartiesian_velocities=self.velocities,
            spherical_coordinates=self.spherical_coordinates
        )
    
    @property
    def spherical_accelerations(self) -> np.ndarray:
        if self.accelerations is None:
            return np.zeros_like(self.velocities)
        return cartesioan_to_spherical_accelerations(
            cartiesian_accelerations=self.accelerations,
            spherical_coordinates=self.spherical_coordinates
        )
    
    @property
    def spherical_phase_space_6D(self) -> np.ndarray:
        return np.column_stack(
            (self.spherical_coordinates, self.spherical_velocities)
        )
    
    @property
    def smallest_nonzero_distance(self) -> float:
        return np.min(self.radial_distances[np.nonzero(self.radial_distances)])
    
    @property
    def inclination_angles(self) -> np.ndarray:
        return np.arctan2(self.positions[:, 1], self.positions[:, 0])
    
    @property
    def azimuthal_angles(self) -> np.ndarray:
        return np.arccos(self.positions[:, 2] / self.radial_distances)
    
    @property
    def inertia_tensor(self) -> np.ndarray:
        return np.einsum('ki,kj->ij', self.positions, self.positions)
    
    @property
    def axis_lengths(self) -> np.ndarray:
        return np.sqrt(np.sort(np.linalg.eigvals(self.inertia_tensor))[::-1])
    
    @property
    def shape_parameters(self) -> dict[str, float]:
        a, b, c = self.axis_lengths
        s, q, p = (c/a), (b/a), (c/b)
        return {
            "major axis": a,
            "intermediate axis": b,
            "minor axis": c,
            "sphericity": s, # short-to-long axis ratio
            "polar flattening": p, # short-to-intermediate axis ratio
            "equatorial flattening":q, # intermediate-to-long axis ratio
            "triaxiality": (1 - q**2)/(1 - s**2)
        }
    
    @property
    def elliptical_radii(self) -> np.ndarray:
        a, b, c = self.axis_lengths
        x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        s, q = (c/a), (b/a)
        return np.sqrt(x**2 + (y/q)**2 + (z/s)**2)
    
    @property
    def reduced_inertia_tensor(self) -> np.ndarray:
        mask = np.nonzero(self.elliptical_radii)[0]
        masked_positions = self.positions[mask]
        masked_elliptical_radii = self.elliptical_radii[mask]
        weighted_r = masked_positions / masked_elliptical_radii[:, np.newaxis]
        return np.einsum('ki,kj->ij', weighted_r, weighted_r) 
    
    @property
    def reduced_axis_lengths(self) -> np.ndarray:
        S_ij = self.reduced_inertia_tensor
        return np.sqrt(np.sort(np.linalg.eigvals(S_ij))[::-1])
    
    @property
    def reduced_shape_parameters(self) -> dict[str, float]:
        a, b, c = self.reduced_axis_lengths
        s, q, p = (c/a), (b/a), (c/b)
        return {
            "major axis": a,
            "intermediate axis": b,
            "minor axis": c,
            "sphericity": s, # short-to-long axis ratio
            "polar flattening": p, # short-to-intermediate axis ratio
            "equatorial flattening":q, # intermediate-to-long axis ratio
            "triaxiality": (1 - q**2)/(1 - s**2)
        }
    

    @property
    def velocity_dispersion_tensor(self) -> np.ndarray:
        n = self.velocities.shape[0]
        return np.einsum('ki,kj->ij', self.velocities, self.velocities) / (n - 1)
    
    @property
    def principal_velocity_dispersions(self) -> np.ndarray:
        return np.sort(np.linalg.eigvals(self.velocity_dispersion_tensor)[::-1])
    
    @property
    def radial_anisotropy(self) -> float:
        tangential_variance = np.nanvar(self.tangent_velocities, axis=0).sum()
        radial_variance = np.nanvar(self.radial_velocities, axis=0).sum()
        return 1 - 0.5 * (tangential_variance / radial_variance)
    
    @property
    def cylindrical_anisotropy(self) -> float:
        var_1, var_2, var_3 = self.principal_velocity_dispersions
        return 1 - 0.5 * (var_2 + var_3) / var_1

    
    @property
    def radial_velocities(self) -> np.ndarray:
        if self.actual.radial_velocities is None:
            raise ValueError("Must provide radial velocities")
        return self.actual.radial_velocities
    
    @property
    def radial_accelerations(self) -> np.ndarray:
        if self.actual.radial_accelerations is None:
            raise ValueError("Must provide radial accelerations")
        return self.actual.radial_accelerations
    
    # @property
    # def momentums(self) -> np.ndarray:
    #     return self.masses * self.velocities

    
    @property
    def angular_velocities(self) -> np.ndarray:
        radial_distances = (self.radial_distances * u.Mpc).to("km").value
        return self.specific_angular_momenta / radial_distances[:, np.newaxis]**2

    @property
    def angular_accelerations(self) -> np.ndarray:
        if self.accelerations is None:
            return np.zeros_like(self.positions)
        
        r = (self.radial_distances * u.Mpc).to("km").value
        r_dot = (self.radial_velocities * u.km / u.s).to("km/s").value
        omega = (self.angular_velocities * u.rad / u.s).to("rad/s").value

        torque_contribution = np.cross(r, r_dot) / r**2
        centripetal_contribution = 2 / r * r_dot * omega

        return torque_contribution - centripetal_contribution

        # return np.cross(r, r_dot) / r**2 - 2 / r * r_dot * omega
    
    @property
    def specific_angular_momenta(self) -> np.ndarray:
        positions = (self.positions * u.Mpc).to("km").value
        velocities = (self.velocities * u.km / u.s).to("km/s").value
        return np.cross(positions, velocities)
    
    @property
    def specific_angular_momentum(self) -> np.ndarray:
        return np.sum(self.specific_angular_momenta, axis=0)

    @property
    def angular_momenta(self) -> np.ndarray:
        positions = (self.positions * u.Mpc).to("km").value
        return np.cross(positions, self.linear_momenta)
        # return np.cross((self.positions * u.Mpc).to("km"), self.linear_momenta)
    
    @property
    def angular_momentum(self) -> np.ndarray:
        return np.sum(self.angular_momenta, axis=0)
    
    @property
    def total_angular_momentum(self) -> float:
        return np.linalg.norm(self.angular_momentum)

    @property
    def tangent_velocities(self) -> np.ndarray:
        omega = (self.angular_velocities * u.rad / u.s).to("rad/s").value
        positions = (self.positions * u.Mpc).to("km").value
        return np.cross(omega, positions)
        # return np.cross(self.angular_velocities, self.positions)


    @property
    def tangent_accelerations(self) -> np.ndarray:
        alpha = (self.angular_accelerations * u.rad / u.s**2).to("rad/s**2").value
        positions = (self.positions * u.Mpc).to("km").value
        return np.cross(alpha, positions)
        # return np.cross(self.angular_accelerations, self.positions)


# Get correlation function here ... 
@define(slots=True)
class BoxParticleProperties(ParticleCollectionProperties):
    ''' The properties of a particles in a simulation box given moment in time. '''
    box_size: float

    # Remove the __kd_tree when looking for particles in a subset of the box
    __kd_tree: cKDTree | None = field(default=None)
    keep_kd_tree: bool = field(default=True)
    compute_kd_tree: bool = field(default=True)

    positions: np.ndarray | None = field(default=None)
    peculiar: PhaseSpaceProperties | None = field(default=None)
    actual: PhaseSpaceProperties | None = field(default=None)
    # accelerations: np.ndarray | None = field(default=None)

    in_comoving_units: bool = field(default=False)

    def __attrs_post_init__(self) -> None:
        if self.positions is None and self.__kd_tree is None:
            raise ValueError("Must provide positions or a KDTree")

        if self.compute_kd_tree:
            self.positions = np.mod(self.positions, self.box_size)
            self.__kd_tree = cKDTree(self.positions, boxsize=self.box_size)


        if not self.keep_kd_tree and self.__kd_tree is not None:
            self.positions = self.__kd_tree.data
            self.__kd_tree = None

    def __getitem__(self, index: int) -> NBodyProperties:
        return NBodyProperties(
            mass=self.masses[index],
            position=self.coordinates[index],
            peculiar=self.peculiar.get_subset(index),
            actual=None,
        )
    
    def get_subset(self, indices: Collection[int]) -> BoxParticleProperties:
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return BoxParticleProperties(
            mass_values=self.mass_values[indices], 
            positions=self.coordinates[indices],
            peculiar=(
                None if self.peculiar is None else 
                self.peculiar.get_subset(indices)
            ),
            actual=None, 
            box_size=self.box_size,
            keep_kd_tree=False,
            compute_kd_tree=False,
            in_comoving_units=self.in_comoving_units,
            softening_length=self.softening_length
        )
    
    @property
    def coordinates(self) -> np.ndarray:
        return (
            self.positions if self.positions is not None else self.__kd_tree.data
        )
    
    @property
    def kd_tree(self) -> cKDTree | None:
        return self.__kd_tree
    
    @kd_tree.setter
    def kd_tree(self, tree: cKDTree) -> None:
        self.__kd_tree = tree
        self.positions = tree.data
    
    
    def column_density(
            self, 
            axis: int, 
            bins: int = 100, 
            range: tuple[float, float] | None = None
        ) -> np.ndarray:
        return np.histogram(
            self.coordinates[:, axis], 
            bins=bins, 
            range=range, 
            weights=self.masses
        )[0] / self.box_size**2
    
    def mean_column_density(  
            self, 
            axis: int, 
            bins: int = 100, 
            range: tuple[float, float] | None = None
        ) -> np.ndarray:
        return self.column_density(axis, bins, range).mean()
    
    def column_density_contrast(
            self, 
            axis: int, 
            bins: int = 100, 
            range: tuple[float, float] | None = None
        ) -> np.ndarray:
        return self.column_density(axis, bins, range) / self.mean_column_density(axis, bins, range)
    
    # def plot_column_density(
    # Put plotting image in Particle collections class 
            
    
    def query(
            self, 
            locations: np.ndarray, 
            radii: float | np.ndarray | None = None,
            num_neighbors: int | None = None,
            tree: cKDTree | None = None,
        ) -> dict[str, list[list[int]] | np.ndarray]: # Make this a dict[str, list[np.ndarray]]
        
        if (radii is not None) and (num_neighbors is not None):
            raise ValueError("Must specify only one metric for query request")

        if sum(arg is None for arg in (radii, num_neighbors, tree)) == 3:
            raise ValueError("Must specify either radius or num_neighbors")
        
        if radii is not None:
            query = (
                tree.query_ball_tree(self.__kd_tree, radii)
                if tree is not None
                else self.__kd_tree.query_ball_point(locations, radii, workers=-1)
            )

            return {"indices": np.asarray(query)}

        else:
            query = self.__kd_tree.query(locations, num_neighbors, workers=-1)
            return {
                "distances": np.asarray(query[0]), 
                "indices": np.asarray(query[1])
            }
        
    def get_queried_halo_properties(
            self,
            locations: np.ndarray,
            group_velocities: np.ndarray,
            radii: float | np.ndarray | None = None,
            num_neighbors: int | None = None,
            tree: cKDTree | None = None,
            hubble: float | None = None,
            run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleProperties | list[BoundedParticleProperties]:

        query_results = self.query(locations, radii, num_neighbors, tree)

        if len(query_results["indices"]) == 0:
            return []

        results = []

        for i, query_idxs in enumerate(query_results["indices"]):
            subset = self.get_subset(query_idxs)
            phase_space_props = get_phase_space_properties(
                coordinates=subset.coordinates,
                velocities=subset.velocities,
                box_size=self.box_size,
                center_of_mass=locations[i],
                group_velocity=group_velocities[i],
                hubble=hubble,
                precomputed_radial_distances=(
                    np.asarray(query_results["distances"][i])
                    if "distances" in query_results
                    else None
                )
            )

            results.append(
                BoundedParticleProperties(
                    mass_values=subset.mass_values, 
                    positions=phase_space_props["positions"],
                    radial_distances=phase_space_props["radial_distances"],
                    peculiar=PhaseSpaceProperties(
                        velocities=phase_space_props["peculiar_relative_velocities"],
                        accelerations=phase_space_props["peculiar_relative_accelerations"],
                        radial_velocities=phase_space_props["peculiar_radial_velocities"],
                        radial_accelerations=phase_space_props["peculiar_radial_accelerations"]
                    ),
                    actual=PhaseSpaceProperties(
                        velocities=phase_space_props["actual_relative_velocities"],
                        accelerations=phase_space_props["actual_relative_accelerations"],
                        radial_velocities=phase_space_props["actual_radial_velocities"],
                        radial_accelerations=phase_space_props["actual_radial_accelerations"]
                    ),
                    center_of_mass=locations[i],
                    in_comoving_units=self.in_comoving_units,
                    softening_length=self.softening_length,
                    run_sort_by_radial_distance=run_sort_by_radial_distance
                )
            )

        return results


