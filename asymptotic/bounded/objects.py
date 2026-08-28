from __future__ import annotations

import numpy as np, pdb
import matplotlib.pyplot as plt

from attrs import define
from pathlib import Path
from tqdm.auto import tqdm
from colossus.lss import peaks
from abc import ABC, abstractmethod
from functools import cached_property
from collections import OrderedDict
from collections.abc import Collection


from ..cosmo.model import Cosmology
from ..simulation.moments import Moment
# from .mass_def import MassDefinitions
from ..mass_def.data import MassDefinitions
from ..simulation.sub_box import SubBoxes
from ..simulation.evo import EvolutionData
# from ..mass_function.data import MassFunctionValue, MassFunctionData
from ..abundance.mass_function import MassFunctionValue, MassFunctionData
from ..particles.collection import (
    BoundedParticleCollection, 
    BoxParticleCollection
)

from ..distribution.slope import get_so_values, get_toms_filtered_mass_distribution
from ..distribution.viz import plot_internal_structure
from ..particles.shapes import ShapeParameterData, ShapeProfileData
from ..particles.anisotropy import AnisotropyParameterData, AnisotropyProfileData

@define(slots=True)
class BoundedObject(ABC): 
    id_number: int
    coordinate: tuple[float, float, float] | np.ndarray
    velocity: tuple[float, float, float] | np.ndarray
    particle_count: int
    spin: float

    moment: Moment
    mass_defs: MassDefinitions

    particles: BoundedParticleCollection | None

    shape: ShapeParameterData | None
    # shape_profile: ShapeProfileData | None

    anisotropy: AnisotropyParameterData | None
    # anisotropy_profile: AnisotropyProfileData | None

    has_physical_boundaries: bool

    def __len__(self) -> int:
        return self.particle_count
    
    @property
    def comoving_coordinate(self) -> tuple[float, float, float]:
        return (
            self.coordinate[0] / self.moment.scale_factor,
            self.coordinate[1] / self.moment.scale_factor,
            self.coordinate[2] / self.moment.scale_factor,
        )
    
    def get_mass_def_value(self, mass_def: str, value: str) -> np.ndarray:
        if not hasattr(self.mass_defs, mass_def):
            raise ValueError(f"Mass definition {mass_def} not found")
        if not hasattr(getattr(self.mass_defs, mass_def), value):
            raise ValueError(f"{value} not found in {mass_def}")
        return getattr(getattr(self.mass_defs, mass_def), value)
    
    @property
    @abstractmethod
    def mass(self) -> float:
        ''' Use the proper mass definition to fill this in '''
        ...

    @property
    @abstractmethod
    def radius(self) -> float:
        ''' Use the proper mass definition to fill this in '''
        ... 

    @property
    def volume(self) -> float:
        return 4/3 * np.pi * self.radius**3
    
    @property
    def lagrangian_radius(self) -> float:
        return peaks.lagrangianR(self.mass) / (1 + self.moment.redshift)
    
    @abstractmethod
    def set_particles(
            self, box_particles: BoxParticleCollection, *args, **kwargs
        ) -> BoundedParticleCollection:
        ...

    @abstractmethod
    def get_particles(
            self, box_particles: BoxParticleCollection, *args, **kwargs
        ) -> BoundedParticleCollection:
        ...

    def get_mass_distribution(
            self, num_radial_points: int, return_in_physical_units: bool = True
        ) -> np.ndarray:

        if self.particles is None:  
            raise ValueError("Particles must be set to compute mass distribution")

        return self.particles.get_mass_distribution(
            num_radial_points=num_radial_points,
            return_in_physical_units=return_in_physical_units
        )
    

    @property
    def phase_space_distribution(self) -> np.ndarray:
        
        if self.particles is None:  
            raise ValueError("Particles must be set to compute phase space distribution")

        return np.array([self.particles.radial_distances, self.particles.radial_velocities])
    
    def set_physical_properties(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> None:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)
        self.spin = particles.get_spin(
            mass=10.0**self.mass_defs[mdef].mass,
            radius=self.mass_defs[mdef].radius,
            velocity=self.mass_defs[mdef].velocity
        )

        try:
            self.shape = ShapeParameterData.from_coordinates(particles.coordinates)
        except np.linalg.LinAlgError as error:
            print(
                f"Shape parameter calculation failed for halo {self.id_number}"
                f" on {self.moment.snapshot_id}. "
            )
            self.shape = None

        try:
            self.anisotropy = AnisotropyParameterData.from_halo(
                velocities=particles.properties.actual.velocities,
                spherical_coordinates=particles.properties.spherical_coordinates
            )
        except np.linalg.LinAlgError as error:
            print(
                f"Anisotropy parameter calculation failed for halo {self.id_number}"
                f" on {self.moment.snapshot_id}. "
            )
            self.anisotropy = None

    def get_radial_velocity_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        if self.particles is None:
            raise ValueError("Particles must be set to compute radial velocity profile")
        return self.particles.get_radial_velocity_profile(num_radial_points=num_radial_points)

    def get_spin(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> float | np.ndarray:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)
            
        return particles.get_spin(
            mass=10.0**self.mass_defs[mdef].mass,
            radius=self.mass_defs[mdef].radius,
            velocity=self.mass_defs[mdef].velocity
        )
        
    def get_anisotropy(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> float | np.ndarray:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)
    
        return particles.properties.radial_anisotropy
    
    def get_cylindrical_anisotropy(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> float | np.ndarray:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)

        return particles.properties.cylindrical_anisotropy
    
    def get_anisotropy_profile(
            self, 
            num_radial_points: int, 
            mdef: str = "bound", 
            get_particle_subset: bool = False
        ) -> np.ndarray:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)

        profile = particles.get_anisotropy_profile(num_radial_points)
        return np.array([
            profile["radial_bins"], 
            profile["anisotropy"],
            profile["anisotropy_cyl"]
        ])
        

    def get_shape_parameters(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> dict[str, float]:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)

        return particles.properties.shape_parameters
    
    def get_reduced_shape_parameters(
            self, mdef: str = "bound", get_particle_subset: bool = False
        ) -> dict[str, float]:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)

        return particles.properties.reduced_shape_parameters
    
    def get_shape_profile(
            self, 
            num_radial_points: int, 
            mdef: str = "bound",
            get_particle_subset: bool = False
        ) -> np.ndarray:

        particles = self._safely_retrieve_particles(get_particle_subset, mdef)

        profile = particles.get_shape_profile(num_radial_points)

        return np.array([
            profile["radial_bins"], 
            profile["major_axis"],
            profile["intermediate_axis"],
            profile["minor_axis"],
            profile["sphericity"],
            profile["polar_flat"],
            profile["equatorial_flat"],
            profile["triaxiality"],
            profile["red_sphericity"],
            profile["red_polar_flat"],
            profile["red_equatorial_flat"],
            profile["red_triaxiality"]
        ])


    def _get_validated_subset(self, error_msg: str, mdef: str) -> BoundedParticleCollection:
        if self.particles is None:
            raise ValueError(error_msg)
        if self.mass_defs[mdef] is None:
            raise ValueError(f"Mass definition {mdef} not found")
        return self.particles.get_radial_subset(self.mass_defs[mdef].radius)
    
    def _safely_retrieve_particles(self, get_particle_subset: bool, mdef: str) -> BoundedParticleCollection:
        if get_particle_subset:
            return self._get_validated_subset(
                "Particles must be set to compute spin", mdef
            )
        else:
            return self.particles



    
@define(slots=True)
class BoundedObjectEvolutionData(EvolutionData):
    data: OrderedDict[int, BoundedObject] 

    def __repr__(self) -> str:
        return super().__repr__() 
    @property
    def descendant(self) -> BoundedObject:
        return self.data[max(self.data.keys())]

    @property
    def progenitors(self) -> dict[int, BoundedObject]:
        return {
            snapshot_id: bound_object
            for snapshot_id, bound_object in self.data.items()
            if snapshot_id != max(self.data.keys())
        }
    
    def get_progenitor(self, snapshot_id: int) -> BoundedObject:
        if snapshot_id not in self.data:
            raise ValueError(f"Snapshot {snapshot_id} not found in progenitors")
        if snapshot_id == max(self.data.keys()):
            raise ValueError(f"Snapshot {snapshot_id} is the descendant, not a progenitor")
        return self.data[snapshot_id]

    @property
    def coordinate_trajectory(self) -> np.ndarray:
        return np.column_stack([
            (bound_object.moment.scale_factor, *bound_object.coordinate)
            for _, bound_object in sorted(self.data.items(), key=lambda x: x[0])
        ])
    
    @property
    def comoving_coordinate_trajectory(self) -> np.ndarray:
        return np.column_stack([
            (bound_object.moment.scale_factor, *bound_object.comoving_coordinate)
            for _, bound_object in sorted(self.data.items(), key=lambda x: x[0])
        ])
    
    def get_mass_accretion_history(self, mass_def: str) -> np.ndarray:
        mah = []

        for _, bound_object in sorted(self.data.items(), key=lambda x: x[0]):
            try:
                mass = 10.0**bound_object.mass_defs[mass_def].mass
            except ValueError:
                mass = np.nan
            
            mah.append((bound_object.moment.scale_factor, mass))

        return np.vstack(mah)
    
    def get_boundary_evolution_history(
            self, mass_def: str, return_in_comoving: bool = False
        ) -> np.ndarray:

        radii_evo = []

        for _, bound_object in sorted(self.data.items(), key=lambda x: x[0]):
            try:
                radius = bound_object.mass_defs[mass_def].radius
            except ValueError:
                radius = np.nan

            if return_in_comoving:
                output_radius = radius / bound_object.moment.scale_factor
            else:
                output_radius = radius
            
            radii_evo.append((bound_object.moment.scale_factor, output_radius))

        return np.vstack(radii_evo)

    def get_common_particles(self, snap_id_1: int, snap_id_2: int) -> dict[int, BoundedParticleCollection]:
        if self.data[snap_id_1].particles is None:
            raise ValueError(f"No particles found for {snap_id_1 = }")
        if self.data[snap_id_2].particles is None:
            raise ValueError(f"No particles found for {snap_id_2 = }")
        
        common_ids = np.intersect1d(
            self.data[snap_id_1].particles.ids, 
            self.data[snap_id_2].particles.ids,
        )

        snap_1_collection = self.data[snap_id_1].particles.get_particles(
            ids=common_ids,
            run_sort_by_radial_distance=False
        )
        snap_1_collection.sort_by_particle_ids()

        snap_2_collection = self.data[snap_id_2].particles.get_particles(
            ids=common_ids, 
            run_sort_by_radial_distance=False
        )
        snap_2_collection.sort_by_particle_ids()

        assert np.array_equal(snap_1_collection.ids, snap_2_collection.ids)

        return {snap_id_1 : snap_1_collection, snap_id_2 : snap_2_collection}
    

    @abstractmethod
    def _get_bounded_object_with_particles(
            self, snapshot_id: int,
            box_particles: BoxParticleCollection | None = None,
            cosmo: Cosmology | None = None
        ) -> BoundedObject:
        
        ...


    def display_internal_evolution_triplet(
            self, 
            snapshot_ids: Collection[int],
            boundaries_to_display: Collection[str],
            normalize_velocity: bool = True,
            num_radial_points: int = 1000,
            num_velocity_points: int = 1000,
            use_density_distribution: bool = True,
            box_particles: Collection[BoxParticleCollection] | None = None,
            cosmo: Cosmology | None = None,
            text_size: int = 14,
            tick_size: int = 14,
            legend_size: int = 14,
            label_size: int = 14,
        ) -> None:

        assert len(snapshot_ids) == 3, "Must provide exactly 3 snapshot ids"

        snaphshot_order = np.argsort(snapshot_ids)
        snapshot_ids = np.array(snapshot_ids)[snaphshot_order]
        
        if box_particles is not None:
            assert len(box_particles) == 3, "Must provide exactly 3 box particles"
            box_particles = np.array(box_particles)[snaphshot_order]

        fig, axes = plt.subplots(
            2, 3, figsize=(8, 4), 
            sharex="col", 
            sharey="row",
            gridspec_kw={'hspace': 0.0, 'wspace': 0.0}
        )

        for i, snapshot_id in tqdm(enumerate(snapshot_ids)):

            bound_object = self._get_bounded_object_with_particles(
                snapshot_id=snapshot_id,
                box_particles=(
                    box_particles[i] if box_particles is not None else None
                ),
                cosmo=cosmo
            )

            dist_data = get_bound_obj_distributions(
                bound_obj=bound_object,
                num_radial_points=num_radial_points,
                num_velocity_points=num_velocity_points,
                use_density_distribution=use_density_distribution
            )

            boundaries = get_boundaries_to_display(
                bound_obj=bound_object,
                boundaries_to_display=boundaries_to_display,
                cosmo=cosmo
            )

            fig, axes[:, i], plt_handles = plot_internal_structure(
                distribution_data=dist_data,
                boundaries=boundaries,
                scale_factor=bound_object.moment.scale_factor,
                v_norm = (
                    bound_object.mass_defs.max_circ.velocity
                    if normalize_velocity else 
                    1.0
                ),
                use_density_distribution=use_density_distribution,
                text_size=text_size,
                tick_size=tick_size,
                legend_size=legend_size,
                label_size=label_size,
                fig=fig,
                ax=axes[:, i],
                return_plot=True,
            )

        if normalize_velocity:
            axes[0, 0].set_ylabel(r"$v_{r}/v_{\rm max}$", fontsize=label_size)
        else:
            axes[0, 0].set_ylabel(r"$v_{r}$ $[{\rm km}/{\rm s}]$", fontsize=label_size)

        axes[1, 0].set_ylabel(r"$\kappa(<r)$", fontsize=label_size)
        axes[1, 2].set_ylim(top=0.5)

        fig.legend(
            handles=plt_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(boundaries),
            fontsize=legend_size,
            frameon=False,
        )

        fig.subplots_adjust(bottom=0.15)
        
            
    
def get_bound_obj_distributions(
        bound_obj: BoundedObject,
        num_radial_points: int = 1000,
        num_velocity_points: int = 1000,
        use_density_distribution: bool = True,
        use_toms_approach: bool = False,
        num_particles_per_dex: int = 60,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    ''' 
    Get the mass distribution and phase space distribution of a bounded object.
    '''

    if use_toms_approach:
        mass_distribution = get_toms_filtered_mass_distribution(
            particle_masses=bound_obj.particles.masses,
            particle_radii=bound_obj.particles.radial_distances,
            num_particles_in_dex=num_particles_per_dex
        )

    else:
        mass_distribution = bound_obj.particles.get_filtered_mass_distribution()

    if use_density_distribution:
        phase_space_dist = bound_obj.particles.get_2D_phase_space_histogram(
            num_radial_points=num_radial_points,
            num_velocity_points=num_velocity_points
        )
    else:
        phase_space_dist = bound_obj.phase_space_distribution

    return {
        "radii" : mass_distribution[:, 0],
        "enclosed_mass" : mass_distribution[:, 1],
        "enclosed_density" : mass_distribution[:, 2],
        "local_density" : mass_distribution[:, 3],
        "mass_slope" : mass_distribution[:, 4],
        "gamma" : mass_distribution[:, 5],
        "kappa" : mass_distribution[:, 6], 
        "phase_space" : phase_space_dist, 
        "radii_slope" : (
            mass_distribution[:, 7]
            if use_toms_approach else 
            1.0 / mass_distribution[:, 4]
        )
    }


def get_boundaries_to_display(
        bound_obj: BoundedObject, 
        boundaries_to_display: Collection[str],
        cosmo: Cosmology | None = None,
    ) -> dict[str, float]:

    boundaries = {}

    for boundary_mdef in boundaries_to_display:

        if boundary_mdef in bound_obj.mass_defs.contained_keys:
            boundaries[boundary_mdef] = bound_obj.mass_defs[boundary_mdef].radius

        elif boundary_mdef == "virial":
            if cosmo is None:
                raise ValueError("Cosmology must be provided to compute virial radius")
            boundaries[boundary_mdef] = get_virial_radius(bound_obj, cosmo)

        elif boundary_mdef in {"scale", "four_scale", "max_circ", "bound", "splashback"}:
            raise ValueError(
                f"Mass definition '{boundary_mdef}' not found. "
                "Verify that physical properties have been set on the object."
            )

        elif boundary_mdef in {"fof", "asymptotic"}:
            raise NotImplementedError(f"Mass definition '{boundary_mdef}' is not implemented yet")

        else:
            raise ValueError(f"Unknown mass definition '{boundary_mdef}'")

    return boundaries


def get_virial_radius(bound_obj: BoundedObject, cosmo: Cosmology) -> float:
    if cosmo is None:
        raise ValueError("Cosmology must be provided for virial radius")
    so_values = get_so_values(
        profiles=bound_obj.particles.get_filtered_mass_distribution(),
        z=bound_obj.moment.redshift,
        cosmo=cosmo
    )
    return so_values["R"]["vir"]
        
        
                                                               
                                                        