from __future__ import annotations

import numpy as np, pdb

from pathlib import Path   
from tqdm.auto import tqdm 
from attrs import define, field
from attrs import fields as attrs_fields
from scipy.interpolate import interp1d  
from scipy.spatial import cKDTree
from abc import ABC, abstractmethod
from collections.abc import Collection
from astropy import units as u
from astropy import constants as const

from time import time
from datetime import timedelta
from matplotlib.figure import Figure
from matplotlib.axes import Axes


from .spin import get_spin
from .nbody import NBodyParticle
from .viz import display_particles

from ..fields.two_point import (
    compute_tpcf,
    DEFAULT_NUM_TARGET_PAIRS,
    DEFAULT_TOLERANCE,
)
from ..simulation.moments import Moment
from ..simulation.sub_box import SubBoxes
from .identify import ParticleIdentifier
from ..utils.get_data import get_snap_data, get_initial_snap_data
from ..cosmo.model import Cosmology
from ..distribution.slope import get_filtered_mass_distribution
from .orbital import make_periodic, get_phase_space_properties
from ..mass_def.data import get_circular_velocity
from ..fields.knn import kNNDistributionData
from ..fields.correlation import TwoPointCorrelationData
from .properties import (
    ParticleCollectionProperties, 
    BoxParticleProperties,
    BoundedParticleProperties,
    PhaseSpaceProperties
)

DEFAULT_NUM_MESH_POINTS = 512
DEFAULT_RESAMPLER = "cic"

DEFAULT_kNN_NUM_QUERY_POINTS = 100 * 100 * 100
DEFAULT_kNN_MAX_NUM_NEIGHBORS = 2
DEFAULT_kNN_NUM_INTERP_POINTS = 1000
DEFAULT_NUM_SPLITS = 1


def get_volume(locations: np.ndarray) -> float:
    x_range = float(locations[:, 0].max() - locations[:, 0].min())
    y_range = float(locations[:, 1].max() - locations[:, 1].min())
    z_range = float(locations[:, 2].max() - locations[:, 2].min())
    return x_range * y_range * z_range


@define(slots=True)
class ParticleCollection(ABC): 
    ''' A collection of particles at a given moment in time. '''
    volume: float 
    moment: Moment
    identifier: ParticleIdentifier
    properties: ParticleCollectionProperties

    def __len__(self) -> int:
        return len(self.identifier)
    
    def __repr__(self) -> str:
        print_volume = f"{self.volume:.3f}" if self.volume < 1e5 else f"{self.volume:.3e}"
        print_n_particles = (
            f"{len(self):,d}" if len(self) < 1_000_000 else f"{len(self):.3e}"
        )
        return (
            f"{self.__class__.__name__}(volume={print_volume} (Mpc/h)^3, "
            f"a={self.moment.scale_factor:.3f}, "
            f"n_particles={print_n_particles})"
        )
    
    # Return a single particle with all of the properties that have values 
    def __getitem__(self, index: int) -> NBodyParticle:
        return NBodyParticle(
            id_number=self.identifier.ids[index],
            properties=self.properties[index]
        )
    
    @abstractmethod
    def get_particles(self, ids: Collection[int], *args, **kwargs) -> ParticleCollection:
        ...
    
    @abstractmethod
    def get_subset_by_idx(self, idxs: Collection[int], *args, **kwargs) -> ParticleCollection:
        ...
    
    @property
    def ids(self) -> np.ndarray:
        return self.identifier.ids

    @property
    def num_particles(self) -> int:
        return self.properties.num_particles
    
    @property
    def coordinates(self) -> np.ndarray:
        return self.properties.coordinates
    
    @property
    def comoving_coordinates(self) -> np.ndarray:
        return self.properties.coordinates / self.moment.scale_factor
    
    @property
    def masses(self) -> np.ndarray:
        return self.properties.masses
    
    @property
    def enclosed_mass(self) -> float:
        return self.properties.enclosed_mass
    
    @property
    def enclosed_density(self) -> float:
        return self.enclosed_mass / self.volume
    
    @property
    def velocities(self) -> np.ndarray:
        return self.properties.velocities
    
    @property
    def accelerations(self) -> np.ndarray | None:
        return self.properties.accelerations
    
    def render_physical_density_map(
            self,
            plane_normal: str,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = True
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        return create_density_map(
            positions=self.coordinates,
            masses=self.masses,
            normal_axes=plane_normal,
            extent=extent,
            num_bins=num_bins,
            scale_factor=(
                1.0 if return_coords_in_physical_units else self.moment.scale_factor
            )
        )

    def render_comoving_density_map(
            self,
            plane_normal: str,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = False
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        map_data = create_density_map(
            positions=self.comoving_coordinates,
            masses=self.masses,
            normal_axes=plane_normal,
            extent=extent,
            num_bins=num_bins,
        )
        
        if return_coords_in_physical_units:
            map_data["x1_edges"] *= self.moment.scale_factor
            map_data["x2_edges"] *= self.moment.scale_factor
            map_data["extent"] = (
                map_data["extent"][0] * self.moment.scale_factor,
                map_data["extent"][1] * self.moment.scale_factor,
                map_data["extent"][2] * self.moment.scale_factor,
                map_data["extent"][3] * self.moment.scale_factor
            )

        return map_data
    
    def render_physical_projected_density_map(
            self,
            plane_normal: str,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = True
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        return create_projected_density_map(
            positions=self.coordinates,
            masses=self.masses,
            normal_axes=plane_normal,
            extent=extent,
            num_bins=num_bins,
            scale_factor=(
                1.0 if return_coords_in_physical_units else self.moment.scale_factor
            )
        )
    
    def render_comoving_projected_density_map(
            self,
            plane_normal: str,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = False
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        map_data = create_projected_density_map(
            positions=self.comoving_coordinates,
            masses=self.masses,
            normal_axes=plane_normal,
            extent=extent,
            num_bins=num_bins,
        )

        if return_coords_in_physical_units:
            convert_back_to_physical = lambda x: x * self.moment.scale_factor
            map_data["x1_edges"] = convert_back_to_physical(map_data["x1_edges"])
            map_data["x2_edges"] = convert_back_to_physical(map_data["x2_edges"])
            map_data["extent"] = (
                convert_back_to_physical(map_data["extent"][0]),
                convert_back_to_physical(map_data["extent"][1]),
                convert_back_to_physical(map_data["extent"][2]),
                convert_back_to_physical(map_data["extent"][3])
            )
        
        return map_data
    
    def render_physical_projected_overdensity_map(
            self,
            plane_normal: str,
            mean_density: float,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = True
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        return create_projected_overdensity_map(
            positions=self.coordinates,
            masses=self.masses,
            normal_axes=plane_normal,
            mean_density=mean_density,
            extent=extent,
            num_bins=num_bins,
            scale_factor=(
                1.0 if return_coords_in_physical_units else self.moment.scale_factor
            )
        )
    
    def render_comoving_projected_overdensity_map(
            self,
            plane_normal: str,
            present_day_mean_density: float,
            extent: tuple[float, ...] | None = None,
            num_bins: int = 1000,
            return_coords_in_physical_units: bool = False
        ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

        proj_density_map = self.render_comoving_projected_density_map(
            plane_normal=plane_normal,
            extent=extent,
            num_bins=num_bins,
            return_coords_in_physical_units=return_coords_in_physical_units
        )

        comoving_density_map = proj_density_map["density"]
        x3_edges = proj_density_map["x3_edges"]
        depth = x3_edges[-1] - x3_edges[0]

        surface_mean_density = present_day_mean_density * depth

        if return_coords_in_physical_units:
            surface_mean_density *= self.moment.scale_factor

        overdensity_map = comoving_density_map / surface_mean_density

        return {
            "overdensity" : overdensity_map,
            "x1_edges" : proj_density_map["x1_edges"],
            "x2_edges" : proj_density_map["x2_edges"],
            "x3_edges" : x3_edges,
            "extent" : proj_density_map["extent"],
            "depth" : depth,
        }
    
    
    def display(
            self,
            spacing: int,
            plt_setup: tuple[Figure, list[Axes], list[Axes]] | None = None,
        ) -> tuple[Figure, list[Axes], list[Axes]] | None:
        return display_particles(self.coordinates, spacing, plt_setup=plt_setup)


@define(slots=True)
class BoxParticleCollection(ParticleCollection):
    ''' A collection of particles in a simulation box at a given moment in time. '''
    properties: BoxParticleProperties

    sub_boxes: SubBoxes | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def has_kd_tree(self) -> bool:
        return self.properties.kd_tree is not None

    @property
    def num_particles(self) -> int:
        return len(self)
    
    @property
    def box_size(self) -> float:
        return self.properties.box_size

    @property
    def comoving_box_size(self) -> float:
        return self.box_size / self.moment.scale_factor

    @property
    def softening_length(self) -> float:
        return self.properties.softening_length

    @property
    def mean_interparticle_spacing(self) -> float:
        return (self.volume / self.num_particles) ** (1.0 / 3.0)

    @property
    def comoving_mean_interparticle_spacing(self) -> float:
        return self.mean_interparticle_spacing / self.moment.scale_factor
    
    @property
    def comoving_softening_length(self) -> float:
        return self.softening_length / self.moment.scale_factor 


    @property
    def default_radial_bounds(self) -> tuple[float, float]:
        min_bound = max(3.0 * self.softening_length, 2.0 * self.mean_interparticle_spacing)
        max_bound = 0.49 * self.box_size # must stay within L/2 
        return (min_bound, max_bound)

    @property
    def default_comoving_radial_bounds(self) -> tuple[float, float]:
        min_bound, max_bound = self.default_radial_bounds
        return (
            min_bound / self.moment.scale_factor, 
            max_bound / self.moment.scale_factor
        )

    @property
    def sub_box_samples_id_array(self) -> np.ndarray:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_sub_box_samples_id_array(self.coordinates) 

    @property
    def sub_box_samples_coordinates(self) -> list[np.ndarray]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_jackknife_subbox_samples(
            coordinates=self.coordinates,
            return_as_mask=False
        )   

    @property
    def sub_box_samples_mask(self) -> list[np.ndarray]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_jackknife_subbox_samples(
            coordinates=self.coordinates,
            return_as_mask=True
        )
    

    @property
    def sub_box_as_comoving_dict(self) -> dict[int, list[tuple[float, float]]]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_comoving_as_dict(self.moment.scale_factor)

    def initialize_empty_sub_boxes(self, num_splits: int) -> None:
        self.sub_boxes = SubBoxes.empty_initialize(self.box_size, num_splits)


    def get_periodic_coordinates(self, origin: np.ndarray) -> np.ndarray:
        return make_periodic(
            coordinates=self.coordinates,
            box_size=self.box_size,
            center_of_mass=origin
        )
    
    def get_particles(self, ids: Collection[int]) -> BoxParticleCollection:
        return self.__class__(
            volume=self.volume,
            moment=self.moment,
            identifier=self.identifier.get_subset(ids),  
            properties=self.properties.get_subset(np.isin(self.ids, ids))
        )
    
    def get_subset_by_idx(self, idxs: Collection[int]) -> BoxParticleCollection:
        return self.get_particles(ids=self.ids[idxs])

    def get_voxel_size(self, num_mesh_points: int) -> float:
        return (
            (self.box_size / float(num_mesh_points))
            if num_mesh_points > 0 else 
            float(num_mesh_points)
        )

    def get_thin_slice(
            self, 
            origin: tuple[float, float, float] | np.ndarray,
            comoving_window_size: int | float,
            comoving_depth: int | float = 10.0,
            plane_normal: str = "z",
        ) -> BoxParticleCollection:
        
        window_size = comoving_window_size * self.moment.scale_factor
        depth = comoving_depth * self.moment.scale_factor

        i, j, k = get_slice_idxs_for_plane(plane_normal)

        periodic_coordinates = self.get_periodic_coordinates(np.asarray(origin))
        window_mask = np.logical_and.reduce([
            np.abs(periodic_coordinates[:, i] - origin[i]) <= (window_size / 2),
            np.abs(periodic_coordinates[:, j] - origin[j]) <= (window_size / 2),
            np.abs(periodic_coordinates[:, k] - origin[k]) <= (depth / 2),
        ])
        
        if window_mask.sum() == 0:
            raise ValueError(
                f"No particles found in the given window: {window_mask.sum()}"
            )
        
        print(f"{window_mask.shape = }")
        
        return self.get_subset_by_idx(window_mask)


    def create_one_plus_delta_pmesh(
            self, 
            nmesh: int = DEFAULT_NUM_MESH_POINTS, 
            resampler: str = DEFAULT_RESAMPLER,
            interlaced: bool = False, 
            comm=None
        ) -> np.ndarray:
        return one_plus_delta_pmesh(
            positions=self.coordinates,
            box_size=self.box_size,
            nmesh=nmesh,
            resampler=resampler,
            interlaced=interlaced,
            comm=comm,
        )


    def get_nn_distributions(
            self, 
            num_query_points: int = DEFAULT_kNN_NUM_QUERY_POINTS,
            max_num_neighbors: int = DEFAULT_kNN_MAX_NUM_NEIGHBORS,
            num_interp_points: int = DEFAULT_kNN_NUM_INTERP_POINTS,
            use_vectorized: bool = True,
            use_jackknifed: bool = False,
            num_splits: int = DEFAULT_NUM_SPLITS,
            return_in_comoving: bool = False,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            use_data_randoms: bool = False,
        ) -> kNNDistributionData:

        if self.sub_boxes is None:
            self.initialize_empty_sub_boxes(num_splits)

        knn_data = kNNDistributionData.from_data(
            comoving_positions=self.coordinates,
            comoving_box_size= self.box_size,
            num_query_points=num_query_points,
            max_num_neighbors=max_num_neighbors,
            num_interp_points=num_interp_points,
            sub_box_info=self.sub_boxes.as_dict, 
            kd_tree=self.properties.kd_tree,
            use_vectorized=use_vectorized,
            run_jackknife= use_jackknifed,
            use_population_jk=use_population_jk,
            use_fast_population_jk=use_fast_population_jk,
            use_data_randoms=use_data_randoms
        )

        if return_in_comoving:
            knn_data.convert_to_comoving(self.moment.scale_factor)

        return knn_data  
    

    def get_two_point_radial_bounds(
            self, 
            num_query_points: int = DEFAULT_kNN_NUM_QUERY_POINTS,
            max_num_neighbors: int = DEFAULT_kNN_MAX_NUM_NEIGHBORS,
            num_interp_points: int = DEFAULT_kNN_NUM_INTERP_POINTS,
            target_pcdf_value: float = 0.01,
            use_vectorized: bool = True,
            use_jackknifed: bool = True,
            return_in_comoving: bool = False,
            num_splits: int = DEFAULT_NUM_SPLITS,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            use_data_randoms: bool = False,
            knn_data: kNNDistributionData | None = None,
        ) -> tuple[float, float]:

        if knn_data is None:
            knn_data = self.get_nn_distributions(
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                num_interp_points=num_interp_points,
                use_vectorized=use_vectorized,
                use_jackknifed=use_jackknifed,
                num_splits=num_splits, 
                return_in_comoving=return_in_comoving,
                use_population_jk=use_population_jk,
                use_fast_population_jk=use_fast_population_jk,
                use_data_randoms=use_data_randoms
            )

        return knn_data.get_two_point_radial_bounds(
            target_pcdf_value=target_pcdf_value,
            for_data_randoms=use_data_randoms
        )

    def get_matter_matter_correlations(
            self, 
            num_query_points: int = DEFAULT_kNN_NUM_QUERY_POINTS,
            max_num_neighbors: int = DEFAULT_kNN_MAX_NUM_NEIGHBORS,
            num_interp_points: int = DEFAULT_kNN_NUM_INTERP_POINTS,
            num_radial_bins: int = 10,
            target_pcdf_value: float = 0.01,
            use_vectorized: bool = True,
            use_jackknifed: bool = True,
            use_natural: bool = True,
            return_in_comoving: bool = True,
            eps: float = DEFAULT_TOLERANCE,
            num_splits: int = DEFAULT_NUM_SPLITS,
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            return_folds: bool = False,
            use_recommended_subsample: bool = False,
            num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
            subsample_secondary: bool = False,   # unused for MM but harmless to expose
            subsample_rng_seed: int | None = None,
            adaptive_randoms: bool = False,
            max_randoms_per_fold: int | None = None,
            min_randoms_per_fold: int = 1024,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            radial_range: tuple[float, float] | None = None,
            use_data_randoms: bool = False,
            knn_data: kNNDistributionData | None = None,
            use_default_radial_bounds: bool = False,
        ) -> TwoPointCorrelationData:

        if (not use_default_radial_bounds) and (radial_range is not None):
            low_bound, high_bound = radial_range
        elif use_default_radial_bounds:
            low_bound, high_bound = self.default_comoving_radial_bounds
        else:
            low_bound, high_bound = self.get_two_point_radial_bounds(
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                num_interp_points=num_interp_points,
                target_pcdf_value=target_pcdf_value,
                use_vectorized=use_vectorized,
                use_jackknifed=use_jackknifed,
                return_in_comoving=True,
                num_splits=num_splits,
                use_population_jk=use_population_jk,
                use_fast_population_jk=use_fast_population_jk,
                use_data_randoms=use_data_randoms,
                knn_data=knn_data
            )

        r_min = max(low_bound, eps)
        r_max = min(high_bound, (self.comoving_box_size / 2.0) - 0.00001)
        print(f"Radial bounds: {r_min} cMpc/h, {r_max} cMpc/h\n")

        # 2) Ensure sub-box info exists if we’ll jackknife
        if use_jackknifed and self.sub_boxes is None:
            self.initialize_empty_sub_boxes(num_splits)

        # 3) Build TwoPointCorrelationData via the unified TPCF API
        tpcf_data = TwoPointCorrelationData.from_data(
            comoving_coordinates=self.comoving_coordinates,  # coordinates/box are comoving in this collection
            rmin=r_min,
            rmax=r_max,
            nbins=num_radial_bins,               # reuse interp count as nbins
            boxsize=self.comoving_box_size,
            sub_box_info=(self.sub_box_as_comoving_dict if use_jackknifed else None),
            use_natural=use_natural,
            run_jackknife=use_jackknifed,
            random_multiplier=random_multiplier,   # optional: NR density control
            rng_seed=rng_seed,                     # optional: reproducibility
            eps=eps,
            return_folds=return_folds,               # always return folds for possible later use
            use_recommended_subsample=use_recommended_subsample,
            num_target_pairs=num_target_pairs,
            subsample_secondary=subsample_secondary,
            subsample_rng_seed=subsample_rng_seed,
            adaptive_randoms=adaptive_randoms,
            max_randoms_per_fold=max_randoms_per_fold,
            min_randoms_per_fold=min_randoms_per_fold,
        )

        # 4) Convert to physical radii if requested
        if not return_in_comoving:
            tpcf_data.convert_to_physical(self.moment.scale_factor)

        return tpcf_data

    @classmethod
    def from_snapshot_file(
            cls, sim_dir: Path, 
            moment: Moment, 
            is_batched: bool = False,
            particle_type: int = 1,
            compute_kd_tree: bool = True, 
            for_initial_condition: bool = False,
        ) -> BoxParticleCollection:
    
        return cls.from_snapshot_data(
            snapshot_data=(
                get_initial_snap_data(directory=sim_dir, is_batched=is_batched)
                if for_initial_condition else
                get_snap_data(
                    directory=sim_dir, 
                    snap_idx=moment.snapshot_id,
                    is_batched=is_batched,
                )
            ),
            moment=moment,
            particle_type=particle_type,
            compute_kd_tree=compute_kd_tree
        )

    @classmethod
    def from_snapshot_data(
            cls, 
            snapshot_data: dict, 
            moment: Moment, 
            particle_type: int = 1,
            compute_kd_tree: bool = True
        ) -> BoxParticleCollection:
        ''' Create a BoxParticleCollection from a snapshot data dictionary. '''
        particles = f"PartType{particle_type:d}"
        box_length = float(snapshot_data["BoxSize"])
        particle_ids = np.asarray(snapshot_data[f"{particles}/ParticleIDs"])
        velocities = np.asarray(snapshot_data[f"{particles}/Velocities"])
        coordinates = np.asarray(snapshot_data[f"{particles}/Coordinates"])

        if f"{particles}/Masses" in snapshot_data:
            masses = np.asarray(snapshot_data[f"{particles}/Masses"])
        else:
            mass_val = snapshot_data["MassTable"][1]
            masses = np.ones(len(particle_ids)) * mass_val

        if f"{particles}/Accelerations" in snapshot_data:
            accelerations = np.asarray(snapshot_data[f"{particles}/Accelerations"])
        else:
            accelerations = None

        return cls(
            volume=get_volume(coordinates),
            moment=moment,
            identifier=ParticleIdentifier(particle_ids),
            properties=BoxParticleProperties(
                mass_values=masses,
                positions=coordinates,
                peculiar=PhaseSpaceProperties(
                    velocities=velocities,
                    accelerations=accelerations
                ),
                actual=None,
                box_size=box_length,
                in_comoving_units=bool(snapshot_data["ComovingIntegrationOn"]),
                softening_length=snapshot_data["Softening"],
                keep_kd_tree=compute_kd_tree,
                compute_kd_tree=compute_kd_tree
            ),
        )


    @classmethod
    def from_kd_tree_file(cls, tree_path: Path, moment: Moment) -> BoxParticleCollection:
        box_particle_data = np.load(tree_path, allow_pickle=True)

        kd_tree: cKDTree = box_particle_data["tree"].item()
        particle_masses = box_particle_data["masses"]
        particle_ids = box_particle_data["ids"]
        particle_velocities = box_particle_data["velocities"]
        particle_accelerations = box_particle_data["accelerations"]
        volume = get_volume(kd_tree.data)

        properties = BoxParticleProperties(
            mass_values=particle_masses,
            positions=kd_tree.data,
            peculiar=PhaseSpaceProperties(
                velocities=particle_velocities,
                accelerations=particle_accelerations
            ),
            actual=None,
            box_size=np.cbrt(volume),
            in_comoving_units=True,
            softening_length=0.0,
            keep_kd_tree=False,
            compute_kd_tree=False

        )
        properties.kd_tree = kd_tree    

        return cls(
            volume=volume,
            moment=moment,
            identifier=ParticleIdentifier(particle_ids),
            properties=properties
        )
    

    def cloud_in_cell_subsample(
            self, *args, **kwargs
        ) -> BoxParticleCollection:
        ''' Subsample particles using cloud-in-cell method. '''
        raise NotImplementedError("Cloud-in-cell subsampling not yet implemented.")
    
    def display(
            self, spacing: int = 100, 
            plt_setup: tuple[Figure, list[Axes], list[Axes]] | None = None
        ) -> tuple[Figure, list[Axes], list[Axes]] | None:
        return super().display(spacing, plt_setup)
    


@define(slots=True)
class BoundedParticleCollection(ParticleCollection):
    ''' A collection of particles in a bounded region at a given moment in time. '''
    properties: BoundedParticleProperties

    num_particles_in_query: list[int] | None = field(default=None)
    # run_sort_by_id: bool = field(default=False)

    # def __attr_post_init__(self):
    #     if self.run_sort_by_id:
    #         self._sort_by_particle_ids()

    def __repr__(self) -> str:
        return super().__repr__()
    

    def get_particles(
            self, ids: Collection[int], 
            run_sort_by_radial_distance: bool = True,
        ) -> BoundedParticleCollection:

        properties = self.properties.get_subset(
            np.isin(self.ids, ids),
            run_sort_by_radial_distance=run_sort_by_radial_distance
        )

        identifier = ParticleIdentifier(
            ids=self.ids[np.isin(self.ids, ids)][properties.order]
        )

        return self.__class__(
            volume=self.volume,
            moment=self.moment,
            identifier=identifier,  # may have to add this here as well... 
            properties=properties
        )
    
    def get_subset_by_idx(
            self, idxs: Collection[int], run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleCollection:
        return self.get_particles(
            ids=self.ids[idxs], 
            run_sort_by_radial_distance=run_sort_by_radial_distance
        )
    
    def get_radial_subset(
            self, radius: float, run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleCollection:
        mask = self.radial_distances <= radius
        subset = BoundedParticleCollection(
            volume=(4./3) * np.pi * radius**3,
            moment=self.moment,
            identifier=self.identifier.get_subset(ids=self.ids[mask]),
            # identifier=ParticleIdentifier(ids=self.ids[mask]),
            properties=self.properties.get_subset(mask, run_sort_by_radial_distance)
        )
        subset.identifier.ids = subset.identifier.ids[subset.properties.order]
        return subset

    def sort_by_particle_ids(self) -> None:
        order = np.argsort(self.identifier.ids) 
        self.update_order(order)

    def update_order(self, order: np.ndarray) -> None:
        self.identifier.ids = self.identifier.ids[order]
        self.properties.update_order(order)

    @property
    def center_of_mass(self) -> np.ndarray:
        return self.properties.center_of_mass
    
    @property
    def enclosed_radius(self) -> float:
        return np.cbrt((3.0 * self.volume) / (np.pi * 4.0))
    
    @property
    def enclosed_circular_velocity(self) -> float | np.ndarray:
        return get_circular_velocity(self.enclosed_mass, self.enclosed_radius)
    
    @property
    def radial_distances(self) -> np.ndarray:
        return self.properties.radial_distances
    
    @property
    def comoving_radial_distances(self) -> np.ndarray:
        return self.radial_distances / self.moment.scale_factor
    
    @property
    def radial_velocities(self) -> np.ndarray:
        return self.properties.radial_velocities
    
    @property
    def radial_accelerations(self) -> np.ndarray | None:
        return self.properties.radial_accelerations

    @classmethod
    def from_dictionary(cls, properties: dict) -> BoundedParticleCollection:
        return cls(
            volume=properties["volume"], 
            moment=properties["moment"],
            identifier=ParticleIdentifier(properties["particle_ids"]),
            properties=BoundedParticleProperties.from_dictionary(properties),
            num_particles_in_query=properties["num_particles_in_query"]
        )

    @classmethod
    def from_box_particles(
            cls, 
            box_particles: BoxParticleCollection, 
            group_coordinates: np.ndarray,
            group_velocities: np.ndarray,
            query_radii: float | None = None,
            num_neighbors: int | None = None,
            cosmo: Cosmology | None = None,
            use_comoving: bool = True, 
            use_vectorized: bool = False,
            run_sort_by_radial_distance: bool = True,
        ) -> BoundedParticleCollection:

        halo_params = {
            "box_particles": box_particles,
            "coordinates": group_coordinates,
            "group_velocities": group_velocities,
            "query_radii": query_radii,
            "num_neighbors": num_neighbors,
            "hubble" : (
                None
                if cosmo is None else
                cosmo.hubble_parameter(z=box_particles.moment.redshift, unit="s")
            ),
            "little_h" : (None if cosmo is None else cosmo.h),
            "use_comoving": use_comoving,
            "run_sort_by_radial_distance": run_sort_by_radial_distance
        }

        particles = (
            get_single_halo_particles(**halo_params)
            if group_coordinates.ndim == 1 else
            get_population_averaged_halo_particles(
                **halo_params, 
                use_vectorized=use_vectorized,
            )
        )

        if particles is None:
            raise ValueError("No particles found in the given region")
        
        return particles

    def get_radial_bins(self, num_radial_points: int) -> np.ndarray:
        return np.logspace(
            np.log10(self.properties.smallest_nonzero_distance),
            np.log10(self.enclosed_radius),
            num_radial_points
        )
    
    def get_mass_distribution(
            self, num_radial_points: int, return_in_physical_units: bool = True
        ) -> np.ndarray:
        return get_mass_distribution(
            masses=self.properties.masses, 
            radii=self.radial_distances, 
            num_radial_points=num_radial_points,
            min_radius=self.properties.softening_length,
            scaling_radius=self.enclosed_radius,
            return_in_physical_units=return_in_physical_units
        )
    
    def get_radial_velocity_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        return compute_velocity_profile(
            radii=self.radial_distances, 
            velocities=self.radial_velocities, 
            radial_bins=self.get_radial_bins(num_radial_points)
        )
    
    def get_radial_dispersion_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        return compute_dispersion_profile(
            radii=self.radial_distances, 
            velocities=self.radial_velocities, 
            radial_bins=self.get_radial_bins(num_radial_points)
        )
    
    def get_tangential_dispersion_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        return compute_dispersion_profile(
            radii=self.radial_distances, 
            velocities=self.properties.tangent_velocities, 
            radial_bins=self.get_radial_bins(num_radial_points)
        )
    
    def get_anisotropy_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        return compute_anisotropy_profile(
            radii=self.radial_distances, 
            velocities=self.velocities,
            radial_velocities=self.properties.tangent_velocities, 
            tangential_velocities=self.radial_velocities,
            radial_bins=self.get_radial_bins(num_radial_points)
        )
    

    def get_shape_profile(self, num_radial_points: int) -> dict[str, np.ndarray]:
        return compute_shape_profiles(
            positions=self.properties.positions,
            radii=self.radial_distances, 
            radial_bins=self.get_radial_bins(num_radial_points)
        )
    
    def get_filtered_mass_distribution(
            self, 
            size_in_dex: float = 1.0,
            poly_order: int = 4,
            mode: str = "nearest",
            num_bins: int = 100,
            radii_are_pre_sorted: bool = True,
        ) -> np.ndarray:
        
        return get_filtered_mass_distribution(
            particle_masses=self.properties.masses, 
            particle_radii=self.radial_distances,
            size_in_dex=size_in_dex,
            poly_order=poly_order,
            mode=mode,
            num_bins=num_bins,
            radii_are_pre_sorted=radii_are_pre_sorted
        )
    
    def get_2D_phase_space_histogram(
            self, 
            num_radial_points: int = 200,
            num_velocity_points: int = 200,
            v_linthresh: float = 1.0,
            v_linscale: float = 1.0,
            v_log_base: float = 10.0
        ) -> dict[str, np.ndarray]:
        
        return get_2D_phase_space_distribution(
            radii = self.radial_distances, 
            velocities = self.radial_velocities, 
            masses = self.properties.masses, 
            num_r_bins = num_radial_points,
            num_v_bins = num_velocity_points,
            v_linthresh = v_linthresh,
            v_linscale=v_linscale,
            v_log_base=v_log_base
        )
    
    def get_spin(
            self, mass: float, radius: float, velocity: float
        ) -> float | np.ndarray:

        return get_spin(
            enclosed_radii=self.radial_distances,
            enclosed_masses=self.properties.masses,
            positions=self.coordinates,
            velocities=self.properties.actual.velocities,
            mass=mass,
            radius=radius,
            velocity=velocity
        )


    def display(
            self, spacing: int = 1, 
            plt_setup: tuple[Figure, list[Axes], list[Axes]] | None = None
        ) -> tuple[Figure, list[Axes], list[Axes]] | None:
        return super().display(spacing, plt_setup)

# Need to develop a way for BoundedObjectCollection when we query particles 
# for a set of FOFGroups or Subhalos. 

def get_plane_idxs(normal_axes: str) -> tuple[int, int]:
    match normal_axes:
        case "x":
            return 1, 2
        case "y":
            return 0, 2
        case "z":
            return 0, 1
        case _:
            raise ValueError(
                f"Invalid normal axes {normal_axes}: must be 'x', 'y', or 'z'"
            )
        
def get_depth_idx_for_plane(normal_axes: str) -> int:
    match normal_axes:
        case "x":
            return 0
        case "y":
            return 1
        case "z":
            return 2    
        case _:
            raise ValueError(
                f"Invalid normal axes {normal_axes}: must be 'x', 'y', or 'z'"
            )
        
def get_slice_idxs_for_plane(normal_axes: str) -> tuple[int, int, int]:
    return (*get_plane_idxs(normal_axes), get_depth_idx_for_plane(normal_axes))

def create_density_map(
        positions: np.ndarray,
        masses: np.ndarray,
        normal_axes: str,
        extent: tuple[float, ...] | None = None,
        num_bins: int = 1000,
        scale_factor: float = 1.0,
    ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

    i, j, k = get_slice_idxs_for_plane(normal_axes)
    
    if extent is None:
        x1_min, x1_max = positions[:, i].min(), positions[:, i].max()
        x2_min, x2_max = positions[:, j].min(), positions[:, j].max()
    else:
        x1_min, x1_max, x2_min, x2_max = extent

    thickness = positions[:, k].max() - positions[:, k].min()
    if thickness <= 0:
        raise ValueError("Thickness must be positive")

    get_edges = lambda lo, hi: np.linspace(lo, hi, num_bins + 1)
    get_width = lambda lo, hi: (hi - lo) / num_bins

    x1_edges, x2_edges = get_edges(x1_min, x1_max), get_edges(x2_min, x2_max)
    dx, dy = get_width(x1_min, x1_max), get_width(x2_min, x2_max)
    bin_volume = dx * dy * thickness

    hist2D, x1_edges, x2_edges = np.histogram2d(
        x=positions[:, i], 
        y=positions[:, j], 
        bins=(x1_edges, x2_edges), 
        weights=masses
    ) 

    return {
        "density" : hist2D / bin_volume,
        "x1_edges" : x1_edges / scale_factor,
        "x2_edges" : x2_edges / scale_factor,
        "extent" : (
            x1_min / scale_factor, 
            x1_max / scale_factor, 
            x2_min / scale_factor, 
            x2_max / scale_factor
        )
    }

def by_bin_size_histogramdd(
        sample: np.ndarray,
        weights: np.ndarray,
        ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        bin_size: float | tuple[float, float, float],
    ) -> tuple[np.ndarray, list[np.ndarray], float, float, float]:

    """
    3D histogram with a fixed bin size (physical width) per axis.
    Returns mass3D, edges, dx1, dx2, dx3.
    """
    (x1_min, x1_max), (x2_min, x2_max), (x3_min, x3_max) = ranges

    # unpack bin_size
    if isinstance(bin_size, tuple):
        bs1, bs2, bs3 = bin_size
    else:
        bs1 = bs2 = bs3 = bin_size

    # number of bins needed (ceil so we cover the full range)
    n1 = np.ceil((x1_max - x1_min) / bs1)
    n2 = np.ceil((x2_max - x2_min) / bs2)
    n3 = np.ceil((x3_max - x3_min) / bs3)

    # exact edge arrays
    edges1 = x1_min + np.arange(n1 + 1) * bs1
    edges2 = x2_min + np.arange(n2 + 1) * bs2
    edges3 = x3_min + np.arange(n3 + 1) * bs3

    mass3D, edges = np.histogramdd(
        sample=sample,
        bins=(edges1, edges2, edges3),
        weights=weights
    )

    return mass3D, edges, bs1, bs2, bs3


# -------------------------------------------------------------------
def by_num_bins_histogramdd(
    sample: np.ndarray,
    weights: np.ndarray,
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    num_bins: int,
) -> tuple[np.ndarray, list[np.ndarray], float, float, float]:
    """
    3D histogram with a fixed number of bins per axis.
    Returns mass3D, edges, dx1, dx2, dx3.
    """
    (x1_min, x1_max), (x2_min, x2_max), (x3_min, x3_max) = ranges

    dx1 = (x1_max - x1_min) / num_bins
    dx2 = (x2_max - x2_min) / num_bins
    dx3 = (x3_max - x3_min) / num_bins

    mass3D, edges = np.histogramdd(
        sample=sample,
        bins=num_bins,
        range=( (x1_min, x1_max),
                (x2_min, x2_max),
                (x3_min, x3_max) ),
        weights=weights
    )

    return mass3D, edges, dx1, dx2, dx3

def create_projected_density_map(
        positions: np.ndarray,
        masses: np.ndarray,
        normal_axes: str,
        extent: tuple[float, ...] | None = None,
        num_bins: int = 1000,
        bin_size: float | tuple[float, float, float] | None = None,
        scale_factor: float = 1.0,
    ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

    i, j, k = get_slice_idxs_for_plane(normal_axes)

    if extent is None:
        x1_min, x1_max = positions[:, i].min(), positions[:, i].max()
        x2_min, x2_max = positions[:, j].min(), positions[:, j].max()
        x3_min, x3_max = positions[:, k].min(), positions[:, k].max()
    else:
        x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = extent

    ranges = (
        (x1_min, x1_max),
        (x2_min, x2_max),
        (x3_min, x3_max),
    )
    sample = positions[:, (i, j, k)]

    # pick the right binning function
    if bin_size is not None:
        mass3D, edges, dx1, dx2, dx3 = by_bin_size_histogramdd(
            sample=sample, 
            weights=masses, 
            ranges=ranges, 
            bin_size=bin_size
        )
    else:
        mass3D, edges, dx1, dx2, dx3 = by_num_bins_histogramdd(
            sample=sample, 
            weights=masses, 
            ranges=ranges, 
            num_bins=num_bins
        )

    density3D = mass3D / (dx1 * dx2 * dx3)
    projected_density = density3D.sum(axis=2)

    return {
        "density" : projected_density,
        "x1_edges" : edges[0] / scale_factor,
        "x2_edges" : edges[1] / scale_factor,
        "x3_edges" : edges[2] / scale_factor,
        "extent" : (
            x1_min / scale_factor, 
            x1_max / scale_factor, 
            x2_min / scale_factor, 
            x2_max / scale_factor
        ),
    }

# def create_projected_density_map(
#         positions: np.ndarray,
#         masses: np.ndarray,
#         normal_axes: str,
#         extent: tuple[float, ...] | None = None,
#         num_bins: int = 1000,
#         scale_factor: float = 1.0,
#     ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

#     i, j, k = get_slice_idxs_for_plane(normal_axes)

#     if extent is None:
#         x1_min, x1_max = positions[:, i].min(), positions[:, i].max()
#         x2_min, x2_max = positions[:, j].min(), positions[:, j].max()
#         x3_min, x3_max = positions[:, k].min(), positions[:, k].max()
#     else:
#         x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = extent

#     get_width = lambda lo, hi: (hi - lo) / num_bins

#     dx1 = get_width(x1_min, x1_max)
#     dx2 = get_width(x2_min, x2_max)
#     dx3 = get_width(x3_min, x3_max)
    

#     mass3D, edges = np.histogramdd(
#         sample=positions[:, (i, j, k)],
#         bins=num_bins,
#         range=(
#             (x1_min, x1_max),
#             (x2_min, x2_max),
#             (x3_min, x3_max),
#         ),
#         weights=masses
#     )
#     density3D = mass3D / (dx1 * dx2 * dx3)
#     projected_density = density3D.sum(axis=2)

#     return {
#         "density" : projected_density,
#         "x1_edges" : edges[0] / scale_factor,
#         "x2_edges" : edges[1] / scale_factor,
#         "x3_edges" : edges[2] / scale_factor,
#         "extent" : (
#             x1_min / scale_factor, 
#             x1_max / scale_factor, 
#             x2_min / scale_factor, 
#             x2_max / scale_factor
#         ),
#     }

def create_projected_overdensity_map(
        positions: np.ndarray,
        masses: np.ndarray,
        normal_axes: str,
        mean_density: float,
        extent: tuple[float, ...] | None = None,
        num_bins: int = 1000,
        bin_size: float | tuple[float, float, float] | None = None,
        scale_factor: float = 1.0,
    ) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:

    projected_density_map_data = create_projected_density_map(
        positions=positions,
        masses=masses,
        normal_axes=normal_axes,
        extent=extent,
        num_bins=num_bins,
        bin_size=bin_size,
        scale_factor=scale_factor,
    )
    projected_density_map = projected_density_map_data["density"]
    x3_edges = projected_density_map_data["x3_edges"]
    depth = (x3_edges[-1] - x3_edges[0]) * scale_factor

    overdensity_map = projected_density_map / (mean_density * depth)

    return {
        "overdensity" : overdensity_map,
        "x1_edges" : projected_density_map_data["x1_edges"],
        "x2_edges" : projected_density_map_data["x2_edges"],
        "x3_edges" : x3_edges,
        "extent" : projected_density_map_data["extent"],
        "depth" : depth,
    }

def is_valid_particle_query(
        query_radii: float | None, 
        num_neighbors: int | None
    ) -> bool:
    return (
        (query_radii is not None and num_neighbors is None) 
        or (query_radii is None and num_neighbors is not None)
    )

def single_tree_query(
        box_particles: BoxParticleCollection,
        coordinates: np.ndarray, 
        query_radii: float | None = None,
        num_neighbors: int | None = None
    ) -> dict[str, list[list[int]] | np.ndarray]:
    
    if query_radii is not None:
        return box_particles.properties.query(
            locations=coordinates, 
            radii=query_radii
        )
    else:
        return box_particles.properties.query(
            locations=coordinates, 
            num_neighbors=num_neighbors
        )
    

def batched_tree_query(
        box_particles: BoxParticleCollection,
        coordinates: np.ndarray, 
        query_radii: float | None = None,
        num_neighbors: int | None = None,
        use_vectorized: bool = False
    ) -> dict[str, list[list[int]] | np.ndarray]:

    if use_vectorized:
        return single_tree_query(
            box_particles=box_particles,
            coordinates=coordinates, 
            query_radii=query_radii,
            num_neighbors=num_neighbors
        )

    if not is_valid_particle_query(query_radii, num_neighbors):
        raise ValueError("Must specify either radius or num_neighbors")

    query_idxs, radii = [], []

    if query_radii is not None:
        for i, r_query in tqdm(
            enumerate(query_radii), 
            total=len(query_radii), 
            desc="Querying Halo Particles"
        ):
            query = single_tree_query(
                box_particles=box_particles,
                coordinates=coordinates[i], 
                query_radii=r_query,
                num_neighbors=num_neighbors
            )
            query_idxs.append(query.get("indices"))
            radii.append(query.get("distances"))

    else:

        for i, n_query in enumerate(num_neighbors):
            query = single_tree_query(
                box_particles=box_particles,
                coordinates=coordinates[i], 
                query_radii=query_radii,
                num_neighbors=n_query
            )
            query_idxs.append(query.get("indices"))
            radii.append(query.get("distances"))

    return {"indices": query_idxs, "distances": radii}

def get_single_halo_particles(
        box_particles: BoxParticleCollection,
        coordinates: np.ndarray, 
        group_velocities: np.ndarray,
        query_radii: float | None = None,
        num_neighbors: int | None = None,
        hubble: float | None = None,
        little_h: float | None = None,
        use_comoving: bool = True,
        run_sort_by_radial_distance: bool = True
    ) -> BoundedParticleCollection | None:

    if not is_valid_particle_query(query_radii, num_neighbors):
        raise ValueError("Must specify either radius or num_neighbors")

    query = single_tree_query(
        box_particles=box_particles,
        coordinates=coordinates, 
        query_radii=query_radii,
        num_neighbors=num_neighbors
    )

    particle_idxs, radii = query.get("indices"), query.get("distances")
    if use_comoving and radii is not None: 
        if not isinstance(radii, np.ndarray):
            radii = np.array([radii])
        radii *= box_particles.moment.scale_factor

    if particle_idxs.size == 0:
        return None


    try:
        phase_space_props = get_phase_space_properties(
            coordinates=box_particles.coordinates[particle_idxs],
            velocities=box_particles.properties.velocities[particle_idxs],
            box_size=box_particles.properties.box_size,
            center_of_mass=coordinates,
            group_velocity=group_velocities,
            hubble=hubble,
            little_h=little_h,
            precomputed_radial_distances=radii
        )
    except IndexError as err: 
        raise IndexError(
            f"Error in computing phase space properties: {err}"
        ) from err

    query_radii = query_radii if query_radii is not None else radii.max()

    return BoundedParticleCollection(
        volume=(4./3) * np.pi * query_radii**3,
            moment=box_particles.moment,
            identifier=ParticleIdentifier(box_particles.ids[particle_idxs]),
            properties=BoundedParticleProperties(
                mass_values=box_particles.properties.mass_values[particle_idxs],
                positions=phase_space_props["positions"],
                radial_distances=phase_space_props["radial_distances"],
                peculiar=PhaseSpaceProperties(
                    velocities=phase_space_props["peculiar_relative_velocities"],
                    accelerations=box_particles.properties.accelerations,
                    radial_velocities=phase_space_props["peculiar_radial_velocities"],
                    radial_accelerations=phase_space_props["peculiar_radial_accelerations"]
                ),
                actual=PhaseSpaceProperties(
                    velocities=phase_space_props["actual_relative_velocities"],
                    accelerations=box_particles.properties.accelerations,
                    radial_velocities=phase_space_props["actual_radial_velocities"],
                    radial_accelerations=phase_space_props["actual_radial_accelerations"]
                ),
                center_of_mass=coordinates,
                in_comoving_units=box_particles.properties.in_comoving_units,
                softening_length=box_particles.properties.softening_length,
                run_sort_by_radial_distance=run_sort_by_radial_distance
            )
        )

def get_enclosed_radii_from_radial_values(
        queried_radial_values: list[np.ndarray],
        scale_factor: float = 1.0,
        use_comoving: bool = True
    ) -> np.ndarray:

    return np.asarray([
        queried_values.max() * (scale_factor if use_comoving else 1.0)
        for queried_values in range(len(queried_radial_values))
    ])


def get_queried_sample_info(
        query: dict[str, list[list[int]] | np.ndarray],
        query_radii: float | Collection[float] | None = None, 
        scale_factor: float = 1.0, 
        use_comoving: bool = True
    ) -> dict[str, list[np.ndarray] | np.ndarray | None]:

    particle_idxs, radii = query.get("indices"), query.get("distances")
    if ((radii is not None) and all(r is None for r in radii)): radii = None
    if ((particle_idxs is None) or len(particle_idxs) == 0): 
        return {}

    num_particles_per_query, entries_to_keep = [], []
    for i, idxs in enumerate(particle_idxs):
        if len(idxs) > 0:
            num_particles_per_query.append(len(idxs))
            entries_to_keep.append(i)

    particle_idxs = [particle_idxs[i] for i in entries_to_keep]
    radii = (
        [radii[i] for i in entries_to_keep]
        if (radii is not None) else 
        None
    )
    query_radii = (
        np.array([query_radii[i] for i in entries_to_keep])
        if (query_radii is not None) else 
        None
    )

    num_halos = len(num_particles_per_query)

    query_weights = [1.0 / (num_halos * n) for n in num_particles_per_query]

    return {
        "particle_idxs": particle_idxs,
        "radii": radii,
        "query_radii": query_radii,
        "num_particles_per_query": num_particles_per_query,
        "query_weights": np.asarray(query_weights),
        "enclosed_radii": (
            query_radii 
            if radii is None else 
            get_enclosed_radii_from_radial_values(
                queried_radial_values=radii, 
                scale_factor=scale_factor, 
                use_comoving=use_comoving
            )
        )
    }

def validate_mass_weights(mass_weights: np.ndarray, final_mass: float) -> bool:

    is_sum_valid = np.isclose(mass_weights.sum(), final_mass)
    is_cum_sum_valid = np.isclose(np.cumsum(mass_weights)[-1], final_mass)

    normalized_mass_weights = mass_weights / final_mass
    is_sum_unity = np.isclose(normalized_mass_weights.sum(), 1.0)
    is_cum_sum_unity = np.isclose(np.cumsum(normalized_mass_weights)[-1], 1.0)

    return np.logical_and.reduce([
        is_sum_valid,
        is_cum_sum_valid,
        is_sum_unity,
        is_cum_sum_unity
    ])

# def validate_radial_weights(radial_weights: np.ndarray, final_radius: float) -> bool:

#     is_sum_valid = np.isclose(radial_weights.sum(), final_radius)
#     is_cum_sum_valid = np.isclose(np.cumsum(radial_weights)[-1], final_radius)

#     normalized_radial_weights = radial_weights / final_radius
#     is_sum_unity = np.isclose(normalized_radial_weights.sum(), 1.0)
#     is_cum_sum_unity = np.isclose(np.cumsum(normalized_radial_weights)[-1], 1.0)

#     return (
#         is_sum_valid and 
#         is_cum_sum_valid and 
#         is_sum_unity and 
#         is_cum_sum_unity
#     )

def get_renormalizing_weights(
        num_particles_per_query: list[int],
        particle_mass_value: float, 
        enclosed_radii: np.ndarray,
        query_weights: np.ndarray
    ) -> dict[str, int | float | np.ndarray]:

    enclosed_masses = np.array([
        particle_mass_value * n for n in num_particles_per_query
    ])

    # new_gadget_mass = np.nanmedian(enclosed_masses) / 1E10
    # new_radius = np.nanmedian(enclosed_radii)

    new_gadget_mass = np.nanmean(enclosed_masses) / 1E10
    new_radius = np.nanmean(enclosed_radii)

    normalized_radial_weights = new_radius / enclosed_radii

    # normalized_mass_weights = np.concatenate([
    #     new_gadget_mass * w * np.ones(n) 
    #     for n, w in zip(num_particles_per_query, query_weights)
    # ])

    # assert validate_mass_weights(normalized_mass_weights, float(new_gadget_mass))

    normalized_mass_weights = np.concatenate([
        new_gadget_mass * w * np.ones(n-1) 
        for n, w in zip(num_particles_per_query, query_weights)
    ])

    validation_mass = float(new_gadget_mass) - particle_mass_value/1E10
    assert validate_mass_weights(normalized_mass_weights, validation_mass)

    return {
        "new_radius": new_radius,
        "normalized_mass_weights": normalized_mass_weights,
        "normalized_radial_weights": normalized_radial_weights
    }


def initialize_property_collection(
        mass_weights: np.ndarray,
        new_radius: float,
        moment: Moment,
        center_of_mass: np.ndarray,
        in_comoving_units: bool = True,
        softening_length: float = 0.0,
        run_sort_by_radial_distance: bool = True,
        num_particles_per_query: list[int] | None = None
    ) -> dict:
    return {
        "volume": (4./3) * np.pi * new_radius**3,
        "moment": moment,
        "mass_values": mass_weights,
        "particle_ids": [],
        "positions": {
            "coordinates": [], "radial": [], "min_radius": [], "max_radius": []
        },
        "peculiar" : {
            "velocities": {"relative": [], "radial": []},
            "accelerations": {"relative": [], "radial": []}
        },
        "actual": {
            "velocities": {"relative": [], "radial": []},
            "accelerations": {"radial": []}
        },
        "center_of_mass": center_of_mass,
        "in_comoving_units": in_comoving_units,
        "softening_length": softening_length,
        "run_sort_by_radial_distance": run_sort_by_radial_distance,
        "num_particles_in_query": num_particles_per_query
    }

# def update_position_arrays(
#         position_arrays: dict[str, list],
#         phase_space_props: dict[str, np.ndarray],
#         radial_weight: float,
#         radial_mask: np.ndarray
#     ) -> None:


#     position_arrays["coordinates"].append(phase_space_props["positions"][radial_mask])

#     radial_distances = phase_space_props["radial_distances"][radial_mask]

#     r_min = radial_distances.min()

#     # Do radial weighting here
#     position_arrays["radial"].append(radial_distances * radial_weight)

def update_position_arrays(
        position_arrays: dict[str, list],
        phase_space_props: dict[str, np.ndarray],
        radial_mask: np.ndarray
    ) -> None:

    position_arrays["coordinates"].append(phase_space_props["positions"][radial_mask])
    radial_distances = phase_space_props["radial_distances"][radial_mask]
    position_arrays["radial"].append(radial_distances)
    position_arrays["min_radius"].append(radial_distances.min())
    position_arrays["max_radius"].append(radial_distances.max())


def scale_radial_distances(
        radial_distances: list[np.ndarray],
        min_radii: np.ndarray,
        max_radii: np.ndarray, 
        radial_weights: np.ndarray
    ) -> np.ndarray:

    r_median_min = 10.0**np.nanmean(np.log10(min_radii))
    r_median_max = 10.0**np.nanmean(np.log10(max_radii))

    # r_median_min = np.nanmedian(min_radii)
    # r_median_max = np.nanmedian(max_radii)
    median_diff = r_median_max - r_median_min

    transformed = []
    for i, r_values in enumerate(radial_distances):
        radial_range = max_radii[i] - min_radii[i]
        slopes = (np.asarray(r_values) - min_radii[i]) / radial_range
        transformed.append((r_median_min + slopes * median_diff) * radial_weights[i])
        # transformed.append((r_median_min + slopes * median_diff))

    return np.concatenate(transformed)

    # slopes = (radial_distances - min_radii) / (max_radii - min_radii) 

    # return (r_median_min + slopes * median_diff) * radial_weights


def update_peculiar_arrays(
        phase_space_props: dict[str, np.ndarray],
        peculiar_arrays: dict[str, dict[str, list]],
        box_particles: BoxParticleCollection,
        particle_locations: np.ndarray,
        radial_mask: np.ndarray
    ) -> None:

    peculiar_arrays["velocities"]["relative"].append(
        phase_space_props["peculiar_relative_velocities"][radial_mask]
    )
    peculiar_arrays["velocities"]["radial"].append(
        phase_space_props["peculiar_radial_velocities"][radial_mask]
    )
    peculiar_arrays["accelerations"]["radial"].append(
        phase_space_props["peculiar_radial_accelerations"][radial_mask]
    )

    if (
            box_particles.properties.accelerations is not None
            and box_particles.properties.accelerations.ndim > 0
    ):
        masked_locations = particle_locations[radial_mask]
        peculiar_arrays["accelerations"]["relative"].append(
            box_particles.properties.accelerations[masked_locations]
        )

def update_actual_arrays(
        phase_space_props: dict[str, np.ndarray],
        actual_arrays: dict[str, dict[str, list]],
        radial_mask: np.ndarray
    ) -> None:

    actual_arrays["velocities"]["relative"].append(
        phase_space_props["actual_relative_velocities"][radial_mask]
    )
    actual_arrays["velocities"]["radial"].append(
        phase_space_props["actual_radial_velocities"][radial_mask]
    )
    actual_arrays["accelerations"]["radial"].append(
        phase_space_props["actual_radial_accelerations"][radial_mask]
    )

    

def update_property_arrays(
        property_arrays: dict[str, list | dict[str, list | dict[str, list]]], 
        phase_space_props: dict[str, np.ndarray],
        box_particles: BoxParticleCollection,
        particle_locations: np.ndarray,
        # radial_weight: float,
    ) -> None:

    mask = phase_space_props["radial_distances"] > 0.0
    masked_locations = particle_locations[mask]

    property_arrays["particle_ids"].append(box_particles.ids[masked_locations])
    update_position_arrays(
        position_arrays=property_arrays["positions"], 
        phase_space_props=phase_space_props, 
        # radial_weight=radial_weight, 
        radial_mask=mask
    )
    update_peculiar_arrays(
        phase_space_props=phase_space_props,
        peculiar_arrays=property_arrays["peculiar"],
        box_particles=box_particles,
        particle_locations=particle_locations,
        radial_mask=mask
    )
    update_actual_arrays(
        phase_space_props=phase_space_props,
        actual_arrays=property_arrays["actual"],
        radial_mask=mask
    )


def concatenate_position_arrays(
        position_arrays: dict[str, list[np.ndarray]],
        normalized_radial_weights: np.ndarray,
        new_radius: float
    ) -> None:

    position_arrays["coordinates"] = np.concatenate(position_arrays["coordinates"])
    position_arrays["radial"] = scale_radial_distances(
        radial_distances=position_arrays["radial"],
        min_radii=np.asarray(position_arrays["min_radius"]),
        max_radii=np.asarray(position_arrays["max_radius"]),
        radial_weights=normalized_radial_weights
    )
    # assert np.isclose(position_arrays["radial"].max(), new_radius)

def concatenate_peculiar_arrays(
        peculiar_arrays: dict[str, dict[str, list | dict[str, list]]]
    ) -> None:

    peculiar_arrays["velocities"]["relative"] = np.concatenate(peculiar_arrays["velocities"]["relative"])
    peculiar_arrays["velocities"]["radial"] = np.concatenate(peculiar_arrays["velocities"]["radial"])
    peculiar_arrays["accelerations"]["radial"] = np.concatenate(peculiar_arrays["accelerations"]["radial"])

    if len(peculiar_arrays["accelerations"]["relative"]) > 0:
        peculiar_arrays["accelerations"]["relative"] = np.concatenate(peculiar_arrays["accelerations"]["relative"])
    else:
        peculiar_arrays["accelerations"]["relative"] = np.asarray([])


def concatenate_actual_arrays(
        actual_arrays: dict[str, dict[str, list | dict[str, list]]]
    ) -> None:

    actual_arrays["velocities"]["relative"] = np.concatenate(actual_arrays["velocities"]["relative"])
    actual_arrays["velocities"]["radial"] = np.concatenate(actual_arrays["velocities"]["radial"])
    actual_arrays["accelerations"]["radial"] = np.concatenate(actual_arrays["accelerations"]["radial"])
    

def concatenate_property_arrays(
        property_arrays: dict[str, list | dict[str, list | dict[str, list]]],
        new_radius: float, 
        radial_weights: np.ndarray
    ) -> None:

    concatenate_position_arrays(property_arrays["positions"], radial_weights, new_radius)
    concatenate_peculiar_arrays(property_arrays["peculiar"])
    concatenate_actual_arrays(property_arrays["actual"])


def get_population_averaged_halo_particles(
        box_particles: BoxParticleCollection,
        coordinates: np.ndarray, 
        group_velocities: np.ndarray,
        query_radii: float | Collection[float] | None = None,
        num_neighbors: int | None = None,
        hubble: float | None = None,
        little_h: float | None = None,
        use_vectorized: bool = False,
        use_comoving: bool = True,
        run_sort_by_radial_distance: bool = True,
    ) -> BoundedParticleCollection | None:

    query = batched_tree_query(
        box_particles=box_particles,
        coordinates=coordinates, 
        query_radii=query_radii,
        num_neighbors=num_neighbors,
        use_vectorized=use_vectorized
    )

    sample_info = get_queried_sample_info(query, query_radii)
    particle_idxs = sample_info.get("particle_idxs")
    radii = sample_info.get("radii")
    query_radii = sample_info.get("query_radii")
    enclosed_radii = sample_info.get("enclosed_radii")
    num_particles_per_query = sample_info.get("num_particles_per_query")
    query_weights = sample_info.get("query_weights")

    if not any((num_particles_per_query, query_weights)): return None
    
    weight_info = get_renormalizing_weights(
        num_particles_per_query=num_particles_per_query,
        particle_mass_value=box_particles.properties.masses[0],
        enclosed_radii=enclosed_radii,
        query_weights=query_weights
    )

    new_radius = weight_info.get("new_radius")
    normalized_mass_weights = weight_info.get("normalized_mass_weights")
    normalized_radial_weights = weight_info.get("normalized_radial_weights")

    properties = initialize_property_collection(
        mass_weights=normalized_mass_weights,
        new_radius=new_radius,
        moment=box_particles.moment,
        center_of_mass=coordinates,
        in_comoving_units=use_comoving,
        softening_length=box_particles.properties.softening_length,
        run_sort_by_radial_distance=run_sort_by_radial_distance,
        num_particles_per_query=num_particles_per_query
    )

    for i, idxs in tqdm(
            enumerate(particle_idxs),
            total=len(particle_idxs),
            desc="Extracting Queried Halo Particle Properties"
        ):

        phase_space_props = get_phase_space_properties(
            coordinates=box_particles.coordinates[idxs],
            velocities=box_particles.properties.velocities[idxs],
            box_size=box_particles.properties.box_size,
            center_of_mass=coordinates[i],
            group_velocity=group_velocities[i],
            hubble=hubble,
            little_h=little_h,
            precomputed_radial_distances=(
                np.asarray(radii[i]) 
                if radii is not None else 
                None
            )
        )

        update_property_arrays(
            property_arrays=properties,
            phase_space_props=phase_space_props,
            box_particles=box_particles,
            particle_locations=idxs,
            # radial_weight=normalized_radial_weights[i]
        )

    properties["particle_ids"] = np.concatenate(properties["particle_ids"])
    if properties["particle_ids"].size == 0:
        return None

    concat_start_time = time()
    
    concatenate_property_arrays(
        property_arrays=properties, 
        radial_weights=normalized_radial_weights, 
        new_radius=new_radius
    )

    concat_end_time = time()

    td = timedelta(seconds=concat_end_time - concat_start_time)
    print(f"Concatenation took {str(td)}")

    collection_start_time = time()

    collection = BoundedParticleCollection.from_dictionary(properties)

    collection_end_time = time()

    td = timedelta(seconds=collection_end_time - collection_start_time)
    print(f"Collection took {str(td)}")

    return collection

def get_interped_radii_by_num_particles_per_bin(
        particle_radii: np.ndarray, 
        bins_per_dex: int = 10,
        num_innermost_particles: int = 1
    ) -> np.ndarray:
    
    r_enc = np.hstack([particle_radii, [particle_radii[-1]]])
    n_enc = np.arange(len(r_enc)+1) + 1
    n_dex = np.log10(len(r_enc) / num_innermost_particles)
    num_bins = int(np.round(n_dex * bins_per_dex))

    log_r_enc, log_num = np.log10(r_enc), np.log10(n_enc)
    bin_idxs = 10**np.linspace(
        np.log10(num_innermost_particles), 
        np.log10(len(r_enc)-1), 
        num_bins,
        dtype=int
    )

    d_log_n = log_num[bin_idxs+1] - log_num[bin_idxs]
    slope = (log_r_enc[bin_idxs+1] - log_r_enc[bin_idxs])/d_log_n

    return 10**(log_r_enc[bin_idxs] + slope * d_log_n)


def get_mass_distribution(
        masses: np.ndarray, 
        radii: np.ndarray, 
        num_radial_points: int, 
        min_radius: float = 0.001,
        scaling_radius: float = 1.0,
        enclosed_masses: np.ndarray | None = None,
        return_in_physical_units: bool = True
    ) -> np.ndarray:

    if enclosed_masses is None:
        enclosed_masses = np.cumsum(masses)

    enclosed_density = (3.0 * enclosed_masses) / (4.0 * np.pi * radii**3.0)
    smallest_nonzero_radius = np.min(radii[radii > 0.0])

    interped_radii = np.logspace(
        np.log10(smallest_nonzero_radius / scaling_radius),
        np.log10(radii[-1] / scaling_radius),
        num_radial_points
    )

    enclosed_mass_interpolator = interp1d(
        np.log10(radii / scaling_radius), 
        np.log10(enclosed_masses / enclosed_masses[-1]),
        fill_value="extrapolate"
    )

    enclosed_density_interpolator = interp1d(
        np.log10(radii / scaling_radius), 
        np.log10(enclosed_density / enclosed_density[-1]),
        fill_value="extrapolate"
    )

    interp_enclosed_mass = 10.0**np.asarray(
        enclosed_mass_interpolator(np.log10(interped_radii))
    )
    interp_enclosed_density = 10.0**np.asarray(
        enclosed_density_interpolator(np.log10(interped_radii))
    )

    mass_slope = np.gradient(
        np.log10(interp_enclosed_mass), 
        np.log10(interped_radii)
    )
    density_profile = (interp_enclosed_density / 3.0) * mass_slope

    if return_in_physical_units:
        interped_radii *= scaling_radius
        interp_enclosed_mass *= enclosed_masses[-1]
        interp_enclosed_density *= enclosed_density[-1]
        density_profile *= enclosed_density[-1]

    return np.column_stack([
        interped_radii, 
        interp_enclosed_mass, 
        interp_enclosed_density,
        density_profile
    ])

def get_2D_phase_space_distribution(
        radii: np.ndarray, 
        velocities: np.ndarray, 
        masses: np.ndarray,
        num_r_bins: int = 200,
        num_v_bins: int = 200,
        v_linthresh: float = 1.0,
        v_linscale: float = 1.0,
        v_log_base: float = 10.0
    ) -> dict[str,np.ndarray]:

    r_min, r_max = np.min(radii[radii > 0]), np.max(radii[radii > 0])
    v_min, v_max = np.min(velocities[radii > 0]), np.max(velocities[radii > 0])
    # r_bins = np.linspace(r_min, r_max, num_r_bins + 1)
    v_bins = np.linspace(v_min, v_max, num_v_bins + 1)

    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), num_r_bins + 1)

    # Creating the 2D histogram with weights
    hist, r_edges, v_edges = np.histogram2d(
        radii[radii > 0], 
        velocities[radii > 0], 
        bins=[r_bins, v_bins], 
        density=True
        # weights=masses[radii > 0]
    )

    return {"density": hist.T, "r" : r_edges, "v" : v_edges}


def compute_velocity_profile(
        radii: np.ndarray, 
        velocities: np.ndarray, 
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])
    nonzero_r_bins, medians, twenty_fives, seventy_fives = [], [], [], []
    for i, r in enumerate(bin_centers):
        bin_weights = velocities[bin_indices == i]
        if len(bin_weights) > 0:
            nonzero_r_bins.append(r)
            medians.append(np.nanmedian(bin_weights))
            twenty_fives.append(np.nanpercentile(bin_weights, 25))
            seventy_fives.append(np.nanpercentile(bin_weights, 75))

    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        "median": np.asarray(medians),
        "25pct": np.asarray(twenty_fives),
        "75pct": np.asarray(seventy_fives),
    }

def compute_dispersion_profile(
        radii: np.ndarray, 
        velocities: np.ndarray, 
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])
    nonzero_r_bins, dispersions = [], []
    for i, r in enumerate(bin_centers):
        bin_weights = velocities[bin_indices == i]
        if len(bin_weights) > 0:
            nonzero_r_bins.append(r)
            dispersions.append(np.nanvar(bin_weights))

    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        "dispersion": np.asarray(dispersions),
    }

def compute_anisotropy_profile(
        radii: np.ndarray, 
        velocities: np.ndarray,
        radial_velocities: np.ndarray, 
        tangential_velocities: np.ndarray,
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])
    nonzero_r_bins, betas, beta_cyl = [], [], []
    for i, r in enumerate(bin_centers):
        velocity_bin_weights = velocities[bin_indices == i]
        radial_bin_weights = radial_velocities[bin_indices == i]
        tangential_bin_weights = tangential_velocities[bin_indices == i]
        if (len(radial_bin_weights) > 0) and (len(tangential_bin_weights) > 0):

            nonzero_r_bins.append(r)
            r_variance = np.nanvar(radial_bin_weights)
            t_variance = np.nanvar(tangential_bin_weights)
            betas.append(1.0 - 0.5 * (r_variance / t_variance))

            n = velocity_bin_weights.shape[0]
            T_ij = np.einsum(
                'ki,kj->ij', velocity_bin_weights, velocity_bin_weights
            ) 
            T_ij /= (n - 1)
            var_1, var_2, var_3 = np.sort(np.linalg.eigvals(T_ij)[::-1])

            beta_cyl.append(1.0 - 0.5 * (var_2 + var_3) / var_1)


    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        "anisotropy": np.asarray(betas),
        "anisotropy_cyl": np.asarray(beta_cyl),
    }


def compute_shape_profiles(
        positions: np.ndarray,
        radii: np.ndarray, 
        radial_bins: np.ndarray,
    ) -> dict[str, np.ndarray]:

    bin_indices = np.digitize(radii, radial_bins) - 1 
    bin_centers = np.sqrt(radial_bins[1:] * radial_bins[:-1])

    nonzero_r_bins, majors, intermediates, minors = [], [], [], []
    sphericity, polar_flat, equatorial_flat, triaxiality = [], [], [], []
    red_sphericity, red_polar_flat, red_equatorial_flat, red_triaxiality = [], [], [], []

    for i, r in enumerate(bin_centers):
        bin_positions = positions[bin_indices == i]
        if len(bin_positions) > 0:

            I_ij = np.einsum('ki,kj->ij', bin_positions, bin_positions)
            a, b, c = np.sqrt(np.sort(np.linalg.eigvals(I_ij))[::-1])
            s, q, p = (c/a), (b/a), (c/b)
            T = (1.0 - q**2) / (1.0 - s**2)
            x, y, z = bin_positions[:, 0], bin_positions[:, 1], bin_positions[:, 2]
            wk = np.sqrt(x**2 + (y/q)**2 + (z/s)**2)
            mask = np.nonzero(wk)[0]
            weighted_r = bin_positions[mask] / wk[mask][:, None]
            S_ij = np.einsum('ki,kj->ij', weighted_r, weighted_r)
            red_a, red_b, red_c = np.sqrt(np.sort(np.linalg.eigvals(S_ij))[::-1])
            red_s, red_q, red_p = (red_c/red_a), (red_b/red_a), (red_c/red_b)
            red_T = (1.0 - red_q**2) / (1.0 - red_s**2)

            nonzero_r_bins.append(r)
            majors.append(a)
            intermediates.append(b)
            minors.append(c)
            sphericity.append(s)
            polar_flat.append(q)
            equatorial_flat.append(p)
            triaxiality.append(T)
            red_sphericity.append(red_s)
            red_polar_flat.append(red_q)
            red_equatorial_flat.append(red_p)
            red_triaxiality.append(red_T)

    return {
        'radial_bins' : np.asarray(nonzero_r_bins),
        "major_axis": np.asarray(majors),
        "intermediate_axis": np.asarray(intermediates),
        "minor_axis": np.asarray(minors),
        "sphericity": np.asarray(sphericity),
        "polar_flat": np.asarray(polar_flat),
        "equatorial_flat": np.asarray(equatorial_flat),
        "triaxiality": np.asarray(triaxiality),
        "red_sphericity": np.asarray(red_sphericity),
        "red_polar_flat": np.asarray(red_polar_flat),
        "red_equatorial_flat": np.asarray(red_equatorial_flat),
        "red_triaxiality": np.asarray(red_triaxiality),
    }


def _aa__as_vec3(value: float | int | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Return a length-3 numpy array from scalar or 3-vector input."""
    arr = np.asarray(value, dtype='f8')
    if arr.ndim == 0:
        return np.array([arr, arr, arr], dtype='f8')
    if arr.shape == (3,):
        return arr.astype('f8', copy=False)
    raise ValueError(f"Expected scalar or length-3 for box size, got shape {arr.shape}")

def _aa__ensure_positions_xyz(positions: np.ndarray) -> np.ndarray:
    """Ensure positions are an (N, 3) float array."""
    pos = np.asarray(positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N,3); got {pos.shape}")
    return pos.astype('f8', copy=False)

def one_plus_delta_pmesh(
        positions: np.ndarray,
        box_size: float | tuple[float, ...] | np.ndarray,
        nmesh: int | tuple[int, int, int],
        *,
        resampler: str = 'cic',
        interlaced: bool = False,
        weights: np.ndarray | None = None,
        dtype: str = 'f4',
        comm=None,
    ) -> np.ndarray:
    """
    Build a 3D (1+δ) density cube using pmesh from particle positions.

    Parameters
    ----------
    positions : (N,3) array
        Particle positions in the same units as `box_size` within [0, L) per axis.
    box_size : float or (3,) array
        Box length(s) (same units as `positions`).
    nmesh : int or (3,) tuple
        Number of mesh cells per axis.
    resampler : {'ngp','cic','tsc','pcs','lanczos2','lanczos3'}
        Mass–assignment window for pmesh.
    interlaced : bool
        If True, average two half-cell–shifted paints to reduce aliasing.
    weights : (N,) array or None
        Optional particle weights (defaults to 1).
    return_field : bool
        If True, return the distributed pmesh `RealField`; else return a (local) ndarray.
    dtype : str
        Floating dtype for mesh storage (default 'f4').
    comm : mpi4py communicator or None
        Passed to ParticleMesh; None runs serial.

    Returns
    -------
    numpy.ndarray or pmesh.pm.RealField
        The 1+δ field; collective mean is 1.
    """

    pos = _aa__ensure_positions_xyz(positions)
    L = _aa__as_vec3(box_size)
    if isinstance(nmesh, int):
        Nmesh = (int(nmesh), int(nmesh), int(nmesh))
    else:
        Nmesh = tuple(int(x) for x in nmesh)

    # lazy: pmesh is deliberately undeclared pending its redistribution review
    from pmesh.pm import ParticleMesh

    pm = ParticleMesh(BoxSize=L, Nmesh=Nmesh, dtype=dtype, comm=comm)

    if weights is None:
        weights = np.ones(pos.shape[0], dtype=dtype)
    else:
        weights = np.asarray(weights).astype(dtype, copy=False)
        if weights.shape[0] != pos.shape[0]:
            raise ValueError("weights must have same length as positions")

    if not interlaced:
        field = pm.paint(pos, mass=weights, resampler=resampler)
    else:
        real1 = pm.paint(pos, mass=weights, resampler=resampler)
        shift = pm.affine.shift(0.5)
        real2 = pm.paint(pos, mass=weights, resampler=resampler, transform=shift)
        field = (real1 + real2) * 0.5

    # Normalize to 1+δ by collective mean
    try:
        mean = field.cmean()
    except Exception:  # single-rank fallback
        mean = field[...].mean()
    if mean == 0:
        mean = 1.0
    field /= mean

    return np.asarray(field[...], dtype=np.float32)


def cic_sample(
        coords: np.ndarray,
        ngrid: int,
        boxsize: float = 1.0,
        weights: np.ndarray | None = None,  
        return_density: bool = False,
        return_delta: bool = False,
        dtype: type = np.float64,
    ) -> tuple[np.ndarray, float]:
    """
    Cloud-in-Cell (CIC) mass assignment on a cubic periodic box.

    Parameters
    ----------
    coords : (N, 3) array
        Particle positions in a periodic cube. Units must match `boxsize`.
        Assumed columns: x,y,z.
    ngrid : int
        Number of cells per dimension (grid shape is (ngrid,ngrid,ngrid)).
    boxsize : float, optional
        Physical box length of the cube. Default is 1.0.
    weights : (N,) array, optional
        Optional particle weights/masses. If None, each particle has weight 1.
    return_density : bool, optional
        If True, converts counts to mass density [weight / (length)^3].
    return_delta : bool, optional
        If True, returns density contrast δ = ρ/⟨ρ⟩ − 1.
        (Implies `return_density=True`.)
    dtype : numpy dtype, optional
        Accumulator dtype. Default float64 for numerical stability.

    Returns
    -------
    grid : (ngrid, ngrid, ngrid) array
        CIC-assigned field. Interpretation:
        - if neither return_density nor return_delta: cell counts (sum of weights)
        - if return_density and not return_delta: mass density ρ
        - if return_delta: density contrast δ
    dx : float
        Grid spacing (boxsize / ngrid).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be an (N, 3) array")

    N = coords.shape[0]
    if weights is None:
        w = np.ones(N, dtype=dtype)
    elif weights.shape == (N,):
        w = weights.astype(dtype, copy=False)

    else:
        raise ValueError("weights must be shape (N,)")
    # Periodic wrap, grid spacing, and cell indices
    dx = boxsize / ngrid
    pos = np.mod(coords, boxsize)  # ensure in [0, boxsize)
    u = pos / dx                   # positions in grid units
    i = np.floor(u).astype(np.int64)  # base cell index
    fx = (u - i).astype(dtype)        # fractional offset within cell [0,1)

    # 1D CIC weights for each axis
    w0 = 1.0 - fx  # weight to lower index
    w1 = fx        # weight to upper index

    # Prepare accumulator
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=dtype)

    # Distribute to 8 neighbors: (bx,by,bz) in {0,1}^3
    # We vectorize by looping only over the 8 combinations (tiny loop),
    # while the particle dimension stays vectorized.
    for bx in (0, 1):
        wx = w0[:, 0] if bx == 0 else w1[:, 0]
        ix = (i[:, 0] + bx) % ngrid
        for by in (0, 1):
            wy = w0[:, 1] if by == 0 else w1[:, 1]
            iy = (i[:, 1] + by) % ngrid
            for bz in (0, 1):
                wz = w0[:, 2] if bz == 0 else w1[:, 2]
                iz = (i[:, 2] + bz) % ngrid

                contrib = (wx * wy * wz) * w
                # Accumulate into cells
                np.add.at(grid, (ix, iy, iz), contrib)

    cell_vol = dx**3
    rho = grid / cell_vol

    if return_delta:
        mean_rho = (w.sum() / (boxsize**3))
        # Avoid divide-by-zero if no particles/weights:
        grid = (rho / mean_rho - 1.0) if mean_rho > 0 else np.zeros_like(rho)
    elif return_density:
        grid = rho

    return grid, dx