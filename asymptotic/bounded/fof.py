from __future__ import annotations

import numpy as np, pdb

from pathlib import Path
from tqdm.auto import tqdm
from attrs import define, field
from collections.abc import Collection
from collections import OrderedDict, defaultdict


from ..simulation.moments import Moment, MomentsInTime
from ..cosmo.model import Cosmology
from ..mass_def.data import MassDefinition, MassDefinitions
from ..simulation.evo import EvolutionData
from ..simulation.sub_box import SubBoxes
from ..utils.get_data import (
    get_fof_subfind_data, 
    load_physical_catalog_data,
    get_all_fof_subfind_catalogs
)
from .subhalo import Subhalos, SubhaloMap, SubhaloMapping, SubhaloMappingEvo
from .objects import BoundedObject, BoundedObjectEvolutionData
from .population import (
    HaloPopulation, HaloPopulationEvolutionData
)
from ..distribution.slope import get_so_values
from ..particles.collection import (
    BoundedParticleCollection, BoxParticleCollection
)
from ..particles.shapes import (
    ShapeParameterData, 
    ShapeProfileData, 
    PopulationShapeParameterData
)
from ..particles.anisotropy import (
    AnisotropyParameterData, 
    AnisotropyProfileData, 
    PopulationAnisotropyParameterData
)

DEFAULT_LINKING_LENGTH = 0.28
DEFAULT_MIN_RESOLVED_MASS = 0.0
DEFAULT_CUMULATIVE_COUNT = 0
DEFAULT_REGION_SCALING = 20.0
DEFAULT_SCALING_MASS_DEF = "200c"


@define(slots=True)
class FOFGroup(BoundedObject):

    region_radius: float
    subhalos: Subhalos | None = field(default=None, repr=False)
    # subhalo_mapping: SubhaloMapping | None = field(default=None, repr=False)
    subhalo_map: SubhaloMap | None = field(default=None, repr=False)
    particles: BoundedParticleCollection | None = field(default=None, repr=False)

    shape: ShapeParameterData | None = field(default=None, repr=False)
    anisotropy: AnisotropyParameterData | None = field(default=None, repr=False)

    has_bounded_region: bool = field(default=False, repr=False)
    has_physical_boundaries: bool = field(default=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.mass_defs.fof is None:
            raise ValueError("FOF mass definition must be provided")

    def __repr__(self) -> str:
        mass = (
            10.0**self.mass_defs.mean200.mass
            if self.mass_defs.mean200 is not None else
            10.0**self.mass_defs.fof.mass
        )
        return (
            f"FOFGroup(id_number={self.id_number}, "
            f"n_subhalos={self.subhalo_count}, "
            f"M200m={mass:.3e} Msun/h)"
        )
        
    @property
    def has_subhalos(self) -> bool:
        return self.subhalos is not None

    @property
    def mass(self) -> float:
        if self.mass_defs.bound is None:
            raise ValueError("Bound mass must be provided")
        return self.mass_defs.bound.mass
    
    @property
    def radius(self) -> float | None:
        if self.mass_defs.bound is None:
            raise ValueError("Bound radius must be provided")
        return self.mass_defs.bound.radius
    
    @property
    def circular_velocity(self) -> float | None:
        if self.mass_defs.bound is None:
            raise ValueError("Bound velocity must be provided")
        return self.mass_defs.bound.velocity
    
    @property
    def comoving_region_radius(self) -> float:
        return self.region_radius / self.moment.scale_factor
    
    def set_particles(
            self, 
            box_particles: BoxParticleCollection,
            r_query: float | None = None,
            cosmo: Cosmology | None = None,
            include_subhalos: bool = False,
            use_comoving: bool = True,
            run_sort_by_radial_distance: bool = True
        ) -> None:
            
        if cosmo is None:
            raise ValueError(
                "Cosmology must be provided to compute radial properties"
            )
        
        coordinates = (
            np.asarray(self.comoving_coordinate)
            if use_comoving else
            np.asarray(self.coordinate)
        )

        if r_query is None:
            r_query = (
                self.comoving_region_radius
                if use_comoving else
                self.region_radius
            )

        try:
            self.particles = BoundedParticleCollection.from_box_particles(
                box_particles=box_particles,
                group_coordinates=np.asarray(coordinates),
                group_velocities=np.asarray(self.velocity),
                query_radii=r_query,
                cosmo=cosmo,
                use_comoving=use_comoving,
                run_sort_by_radial_distance=run_sort_by_radial_distance
            )
        except ValueError as error:
            print(f"FOFGroup {self.id_number}: {error}")
            self.particles = None


        if include_subhalos and (self.subhalos is not None):
            print("Collecting subhalo particles")
            self.subhalos.set_particles(
                box_particles=box_particles,
                cosmo=cosmo,
                use_comoving=use_comoving
            )


    def set_physical_boundaries(
            self, 
            cosmo: Cosmology,
            min_rvir: float = 0.03,
            max_rvir: float = 10.0,
            num_radial_points: int = 100,  
            size_in_dex: float = 1.0,
            poly_order: int = 4,
            mode: str = "nearest",
            get_virial_boundaries: bool = True,
            set_bound_particle_boundaries: bool = False,
        ) -> None:

        if self.particles is None:
            raise ValueError("Particles must be set to compute boundaries")

        region_mass_distribution = self.particles.get_filtered_mass_distribution(  
            size_in_dex=size_in_dex,
            poly_order=poly_order,
            mode=mode,
            num_bins=num_radial_points,
            radii_are_pre_sorted=True
        )


        # if get_virial_boundaries:

        #     so_values = get_so_values(
        #         profiles=region_mass_distribution,
        #         z=self.moment.redshift,
        #         cosmo=cosmo
        #     )

        #     self.mass_defs.virial = MassDefinition(
        #         mapping=self.mass_defs.fof.mapping,
        #         mass=np.log10(so_values["M"]["vir"]),
        #         radius=so_values["R"]["vir"],
        #         hubble=self.mass_defs.fof.hubble,
        #     )
        #     self.mass_defs.virial.density = so_values["rho"]["vir"]

        Rmin = min_rvir * self.mass_defs.virial.radius
        Rmax = max_rvir * self.moment.scale_factor * self.mass_defs.virial.radius

        self.mass_defs.set_physical_mass_defs(
            profile_array=region_mass_distribution,
            min_radius=Rmin,
            max_radius=Rmax
        )

        if set_bound_particle_boundaries:
            self.particles = self.particles.get_radial_subset(
                radius=self.mass_defs.bound.radius  
            )
            self.has_bounded_region = True

        self.has_physical_boundaries = True
    @property
    def subhalo_count(self) -> int:
        return len(self.subhalos) if self.has_subhalos else 0

    @property
    def subhalo_ids(self) -> np.ndarray | None:
        if self.subhalos is None:
            print("This FOF group does not contain any subhalos")
            return None
        return self.subhalos.ids


    @property
    def lead_subhalo_id(self) -> int:
        if self.subhalo_map is None:
            raise ValueError("No subhalo mapping is present in this instance")
        
        return (
            self.subhalo_map.first_subhalo_id
            if self.subhalo_map.first_subhalo_id is not None else
            -1
        )

    
    @property
    def contained_subhalo(self) -> np.ndarray | None:
        if self.subhalos is None:
            print("This FOF group does not contain any subhalos")
            return None
        return self.subhalos.ids

    
    @property
    def descendant_subhalo_ids(self) -> np.ndarray:
        if self.subhalo_map is None:
            raise ValueError("No subhalo mapping is present in this instance")
        return (
            self.subhalo_map.descendant_ids
            if self.subhalo_map.descendant_ids is not None else
            np.array([-1], dtype=int)
        )
    
    @property
    def progenitor_subhalo_ids(self) -> np.ndarray:
        if self.subhalo_map is None:
            raise ValueError("No subhalo mapping is present in this instance")
        return (
            self.subhalo_map.progenitor_ids
            if self.subhalo_map.progenitor_ids is not None else
            np.array([-1], dtype=int)
        )
    
    
    def get_particles(
            self, box_particles: BoxParticleCollection, 
            r_query: float | None = None,
            cosmo: Cosmology | None = None,
            include_subhalos: bool = False,
            use_comoving: bool = True,
            run_sort_by_radial_distance: bool = True
        ) -> BoundedParticleCollection:

        if self.particles is None:
            self.set_particles(
                box_particles=box_particles,
                r_query=r_query,
                cosmo=cosmo,
                include_subhalos=include_subhalos,
                use_comoving=use_comoving,
                run_sort_by_radial_distance=run_sort_by_radial_distance
            )
        return self.particles


    # @property
    # def spin(self) -> float:
    #     return float(self.get_spin(get_particle_subset=(not self.has_bounded_region)))
    
    # @property
    # def anisotropy(self) -> float:
    #     return float(self.get_anisotropy(get_particle_subset=(not self.has_bounded_region)))
    
    # @property
    # def cylindrical_anisotropy(self) -> float:
    #     return float(
    #         self.get_cylindrical_anisotropy(get_particle_subset=(not self.has_bounded_region))
    #     )
    
    def anisotropy_profile(self, num_radial_points: int = 50) -> np.ndarray:
        return self.get_anisotropy_profile(
            num_radial_points=num_radial_points,
            get_particle_subset=(not self.has_bounded_region)
        )

    @property
    def shape_parameters(self) -> dict[str, float]:
        return self.get_shape_parameters(
            get_particle_subset=(not self.has_bounded_region)
        )
    
    @property
    def reduced_shape_parameters(self) -> dict[str, float]:
        return self.get_reduced_shape_parameters(
            get_particle_subset=(not self.has_bounded_region)
        )
    
    def shape_profile(self, num_radial_points: int = 50) -> np.ndarray:
        return self.get_shape_profile(
            num_radial_points=num_radial_points,
            get_particle_subset=(not self.has_bounded_region)
        )
    


@define(slots=True)
class FOFGroupEvoData(BoundedObjectEvolutionData):
    data: OrderedDict[int, FOFGroup]


    def __repr__(self) -> str:
        return super().__repr__()       

    def _get_bounded_object_with_particles(
            self, snapshot_id: int,
            box_particles: BoxParticleCollection | None = None,
            cosmo: Cosmology | None = None
        ) -> FOFGroup:

        group = self.data[snapshot_id]

        if group.particles is None:
            if (box_particles is None):
                raise ValueError(f"No particles found for {snapshot_id = }")
            elif (cosmo is None):
                raise ValueError("Cosmology must be provided to compute radial properties")
            else:
                group.set_particles(
                    box_particles=box_particles, 
                    cosmo=cosmo,
                    use_comoving=False
                )

        if not group.has_physical_boundaries:
            group.set_physical_boundaries(cosmo=cosmo)

        return group



@define(slots=True)
class FOFGroups(HaloPopulation): 
    linking_length_factor: float # set this in config file (maybe perform the read in there too)
    
    subhalos: Subhalos | None
    subhalo_mapping: SubhaloMapping

    spins: np.ndarray | None = field(default=None, repr=False)

    region_scaling: float = field(default=DEFAULT_REGION_SCALING, repr=False)
    scaling_def: str = field(default=DEFAULT_SCALING_MASS_DEF, repr=False)

    sub_boxes: SubBoxes | None = field(default=None, repr=False)

    min_resolved_mass: float = field(default=DEFAULT_MIN_RESOLVED_MASS)
    min_count_above_max_bin_hist: int = field(default=DEFAULT_CUMULATIVE_COUNT)

    shapes: PopulationShapeParameterData | None = field(default=None)
    anisotropies: PopulationAnisotropyParameterData | None = field(default=None)

    has_physical_boundaries: bool = field(default=False, repr=False)
    def __repr__(self) -> str:
        n_subhalos = self.subhalos.n_subhalos if self.subhalos is not None else 0
        return (
            f"FOFGroups(n_groups={self.n_groups:,} , "
            f"n_subhalos={n_subhalos:,} , "
            f"b={self.linking_length_factor})"
        )
    
    def __getitem__(self, group_id: int) -> FOFGroup:
        
        location = np.where(self.ids == group_id)[0]

        if len(location) == 0:
            raise IndexError(f"Group with id {group_id} not found")
        
        if len(location) > 1:
            raise ValueError(f"Multiple groups with id {group_id} found")
        
        halo_loc, halo_idx = location[0], int(self.ids[location[0]])

        mapping = self.subhalo_mapping.get_subset(halo_idx)

        return FOFGroup(
            id_number=halo_idx,
            coordinate=self.positions[halo_loc],
            velocity=self.velocities[halo_loc],
            particle_count=int(self.particle_counts[halo_loc]),
            region_radius=float(self.region_radii[halo_loc]),
            moment=self.moment,
            mass_defs=self.mass_defs.get_mass_defs_for_object(halo_loc),
            subhalos=(
                self.subhalos.get_subset(mapping.subhalo_ids)
                if self.has_subhalos and (mapping.subhalo_ids.size > 0) else
                None
            ),
            spin=self.spins[halo_loc] if self.spins is not None else None,
            subhalo_map=self.subhalo_mapping[halo_idx],
            shape=self.shapes[halo_loc] if self.shapes is not None else None,
            anisotropy=self.anisotropies[halo_loc] if self.anisotropies is not None else None,
            has_physical_boundaries=self.has_physical_boundaries
        )

    
    def get_subset(self, group_ids: int | Collection[int]) -> FOFGroups:
        if not isinstance(group_ids, np.ndarray):
            group_ids = np.array(group_ids)

        locations = np.isin(self.ids, group_ids)

        if ((locations.size == 0) or (locations.sum() == 0)):
            raise IndexError("No groups found with the given ids")

        halo_idxs = self.ids[locations]
        mapping = self.subhalo_mapping.get_subset(halo_idxs)

        assert np.array_equal(mapping.group_ids, halo_idxs)

        subhalos = (
            self.subhalos.get_subset(mapping.subhalo_ids)
            if self.has_subhalos and (mapping.subhalo_ids.size > 0) else 
            None
        )

        if subhalos is None: 
            assert mapping.subhalo_ids.size == 0

        assert np.array_equal(mapping.subhalo_ids, subhalos.ids)

        return FOFGroups(
            ids=halo_idxs,
            positions=self.positions[locations],
            velocities=self.velocities[locations],
            particle_counts=self.particle_counts[locations],
            moment=self.moment,
            spins=self.spins[locations] if self.spins is not None else None,
            mass_defs=self.mass_defs.get_subset(locations),
            linking_length_factor=self.linking_length_factor,
            region_scaling=self.region_scaling,
            scaling_def=self.scaling_def,
            subhalos=subhalos,
            subhalo_mapping=mapping,
            shapes=self.shapes[locations] if self.shapes is not None else None,
            anisotropies=self.anisotropies[locations] if self.anisotropies is not None else None
        )

    def get_subset_by_index(self, idxs: int | Collection[int]) -> FOFGroups:
        if not isinstance(idxs, np.ndarray):
            idxs = np.array(idxs)
        return self.get_subset(self.ids[idxs])

    def get_resolved_groups(self, mass_def_key: str, min_resolved_mass: float) -> FOFGroups:
        mask = self.mass_defs[mass_def_key].mass >= np.log10(min_resolved_mass)
        return self.get_subset(self.ids[mask])
    
    def get_subset_in_mass_range(self, mass_def_key: str, min_mass: float, max_mass: float) -> FOFGroups:
        mask = np.logical_and(
            self.mass_defs[mass_def_key].mass >= np.log10(min_mass),
            self.mass_defs[mass_def_key].mass <= np.log10(max_mass)
        )
        return self.get_subset(self.ids[mask])

    def get_by_subhalos(
            self, subhalo_ids: int | Collection[int], return_as_groups: bool = False
        ) -> FOFGroup | FOFGroups:

        if self.subhalos is None:
            raise ValueError("No subhalos are present in this data")

        if not isinstance(subhalo_ids, np.ndarray):
            subhalo_ids = np.array(subhalo_ids)

        group_ids = self.subhalo_mapping.get_group_ids_from_subhalo_ids(subhalo_ids)

        if (len(group_ids) == 1) and (not return_as_groups):
            return self[group_ids[0]]
        
        return self.get_subset(group_ids)


    def get_subset_on_subhalo_particle_counts(self, min_count: int) -> FOFGroups:
        idxs = np.where(self.lead_subhalos.particle_counts >= min_count)[0]
        return self.get_subset_by_index(idxs)


    def setup_sub_boxes(self, box_size: float, num_splits: int) -> None:
        self.sub_boxes = SubBoxes.from_fof_groups(
            box_size=box_size,
            num_splits=num_splits,
            group_ids=self.ids,
            group_positions=self.comoving_positions,
            subhalo_ids=self.subhalo_mapping.subhalo_ids,
            subhalo_positions=self.subhalos.comoving_positions,
        )

    @property
    def n_groups(self) -> int:
        return len(self)
    
    @property
    def min_group_size(self) -> int:
        return self.particle_counts.min()
    
    @property
    def max_group_size(self) -> int:
        return self.particle_counts.max()
    
    @property
    def has_subhalos(self) -> bool:
        return self.subhalos is not None
    
    
    def get_linking_length(self, box_length: float, num_particles: int) -> float:
        ''' Linking length for FOF Groups in units of Mpc/h'''
        mean_interparticle_separation = (box_length / num_particles) ** (1 / 3)
        return self.linking_length_factor * mean_interparticle_separation

    
    def sample(self, n_samples: int, seed: int | None = None) -> FOFGroups:
        if n_samples > len(self):
            raise ValueError("Number of samples must be less than the number of groups")
        # Local RNG so the draw is reproducible when a seed is given. seed=None
        # falls back to fresh OS entropy (the historical unseeded behavior).
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self), size=n_samples, replace=False)
        return self.get_subset_by_index(idxs)

    def get_top_n_by_mass(self, mass_def: str, n: int) -> FOFGroups:
        masses = self.mass_defs[mass_def].mass
        # Replace NaNs with -infinity so they are not chosen among the top n
        safe_masses = np.where(np.isnan(masses), -np.inf, masses)
        top_n_indices = np.argsort(safe_masses)[-n:]
        return self.get_subset(self.ids[top_n_indices])
        
    def get_bottom_n_by_mass(self, mass_def: str, n: int) -> FOFGroups:
        masses = self.mass_defs[mass_def].mass
        # Replace NaNs with +infinity so they are not chosen among the bottom n
        safe_masses = np.where(np.isnan(masses), np.inf, masses)
        bottom_n_indices = np.argsort(safe_masses)[:n]
        return self.get_subset(self.ids[bottom_n_indices])

    
    @property
    def region_radii(self) -> np.ndarray:
        # time_scaling = max(1.0, self.moment.scale_factor)
        scaling = self.region_scaling * self.moment.scale_factor
        return scaling * self.mass_defs[self.scaling_def].radius
    
    @property
    def comoving_region_radii(self) -> np.ndarray:
        return self.region_radii / self.moment.scale_factor

    @property
    def lead_subhalo_ids(self) -> np.ndarray:
        return self.subhalo_mapping.first_subhalo_ids

    @property
    def lead_subhalos(self) -> Subhalos:
        if self.subhalos is None:
            raise ValueError("No subhalos are present in this data")
        return self.subhalos.get_subset(self.lead_subhalo_ids)
    
    @property
    def descendant_subhalo_ids(self) -> np.ndarray:
        return self.subhalo_mapping.get_descendant_ids_from_group_ids(self.ids)
    
    @property
    def progenitor_subhalo_ids(self) -> np.ndarray:
        return self.subhalo_mapping.get_progenitor_ids_from_group_ids(self.ids)
    
    @classmethod
    def from_catalog(
            cls,
            catalog_dir: Path,
            snapshot_id: int,
            cosmo: Cosmology,
            b: float,
            min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
            min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
            region_scaling: float = DEFAULT_REGION_SCALING,
            scaling_def: str = DEFAULT_SCALING_MASS_DEF,
            is_batched: bool = False,
            in_comoving: bool = True,
            use_full_catalog: bool = False,
            full_catalog_dir: Path | None = None,
        ) -> FOFGroups:

        data = get_fof_subfind_data(
            directory=catalog_dir,
            snap_idx=snapshot_id,
            is_batched=is_batched,
            in_comoving=in_comoving,
            display_if_empty=False,
            use_full_catalog=use_full_catalog,
            full_catalog_dir=full_catalog_dir,
        )

        try:
            scale_factor = data["fof"]["scale_factor"]
        except KeyError as error:
            raise ValueError(f"No FOF data found on {snapshot_id = }") from error


        moment = Moment(snapshot_id=snapshot_id, scale_factor=scale_factor)

        moment.add_times(cosmo)

        return cls.from_catalog_data(
            data=data,
            moment=moment,
            hubble=cosmo.hubble_parameter(z=moment.redshift),
            b=b,
            region_scaling=region_scaling,
            scaling_def=scaling_def,
            min_resolved_mass=min_resolved_mass,
            min_count_above_max_bin_hist=min_count_above_max_bin_hist
        )

    @classmethod
    def from_catalog_data(
            cls, data: dict,
            moment: Moment,
            hubble: float,
            b: float = DEFAULT_LINKING_LENGTH,
            min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
            min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
            region_scaling: float = DEFAULT_REGION_SCALING,
            scaling_def: str = DEFAULT_SCALING_MASS_DEF,
        ) -> FOFGroups:

        mass_defs = MassDefinitions.from_fof_data(data["fof"], hubble)

        # Auto-populate the solo-derived mass_def slots when the cleaned
        # dict carries them (use_full_catalog=True upstream). Detection
        # is the presence of any "masses_<descriptor>" key — see
        # MassDefinitions.add_full_catalog_solo_data for the descriptor
        # set. This removes the need for a manual add_solo_boundaries(...)
        # call after construction; FOFGroups.from_catalog(..., use_full_catalog=True)
        # comes back with splashback / bound / lambda / hydrostatic /
        # turnaround / infall / four_scale / scale / max_circ already wired.
        cleaned_fof = data["fof"]
        if any(k.startswith("masses_") and k not in {"masses", "masses200c",
                                                     "masses200m", "masses500c"}
               for k in cleaned_fof):
            mass_defs.add_full_catalog_solo_data(
                cleaned_fof_data=cleaned_fof,
                fof_group_ids=cleaned_fof["group_ids"],
            )

        return cls(
            ids=cleaned_fof["group_ids"],
            positions=cleaned_fof["positions"],
            velocities=cleaned_fof["velocities"],
            particle_counts=cleaned_fof["particle_counts"],
            linking_length_factor=b,
            moment=moment,
            mass_defs=mass_defs,
            subhalos=Subhalos.from_catalog_data(data, moment, hubble),
            region_scaling=region_scaling,
            scaling_def=scaling_def,
            subhalo_mapping=SubhaloMapping.from_data(data),
            min_resolved_mass=min_resolved_mass,
            min_count_above_max_bin_hist=min_count_above_max_bin_hist
        )

    def get_by_mass_bin(self, mass_def: str) -> dict[int, FOFGroups]:
        mass_values = getattr(self.mass_defs, mass_def).mass
        mass_bins = np.unique(np.asarray(mass_values).astype(int))
        bin_locs = np.digitize(mass_values, mass_bins)
        return {
            mass_bin: self.get_subset(np.where(bin_locs == (i + 1))[0])
            for i, mass_bin in enumerate(mass_bins)
        }
    
    def add_physical_mass_defs_from_spin_and_shapes_data(
            self, spin_and_shapes_data: dict[int, dict[str, np.ndarray]],
        ) -> None:
        self.mass_defs.add_physical_catalog_data(
            physical_catalog=spin_and_shapes_data,
            fof_group_ids=self.ids
        )


    def add_solo_boundaries(
            self,
            solo_boundaries_path: Path,
            mdef: str,
        ) -> None:
        """Inject mass-distribution + kinematic boundaries from a
        save_solo_profiles_phusis
        ``..._solo_boundaries_..._snapNNN_{mdef}.hdf5``.

        Forwards into :meth:`MassDefinitions.add_solo_boundary_data` after
        flattening the loader's nested per-(snap, mdef) layout into the
        flat ``{label}_M`` / ``{label}_R`` form that method expects. Sets
        ``has_physical_boundaries=True`` on success, matching the
        invariant maintained by the spin-and-shapes catalog path.

        Label re-mapping between solo HDF5 milestone names and the
        ``mass_defs`` slot they populate (see
        :meth:`MassDefinitions.add_solo_boundary_data` for the full
        loop):

            splashback   -> mass_defs["sp"]
            scale        -> mass_defs["scale"]
            four_scale   -> mass_defs["<4s"]
            vmax         -> mass_defs["max"]
            target       -> mass_defs["bound"]
            hs           -> mass_defs["hs"]            (hydrostatic)
            turnaround   -> mass_defs["ta"]
            inf          -> mass_defs["inf"]           (infall)
            lambda_so/   -> mass_defs["marg_bound"]    (Pavlidou-Tomaras)
              lambda

        ``marginally_bound_M / _R`` is synthesised per-halo from
        ``lambda_so_M / _R`` when present (SO-bisection on particles is
        the authoritative downstream value), falling back to the
        profile-interp ``lambda_M / _R`` for halos with non-finite
        lambda_so. When the solo file has neither, marginally_bound is
        skipped and the slot is left at None.
        """
        from phusis.io.solo_profiles import load_solo_boundaries

        data = load_solo_boundaries(solo_boundaries_path)
        block = data[self.moment.snapshot_id][mdef]
        ms = block["milestones"]

        solo_data = {
            "group_ids":     block["group_ids"],
            "splashback_M":  ms["splashback_M"],
            "splashback_R":  ms["splashback_R"],
            "scale_M":       ms["scale_M"],
            "scale_R":       ms["scale_R"],
            "four_scale_M":  ms["four_scale_M"],
            "four_scale_R":  ms["four_scale_R"],
            "vmax_M":        ms["vmax_M"],
            "vmax_R":        ms["vmax_R"],
            "bound_M":       ms["target_M"],
            "bound_R":       ms["target_R"],
        }

        # Kinematic outer-boundary milestones. These get re-keyed from
        # solo's storage names to the receiver-side mass_def labels:
        # hs -> hydrostatic, inf -> infall, turnaround stays as
        # turnaround. Each (M, R) pair is added only when present, so
        # older solo files (or ones written before the kinematic suite
        # landed) skip the corresponding slot gracefully.
        if "hs_M" in ms and "hs_R" in ms:
            solo_data["hydrostatic_M"] = ms["hs_M"]
            solo_data["hydrostatic_R"] = ms["hs_R"]
        if "turnaround_M" in ms and "turnaround_R" in ms:
            solo_data["turnaround_M"] = ms["turnaround_M"]
            solo_data["turnaround_R"] = ms["turnaround_R"]
        if "inf_M" in ms and "inf_R" in ms:
            solo_data["infall_M"] = ms["inf_M"]
            solo_data["infall_R"] = ms["inf_R"]

        # marginally_bound: SO-bisection lambda_so preferred per halo,
        # profile-interp lambda as fallback. NumPy is already a transitive
        # dependency; do the merge inline to keep the helper local.
        import numpy as _np
        lso_M = ms.get("lambda_so_M")
        lso_R = ms.get("lambda_so_R")
        l_M   = ms.get("lambda_M")
        l_R   = ms.get("lambda_R")

        def _prefer(primary, fallback):
            """Per-element: primary if finite and >0, else fallback."""
            if primary is None:
                return fallback
            if fallback is None:
                return primary
            with _np.errstate(invalid="ignore"):
                use_primary = _np.isfinite(primary) & (primary > 0)
            return _np.where(use_primary, primary, fallback)

        mb_M = _prefer(lso_M, l_M)
        mb_R = _prefer(lso_R, l_R)
        if mb_M is not None and mb_R is not None:
            solo_data["marginally_bound_M"] = mb_M
            solo_data["marginally_bound_R"] = mb_R

        self.mass_defs.add_solo_boundary_data(
            solo_data=solo_data,
            fof_group_ids=self.ids,
        )
        self.has_physical_boundaries = True


    def load_full_catalog(self, full_catalog_path: Path) -> None:
        """Placeholder for the future joint catalog format that will fold
        spin-and-shapes catalog data and solo boundary milestones into a
        single per-snap HDF5. Not built yet — for the splashback / scale
        / four_scale / vmax mass-defs in the meantime use
        :meth:`add_solo_boundaries`; for the existing 5-mass-def catalog
        path use :meth:`add_physical_mass_defs_from_spin_and_shapes_data`.
        """
        raise NotImplementedError(
            "Joint full-catalog format not yet defined. Use "
            "add_solo_boundaries for splashback/scale/four_scale/vmax in "
            "the meantime."
        )



@define(slots=True)
class FOFEvoData(HaloPopulationEvolutionData):
    data: OrderedDict[int, FOFGroups]
    mapping: SubhaloMappingEvo = field(init=False, repr=False)

    def __attrs_post_init__(self):
        mappings = {
            snap_id: fof_group.subhalo_mapping
            for snap_id, fof_group in self.data.items()
        }
        self.mapping = SubhaloMappingEvo(
            moments=self.moments,
            data=OrderedDict(sorted(mappings.items()))
        )


    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_data(
            cls, data: dict,
            moments: MomentsInTime,
            hubble: float,
            b: float = DEFAULT_LINKING_LENGTH,
            min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
            min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
            region_scaling: float = DEFAULT_REGION_SCALING,
            scaling_def: str = DEFAULT_SCALING_MASS_DEF,
        ) -> FOFEvoData:
        """Build FOFEvoData from a pre-loaded per-snap data dict.

        No ``use_full_catalog`` flag is needed: ``FOFGroups.from_catalog_data``
        auto-detects whether each snap's cleaned dict carries the
        solo-derived ``masses_<descriptor>`` / ``radii_<descriptor>``
        keys (which it does when the upstream
        :func:`get_data.get_fof_subfind_data` was called with
        ``use_full_catalog=True``) and routes the mass_defs slot
        population accordingly via
        :meth:`MassDefinitions.add_full_catalog_solo_data`.
        """
        fof_group_data = {
            snap_id: FOFGroups.from_catalog_data(
                data=data[snap_id],
                moment=moments[snap_id],
                hubble=hubble,
                b=b,
                region_scaling=region_scaling,
                scaling_def=scaling_def,
                min_resolved_mass=min_resolved_mass,
                min_count_above_max_bin_hist=min_count_above_max_bin_hist
            )
            for snap_id in data
        }
        
        return cls(
            moments=moments,
            data=OrderedDict(sorted(fof_group_data.items())),
        )
    
    @classmethod
    def from_directory(
            cls, sim_dir: Path,
            cosmo: Cosmology,
            b: float = DEFAULT_LINKING_LENGTH,
            min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
            min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
            region_scaling: float = DEFAULT_REGION_SCALING,
            scaling_def: str = DEFAULT_SCALING_MASS_DEF,
            is_batched: bool = False,
            in_comoving: bool = True,
            run_snap_by_snap: bool = False,
            snapshots_to_skip: Collection[int] | None = None,
            use_full_catalog: bool = False,
            full_catalog_dir: Path | None = None,
        ) -> FOFEvoData:
        """Multi-snap FOFEvoData loader.

        ``use_full_catalog=True`` routes per-snap loads through the
        Phase 1 full_catalog product, so each snap's FOFGroups comes
        back with the solo-derived mass_defs slots already populated
        (Phase 4 v1). Snaps lacking a full_catalog HDF5 are skipped —
        same behavior as missing GADGET-4 snaps. ``from_data`` needs
        no flag plumbing: ``FOFGroups.from_catalog_data`` auto-detects
        the cleaned-dict solo keys regardless of how the data dict was
        built.
        """
        fn_params = {
            "sim_dir": sim_dir,
            "cosmo": cosmo,
            "b": b,
            "min_resolved_mass": min_resolved_mass,
            "min_count_above_max_bin_hist": min_count_above_max_bin_hist,
            "region_scaling": region_scaling,
            "scaling_def": scaling_def,
            "is_batched": is_batched,
            "in_comoving": in_comoving,
            "snapshots_to_skip": snapshots_to_skip,
            "use_full_catalog": use_full_catalog,
            "full_catalog_dir": full_catalog_dir,
        }

        return (
            load_fof_evo_snap_by_snap(**fn_params)
            if run_snap_by_snap else
            load_fof_evo_all_at_once(**fn_params)
        )

    def get_subset(self, group_id_evo: dict[int, Collection[int] | np.ndarray]) -> FOFEvoData:
        data = OrderedDict()
        moments = MomentsInTime()
        for snap_id, group_ids in sorted(group_id_evo.items(), key=lambda x: x[0]):
            if snap_id not in self.data: continue
            data[snap_id] = self.data[snap_id].get_subset(group_ids)
            moments.add_moment(data[snap_id].moment)
        return FOFEvoData(moments=moments, data=data)


    def get_tracked_halo_evo(
            self, subhalo_id: int, reference_snap_id: int
        ) -> FOFEvoData: # Make a group tree

        ref_group = self.data[reference_snap_id].get_by_subhalos(
            subhalo_id, return_as_groups=False
        )

        if len(ref_group) == 0:
            raise ValueError(
                f"Was not able to get subset for the provided subhalo_ids "
                f"on snap_idx={reference_snap_id}"
            )
        
        tracked_moments = MomentsInTime() 
        tracked_moments.add_moment(ref_group.moment)

        tracked_group_dict = OrderedDict({reference_snap_id : ref_group})

        all_snap_ids = sorted(self.moments.snapshot_ids)

        # get_tracked_progs_and_descs(
        #     fof_group_evo=self,
        #     reference_fof_groups=ref_group,
        #     snap_ids=all_snap_ids,
        #     reference_snap_id=reference_snap_id,
        #     tracked_moments=tracked_moments,
        #     tracked_groups_dict=tracked_group_dict
        # )

        # return FOFEvoData(
        #     moments=tracked_moments,
        #     data=OrderedDict(sorted(tracked_group_dict.items()))
        # )

        current_forward, current_backward = ref_group, ref_group

        for snap_idx in all_snap_ids:
            if snap_idx <= reference_snap_id:
                continue

            # Use the descendants from the previous snapshot's halos
            desc_sub_ids = current_forward.descendant_subhalo_ids

            if ((desc_sub_ids is not None) and (len(desc_sub_ids) > 0)):

                try:
                    future_halo = self.data[snap_idx].get_by_subhalos(
                        subhalo_ids=desc_sub_ids[desc_sub_ids > -1], 
                        return_as_groups=True
                    )
                except (ValueError, IndexError) as error:
                    print(f"Error on snap_idx={snap_idx}: {error}")
                    continue

                if future_halo is None: continue

                tracked_group_dict[snap_idx] = future_halo
                tracked_moments.add_moment(future_halo.moment)
                current_forward = future_halo
            else:
                current_forward = self.data[snap_idx]

        # Next, handle snapshots before ref_snap_id (past progenitors)
        for snap_idx in reversed(all_snap_ids):

            if snap_idx >= reference_snap_id: continue
            # Use the progenitors from the next snapshot's halos

            prog_sub_ids = current_backward.progenitor_subhalo_ids
            
            if ((prog_sub_ids is not None) and (len(prog_sub_ids) > 0)):

                try:
                    prev_halo = self.data[snap_idx].get_by_subhalos(
                        subhalo_ids=prog_sub_ids[prog_sub_ids > -1], 
                        return_as_groups=True
                    )
                except (ValueError, IndexError) as error:
                    print(f"Error on snap_idx={snap_idx}: {error}")
                    continue

                if prev_halo is None: continue

                tracked_group_dict[snap_idx] = prev_halo
                tracked_moments.add_moment(prev_halo.moment)
                current_backward = prev_halo
            else:
                current_backward = self.data[snap_idx]

        return FOFEvoData(
            moments=tracked_moments,
            data=OrderedDict(sorted(tracked_group_dict.items()))
        )
    
    def get_main_progenitor_branch(self, evo_path: dict[int, int]) -> FOFGroupEvoData:

        moments = MomentsInTime()   
        data = OrderedDict()

        for snap_idx, group_id in sorted(evo_path.items(), key=lambda x: x[0]):

            if (groups := self.data.get(snap_idx)) is None: continue

            if group_id not in groups.ids: continue

            data[snap_idx] = groups[group_id]
            moments.add_moment(groups.moment)

        return FOFGroupEvoData(moments=moments, data=data)


    def get_main_progenitor_branch_set(
            self, evo_paths: dict[int, dict[int, int]]
        ) -> dict[int, FOFGroupEvoData]:

        return {
            halo_id : self.get_main_progenitor_branch(evo_path)
            for halo_id, evo_path in sorted(evo_paths.items(), key=lambda x: x[0])
        }
    
    def get_subset_by_subhalo_tracking(
            self, subhalo_ids: Collection[int], reference_snap_id: int
        ) -> FOFEvoData:

        ref_groups = self.data[reference_snap_id].get_by_subhalos(
            subhalo_ids, return_as_groups=True
        )

        if len(ref_groups) == 0:
            print(
                f"Was not able to get subset for the provided subhalo_ids "
                f"on snap_idx={reference_snap_id}"
            )
            return self

        tracked_moments = MomentsInTime()
        tracked_moments.add_moment(self.data[reference_snap_id].moment)

        tracked_groups_dict = OrderedDict({reference_snap_id: ref_groups})

        all_snap_ids = sorted(self.moments.snapshot_ids)

        # get_tracked_progs_and_descs(
        #     fof_group_evo=self,
        #     reference_fof_groups=ref_groups,
        #     snap_ids=all_snap_ids,
        #     reference_snap_id=reference_snap_id,
        #     tracked_moments=tracked_moments,
        #     tracked_groups_dict=tracked_groups_dict
        # )

        # return FOFEvoData(
        #     moments=tracked_moments,
        #     data=OrderedDict(sorted(tracked_groups_dict.items()))
        # )

        current_forward, current_backward = ref_groups, ref_groups

        all_snap_ids = sorted(self.moments.snapshot_ids)

        # First, handle snapshots after ref_snap_id (future descendants)
        for snap_idx in all_snap_ids:
            if snap_idx <= reference_snap_id:
                continue

            # Use the descendants from the previous snapshot's halos
            desc_sub_ids = current_forward.descendant_subhalo_ids

            if ((desc_sub_ids is not None) and (len(desc_sub_ids) > 0)):

                try: 
                    future_halos = self.data[snap_idx].get_by_subhalos(
                        desc_sub_ids, return_as_groups=True
                    )
                except (ValueError, IndexError) as error:
                    print(f"Error on snap_idx={snap_idx}: {error}")
                    continue
                
                if future_halos is None: continue

                tracked_groups_dict[snap_idx] = future_halos
                tracked_moments.add_moment(future_halos.moment)
                current_forward = future_halos
            else:
                current_forward = self.data[snap_idx]

        # Next, handle snapshots before ref_snap_id (past progenitors)
        for snap_idx in reversed(all_snap_ids):

            if snap_idx >= reference_snap_id: continue

            # Use the progenitors from the next snapshot's halos
            prog_sub_ids = current_backward.progenitor_subhalo_ids
            if ((prog_sub_ids is not None) and (len(prog_sub_ids) > 0)):

                try:
                    prev_halos = self.data[snap_idx].get_by_subhalos(
                        prog_sub_ids, return_as_groups=True
                    )
                except (ValueError, IndexError) as error:
                    print(f"Error on snap_idx={snap_idx}: {error}")
                    continue

                if prev_halos is None: continue

                tracked_groups_dict[snap_idx] = prev_halos
                tracked_moments.add_moment(prev_halos.moment)
                current_backward = prev_halos

            else:
                current_backward = self.data[snap_idx]


        return FOFEvoData(
            moments=tracked_moments,
            data=OrderedDict(sorted(tracked_groups_dict.items()))
        )


    def get_progenitor(
            self, 
            descendant_id: int,
            descendant_snap_id: int,
            progenitor_snap_id: int
        ) -> FOFGroup:

        track_halo = self.get_tracked_halo_evo(
            subhalo_id=descendant_id,
            reference_snap_id=descendant_snap_id
        )

        return track_halo.data[progenitor_snap_id]


    def get_descendant(
            self, 
            progenitor_id: int,
            progenitor_snap_id: int,
            descendant_snap_id: int
        ) -> FOFGroup:

        track_halo = self.get_tracked_halo_evo(
            subhalo_id=progenitor_id,
            reference_snap_id=progenitor_snap_id
        )

        # Have to account for the fact that the descendant may not exist at last snapshot
        return track_halo.data[descendant_snap_id]
    

    def get_subset_by_prog_orders(
            self, 
            prog_orders: dict[int, dict[int, np.ndarray]],
            target_desc_ids: Collection[int] | None = None
        ) -> FOFEvoData:

        return self.get_subset(
            group_id_evo=organize_prog_info_by_snapshot(
                prog_info=prog_orders,
                target_desc_ids=target_desc_ids
            ) # type: ignore
        )
    

    @property
    def organized_by_fof_group_evo(self) -> OrderedDict[int, FOFGroupEvoData]:
        max_snap_id = max(self.moments.snapshot_ids)
        return OrderedDict({
            halo.id_number : self.get_tracked_halo_evo(
                subhalo_id=halo.lead_subhalo_id,
                reference_snap_id=max_snap_id
            )
            for halo in self.data[max_snap_id]
        })
    

    def add_physical_mass_defs_from_spin_and_shapes_data(
            self, spin_and_shapes_evo_data: dict,
        ) -> None:

        for snap_id, spin_and_shapes_data in spin_and_shapes_evo_data.items():
            if snap_id not in self.data: continue
            if len(spin_and_shapes_data) == 0: continue
            if ((snap_data := spin_and_shapes_data.get('M200m')) is None): continue
            self.data[snap_id].add_physical_mass_defs_from_spin_and_shapes_data(snap_data)


    def add_solo_boundaries(
            self,
            solo_boundaries_dir: Path,
            mdef: str,
            sim_save_name: str,
            units_tag: str = "physical",
        ) -> None:
        """Apply :meth:`FOFGroups.add_solo_boundaries` across every snap.

        Per-snap path is built as::

            {solo_boundaries_dir}/{sim_save_name}_solo_boundaries
                _phusis_{units_tag}_snap{NNN}_{mdef}.hdf5

        which matches the standard (non-descendant, non-allresolved) layout
        emitted by ``save_solo_profiles_phusis._get_save_path``. Snaps
        without a matching file are skipped silently — they may have failed
        solo's all_resolved cut at write time, or simply not been processed
        yet. Snaps that are missing from ``self.data`` are also skipped, so
        callers can pass a directory containing more snaps than the evo
        currently spans.
        """
        for snap_id, fof_group in self.data.items():
            path = (
                solo_boundaries_dir
                / f"{sim_save_name}_solo_boundaries_phusis_{units_tag}"
                  f"_snap{int(snap_id):03d}_{mdef}.hdf5"
            )
            if not path.exists():
                continue
            fof_group.add_solo_boundaries(path, mdef)


    def load_full_catalog(self, full_catalog_dir: Path) -> None:
        """Placeholder — see :meth:`FOFGroups.load_full_catalog`."""
        raise NotImplementedError(
            "Joint full-catalog format not yet defined. Use "
            "add_solo_boundaries for splashback/scale/four_scale/vmax in "
            "the meantime."
        )


def get_tracked_descendants(
        fof_group_evo: FOFEvoData,
        reference_fof_groups: FOFGroups,
        snap_ids: Collection[int],
        reference_snap_id: int,
        tracked_moments: MomentsInTime,
        tracked_groups_dict: OrderedDict[int, FOFGroups],
    ) -> None:


    current_forward = reference_fof_groups

    for snap_idx in snap_ids:
        if snap_idx <= reference_snap_id:
            continue

        # Use the descendants from the previous snapshot's halos
        desc_sub_ids = current_forward.descendant_subhalo_ids

        if ((desc_sub_ids is not None) and (len(desc_sub_ids) > 0)):

            try: 
                future_halos = fof_group_evo.data[snap_idx].get_by_subhalos(
                    desc_sub_ids[desc_sub_ids > -1], return_as_groups=True
                )
            except (ValueError, IndexError) as error:
                print(f"Error on snap_idx={snap_idx}: {error}")
                continue
            
            if future_halos is None: continue

            tracked_groups_dict[snap_idx] = future_halos
            tracked_moments.add_moment(future_halos.moment)
            current_forward = future_halos


def get_tracked_progenitors(
        fof_group_evo: FOFEvoData,
        reference_fof_groups: FOFGroups,
        snap_ids: Collection[int],
        reference_snap_id: int,
        tracked_moments: MomentsInTime,
        tracked_groups_dict: OrderedDict[int, FOFGroups],
    ) -> None:

    current_backward = reference_fof_groups

    for snap_idx in reversed(snap_ids):

        if snap_idx >= reference_snap_id: continue

        # Use the progenitors from the next snapshot's halos
        prog_sub_ids = current_backward.progenitor_subhalo_ids
        if ((prog_sub_ids is not None) and (len(prog_sub_ids) > 0)):

            try:
                prev_halos = fof_group_evo.data[snap_idx].get_by_subhalos(
                    prog_sub_ids[prog_sub_ids > -1], return_as_groups=True
                )
            except (ValueError, IndexError) as error:
                print(f"Error on snap_idx={snap_idx}: {error}")
                continue

            if prev_halos is None: continue

            tracked_groups_dict[snap_idx] = prev_halos
            tracked_moments.add_moment(prev_halos.moment)
            current_backward = prev_halos
            
        else:
            current_backward = fof_group_evo.data[snap_idx]

def get_tracked_progs_and_descs(
        fof_group_evo: FOFEvoData,
        reference_fof_groups: FOFGroups,
        snap_ids: Collection[int],
        reference_snap_id: int,
        tracked_moments: MomentsInTime,
        tracked_groups_dict: OrderedDict[int, FOFGroups],
    ) -> None:

    get_tracked_descendants(
        fof_group_evo=fof_group_evo,
        reference_fof_groups=reference_fof_groups,
        snap_ids=snap_ids,
        reference_snap_id=reference_snap_id,
        tracked_moments=tracked_moments,
        tracked_groups_dict=tracked_groups_dict
    )

    get_tracked_progenitors(
        fof_group_evo=fof_group_evo,
        reference_fof_groups=reference_fof_groups,
        snap_ids=snap_ids,
        reference_snap_id=reference_snap_id,
        tracked_moments=tracked_moments,
        tracked_groups_dict=tracked_groups_dict
    )


def organize_prog_info_by_snapshot(
        prog_info: dict[int, dict[int, np.ndarray]],
        target_desc_ids: Collection[int] | None = None
    ) -> dict[int, np.ndarray]:

    organized_info = defaultdict(list)

    if target_desc_ids is None:
        target_desc_ids = set(prog_info.keys())

    for desc_group_id, prog_line in prog_info.items():
        if desc_group_id not in target_desc_ids: continue
        for snap_id, prog_group_ids in prog_line.items():
            organized_info[snap_id].extend(prog_group_ids)

    return {
        snap_id: np.unique(np.asarray(group_ids))
        for snap_id, group_ids in organized_info.items()
    }


def load_fof_evo_all_at_once(
        sim_dir: Path,
        cosmo: Cosmology,
        b: float = DEFAULT_LINKING_LENGTH,
        min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
        min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
        region_scaling: float = DEFAULT_REGION_SCALING,
        scaling_def: str = DEFAULT_SCALING_MASS_DEF,
        is_batched: bool = False,
        in_comoving: bool = True,
        snapshots_to_skip: Collection[int] | None = None,
        use_full_catalog: bool = False,
        full_catalog_dir: Path | None = None,
    ) -> FOFEvoData:

    catalogs = get_all_fof_subfind_catalogs(
        directory=sim_dir,
        is_batched=is_batched,
        in_comoving=in_comoving,
        display_if_empty=False,
        snapshots_to_skip=snapshots_to_skip,
        use_full_catalog=use_full_catalog,
        full_catalog_dir=full_catalog_dir,
    )
    
    data = {}

    moments = MomentsInTime()

    for snapshot_id, catalog in tqdm(
        catalogs.items(), desc="Setting up FOFEvoData"
    ):

        if not catalog: continue

        try:
            scale_factor = catalog["fof"]["scale_factor"]
        except KeyError as error:
            print(f"No FOF data found on {snapshot_id = }: {error}")
            continue


        moment = Moment(snapshot_id=snapshot_id, scale_factor=scale_factor)

        groups = FOFGroups.from_catalog_data(
            data=catalog,
            moment=moment, 
            hubble=float(cosmo.hubble_parameter(a=moment.scale_factor)),
            b=b,
            min_resolved_mass=min_resolved_mass,
            min_count_above_max_bin_hist=min_count_above_max_bin_hist,
            region_scaling=region_scaling,
            scaling_def=scaling_def,
        ) 

        if groups.ids.size == 0:
            print(f"No groups found in {snapshot_id = }")
            continue

        data[snapshot_id] = groups
        moments.add_moment(moment)

    moments.add_times(cosmo)

    return FOFEvoData(moments=moments, data=OrderedDict(sorted(data.items())))
    


def load_fof_evo_snap_by_snap(
        sim_dir: Path,
        cosmo: Cosmology,
        b: float,
        min_resolved_mass: float = DEFAULT_MIN_RESOLVED_MASS,
        min_count_above_max_bin_hist: int = DEFAULT_CUMULATIVE_COUNT,
        region_scaling: float = DEFAULT_REGION_SCALING,
        scaling_def: str = DEFAULT_SCALING_MASS_DEF,
        is_batched: bool = False,
        in_comoving: bool = True,
        snapshots_to_skip: Collection[int] | None = None,
        use_full_catalog: bool = False,
        full_catalog_dir: Path | None = None,
    ) -> FOFEvoData:

    data = {}
    moments = MomentsInTime()
    pattern = "groups_*" if is_batched else "fof_subhalo_tab_*.hdf5"
    iter_dir = sim_dir.glob(pattern)

    if snapshots_to_skip is None:
        snapshots_to_skip = set()

    for fof_path in iter_dir:
        snap_idx = int(fof_path.name.split("_")[-1].split(".")[0])

        if snap_idx in snapshots_to_skip:
            print(f"Skipping snapshot {snap_idx}")
            continue

        try:

            groups = FOFGroups.from_catalog(
                catalog_dir=fof_path.parent,
                is_batched=is_batched,
                snapshot_id=snap_idx,
                cosmo=cosmo,
                b=b,
                region_scaling=region_scaling,
                scaling_def=scaling_def,
                in_comoving=in_comoving,
                min_resolved_mass=min_resolved_mass,
                min_count_above_max_bin_hist=min_count_above_max_bin_hist,
                use_full_catalog=use_full_catalog,
                full_catalog_dir=full_catalog_dir,
            )

        except ValueError:

            continue

        if groups.ids.size == 0:
            print(f"No groups found in {snap_idx = }")
            continue

        data[snap_idx] = groups

        moments.add_moment(
            Moment(
                snapshot_id=snap_idx,
                scale_factor=data[snap_idx].moment.scale_factor
            )
        )

    moments.add_times(cosmo)

    return FOFEvoData(moments=moments, data=OrderedDict(sorted(data.items())))


def get_loss_cfg_linking_length(seed_num: int) -> float:
    return 0.28 if seed_num in {-1, 0} else 0.2
    