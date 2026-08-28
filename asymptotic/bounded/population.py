from __future__ import annotations

import numpy as np, pdb
import matplotlib.pyplot as plt

from attrs import define
from attrs import fields as attrs_fields
from pathlib import Path
from tqdm.auto import tqdm
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from colossus.lss import peaks
from typing import Iterator
from abc import ABC, abstractmethod
from functools import cached_property
from collections import OrderedDict
from collections.abc import Collection


from ..cosmo.model import Cosmology
from ..simulation.moments import Moment, MomentsInTime
from ..mass_def.data import (
    MassDefinitions, MassDefinitionEvo, MassDefinitionsEvo
)
from ..model.mass_function import (
    AsymptoticAbundanceFit, 
    AsymptoticAbundanceFitEvo,
    AsymptoticAbundanceModel,
    AsymptoticAbundanceEvoModel
)
from ..utils.get_data import (
    get_fof_subfind_data,
    load_all_physical_catalogs,
    load_physical_catalog_data,
    get_all_fof_subfind_catalogs
)
from ..utils.physical_catalog import get_physical_catalog_path
from ..simulation.sub_box import SubBoxes
from ..simulation.evo import EvolutionData
from .objects import BoundedObject, BoundedObjectEvolutionData
from ..abundance.halo import HaloMassFunction, HaloMassFunctionEvoData
from ..abundance.mass_function import (
    MassFunctionValue, MassFunctionData, MassFunctionEvoData
)
from ..abundance.accumulation import PopulationAbundanceAccumulationHistories
from ..particles.collection import (
    BoundedParticleCollection, BoxParticleCollection
)

from ..particles.shapes import PopulationShapeParameterData
from ..particles.anisotropy import PopulationAnisotropyParameterData

from ..fields.knn import kNNDistributionData, kNNDistributionEvoData
from ..fields.peaks import DensityPeaks, DensityPeaksEvo
from ..fields.correlation import (
    TwoPointCorrelationData, 
    TwoPointCorrelationEvoData,
    group_halo_halo_correlations_by_property
)
from .bias import ( # Integrate these in ...
    HaloBiasData,
    HaloBiasDataset,
    HaloBiasEvoData,
    HaloBiasEvoDataset
)

@define(slots=True)
class HaloPopulation(ABC): 
    ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    particle_counts: np.ndarray

    moment: Moment
    mass_defs: MassDefinitions

    sub_boxes: SubBoxes | None

    min_resolved_mass: float 
    min_count_above_max_bin_hist: int

    spins: np.ndarray | None
    shapes: PopulationShapeParameterData | None
    anisotropies: PopulationAnisotropyParameterData | None

    has_physical_boundaries: bool


    def __len__(self) -> int:
        return self.ids.size
    
    @abstractmethod
    def __getitem__(self, idx: int) -> BoundedObject:
        ... 

    def __iter__(self) -> Iterator[BoundedObject]:
        for idx in self.ids:
            yield self[idx]    

    @property
    def size(self) -> int:
        return len(self)
    def get_masses(self, mass_def: str) -> np.ndarray:
        if not hasattr(self.mass_defs, mass_def):
            raise ValueError(f"{mass_def} is not a valid mass definition")
        return getattr(self.mass_defs, mass_def).mass
    
    def get_masses_by_key(self, mass_def_key: str) -> np.ndarray:
        return self.mass_defs[mass_def_key].mass
    
    @property
    def comoving_positions(self) -> np.ndarray:
        return self.positions / self.moment.scale_factor

    def initialize_empty_sub_boxes(
            self, box_size: int, num_splits: int
        ) -> None:
        self.sub_boxes = SubBoxes.empty_initialize(box_size, num_splits)

    def get_mass_ratios(
            self, 
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> np.ndarray:

        ratios = self.mass_defs.get_attribute_ratios(
            attr_name="mass", 
            mass_def_key_numerator=mass_def_key_numerator, 
            mass_def_key_denominator=mass_def_key_denominator,
            return_full=return_full
        )

        return np.column_stack([self.ids, ratios])

    def get_radii_ratios(
            self, 
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> np.ndarray:

        ratios = self.mass_defs.get_attribute_ratios(
            attr_name="radius", 
            mass_def_key_numerator=mass_def_key_numerator, 
            mass_def_key_denominator=mass_def_key_denominator,
            return_full=return_full
        )

        return np.column_stack([self.ids, ratios])
    
    def get_velocity_ratios(
            self, 
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> np.ndarray:

        ratios = self.mass_defs.get_attribute_ratios(
            attr_name="velocity", 
            mass_def_key_numerator=mass_def_key_numerator, 
            mass_def_key_denominator=mass_def_key_denominator,
            return_full=return_full
        )

        return np.column_stack([self.ids, ratios])

    # @abstractmethod
    # def save(self, path: Path) -> None:
    #     ...

    # @classmethod
    # @abstractmethod     
    # def load(cls, path: Path) -> HaloPopulation:
    #     ...

    @classmethod
    @abstractmethod
    def from_catalog(
            cls, 
            catalog_dir: Path, 
            snapshot_id: int,
            cosmo: Cosmology,
            in_comoving: bool,
            compute_mass_function: bool,
            *args, **kwargs
        ) -> HaloPopulation:
        ...

    @classmethod
    @abstractmethod
    def from_catalog_data(
            cls, data: dict, moment: Moment, hubble: float, *args, **kwargs
        ) -> HaloPopulation:
        ...

    def add_physical_catalog(
            self, 
            catalog_dir: Path,
            comoving_box_size: int,
            num_particles: int,
            cosmo: Cosmology,
            seed_num: int = 1,
            sampling_mdef: str = "200m",
        ) -> None:
        
        try:
            phys_catalog_path = get_physical_catalog_path(
                sim_data_dir=catalog_dir,
                box_size=comoving_box_size,
                num_particles=num_particles,
                snap_idx=self.moment.snapshot_id,
                seed_num=seed_num,
                is_from_primary=(cosmo.name == "primary"),
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"No physical catalog found for {self.moment.snapshot_id = }"
            ) from error
        
        catalog = load_physical_catalog_data(
            filepath=phys_catalog_path, 
            return_concatenated=True
        )

        self.mass_defs.add_physical_catalog_data(catalog[sampling_mdef], self.ids)
        self.has_physical_boundaries = True

    def get_objects(self, idxs: int | Collection[int], **subclass_kwargs) -> HaloPopulation:
        if not isinstance(idxs, np.ndarray):
            idxs = np.array(idxs)
        return self.__class__(
            ids=self.ids[idxs],
            positions=self.positions[idxs],
            velocities=self.velocities[idxs],
            particle_counts=self.particle_counts[idxs],
            moment=self.moment,
            mass_defs=self.mass_defs.get_subset(idxs),
            sub_boxes=self.sub_boxes,
            min_resolved_mass=self.min_resolved_mass,
            min_count_above_max_bin_hist=self.min_count_above_max_bin_hist,
            **subclass_kwargs
        )
    
    
    def _get_property_array(
            self,
            property_name: str,
            mass_def_key: str | None = None,
        ) -> np.ndarray:
        """Return a 1D array of per-halo property values for bias computations.

        Some properties depend on the chosen mass definition (e.g., mass,
        concentration, peak height, velocity). For those, `mass_def_key`
        must be provided (e.g., "200m", "500c", "vir", or a mapping key
        exposed by `self.mass_defs`).
        """
        prop = property_name.lower()

        # Mass-definition–dependent properties
        if prop in {"mass", "log_mass", "m"}:
            if mass_def_key is None:
                raise ValueError("mass_def_key must be provided when requesting 'mass'.")
            return self.get_masses_by_key(mass_def_key)

        if prop in {"peak_height", "nu", "peak_heights"}:
            if mass_def_key is None:
                raise ValueError("mass_def_key must be provided when requesting 'peak_height'.")
            md = self.mass_defs[mass_def_key]
            if getattr(md, "peak_heights", None) is None:
                raise ValueError("Peak heights not set; call mass_defs.add_peak_heights(...) first.")
            return np.asarray(md.peak_heights, dtype=float)

        if prop in {"concentration", "c"}:
            if mass_def_key is None:
                raise ValueError("mass_def_key must be provided when requesting 'concentration'.")
            # Use the canonical accessor on MassDefinitions as requested
            if not hasattr(self.mass_defs, "get_concentrations"):
                raise ValueError("MassDefinitions.get_concentrations not available in this build.")
            return np.asarray(self.mass_defs.get_concentrations(mass_def_key), dtype=float)

        if prop in {"velocity", "v_circ", "circular_velocity"}:
            if mass_def_key is None:
                raise ValueError("mass_def_key must be provided when requesting 'velocity'.")
            md = self.mass_defs[mass_def_key]
            v = md.velocity
            if v is None:
                raise ValueError("Velocity not available; ensure radii are set for the mass definition.")
            return np.asarray(v, dtype=float)

        # Properties not tied to a mass definition
        if prop in {"spin", "lambda", "lambda_peebles"}:
            if self.spins is None:
                raise ValueError("Spin data not attached to this HaloPopulation.")
            return np.asarray(self.spins, dtype=float)

        if prop in {"sphericity", "ellipticity", "triaxiality"}:
            if self.shapes is None:
                raise ValueError("Shape parameters not attached to this HaloPopulation.")
            if not hasattr(self.shapes, prop):
                raise ValueError(f"Shape parameter '{prop}' not found on shapes data.")
            return np.asarray(getattr(self.shapes, prop), dtype=float)

        if prop in {"velocity_dispersion", "radial_velocity"}:
            # If you store these on a dedicated data container, expose here.
            if self.anisotropies is None or not hasattr(self.anisotropies, prop):
                raise ValueError(
                    f"Property '{prop}' not available; attach anisotropy data or implement retrieval."
                )
            return np.asarray(getattr(self.anisotropies, prop), dtype=float)

        if prop in {"speed", "speed_magnitude"}:
            # Magnitude of 3D halo velocity vector (per-halo)
            return np.linalg.norm(np.asarray(self.velocities, dtype=float), axis=1)

        raise ValueError(
            f"Unrecognized property '{property_name}'. Supported examples: mass, peak_height, concentration, "
            f"spin, sphericity/ellipticity/triaxiality, velocity, speed, velocity_dispersion, radial_velocity."
        )

    @property
    def sub_box_as_comoving_dict(self) -> dict[int, list[tuple[float, float]]]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_comoving_as_dict(self.moment.scale_factor)

    @property
    def sub_box_samples_id_array(self) -> np.ndarray:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_sub_box_samples_id_array(self.comoving_positions) 

    @property
    def sub_box_samples_coordinates(self) -> list[np.ndarray]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_jackknife_subbox_samples(
            coordinates=self.positions,
            return_as_mask=False
        )   

    @property
    def sub_box_samples_mask(self) -> list[np.ndarray]:
        if self.sub_boxes is None:
            raise ValueError("Sub-boxes have not been initialized.")
        return self.sub_boxes.get_jackknife_subbox_samples(
            coordinates=self.positions,
            return_as_mask=True
        )
    
    def get_property_halo_bias(
            self,
            comoving_particle_coordinates: np.ndarray,
            comoving_box_size: float,
            property_name: str,
            rmin: float,
            rmax: float,
            nbins: int,
            property_bins: dict[int | str, tuple[float, float]] | None = None,
            mass_def_key: str | None = None,
            target_radius: float | None = None,
            radial_index: int | None = None,
            use_natural: bool = False,
            eps: float = 1e-12,
            positive_only: bool = True,
            return_folds: bool = False,
            interpolate_to_radius: float | None = None,
            allow_loglog: bool = False,
            run_jackknife: bool = True,
            num_splits: int = 1,
            # Subsampling controls:
            use_recommended_subsample: bool = False,
            num_target_pairs: int = 100_000,
            subsample_secondary: bool = False,
            subsample_rng_seed: int | None = None,
        ) -> HaloBiasData:
        """Compute halo bias for a single property using jackknife over sub-boxes.

        Parameters
        ----------
        comoving_particle_coordinates : (Nm, 3)
            Particle catalog (matter) in comoving coordinates.
        comoving_box_size : float
            Periodic box size in comoving units.
        property_name : str
            Name of the halo property (e.g., 'mass', 'concentration', 'spin', ...).
        property_bins : dict or None
            If None, 10 quantile bins are used.
        mass_def_key : str or None
            Required for mass-definition–dependent properties (mass, concentration, peak_height, velocity).
        """
        # Ensure sub-boxes exist if we jackknife
        if run_jackknife and self.sub_boxes is None:
            self.initialize_empty_sub_boxes(comoving_box_size, num_splits)

        # Halo positions in comoving units (required by TPCF/bias routines)
        halo_pos = self.comoving_positions

        # Gather the property array
        prop_vals = self._get_property_array(property_name, mass_def_key=mass_def_key)
        if prop_vals.shape[0] != halo_pos.shape[0]:
            raise ValueError("Property array length does not match number of halos.")

        # Build sub-box info if jackknifing
        sub_box_info = (self.sub_boxes.as_dict if run_jackknife else None)

        return HaloBiasData.from_data(
            comoving_halo_positions=halo_pos,
            comoving_particle_coordinates=comoving_particle_coordinates,
            property_values=prop_vals,
            property_bins=property_bins,
            sub_box_info=sub_box_info,  # required when run_jackknife=True inside from_data
            boxsize=comoving_box_size,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            target_radius=target_radius,
            radial_index=radial_index,
            use_natural=use_natural,
            eps=eps,
            positive_only=positive_only,
            return_folds=return_folds,
            interpolate_to_radius=interpolate_to_radius,
            allow_loglog=allow_loglog,
            # NEW subsampling controls:
            # use_recommended_subsample=use_recommended_subsample,
            # num_target_pairs=num_target_pairs,
            # subsample_secondary=subsample_secondary,
            # subsample_rng_seed=subsample_rng_seed,
        )

    def get_halo_bias_dataset(
            self,
            comoving_particle_coordinates: np.ndarray,
            comoving_box_size: float,
            rmin: float,
            rmax: float,
            nbins: int,
            properties: list[str] | None = None,
            bins_dict: dict[str, dict[int | str, tuple[float, float]]] | None = None,
            mass_def_key_for_mass_based: str | None = None,
            target_radius: float | None = None,
            radial_index: float | None = None,
            use_natural: bool = False,
            eps: float = 1e-12,
            positive_only: bool = True,
            return_folds: bool = False,
            interpolate_to_radius: float | None = None,
            allow_loglog: bool = False,
            run_jackknife: bool = True,
            num_splits: int = 1,
            # Subsampling controls:
            use_recommended_subsample: bool = False,
            num_target_pairs: int = 100_000,
            subsample_secondary: bool = False,
            subsample_rng_seed: int | None = None,
        ) -> HaloBiasDataset:
        """Compute a multi-property `HaloBiasDataset` using jackknife.

        Notes
        -----
        For properties that depend on the mass definition (mass, concentration,
        peak_height, velocity), provide `mass_def_key_for_mass_based`.
        Only properties provided in `properties` and recognized by
        `HaloBiasDataset` will be computed; others remain `None`.
        """
        # Ensure sub-boxes exist if we jackknife
        if run_jackknife and self.sub_boxes is None:
            self.initialize_empty_sub_boxes(comoving_box_size, num_splits)

        # Default property list: use fields present on HaloBiasDataset
        if properties is None:
            properties = [
                f.name for f in attrs_fields(HaloBiasDataset)
            ]

        # Assemble per-property arrays
        halo_props: dict[str, np.ndarray] = {}
        for name in properties:
            try:
                if name in {"mass", "peak_height", "concentration", "velocity"}:
                    arr = self._get_property_array(name, mass_def_key=mass_def_key_for_mass_based)
                else:
                    arr = self._get_property_array(name)
                halo_props[name] = arr
            except Exception:
                # Skip properties we cannot provide in this configuration
                continue

        if not halo_props:
            raise ValueError("No valid properties could be assembled for HaloBiasDataset.")

        sub_box_info = (self.sub_boxes.as_dict if run_jackknife else None)

        return HaloBiasDataset.from_data(
            comoving_halo_positions=self.comoving_positions,
            comoving_particle_coordinates=comoving_particle_coordinates,
            halo_properties=halo_props,
            bins_dict=bins_dict,
            sub_box_info=sub_box_info,
            boxsize=comoving_box_size,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            target_radius=target_radius,
            radial_index=radial_index,
            use_natural=use_natural,
            eps=eps,
            positive_only=positive_only,
            return_folds=return_folds,
            interpolate_to_radius=interpolate_to_radius,
            allow_loglog=allow_loglog,
            # NEW subsampling controls:
            # use_recommended_subsample=use_recommended_subsample,
            # num_target_pairs=num_target_pairs,
            # subsample_secondary=subsample_secondary,
            # subsample_rng_seed=subsample_rng_seed,
        )

    # def sample(self, n_samples: int) -> FOFGroups:
    #     if n_samples > len(self):
    #         raise ValueError("Number of samples must be less than the number of groups")
    #     idxs = np.random.choice(len(self), size=n_samples, replace=False)
    #     return self.get_subset_by_index(idxs)

    def get_objects_above_min_particles(self, min_particles: int) -> HaloPopulation:
        return self.get_objects(self.particle_counts > min_particles)
    
    @abstractmethod
    def setup_sub_boxes(self, box_size: float, num_splits: int) -> None:
        ...

    def get_nn_distribution(
            self,
            comoving_box_size: float,
            num_query_points: int,
            max_num_neighbors: int,
            num_interp_points: int = 1000,
            sampling_mass_def: str = "all",
            lower_log_mass_bin_limit: float = -np.inf,
            upper_log_mass_bin_limit: float = np.inf,
            use_vectorized: bool = True,
            return_in_comoving: bool = False,
            use_jackknifed: bool = False,
            num_splits: int = 1,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            use_data_randoms: bool = False,
            use_full_population: bool = True,
        ) -> kNNDistributionData:

        locs = self.comoving_positions if return_in_comoving else self.positions
        L = (
            comoving_box_size
            if return_in_comoving else
            comoving_box_size * self.moment.scale_factor
        )

        if (sampling_mass_def == "all"):
            positions = locs
        else:
            log_masses = self.mass_defs[sampling_mass_def].mass
            mass_sample_mask = np.logical_and(
                log_masses >= lower_log_mass_bin_limit,
                log_masses <= upper_log_mass_bin_limit
            )
            positions = locs[mass_sample_mask]

        if use_jackknifed and self.sub_boxes is None:
            self.initialize_empty_sub_boxes(L, num_splits)

        return kNNDistributionData.from_data(
            comoving_positions=positions,
            comoving_box_size=L,
            num_query_points=(
                positions.shape[0] if use_full_population else num_query_points
            ),
            max_num_neighbors=max_num_neighbors,
            num_interp_points=num_interp_points,
            use_vectorized=use_vectorized,
            sub_box_info=self.sub_boxes.as_dict if use_jackknifed else None,
            run_jackknife=use_jackknifed,
            use_population_jk=use_population_jk,
            use_fast_population_jk=use_fast_population_jk,
            use_data_randoms=use_data_randoms
        )

    def get_two_point_radial_bounds(
            self,
            comoving_box_size: float,
            num_query_points: int,
            max_num_neighbors: int,
            target_pcdf_value: float = 0.01,
            num_interp_points: int = 1000,
            sampling_mass_def: str = "all",
            lower_log_mass_bin_limit: float = -np.inf,
            upper_log_mass_bin_limit: float = np.inf,
            use_vectorized: bool = True,
            use_comoving: bool = True,
            use_jackknifed: bool = True,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            num_splits: int = 1,
            use_data_randoms: bool = False,
        ) -> tuple[float, float]:

        knn_data = self.get_nn_distribution(
            comoving_box_size=comoving_box_size,
            num_query_points=num_query_points,
            max_num_neighbors=max_num_neighbors,
            num_interp_points=num_interp_points,
            sampling_mass_def=sampling_mass_def,
            lower_log_mass_bin_limit=lower_log_mass_bin_limit,
            upper_log_mass_bin_limit=upper_log_mass_bin_limit,
            use_vectorized=use_vectorized,
            return_in_comoving=use_comoving,
            use_jackknifed=use_jackknifed,
            num_splits=num_splits,
            use_population_jk=use_population_jk,
            use_fast_population_jk=use_fast_population_jk,
            use_data_randoms=use_data_randoms,
        )

        r_low, r_high = knn_data.get_two_point_radial_bounds(
            target_pcdf_value=target_pcdf_value,
            for_data_randoms=use_data_randoms
        )

        return (
            (r_low, r_high)
            if use_comoving else 
            (r_low * self.moment.scale_factor, r_high * self.moment.scale_factor)
        )


    @property
    def abundance_model(self) -> AsymptoticAbundanceModel:
        return self.mass_defs.abundance_model
    
    def get_abundance_fit(self, mass_def: str) -> AsymptoticAbundanceFit:
        return self.abundance_model[mass_def]
    
    def compute_mass_functions(
            self, 
            volume: float,
            present_day_density: float,
            bin_count: int = 50,
            transfer: str = "eisenstein98",
            corrections: bool = True,
            is_comoving_volume: bool = True,
            run_jackknife: bool = False,
            run_both_jk: bool = False,
            num_splits: int = 1, 
            run_fits: bool = True, 
            run_cleaning: bool = True,
            use_ps_file: bool = False,
            ps_dir: Path | None = None,
            verbose: bool = False,
            cosmo: Cosmology | None = None,
            mass_bins: dict[str, dict[str, np.ndarray]] | None = None,
            return_fold_samples: bool = False,
        ) -> None:
        ''' Compute the mass function of the objects in this collection '''

        if not (run_both_jk or run_jackknife):
            group_indices, subhalo_indices = None, None

        if self.sub_boxes is None:      
            self.setup_sub_boxes(np.cbrt(volume), num_splits)

        group_indices = self.sub_boxes.group_indices
        subhalo_indices = self.sub_boxes.subhalo_indices

        if use_ps_file:
            if ps_dir is None:
                raise ValueError(
                    "Must provide a path to the directory containing the power spectrum files"
                )
            ps_file_path = Path(ps_dir, f"for_colossus_{self.moment.snapshot_id:03d}.txt")
        else:
            ps_file_path = None

        self.mass_defs.compute_mass_functions(
            volume=volume,
            present_day_density=present_day_density,
            redshift=self.moment.redshift,
            bin_count=bin_count,
            transfer=transfer,
            corrections=corrections,
            run_jackknife=run_jackknife,
            run_both_jk=run_both_jk,
            is_comoving_volume=is_comoving_volume,
            sub_box_group_idxs=group_indices,
            sub_box_subhalo_idxs=subhalo_indices,
            run_fits=run_fits,
            run_cleaning = run_cleaning,
            min_resolved_mass=self.min_resolved_mass,
            min_count_above_max_bin_hist=self.min_count_above_max_bin_hist,
            use_ps_file=use_ps_file,
            ps_file_path=ps_file_path,
            verbose=verbose,
            mass_bins=mass_bins,
            cosmo=cosmo,
            return_fold_samples=return_fold_samples
        )

        # pdb.set_trace()

    @property
    def halo_mass_function(self) -> HaloMassFunction:
        return self.mass_defs.halo_mass_function

    # def get_top_n_by_mass(self, mass_def: str, n: int) -> HaloPopulation:
    #     masses = self.mass_defs[mass_def].mass
    #     top_n_indices = np.argsort(masses)[-n:]
    #     return self.get_objects(self.ids[top_n_indices])
    
    # def get_bottom_n_by_mass(self, mass_def: str, n: int) -> HaloPopulation:
    #     masses = self.mass_defs[mass_def].mass
    #     bottom_n_indices = np.argsort(masses)[:n]
    #     return self.get_objects(self.ids[bottom_n_indices])

    def get_mass_function(self, mass_def: str) -> MassFunctionData:
        return self.halo_mass_function[mass_def]
    
    def get_mf_at_mass(
            self, 
            mass_def: str, 
            mass: float, 
            is_logged: bool = False,
            for_data: bool = True
        ) -> MassFunctionValue | None:
        return self.get_mass_function(mass_def).get_value(mass, is_logged, for_data)

    def display_mass_function(
            self, 
            mass_def: str, 
            mf_type: str, 
            show_raw: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:
        ''' Display the mass function of the objects in this collection '''

        self.mass_defs.display_mass_function(
            mass_def, 
            mf_type, 
            show_raw=show_raw,
            ax_main=ax_main,
            ax_resid=ax_resid,
        )

    def add_peak_heights(self, peaks: DensityPeaks) -> None:
        self.mass_defs.add_peak_heights(peaks)

    def get_mass_bins(self, mass_def: str) -> np.ndarray:
        return getattr(self.mass_defs, mass_def).mass_bins
    
    def get_round_one_mass_bins(self, mass_def: str) -> np.ndarray:
        return getattr(self.mass_defs, mass_def).round_one_mass_bins
    
    @property
    def mass_bins_by_mass_def(self) -> dict[str, np.ndarray]:
        return {
            getattr(self.mass_defs, mass_def).mapping.key : self.get_mass_bins(mass_def) 
            for mass_def in self.mass_defs.contained_definitions
            if getattr(self.mass_defs, mass_def) is not None
        }
    
    @property
    def round_one_mass_bins_by_mass_def(self) -> dict[str, np.ndarray]:
        return {
            getattr(self.mass_defs, mass_def).mapping.key : self.get_round_one_mass_bins(mass_def) 
            for mass_def in self.mass_defs.contained_definitions
            if getattr(self.mass_defs, mass_def) is not None
        }

    def split_by_mass(self, mass_def: str) -> dict[int, HaloPopulation]:
        mass_bins = self.get_mass_bins(mass_def)
        mass_values = getattr(self.mass_defs, mass_def).mass
        return {
            i: self.get_objects(np.where((mass_values > bin_min) & (mass_values <= bin_max))[0])
            for i, (bin_min, bin_max) in enumerate(zip(mass_bins[:-1], mass_bins[1:]))
        }

    # Add any additional arguments that you may need to as well
    def compute_halo_halo_correlations(
            self,
            comoving_box_size: float,
            r_min: float,
            r_max: float,
            num_bins: int, 
            position_mask: np.ndarray | None = None,
            use_natural: bool = False,
            return_in_comoving: bool = True,
            run_jackknife: bool = True,
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            eps: float = 1e-12,
            return_folds: bool = False
        ) -> TwoPointCorrelationData:

        # Select positions (in comoving units for the TPCF engine)
        positions = (
            self.comoving_positions[position_mask]
            if position_mask is not None else
            self.comoving_positions
        )

        # Ensure sub-box info is available for jackknife estimation
        if run_jackknife and self.sub_boxes is None:
            self.setup_sub_boxes(comoving_box_size, num_splits=1)
        
        # pdb.set_trace()

        tpcf_data = TwoPointCorrelationData.from_data(
            comoving_coordinates=positions,
            rmin=r_min,
            rmax=r_max,
            nbins=num_bins,
            boxsize=comoving_box_size,
            sub_box_info=self.sub_boxes.as_dict if run_jackknife else None,
            use_natural=use_natural,
            run_jackknife=run_jackknife,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
            eps=eps,
            return_folds=return_folds
        )

        # Optionally return radii in physical (non-comoving) units
        if not return_in_comoving:
            tpcf_data.convert_to_physical(self.moment.scale_factor)

        return tpcf_data
    

    # Add any additional arguments that you may need to as well
    def get_halo_halo_correlations(
            self,
            comoving_box_size: float, 
            num_query_points: int,
            max_num_neighbors: int,
            num_splits: int = 1,
            num_interp_points: int = 1000,
            num_radial_bins: int = 10, 
            mass_def: str = "vir",
            target_pcdf_value: float = 0.01,
            min_resolved_mass: float = 0.0,
            position_mask: np.ndarray | None = None,
            use_natural: bool = False,
            return_in_comoving: bool = True,
            run_jackknife: bool = True,
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            eps: float = 1e-12,
            return_folds: bool = False,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            radial_range: tuple[float, float] | None = None,
            use_data_randoms: bool = False
        ) -> TwoPointCorrelationData:

        if radial_range is not None:
            low_bound, high_bound = radial_range
        else:
            low_bound, high_bound = self.get_two_point_radial_bounds(
                comoving_box_size=comoving_box_size,
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                num_interp_points=num_interp_points,
                target_pcdf_value=target_pcdf_value,
                use_jackknifed=run_jackknife,
                use_comoving=True,
                num_splits=num_splits,
                use_population_jk=use_population_jk,
                use_fast_population_jk=use_fast_population_jk,
                use_data_randoms=use_data_randoms
            )

        r_min = max(low_bound, eps)
        r_max = min(high_bound, (comoving_box_size / 2.0) - 0.00001)

        print(f"Radial bounds: {r_min} cMpc/h, {r_max} cMpc/h \n")

        resolved_mask = 10.0**self.get_masses_by_key(mass_def) >= min_resolved_mass
        if position_mask is not None:
            population_mask = np.logical_and(position_mask, resolved_mask)
        else:
            population_mask = resolved_mask

        if run_jackknife and self.sub_boxes is None:
            self.setup_sub_boxes(comoving_box_size, num_splits)

        # Figure out empty jackknife case...
        return self.compute_halo_halo_correlations(
            comoving_box_size=comoving_box_size,
            r_min=r_min,
            r_max=r_max,
            num_bins=num_radial_bins,
            position_mask=population_mask,
            use_natural=use_natural,
            return_in_comoving=return_in_comoving,
            run_jackknife=run_jackknife,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
            eps=eps,
            return_folds=return_folds
        )


    
    '''
      Make a property binned halo-halo correlations function that this method
      calls on, so that we can easily compute correlations for different halo properties.

      Good for mass assembly bias studies!

    '''
    
    def get_mass_binned_halo_halo_correlations(
            self,
            comoving_box_size: float, 
            num_query_points: int,
            max_num_neighbors: int,
            num_splits: int = 1,
            num_interp_points: int = 1000,
            num_radial_bins: int = 10,
            target_pcdf_value: float = 0.01,
            min_resolved_mass: float = 0.0,  
            mass_def: str = "vir", 
            target_mass_bins: dict[str, tuple[float, float]] | None = None,
            use_natural: bool = False,
            return_in_comoving: bool = True,
            use_jackknifed: bool = True,
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            eps: float = 1e-12,
            use_round_one_bins: bool = True,
            return_folds: bool = False,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            radial_range: tuple[float, float] | None = None,
            use_data_randoms: bool = False,
        ) -> dict[int | str, TwoPointCorrelationData]:

        if (target_mass_bins is None) and use_round_one_bins:
            round_one_bins = self.round_one_mass_bins_by_mass_def[mass_def]
            mass_bins = {
                round_one_bins[i] : (round_one_bins[i], round_one_bins[i + 1])
                for i in range(len(round_one_bins) - 1)
            }
        elif (target_mass_bins is None):
            int_mass_bins = self.mass_bins_by_mass_def[mass_def]
            mass_bins = {
                int_mass_bins[i] : (int_mass_bins[i], int_mass_bins[i + 1])
                for i in range(len(int_mass_bins) - 1)
            }
        else:
            mass_bins = target_mass_bins

        if radial_range is not None:
            low_bound, high_bound = radial_range
        else:
            low_bound, high_bound = self.get_two_point_radial_bounds(
                comoving_box_size=comoving_box_size,
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                num_interp_points=num_interp_points,
                target_pcdf_value=target_pcdf_value,
                use_jackknifed=use_jackknifed,
                num_splits=num_splits,
                use_comoving=True,
                use_population_jk=use_population_jk,
                use_fast_population_jk=use_fast_population_jk,
                use_data_randoms=use_data_randoms
            )

        r_min = max(low_bound, eps)
        r_max = min(high_bound, comoving_box_size / 2.0)

        print(f"Radial bounds: {r_min} cMpc/h, {r_max} cMpc/h \n")

        # Ensure sub-box info is available for jackknife estimation when requested
        if use_jackknifed and self.sub_boxes is None:
            raise ValueError(
                "sub_boxes must be initialized to compute jackknifed two-point correlations. "
                "Call initialize_empty_sub_boxes(box_size, num_splits) or setup_sub_boxes(...) first."
            )

        mass_values = self.get_masses_by_key(mass_def)
        resolved_mask = 10.0**mass_values >= min_resolved_mass

        return group_halo_halo_correlations_by_property(
            comoving_positions=self.comoving_positions,
            prop_values=mass_values,
            resolved_mask=resolved_mask,
            value_bins=mass_bins,
            r_min=r_min,
            r_max=r_max,
            num_radial_bins=num_radial_bins,
            comoving_box_size=comoving_box_size,
            scale_factor=self.moment.scale_factor,
            sub_box_info=(self.sub_boxes.as_dict if use_jackknifed else None),
            use_natural=use_natural,
            use_jackknifed=use_jackknifed,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
            eps=eps,
            return_in_comoving=return_in_comoving,
            return_folds=return_folds
        )



@define(slots=True)
class HaloPopulationEvolutionData(EvolutionData, ABC): 
    data: OrderedDict[int, HaloPopulation] 


    def __repr__(self) -> str:
        return super().__repr__()
        

    @classmethod
    @abstractmethod
    def from_data(
            cls, data: dict, 
            moments: MomentsInTime, 
            hubble: float, 
            min_resolved_mass: float,
            min_count_above_max_bin_hist: int,
            *args, **kwargs
        ) -> HaloPopulationEvolutionData:

        ... 

    @classmethod
    @abstractmethod
    def from_directory(
            cls, sim_dir: Path, 
            cosmo: Cosmology,
            in_comoving: bool,
            is_batched: bool,
            # fof_data_dir_name: str | None,
            min_resolved_mass: float,
            min_count_above_max_bin_hist: int,
            *args, **kwargs
        ) -> HaloPopulationEvolutionData:

        ... 

    @property
    def descendants(self) -> HaloPopulation:
        return self.data[max(self.data.keys())]
    
    def add_physical_catalogs(
            self,
            phys_catalog_dir: Path,
            comoving_box_size: int,
            num_particles: int,
            cosmo: Cosmology,
            seed_num: int = 1,
            sampling_mdef: str = "200m",
        ) -> None:

        full_phys_catalogs = load_all_physical_catalogs(
            directory=phys_catalog_dir,
            comoving_box_size=comoving_box_size,
            num_particles=num_particles,
            sampling_mdef=sampling_mdef,
            seed_num = (-1 if (cosmo.name == "primary") else seed_num),
            is_from_primary=(cosmo.name == "primary"),
            return_concatenated=True
        )

        for snapshot_id in self.data:
            if snapshot_id not in full_phys_catalogs: continue
            self.data[snapshot_id].mass_defs.add_physical_catalog_data(
                physical_catalog=full_phys_catalogs[snapshot_id],
                fof_group_ids=self.data[snapshot_id].ids
            )
    
    def get_mass_ratios_evo(
            self,
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> dict[int, np.ndarray]:

        ratios = {}
        for snapshot_id in self.data:
            try:
                ratios[snapshot_id] = self.data[snapshot_id].get_mass_ratios(
                    mass_def_key_numerator=mass_def_key_numerator, 
                    mass_def_key_denominator=mass_def_key_denominator,
                    return_full=return_full
                )
            except ValueError as error:
                continue    

        return ratios
    
    def get_radii_ratios_evo(
            self,
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> dict[int, np.ndarray]:
        
        ratios = {}
        for snapshot_id in self.data:
            try:
                ratios[snapshot_id] = self.data[snapshot_id].get_radii_ratios(
                    mass_def_key_numerator=mass_def_key_numerator, 
                    mass_def_key_denominator=mass_def_key_denominator,
                    return_full=return_full
                )
            except ValueError as error:
                continue    

        return ratios
    
    def get_velocity_ratios_evo(
            self,
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> dict[int, np.ndarray]:

        ratios = {}
        for snapshot_id in self.data:
            try:
                ratios[snapshot_id] = self.data[snapshot_id].get_velocity_ratios(
                    mass_def_key_numerator=mass_def_key_numerator, 
                    mass_def_key_denominator=mass_def_key_denominator,
                    return_full=return_full
                )
            except ValueError as error:
                continue    

        return ratios

    def add_peak_heights(self, peaks_evo: DensityPeaksEvo) -> None:
        for snapshot_id, peaks in peaks_evo.data.items():
            if snapshot_id not in self.data: continue
            self.data[snapshot_id].add_peak_heights(peaks)

    def compute_evo_mass_functions(
            self, 
            volume: float,
            present_day_density: float,
            bin_count: int = 50,
            transfer: str = "eisenstein98",
            corrections: bool = True,
            is_comoving_volume: bool = True,
            run_jackknife: bool = False,
            run_both_jk: bool = False,
            num_splits: int = 1,
            run_fits: bool = True,
            run_cleaning: bool = True,
            use_ps_from_sim: bool = False,
            ps_dir: Path | None = None,
            verbose: bool = False,
            cosmo: Cosmology | None = None,
            mass_bins_evo: dict[int, dict[str, dict[str, np.ndarray]]] | None = None,
            return_fold_samples: bool = False,
        ) -> None:

        for snapshot_id in self.data:

            use_mass_bins = (
                (mass_bins_evo is not None)
                and 
                (snapshot_id in mass_bins_evo)
            )
            self.data[snapshot_id].compute_mass_functions(
                volume=volume,
                present_day_density=present_day_density,
                bin_count=bin_count,
                transfer=transfer,
                corrections=corrections,
                is_comoving_volume=is_comoving_volume,
                run_jackknife=run_jackknife,
                run_both_jk=run_both_jk,
                num_splits=num_splits,
                run_fits=run_fits,
                run_cleaning=run_cleaning,
                use_ps_file=use_ps_from_sim,
                ps_dir=ps_dir,
                verbose=verbose,
                mass_bins=(
                    mass_bins_evo[snapshot_id]
                    if use_mass_bins else 
                    None 
                ),
                cosmo=cosmo,
                return_fold_samples=return_fold_samples
            )

    def snapshots_with_mass_def_data(self, mass_def: str) -> list[int]:
        snapshot_ids = []
        for snapshot_id in self.data:
            try:
                self.data[snapshot_id].mass_defs[mass_def]
            except (ValueError, KeyError) as error:
                print(f"Snapshot {snapshot_id} does not have {mass_def} data")
                print(f"Error: {error}")
                continue
            if self.data[snapshot_id].mass_defs[mass_def] is not None:
                snapshot_ids.append(snapshot_id)
        return snapshot_ids

    
    @property
    def mass_def_evos(self) -> MassDefinitionsEvo:
        evo_data = {
            snapshot_id : halos.mass_defs
            for snapshot_id, halos in self.data.items()
        }

        final_mass_defs = self.data[max(self.data)].mass_defs

        mass_def_evo = MassDefinitionsEvo()

        for mass_def in final_mass_defs.contained_keys:

            if (mass_def in  ("__weakref__", "__dict__")): continue

            snapshot_ids = self.snapshots_with_mass_def_data(mass_def)
            
            if len(snapshot_ids) == 0: continue

            mass_def_evo[mass_def] = MassDefinitionEvo(
                moments=self.moments.get_subset_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snapshot_ids
                ),
                data=OrderedDict({
                    snapshot_id : evo_data[snapshot_id][mass_def]
                    for snapshot_id in snapshot_ids
                })
            )

        return mass_def_evo


    @property
    def mass_function_evo(self) -> HaloMassFunctionEvoData:
        return self.mass_def_evos.halo_mass_function_evo

    # def get_mass_function_evo(self, mass_defs: str | Collection[str]) -> HaloMassFunctionEvoData:
    #     if isinstance(mass_defs, str):
    #         mass_defs = [mass_defs]
    #     return self.mass_def_evos.get_halo_mass_function_evo(mass_defs)
    
    @property
    def jackknife_mass_function_evo(self) -> HaloMassFunctionEvoData:
        return self.mass_def_evos.jackknife_mass_function_evo
    
    @property
    def universal_mass_function_evo(self) -> HaloMassFunctionEvoData:
        return self.mass_def_evos.universal_mass_function_evo
    
    @property
    def abundance_evo_model(self) -> AsymptoticAbundanceEvoModel:
        return self.mass_def_evos.abundance_evo_model

    @property
    def abundance_accumulation(self) -> PopulationAbundanceAccumulationHistories:
        return self.mass_function_evo.accumulation_histories
    
    def get_mass_def_evo(self, mass_def: str) ->  MassDefinitionEvo:
        return self.mass_def_evos[mass_def]
    
    def get_mass_function_evo(self, mass_def: str) -> MassFunctionEvoData:
        return self.mass_def_evos.get_mass_function_evo(mass_def)
    
    def get_mass_function_fit_evo(self, mass_def: str) -> AsymptoticAbundanceFitEvo:
        return self.abundance_evo_model[mass_def]
    

    def get_nn_distribution_evo(
            self, 
            comoving_box_size: float,
            num_query_points: int,
            max_num_neighbors: int,
            num_interp_points: int = 1000,
            sampling_mass_def: str = "all",
            return_in_comoving: bool = False,
            use_vectorized: bool = True,
            use_jackknifed: bool = False,
            num_splits: int = 1,
            lower_log_mass_bin_limit: float = -np.inf,
            upper_log_mass_bin_limit: float = np.inf,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
        ) -> kNNDistributionEvoData:

        data = OrderedDict()
        moments = MomentsInTime()

        for snapshot_id, halos in tqdm(
            sorted(self.data.items(), key=lambda item: item[0]),
            desc="Computing kNN Distribution for halos", 
            unit="snapshot",
        ):
            if halos.size == 0: continue
            try:
                data[snapshot_id] = halos.get_nn_distribution(
                    comoving_box_size=comoving_box_size,
                    num_query_points=num_query_points,
                    max_num_neighbors=max_num_neighbors,
                    num_interp_points=num_interp_points,
                    sampling_mass_def=sampling_mass_def,
                    return_in_comoving=return_in_comoving,
                    use_vectorized=use_vectorized,
                    use_jackknifed=use_jackknifed,
                    num_splits=num_splits,
                    lower_log_mass_bin_limit=lower_log_mass_bin_limit,
                    upper_log_mass_bin_limit=upper_log_mass_bin_limit,
                    use_population_jk=use_population_jk,
                    use_fast_population_jk=use_fast_population_jk
                )
            except ValueError as error:
                print(f"Skipping snapshot {snapshot_id} due to error: {error}")
                continue
            moments.add_moment(halos.moment)

        return kNNDistributionEvoData(moments=moments, data=data)
    

    def get_halo_halo_correlations_evo(
            self,
            comoving_box_size: float, 
            num_query_points: int,
            max_num_neighbors: int,
            num_splits: int = 1,
            num_interp_points: int = 1000,
            num_radial_bins: int = 10, 
            mass_def: str = "vir",
            target_pcdf_value: float = 0.01,
            min_resolved_mass: float = 0.0,
            position_mask: np.ndarray | None = None,
            use_natural: bool = False,
            return_in_comoving: bool = True,
            run_jackknife: bool = True,
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            eps: float = 1e-12,
            return_folds: bool = False,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            radial_range: tuple[float, float] | None = None
        ) -> TwoPointCorrelationEvoData:

        data = OrderedDict()
        moments = MomentsInTime()

        for snapshot_id, halos in tqdm(
            sorted(self.data.items(), key=lambda item: item[0]),
            desc="Computing Two-Point Correlations for halos", 
            unit="snapshot",
        ):
            if halos.size == 0: continue
            try:
                data[snapshot_id] = halos.get_halo_halo_correlations(
                    comoving_box_size=comoving_box_size,
                    num_query_points=num_query_points,
                    max_num_neighbors=max_num_neighbors,
                    num_interp_points=num_interp_points,
                    num_splits=num_splits,
                    num_radial_bins=num_radial_bins,
                    mass_def=mass_def,
                    target_pcdf_value=target_pcdf_value,
                    min_resolved_mass=min_resolved_mass,
                    position_mask=position_mask,
                    use_natural=use_natural,
                    return_in_comoving=return_in_comoving,
                    run_jackknife=run_jackknife,
                    random_multiplier=random_multiplier,
                    rng_seed=rng_seed,
                    eps=eps,
                    return_folds=return_folds,
                    use_population_jk=use_population_jk,
                    use_fast_population_jk=use_fast_population_jk,
                    radial_range=radial_range
                )

            except ValueError as value_error:
                print(f"Skipping snapshot {snapshot_id} due to error: {value_error}")
                continue

            except RuntimeError as runtime_error:
                print(
                    f"Skipping snapshot {snapshot_id} due to pair counting "
                    f"error: {runtime_error}"
                )
                continue


            moments.add_moment(halos.moment)

        return TwoPointCorrelationEvoData(moments=moments, data=data)
    

    def display_mass_def_accumulation_histories(
            self, 
            mass_def_key: str,
            with_fits: bool = True,
            mf_type: str = "differential",
            normalize: bool = True, 
            normalizing_scale_factor: float = 10.0,
            top_resid_min: float | None = None,
            top_resid_max: float | None = None,
            bottom_resid_min: float | None = None,
            bottom_resid_max: float | None = None,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            top_main_y_label_text_size: int = 10,
            top_resid_y_label_text_size: int = 10,
            bottom_main_y_label_text_size: int = 10,
            bottom_resid_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            top_main_y_tick_text_size: int = 10,
            top_resid_y_tick_text_size: int = 10,
            bottom_main_y_tick_text_size: int = 10,
            bottom_resid_y_tick_text_size: int = 10,
            show_top_log_y: bool = False,
            show_bottom_log_y: bool = False,
            return_fig: bool = False,
            ax_main_top: plt.Axes | None = None,
            ax_resid_top: plt.Axes | None = None,
            ax_main_bottom: plt.Axes | None = None,
            ax_resid_bottom: plt.Axes | None = None,
            color_palette: str = "viridis",
            populations_to_display: list[int] | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:

        self.abundance_accumulation.display_population_accumulation_histories(
            mass_def_key=mass_def_key,
            with_fits=with_fits,
            mf_type=mf_type,
            normalize=normalize,
            normalizing_scale_factor=normalizing_scale_factor,
            top_resid_min=top_resid_min,
            top_resid_max=top_resid_max,
            bottom_resid_min=bottom_resid_min,
            bottom_resid_max=bottom_resid_max,
            legend_text_size=legend_text_size,
            x_label_text_size=x_label_text_size,
            top_main_y_label_text_size=top_main_y_label_text_size,
            top_resid_y_label_text_size=top_resid_y_label_text_size,
            bottom_main_y_label_text_size=bottom_main_y_label_text_size,
            bottom_resid_y_label_text_size=bottom_resid_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            top_main_y_tick_text_size=top_main_y_tick_text_size,
            top_resid_y_tick_text_size=top_resid_y_tick_text_size,
            bottom_main_y_tick_text_size=bottom_main_y_tick_text_size,
            bottom_resid_y_tick_text_size=bottom_resid_y_tick_text_size,
            show_top_log_y=show_top_log_y,
            show_bottom_log_y=show_bottom_log_y,
            return_fig=return_fig,
            ax_main_top=ax_main_top,
            ax_resid_top=ax_resid_top,
            ax_main_bottom=ax_main_bottom,
            ax_resid_bottom=ax_resid_bottom,
            color_palette=color_palette,
            populations_to_display=populations_to_display
        )

    def display_accumulation_histories(
            self, 
            with_fits: bool = True,
            mf_type: str = "differential",
            normalize: bool = True, 
            normalizing_scale_factor: float = 10.0,
            top_resid_min: float | None = None,
            top_resid_max: float | None = None,
            bottom_resid_min: float | None = None,
            bottom_resid_max: float | None = None,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            top_main_y_label_text_size: int = 10,
            top_resid_y_label_text_size: int = 10,
            bottom_main_y_label_text_size: int = 10,
            bottom_resid_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            top_main_y_tick_text_size: int = 10,
            top_resid_y_tick_text_size: int = 10,
            bottom_main_y_tick_text_size: int = 10,
            bottom_resid_y_tick_text_size: int = 10,
            show_top_log_y: bool = False,
            show_bottom_log_y: bool = False,
            return_fig: bool = False,
            ax_main_top: plt.Axes | None = None,
            ax_resid_top: plt.Axes | None = None,
            ax_main_bottom: plt.Axes | None = None,
            ax_resid_bottom: plt.Axes | None = None,
            color_palette: str = "viridis",
            populations_to_display: list[int] | None = None,
            mass_defs_to_display: list[str] | None = None,
            # top_legend_xloc: float = 1.0,
            # top_legend_yloc: float = 1.0,
            bottom_legend_xloc: float = 1.0,
            bottom_legend_yloc: float = -0.25,
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:

        self.abundance_accumulation.display(
            with_fits=with_fits,
            mf_type=mf_type,
            normalize=normalize,
            normalizing_scale_factor=normalizing_scale_factor,
            top_resid_min=top_resid_min,
            top_resid_max=top_resid_max,
            bottom_resid_min=bottom_resid_min,
            bottom_resid_max=bottom_resid_max,
            legend_text_size=legend_text_size,
            x_label_text_size=x_label_text_size,
            top_main_y_label_text_size=top_main_y_label_text_size,
            top_resid_y_label_text_size=top_resid_y_label_text_size,
            bottom_main_y_label_text_size=bottom_main_y_label_text_size,
            bottom_resid_y_label_text_size=bottom_resid_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            top_main_y_tick_text_size=top_main_y_tick_text_size,
            top_resid_y_tick_text_size=top_resid_y_tick_text_size,
            bottom_main_y_tick_text_size=bottom_main_y_tick_text_size,
            bottom_resid_y_tick_text_size=bottom_resid_y_tick_text_size,
            show_top_log_y=show_top_log_y,
            show_bottom_log_y=show_bottom_log_y,
            return_fig=return_fig,
            ax_main_top=ax_main_top,
            ax_resid_top=ax_resid_top,
            ax_main_bottom=ax_main_bottom,
            ax_resid_bottom=ax_resid_bottom,
            color_palette=color_palette,
            populations_to_display=populations_to_display,
            mass_defs_to_display=mass_defs_to_display,
            bottom_legend_xloc=bottom_legend_xloc,
            bottom_legend_yloc=bottom_legend_yloc
        )

    def display_mass_function_evo(
            self,
            target_scale_factors: Collection[float],
            mf_type: str = "differential",
            with_fits: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
            show_raw: bool = False,
            show_legend: bool = True,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            main_y_label_text_size: int = 10,
            resid_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            main_y_tick_text_size: int = 10,
            resid_y_tick_text_size: int = 10,
            return_fig: bool = False,
            resid_min: float | None = None,
            resid_max: float | None = None,
            color_palette: str = "crest",
            mass_defs_to_display: list[str] | None = None,
            bottom_legend_xloc: float = 0.9,
            bottom_legend_yloc: float = -0.15,
        ) -> None: 

        self.mass_function_evo.display(
            target_scale_factors=target_scale_factors,
            mf_type=mf_type,
            with_fits=with_fits,
            ax_main=ax_main,
            ax_resid=ax_resid,
            show_raw=show_raw,
            legend_text_size=legend_text_size,
            x_label_text_size=x_label_text_size,
            main_y_label_text_size=main_y_label_text_size,
            resid_y_label_text_size=resid_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            main_y_tick_text_size=main_y_tick_text_size,
            resid_y_tick_text_size=resid_y_tick_text_size,
            show_legend=show_legend,
            return_fig=return_fig,
            resid_min=resid_min,
            resid_max=resid_max,
            color_palette=color_palette,
            mass_defs_to_display=mass_defs_to_display,
            bottom_legend_xloc=bottom_legend_xloc,
            bottom_legend_yloc=bottom_legend_yloc
        )

    def display_mass_def_mass_function_evo(
            self,
            mass_def_key: str,
            target_scale_factors: Collection[float],
            mf_type: str = "differential",
            with_fits: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
            show_raw: bool = False,
            show_legend: bool = True,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            main_y_label_text_size: int = 10,
            resid_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            main_y_tick_text_size: int = 10,
            resid_y_tick_text_size: int = 10,
            return_fig: bool = False,
            resid_min: float | None = None,
            resid_max: float | None = None,
            plot_linestyle: tuple | None = None,
            marker_style: str | None = '.',
            color_palette: str = "crest",
        ) -> None:

        mf_evo = self.mass_function_evo[mass_def_key]

        if color_palette != mf_evo.color_palette:
            mf_evo.update_colormap(color_palette)

        mf_evo.display(
            mass_def_key=mass_def_key,
            target_scale_factors=target_scale_factors,
            mf_type=mf_type,
            with_fits=with_fits,
            ax_main=ax_main,
            ax_resid=ax_resid,
            show_raw=show_raw,
            show_legend=show_legend,
            legend_text_size=legend_text_size,
            x_label_text_size=x_label_text_size,
            main_y_label_text_size=main_y_label_text_size,
            resid_y_label_text_size=resid_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            main_y_tick_text_size=main_y_tick_text_size,
            resid_y_tick_text_size=resid_y_tick_text_size,
            return_fig=return_fig,
            resid_min=resid_min,
            resid_max=resid_max,
            plot_linestyle=plot_linestyle,
            marker_style=marker_style
        )


