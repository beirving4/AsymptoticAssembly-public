from __future__ import annotations

import numpy as np, pdb 
import matplotlib.pyplot as plt

from pathlib import Path    
from collections import defaultdict
from astropy import units as u
from attrs import define, field, fields
from astropy import constants as const
from collections import OrderedDict, Counter
from collections.abc import Collection

from ..simulation.evo import EvolutionData
from ..abundance.mass_function import MassFunctionData, MassFunctionEvoData
from ..abundance.halo import HaloMassFunction, HaloMassFunctionEvoData
from .base import MassDefinitionDependentType, MassDefinitionDependentEvoType
from ..abundance.accumulation import (
    AccumulationHistoryData, 
    AbundanceAccumulationHistories,
    PopulationAbundanceAccumulationHistories
)
from ..distribution.slope import (
    get_circular_velocity,
    get_filtered_mass_distribution_milestones, 
    get_mass_distribution_milestones
)
from ..cosmo.model import Cosmology
from ..fields.peaks import DensityPeaks, DensityPeaksEvo
from ..model.mass_function import (
    AsymptoticAbundanceModel,
    AsymptoticAbundanceEvoModel
)
from .mapping import (
    DensityThresholdMap, DensityThresholdProperty, global_mass_def_mapping
)

# @define(slots=True) 
# class MassDefinitionValue:
#     mass: float
#     radius: float | None = field(default=None)
#     density: float | None = field(default=None)
#     velocity: float | None = field(default=None)
#     hubble: float | None = field(default=None, repr=False)

#     def __attrs_post_init__(self):


@define(slots=True)
class MassDefinition:
    mapping: DensityThresholdMap 
    mass: float | np.ndarray
    radius: float | np.ndarray | None = field(default=None)
    hubble: float | None = field(default=None, repr=False) 

    peak_heights: float | np.ndarray | None = field(default=None, repr=False)

    _density: float | None = field(default=None, repr=False)

    jackknifed_mf: MassFunctionData | None = field(default=None, repr=False)
    universal_mf: MassFunctionData | None = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._density is None and self.radius is not None:
            self._set_density_by_radius(self.radius)

    def __getitem__(self, idxs: int | Collection[int] | None = None) -> MassDefinition:
        if isinstance(self.mass, (float, np.float32, np.float64)):
            mass = self.mass
        elif isinstance(self.mass, np.ndarray) and idxs is not None:
            mass = self.mass[idxs]
        else:
            raise ValueError("Index must be provided for array of masses")

        if self.radius is None:
            radius = None
        elif isinstance(self.radius,  (float, np.float32, np.float64)):
            radius = self.radius
        elif isinstance(self.radius, np.ndarray) and idxs is not None:
            radius = self.radius[idxs]
        else:
            raise ValueError("Index must be provided for array of radii")
        
        return MassDefinition(
            mass=mass,
            radius=radius,
            hubble=self.hubble,
            mapping=self.mapping
        )

    def __repr__(self) -> str:
        if isinstance(self.mass, np.ndarray):
            return (
                f"MassDefinition({self.mapping.key}, "
                f"M={10.**min(self.mass):.2e}-{10.0**max(self.mass):.2e})"
            )
        else:
            return f"MassDefinition({mapping.key}, M={10.0**self.mass:.2e})"
        

    def get_subset(self, idxs: Collection[int]) -> MassDefinition:
        if not isinstance(idxs, np.ndarray):
            idxs = np.asarray(idxs)

        return MassDefinition(
            mass=self.mass[idxs],
            radius=self.radius[idxs] if self.radius is not None else None,
            hubble=self.hubble,
            mapping=self.mapping
        )
    
    @classmethod
    def for_template(
            cls, 
            mapping: DensityThresholdMap, 
            num_halos: int, 
            hubble: float | None = None
        ) -> MassDefinition:
        return cls(
            mapping=mapping,
            mass=np.full(num_halos, np.nan),
            radius=np.full(num_halos, np.nan),
            hubble=hubble
        )  

    def get_lagrangian_radius(self, critical_density: float) -> float | np.ndarray:
        return ((3 * 10.**self.mass) / (4 * np.pi * critical_density))**(1/3)
    
    # @classmethod
    # def from_spin_and_shapes_data(
    #         cls, 
    #         spin_and_shapes_data: dict[int, dict[str, np.ndarray]],
    #         mapping: DensityThresholdMap,
    #         fof_group_ids: np.ndarray,
    #         mass_def_idx: int,
    #         hubble: float | None = None
    #     ) -> MassDefinition:
    
    #     base = cls.for_template(mapping, fof_group_ids.size, hubble=hubble)
    
    #     collected_group_ids, mass, radii = [], [], []

    #     for mass_bin_data in spin_and_shapes_data.values():
    #         pdb.set_trace()
    #         collected_group_ids.append(mass_bin_data['group_ids'])
    #         mass.append(mass_bin_data['masses'][:, mass_def_idx])
    #         radii.append(mass_bin_data['radii'][:, mass_def_idx])
        

    #     collected_group_ids = np.concatenate(collected_group_ids)
    #     group_array_locs = np.intersect1d(
    #         fof_group_ids, 
    #         collected_group_ids,
    #         assume_unique=True,
    #         return_indices=True
    #     )[1]

    #     base.mass[group_array_locs] = np.concatenate(mass)
    #     base.radius[group_array_locs] = np.concatenate(radii)

    #     return base
    
    @classmethod
    def add_physical_catalog_data(
            cls, 
            physical_catalog_data: dict[int, dict[str, np.ndarray]],
            mapping: DensityThresholdMap,
            fof_group_ids: np.ndarray,
            mass_def_idx: int,
            hubble: float | None = None
        ) -> MassDefinition:
    
        # Move some of this logic into the MassDefinitions class as well
        # so that it doesn't have to be repeated
        base = cls.for_template(mapping, fof_group_ids.size, hubble=hubble)
    
        collected_group_ids, mass, radii = [], [], []

        for mass_bin_data in physical_catalog_data.values():
            collected_group_ids.append(mass_bin_data['group_ids'])
            mass.append(mass_bin_data['masses'][:, mass_def_idx])
            radii.append(mass_bin_data['radii'][:, mass_def_idx])
        

        collected_group_ids = np.concatenate(collected_group_ids)
        group_array_locs = np.intersect1d(
            fof_group_ids, 
            collected_group_ids,
            assume_unique=True,
            return_indices=True
        )[1]

        base.mass[group_array_locs] = np.concatenate(mass)
        base.radius[group_array_locs] = np.concatenate(radii)

        return base

    
    def add_peak_heights(self, peaks: DensityPeaks) -> None:
        self.peak_heights = peaks.heights(self.mass, from_masses=True)

    
    @property
    def velocity(self) -> float | np.ndarray | None:
        if self.radius is None:
            print("Radius has not yet been defined for this mass definition")
            return None
        return get_circular_velocity(10.0**self.mass, self.radius)
    
    @property
    def density(self) -> float | None:
        return self._density
    
    @property
    def volume(self) -> float | np.ndarray | None:
        if self.radius is None:
            print("Radius has not yet been defined for this mass definition")
            return None
        return 4/3 * np.pi * self.radius**3
    
    @density.setter
    def density(self, value: float) -> None:
        self._density = value

    @property
    def has_density(self) -> bool:
        return self._density is not None

    @property
    def mass_bins(self) -> np.ndarray:
        return np.unique(np.asarray(self.mass, dtype=int))

    @property
    def crossing_time(self) -> float | np.ndarray | None:
        if self.radius is None or self.velocity is None:
            print("Radius and velocity must be defined to compute crossing time")
            return None
        return 2 * self.radius / self.velocity
    
    @property
    def peak_height_bins(self) -> np.ndarray:
        if self.peak_heights is None:
            print("Peak heights have not yet been defined for this mass definition")
            return np.empty(0)
        return 2.0**np.unique(np.asarray(np.log2(self.peak_heights), dtype=int))

    @property
    def round_one_mass_bins(self) -> np.ndarray:
        return np.unique(np.asarray(np.round(self.mass, decimals=1)))

    @property
    def round_one_peak_height_bins(self) -> np.ndarray:
        if self.peak_heights is None:
            print("Peak heights have not yet been defined for this mass definition")
            return np.empty(0)
        log_heights = np.log2(self.peak_heights)
        return 2.0**np.unique(np.asarray(np.round(log_heights, decimals=1)))

    @property
    def num_halos_in_each_mass_bins(self) -> Counter:
        # So keys can be sorted in ascending order
        return Counter(np.asarray(self.mass[::-1], dtype=int))

    @property
    def mass_bin_idxs(self) -> np.ndarray:
        return np.digitize(self.mass, self.mass_bins)

    @property
    def mass_bin_locs(self) -> dict[int, np.ndarray]:
        all_locations = np.arange(len(self.mass))
        return {
            mass_bin : all_locations[self.mass_bin_idxs == (i+1)]
            for i, mass_bin in enumerate(self.mass_bins)
        }
    
    @property
    def mass_function(self) -> MassFunctionData | None:
        return (
            self.jackknifed_mf 
            if self.jackknifed_mf is not None 
            else self.universal_mf
        )
    
    @property
    def has_jackknifed_mf(self) -> bool:
        return self.jackknifed_mf is not None
    
    @property
    def has_universal_mf(self) -> bool:
        return self.universal_mf is not None
    
    @property
    def has_mf(self) -> bool:
        return self.mass_function is not None

    def _set_density_by_radius(self, radius: float | np.ndarray) -> None:
        
        if (not np.isfinite(radius).all()) or (not np.isfinite(self.mass).all()):
            self._density = None

        try:
            density = (3 * 10.**self.mass) / (4 * np.pi * radius**3)
        except OverflowError as error:
            print(f"Can't compute density: {error}")
            density = None

        # I believe this is here to account for little h, but still verify this
        # density /= (1 if self.hubble is None else self.hubble**2)

        if isinstance(self.mass, float) and isinstance(self.radius, float):
            self._density = density
        else: 
            self._density = np.asarray(density).mean()

    def update_radius(self, radius: float | np.ndarray) -> None:
        self.radius = radius
        self._set_density_by_radius(radius)

    def estimate_mass_def(self, mass: float) -> MassDefinition:
        radii = np.cbrt((3.0 * 10.**mass) / (4.0 * np.pi * self.density))
        return MassDefinition(
            mass=mass,
            radius=radii,
            hubble=self.hubble,
            mapping=self.mapping
        )
    
    def get_mass_sub_box_splits(self, ids: list[np.ndarray]) -> list[np.ndarray]:
        if isinstance(self.mass, float):    
            raise ValueError("Mass is not an array")
        return [self.mass[idxs] for idxs in ids]

    def display_mass_function(
            self, 
            mf_type: str, 
            show_raw: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None]:
        
        return self.mass_function.display(
            mass_def_key=self.mapping.mass.eqn, 
            mf_type=mf_type, 
            show_raw=show_raw,
            ax_main=ax_main,
            ax_resid=ax_resid,
        )

@define(slots=True)
class MassDefinitionEvo(EvolutionData):
    data: OrderedDict[int, MassDefinition]

    def __repr__(self) -> str:
        return super().__repr__()
    
    @property
    def num_halos_in_each_mass_bins(self) -> dict[int, Counter]:
        return {
            snap_idx : mass_def_data.num_halos_in_each_mass_bins
            for snap_idx, mass_def_data in self.data.items()
        }

    @property
    def snapshots_with_mf(self) -> np.ndarray:
        return np.asarray(
            [i for i, mass_def in self.data.items() if mass_def.has_mf]
        )
    
    @property
    def snapshots_with_jackknifed_mf(self) -> np.ndarray:
        return np.asarray(
            [i for i, mass_def in self.data.items() if mass_def.has_jackknifed_mf]
        )
    
    @property
    def snapshots_with_universal_mf(self) -> np.ndarray:
        return np.asarray(
            [i for i, mass_def in self.data.items() if mass_def.has_universal_mf]
        )
    
    @property
    def mass_bins(self) -> np.ndarray:
        return self.data[max(self.data)].mass_bins
    
    @property
    def mapping(self) -> DensityThresholdMap | None: 
        return self.data[max(self.data)].mapping
    
    @property
    def display_name(self) -> str:
        return self.mapping.mass.eqn if self.mapping is not None else ""
    
    def get_mass_function_evo(self, is_universal: bool = False) -> MassFunctionEvoData:
        mf_attr = "universal_mf" if is_universal else "jackknifed_mf"
        snap_attr = (
            "snapshots_with_universal_mf" 
            if is_universal else 
            "snapshots_with_jackknifed_mf"
        )

        return MassFunctionEvoData(
            moments=self.moments.get_subset_by_attribute( 
                key_attr="snapshot_id", attr_value=getattr(self, snap_attr)
            ),
            data=OrderedDict({
                snapshot_id: getattr(self.data[snapshot_id], mf_attr)
                for snapshot_id in getattr(self, snap_attr)
            })
        )    
    @property
    def mass_function_evo(self) -> MassFunctionEvoData:
        has_jackknifed = all(
            self.data[snap_idx].has_jackknifed_mf
            for snap_idx in self.snapshots_with_mf
        )
        return self.get_mass_function_evo(is_universal=not has_jackknifed)
    
    @property
    def jackknifed_mass_function_evo(self) -> MassFunctionEvoData:
        return self.get_mass_function_evo(is_universal=False)
    
    @property
    def universal_mass_function_evo(self) -> MassFunctionEvoData:
        return self.get_mass_function_evo(is_universal=True)


    @property
    def accumulation_histories(self) -> dict[int, AccumulationHistoryData]:
        return self.mass_function_evo.accumulation_histories
    

    def add_density_peaks(self, peaks_evo: DensityPeaksEvo) -> None:
        for snap_id, peaks in peaks_evo.data.items():
            self.data[snap_id].add_peak_heights(peaks)

    # def display_accumulation_histories(
    #         self, 
    #         mf_type: str = "differential",
    #         normalize: bool = True, 
    #         normalizing_scale_factor: float = 10.0,
    #         resid_min: float | None = None,
    #         resid_max: float | None = None,
    #         text_size: int = 12,
    #         show_log_y: bool = True,
    #         verbose: bool = False
    #     ) -> None:

    #     self.mass_function_evo.display_accumulation_histories(
    #         mass_def=self.display_name,
    #         mf_type=mf_type, 
    #         normalize=normalize,
    #         normalizing_scale_factor=normalizing_scale_factor,
    #         resid_min=resid_min,
    #         resid_max=resid_max,
    #         text_size=text_size,
    #         show_log_y=show_log_y,
    #         verbose=verbose
    #     )


    def display_mass_function_evo(
            self,
            target_scale_factors: Collection[float],
            mf_type: str = "differential", 
            show_raw: bool = False,
            text_size: int = 10,
            resid_min: float | None = None,
            resid_max: float | None = None,
            error_scale: float = 0.0,
            save_fig: bool = False,
        ) -> None: 

        self.mass_function_evo.display(
            target_scale_factors=target_scale_factors, 
            mass_def_key=self.display_name,
            mf_type=mf_type, 
            show_raw=show_raw,
            # text_size=text_size,
            resid_min=resid_min,
            resid_max=resid_max,
            # error_scale=error_scale,
            # save_fig=save_fig
        )


# Map from clean_fof_groups's cleaned-dict descriptor suffix to the
# MassDefinitions slot it populates. Mirrors the (solo_label, mass_def)
# pairs in MassDefinitions.add_solo_boundary_data; busha05 is intentionally
# absent because MassDefinitions has no slot for it. Module-level constant
# rather than a class attribute so attrs doesn't pick it up as a field.
_FULL_CATALOG_SOLO_SLOTS: tuple[tuple[str, str], ...] = (
    ("splashback",     "splashback"),
    ("bound",          "bound"),
    ("lambda",         "marginally_bound"),
    ("hydrostaticeq",  "hydrostatic"),
    ("turnaround",     "turnaround"),
    ("infall",         "infall"),
    ("fourscale",      "four_scale"),
    ("scale",          "scale"),
    ("vmax",           "max_circ"),
)


@define(slots=True)
class MassDefinitions(MassDefinitionDependentType):
    # Algorithmic 
    fof: MassDefinition | None = field(default=None)
    subfind: MassDefinition | None = field(default=None)

    # Spherical Overdensity
    crit200: MassDefinition | None = field(default=None)
    crit500: MassDefinition| None = field(default=None)
    mean200: MassDefinition | None = field(default=None)
    virial: MassDefinition | None = field(default=None)

    # Physical
    splashback: MassDefinition | None = field(default=None)
    scale: MassDefinition | None = field(default=None)
    four_scale: MassDefinition | None = field(default=None)
    max_circ: MassDefinition | None = field(default=None)
    bound: MassDefinition | None = field(default=None)

    # Kinematic outer-boundary milestones (mean_vr-derived). Populated
    # by FOFGroups.add_solo_boundaries from compute_outer_boundary_milestones
    # output. marginally_bound prefers SO-bisection lambda_so per halo,
    # falling back to profile lambda when lambda_so is absent / NaN.
    hydrostatic: MassDefinition | None = field(default=None)
    turnaround: MassDefinition | None = field(default=None)
    infall: MassDefinition | None = field(default=None)
    marginally_bound: MassDefinition | None = field(default=None)

    # This can be computed then stored in the object
    asymptotic: MassDefinition | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        for mdef in self.__slots__:
            if (getattr(self, mdef) is not None) and (getattr(self, mdef).mapping is None):
                getattr(self, mdef).mapping = getattr(global_mass_def_mapping, mdef)

    def __repr__(self) -> str:
        return super().__repr__()  

    @property
    def mass_bins(self) -> np.ndarray:
        return np.unique(np.concatenate([
            getattr(self, mass_def).mass_bins
            for mass_def in self.contained_definitions
        ]))
    
    @property
    def mass_bins_by_mass_def(self) -> dict[str, np.ndarray]:
        return {
            self.contained_keys[i] : getattr(self, mass_def).mass_bins
            for i, mass_def in enumerate(self.contained_definitions)
        }
    
    @property
    def num_halos_in_mass_bin_by_mass_def(self) -> dict[str, Counter]:  
        return {
            self.contained_keys[i] : getattr(self, mass_def).num_halos_in_each_mass_bins
            for i, mass_def in enumerate(self.contained_definitions)
        }


    @property
    def threshold_density_values(self) -> dict[str, float]:
        return {
            self.contained_keys[i] : getattr(self, mass_def).density
            for i, mass_def in enumerate(self.contained_definitions)
            if getattr(self, mass_def).density is not None
        }
    

    @property
    def crossing_times(self) -> dict[str, float | np.ndarray]:
        return {
            self.contained_keys[i] : getattr(self, mass_def).crossing_time
            for i, mass_def in enumerate(self.contained_definitions)
            if getattr(self, mass_def).crossing_time is not None
        }


    def get_mass_defs_for_object(self, index: int) -> MassDefinitions:
        return MassDefinitions(
            **{
                mass_def : getattr(self, mass_def)[index]
                for mass_def in self.contained_definitions
            }
        )
    
    def get_subset(self, idxs: Collection[int]) -> MassDefinitions:
        if not isinstance(idxs, np.ndarray):
            idxs = np.array(idxs)
        return MassDefinitions(
            **{
                mass_def: getattr(self, mass_def).get_subset(idxs)
                for mass_def in self.contained_definitions
            }
        )

    def get_attribute_ratios(
            self, 
            attr_name: str, 
            mass_def_key_numerator: str, 
            mass_def_key_denominator: str,
            return_full: bool = False
        ) -> np.ndarray:

        try: 
            mdef_numerator = self[mass_def_key_numerator]
        except ValueError as error:
            raise ValueError(
                f"Mass definition {mass_def_key_numerator} not been defined:"
                f" {error}"
            ) from error

        try:
            mdef_denominator = self[mass_def_key_denominator]
        except ValueError as error:
            raise ValueError(
                f"Mass definition {mass_def_key_denominator} not been defined:"
                f" {error}"
            ) from error

        numerator = getattr(mdef_numerator, attr_name)
        denominator = getattr(mdef_denominator, attr_name)

        if attr_name == "mass": 
            numerator, denominator = 10.0**numerator, 10.0**denominator

        ratios = numerator / denominator

        return np.vstack([ratios, numerator, denominator]).T if return_full else ratios
    
    def estimate_mass_defs(self, mass: float) -> MassDefinitions:
        return MassDefinitions(
            fof=(
                self.fof.estimate_mass_def(mass) 
                if (self.has_fof and self.fof.has_density)
                else None
            ),
            subfind=(
                self.subfind.estimate_mass_def(mass) 
                if (self.has_subfind and self.subfind.has_density) 
                else None
            ),
            crit200=(
                self.crit200.estimate_mass_def(mass) 
                if (self.has_200c and self.crit200.has_density)
                else None
            ),
            crit500=(
                self.crit500.estimate_mass_def(mass) 
                if (self.has_500c and self.crit500.has_density) 
                else None
            ),
            mean200=(
                self.mean200.estimate_mass_def(mass) 
                if (self.has_200m and self.mean200.has_density) 
                else None
            ),
            virial=(
                self.virial.estimate_mass_def(mass) 
                if (self.has_virial and self.virial.has_density) 
                else None
            ),
            splashback=(
                self.splashback.estimate_mass_def(mass) 
                if (self.has_splashback and self.splashback.has_density) 
                else None
            ),
            scale=(
                self.scale.estimate_mass_def(mass) 
                if (self.has_scale and self.scale.has_density) 
                else None
            ),
            four_scale=(
                self.four_scale.estimate_mass_def(mass)
                if (self.has_four_scale and self.four_scale.has_density)
                else None
            ),
            max_circ=(
                self.max_circ.estimate_mass_def(mass)
                if (self.has_max_circ and self.max_circ.has_density)
                else None
            ),
            bound=(
                self.bound.estimate_mass_def(mass)
                if (self.has_bound and self.bound.has_density)
                else None  
            ),
            asymptotic=(
                self.asymptotic.estimate_mass_def(mass) 
                if (self.has_asymptotic and self.asymptotic.has_density) 
                else None
            )
        )
    
    def add_physical_catalog_data(
            self,
            physical_catalog: dict[str, np.ndarray],
            fof_group_ids: np.ndarray,
        ) -> None:

        hubble = getattr(self.fof, "hubble", None)
        hubble = getattr(self.crit200, "hubble", hubble)
        hubble = getattr(self.mean200, "hubble", hubble)

        mask = np.isin(fof_group_ids, physical_catalog["group_ids"])
        masses = physical_catalog["masses"]
        radii = physical_catalog["radii"]

        for i, mass_def in enumerate(
                ("bound", "splashback", "scale", "four_scale", "max_circ")
            ):
            setattr(
                self,
                mass_def,
                MassDefinition.for_template(
                    mapping=getattr(global_mass_def_mapping, mass_def),
                    num_halos=fof_group_ids.size,
                    hubble=hubble,
                )
            )
            getattr(self, mass_def).mass[mask] = masses[:, i]
            getattr(self, mass_def).radius[mask] = radii[:, i]


    def add_solo_boundary_data(
            self,
            solo_data: dict[str, np.ndarray],
            fof_group_ids: np.ndarray,
        ) -> None:
        """Inject mass-def boundaries from a save_solo_profiles_phusis output.

        Mirrors :meth:`add_physical_catalog_data` for solo's milestone
        layout: ``splashback``, ``scale``, ``four_scale``, ``vmax`` (mapped
        onto ``max_circ``), and ``bound`` (sourced from solo's ``target``
        κ-minimum milestone, repackaged as ``bound_M`` / ``bound_R`` by
        the caller in :meth:`FOFGroups.add_solo_boundaries`). Solo writes
        per-halo linear masses, so they are log10'd here to match the
        in-memory convention used elsewhere by ``MassDefinition.mass``.

        ``solo_data`` is the flat per-(snap, mdef) layout produced by
        callers from :func:`phusis.io.solo_profiles.load_solo_boundaries`:
        keys ``group_ids`` plus ``{label}_M`` and ``{label}_R`` for each
        solo label.

        Each label entry in ``solo_data`` is optional — when missing (older
        solo runs, or callers that pass only a subset), the corresponding
        mass-def attribute is left untouched.
        """
        hubble = getattr(self.fof, "hubble", None)
        hubble = getattr(self.crit200, "hubble", hubble)
        hubble = getattr(self.mean200, "hubble", hubble)

        mask = np.isin(fof_group_ids, solo_data["group_ids"])

        for solo_label, mass_def in (
                ("splashback",       "splashback"),
                ("scale",            "scale"),
                ("four_scale",       "four_scale"),
                ("vmax",             "max_circ"),
                ("bound",            "bound"),
                ("hydrostatic",      "hydrostatic"),
                ("turnaround",       "turnaround"),
                ("infall",           "infall"),
                ("marginally_bound", "marginally_bound"),
            ):
            m_key = f"{solo_label}_M"
            r_key = f"{solo_label}_R"
            if m_key not in solo_data or r_key not in solo_data:
                # Caller didn't supply this milestone — leave the slot
                # at its current value (None for un-populated defs).
                continue
            setattr(
                self,
                mass_def,
                MassDefinition.for_template(
                    mapping=getattr(global_mass_def_mapping, mass_def),
                    num_halos=fof_group_ids.size,
                    hubble=hubble,
                )
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                getattr(self, mass_def).mass[mask] = np.log10(
                    solo_data[m_key]
                )
            getattr(self, mass_def).radius[mask] = solo_data[r_key]


    def add_full_catalog_solo_data(
            self,
            cleaned_fof_data: dict[str, np.ndarray],
            fof_group_ids: np.ndarray,
        ) -> None:
        """Inject the solo-derived mass-def slots from a full_catalog load.

        Equivalent in spirit to :meth:`add_solo_boundary_data` but reads
        from the flat-key layout produced by
        :func:`get_data.clean_fof_groups` when called with
        ``use_full_catalog=True`` — keys like ``"masses_splashback"``,
        ``"radii_bound"``, etc. Those masses are already ``log10(M/Msun_h)``
        (``convert_mass`` was applied upstream), so we copy them as-is
        rather than re-log10-ing like the solo_boundaries path does.

        Because the cleaned arrays are already shaped to the post-mask
        halo set, length-matched to ``fof_group_ids``, the per-halo
        ``np.isin`` filter ``add_solo_boundary_data`` performs is
        unnecessary here — assignment is direct.

        Quietly skips any (descriptor, slot) pair whose cleaned keys are
        missing, leaving the slot untouched. ``busha05`` is intentionally
        not mapped — :class:`MassDefinitions` has no slot for it.

        Note on ``marginally_bound``: ``Group_M_Lambda`` /
        ``Group_R_Lambda`` in full_catalog now already encode the
        per-halo SO-bisection-preferred synthesis
        (``lambda_so_<q>`` when finite, ``lambda_<q>`` fallback). The
        compose-side synthesis in
        :func:`postprocessing.compose_full_catalog._load_solo_milestones`
        mirrors :meth:`FOFGroups.add_solo_boundaries`, so the
        ``marg_bound`` slot loaded from full_catalog matches the one
        built directly from solo_boundaries.
        """
        hubble = getattr(self.fof, "hubble", None)
        hubble = getattr(self.crit200, "hubble", hubble)
        hubble = getattr(self.mean200, "hubble", hubble)

        n_halos = int(fof_group_ids.size)

        for cleaned_descr, slot in _FULL_CATALOG_SOLO_SLOTS:
            m_key = f"masses_{cleaned_descr}"
            r_key = f"radii_{cleaned_descr}"
            if m_key not in cleaned_fof_data or r_key not in cleaned_fof_data:
                continue
            setattr(
                self,
                slot,
                MassDefinition.for_template(
                    mapping=getattr(global_mass_def_mapping, slot),
                    num_halos=n_halos,
                    hubble=hubble,
                ),
            )
            getattr(self, slot).mass[:] = np.asarray(cleaned_fof_data[m_key])
            getattr(self, slot).radius[:] = np.asarray(cleaned_fof_data[r_key])

    def set_physical_mass_defs(
            self,
            profile_array: np.ndarray,
            min_radius: float = 0.0,
            max_radius: float = np.inf
        ) -> None:

        milestones = get_filtered_mass_distribution_milestones(
            filtered_profiles=profile_array,
            min_radius=min_radius,
            max_radius=max_radius
        )

        self.splashback = MassDefinition(
            mapping=self.fof.mapping,
            mass=np.log10(milestones["M_splashback"]),
            radius=milestones["r_splashback"],
            hubble=self.fof.hubble,
        )
        self.splashback.density = milestones["rho_enc_splashback"]

        self.scale = MassDefinition(
            mapping=self.fof.mapping,
            mass=np.log10(milestones["M_scale"]),
            radius=milestones["r_scale"],
            hubble=self.fof.hubble,
        )
        self.scale.density = milestones["rho_enc_scale"]

        self.four_scale = MassDefinition(
            mapping=self.fof.mapping,
            mass=np.log10(milestones["M4rs"]),
            radius=milestones["r_4s"],
            hubble=self.fof.hubble,
        )
        self.four_scale.density = milestones["rho_enc_4s"]


        self.max_circ = MassDefinition(
            mapping=self.fof.mapping,
            mass=np.log10(milestones["M_v_max"]),
            radius=milestones["r_v_max"],
            hubble=self.fof.hubble,
        )
        self.max_circ.density = milestones["rho_enc_v_max"]

        self.bound = MassDefinition(
            mapping=self.fof.mapping,
            mass=np.log10(milestones["M_bound"]),
            radius=milestones["r_bound"],
            hubble=self.fof.hubble,
        )
        self.bound.density = milestones["rho_enc_bound"]

    
    def get_concentrations(self, mass_def_key: str = "vir") -> np.ndarray:
        if (mass_def := getattr(self, mass_def_key)) is None:
            raise ValueError(f"Mass definition {mass_def_key} not defined")
        if (max_circ := getattr(self, "max_circ")) is None:
            raise ValueError("Max circular mass definition not defined")
        
        return (mass_def.velocity / max_circ.velocity)**2
    
    @classmethod
    def from_fof_data(cls, fof_data: dict, hubble: float) -> MassDefinitions:

        return cls(
            fof=MassDefinition(
                mapping=global_mass_def_mapping.fof,
                mass=fof_data[global_mass_def_mapping.fof.mass.name],
                hubble=hubble
            ),
            crit200=MassDefinition(
                mapping=global_mass_def_mapping.crit200,
                mass=fof_data[global_mass_def_mapping.crit200.mass.name],
                radius=fof_data[global_mass_def_mapping.crit200.radius.name],
                hubble=hubble
            ),
            crit500=MassDefinition(
                mapping=global_mass_def_mapping.crit500,
                mass=fof_data[global_mass_def_mapping.crit500.mass.name],
                radius=fof_data[global_mass_def_mapping.crit500.radius.name],
                hubble=hubble
            ),
            mean200=MassDefinition(
                mapping=global_mass_def_mapping.mean200,
                mass=fof_data[global_mass_def_mapping.mean200.mass.name],
                radius=fof_data[global_mass_def_mapping.mean200.radius.name],
                hubble=hubble
            ),
            virial=MassDefinition(
                mapping=global_mass_def_mapping.virial,
                mass=fof_data[global_mass_def_mapping.virial.mass.name],
                radius=fof_data[global_mass_def_mapping.virial.radius.name],
                hubble=hubble
            ),
        )
    

    def add_peak_heights(self, peaks: DensityPeaks) -> None:
        for mass_def in self.contained_definitions:
            getattr(self, mass_def).add_peak_heights(peaks)

    
    @classmethod
    def from_subfind_data(cls, subfind_data: dict, hubble: float) -> MassDefinitions:
        return cls(
            subfind=MassDefinition(
                mapping=global_mass_def_mapping.subfind,
                mass=subfind_data[global_mass_def_mapping.subfind.mass.name],
                # radius=subfind_data[global_mass_def_mapping.subfind.radius.name],
                hubble=hubble
            )
        )
        

    def normalized_definition(
            self, target_def: str, normalizing_def: str, overdenisty: float = 1
        ) -> MassDefinition:
        target = getattr(self, target_def)
        normalizing = getattr(self, normalizing_def)
        new_def = MassDefinition(
            mapping=DensityThresholdMap(
                key=f"{target.mapping.key}_{normalizing.mapping.key}",
                title=f"{target.mapping.title} Scaled by {normalizing.mapping.title}",
                display=f"{target.mapping.display} Scaled by {normalizing.mapping.display}",
                mass=DensityThresholdProperty(
                    f"{target.mapping.mass.name}_{normalizing.mapping.mass.name}",
                    f"{target.mapping.mass.display} Scaled by {normalizing.mapping.mass.display}"
                ),
                radius=DensityThresholdProperty(
                    f"{target.mapping.radius.name}_{normalizing.mapping.radius.name}",
                    f"{target.mapping.radius.display} Scaled by {normalizing.mapping.radius.display}"
                )
            ),
            mass=10.0**(target.mass - normalizing.mass),
            radius=target.radius / normalizing.radius,
            hubble=target.hubble,
        )
        new_def.density = (target.density / normalizing.density) / overdenisty
        return new_def
    
    def compute_mass_functions(
            self,
            volume: float,
            present_day_density: float,
            redshift: float,
            bin_count: int = 50,
            transfer: str = "eisenstein98",
            corrections: bool = True,
            is_comoving_volume: bool = True,
            run_jackknife: bool = False,
            run_both_jk: bool = False,
            sub_box_group_idxs: Collection[int] | list[np.ndarray] | None = None,
            sub_box_subhalo_idxs: Collection[int] | list[np.ndarray] | None = None,
            run_fits: bool = True, 
            run_cleaning: bool = True,
            min_resolved_mass: float = 0.0, 
            min_count_above_max_bin_hist: int = 0,
            use_ps_file: bool = False,
            ps_file_path: Path | None = None,
            verbose: bool = False,
            cosmo: Cosmology | None = None,
            mass_bins: dict[str, dict[str, np.ndarray]] | None = None,
            return_fold_samples: bool = False
        ) -> None:

        comoving_volume = (
            volume if is_comoving_volume else volume * (1.0 + redshift)**3
        )

        if (
            (sub_box_group_idxs is None) 
            and (sub_box_subhalo_idxs is None) 
            and run_jackknife
        ):
            raise ValueError("Sub-box indices must be provided for jackknife")


        for i, mass_def in enumerate(self.contained_definitions):

            if (mass_def_data := getattr(self, mass_def)) is not None:

                run_subhalo_jackknife = run_jackknife and (mass_def == "subfind")
                run_group_jackknife = run_jackknife and not run_subhalo_jackknife

                if run_subhalo_jackknife:
                    mass_samples = mass_def_data.get_mass_sub_box_splits(sub_box_subhalo_idxs)
                elif run_group_jackknife:
                    mass_samples = mass_def_data.get_mass_sub_box_splits(sub_box_group_idxs)
                else:
                    mass_samples = mass_def_data.mass

                mf_config = {
                    "box_size": np.cbrt(comoving_volume),
                    "present_day_density": present_day_density,
                    "redshift": redshift,
                    "bin_count": bin_count,
                    "transfer": transfer,
                    "corrections": corrections,
                    "run_fit": run_fits,
                    "use_ps_file": use_ps_file,
                    "ps_file_path": ps_file_path,
                    "cosmo": cosmo,
                    "return_fold_samples": return_fold_samples
                }

                if run_jackknife:

                    if run_both_jk:

                        try: 
                            mass_def_data.universal_mf = MassFunctionData.from_data(
                                samples=mass_def_data.mass,
                                relation_type="mass",
                                run_jackknife=False,
                                sample_bins=(
                                    mass_bins[self.contained_keys[i]] 
                                    if mass_bins is not None else 
                                    None
                                ),
                                **mf_config
                            )
                        except Exception as error:
                            print(
                                (
                                    f"Could not compute universal mass function "
                                    f"for {mass_def} at z={redshift}: {error}"
                                )
                            )

                        if run_cleaning and mass_def_data.has_universal_mf:
                            try: 
                                mass_def_data.universal_mf.set_clean_abundance_relationship(
                                    min_sample_value=min_resolved_mass,
                                    min_above_max_mass_bin_count=min_count_above_max_bin_hist,
                                    verbose=verbose
                                )
                            except Exception as error:
                                print(
                                    (
                                        f"Could not clean mass function "
                                        f"for {mass_def} at z={redshift}: {error}"
                                    )
                                )

                    try: 
                        mass_def_data.jackknifed_mf = MassFunctionData.from_data(
                            samples=mass_samples,
                            relation_type="mass",
                            run_jackknife=True,
                            sample_bins=(
                                mass_def_data.universal_mf.mass_bins 
                                if (run_both_jk and mass_def_data.has_universal_mf) else 
                                None
                            ),
                            **mf_config
                        )
                    except Exception as error:
                        print(
                            (
                                f"Could not compute jackknifed mass function "
                                f"for {mass_def} at z={redshift}: {error}"
                            )
                        )

                
                else:

                    try:
                        mass_def_data.universal_mf = MassFunctionData.from_data(
                            samples=mass_samples,
                            relation_type="mass",
                            run_jackknife=False,
                            **mf_config
                        )
                    except Exception as error:
                        print(
                            (
                                f"Could not compute universal mass function "
                                f"for {mass_def} at z={redshift}: {error}"
                            )
                        )

                if run_cleaning and mass_def_data.has_jackknifed_mf:

                    try:
                        mass_def_data.jackknifed_mf.set_clean_abundance_relationship(
                            min_sample_value=min_resolved_mass,
                            min_above_max_mass_bin_count=min_count_above_max_bin_hist,
                            verbose=verbose
                        )
                    except Exception as error:
                        print(
                            (
                                f"Could not clean mass function "
                                f"for {mass_def} at z={redshift}: {error}"
                            )
                        )

                


    @property
    def halo_mass_function(self) -> HaloMassFunction:
        return HaloMassFunction(**self.get_mass_def_attrs("mass_function"))

    @property
    def abundance_model(self) -> AsymptoticAbundanceModel: 
        return self.halo_mass_function.model
    

    def display_model_params(self, ref_density: float, text_size: int = 12) -> None:

        _, ax = plt.subplots(
            4, 1, figsize=(4, 6), sharex=True, gridspec_kw={"hspace": 0.0}
        )

        model =self.abundance_model

        
        params_A = model.parameters_A
        params_a = model.parameters_a
        params_b = model.parameters_b
        params_c = model.parameters_c


        for mdef, rho in self.threshold_density_values.items():

            overdensity = rho / ref_density

            ax[0].plot(overdensity, params_A[mdef], 'o')
            ax[1].plot(overdensity, params_a[mdef], 'o')
            ax[2].plot(overdensity, params_b[mdef], 'o')
            ax[3].plot(overdensity, params_c[mdef], 'o')

        ax[0].set_ylabel(r"$A$")
        ax[1].set_ylabel(r"$\alpha$")
        ax[2].set_ylabel(r"$\beta$")
        ax[3].set_ylabel(r"$\gamma$")

        ax[3].set_xlabel(r"$\Delta$")
    
        ax[3].legend(
            ax[0].lines, 
            labels=[
                rf"$\Delta_{{\rm {mdef}}}$" 
                for mdef in self.threshold_density_values.keys()
            ],
            loc="center", ncol=len(self.threshold_density_values),
            bbox_to_anchor=(0.5, -0.5),
            fontsize=text_size
        )
    

    def display_mass_function(
            self, 
            mass_def: str, 
            mf_type: str, 
            show_raw: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None]:

        return self[mass_def].display_mass_function(
            mf_type=mf_type, 
            show_raw=show_raw,
            ax_main=ax_main,
            ax_resid=ax_resid,
        )
    
@define(slots=True)
class MassDefinitionsEvo(MassDefinitionDependentEvoType):
    fof: MassDefinitionEvo | None = field(default=None)
    subfind: MassDefinitionEvo | None = field(default=None)
    crit200: MassDefinitionEvo | None = field(default=None)
    crit500: MassDefinitionEvo | None = field(default=None)
    mean200: MassDefinitionEvo | None = field(default=None)
    virial: MassDefinitionEvo | None = field(default=None)
    splashback: MassDefinitionEvo | None = field(default=None)
    hydrostatic: MassDefinitionEvo | None = field(default=None)
    turnaround: MassDefinitionEvo | None = field(default=None)
    infall: MassDefinitionEvo | None = field(default=None)
    marginally_bound: MassDefinitionEvo | None = field(default=None)
    asymptotic: MassDefinitionEvo | None = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()

    def get_snapshot_instance(self, snapshot_id: int) -> MassDefinitions:
        mass_defs_data =  MassDefinitions(
            **self.get_mass_def_attrs_instance("data", snapshot_id)
        )
        
        if (
            set(mass_defs_data.contained_definitions).issubset(
                set(self.contained_definitions)
            )
        ):
            return mass_defs_data
        
        raise ValueError(f"Mass Definitions for {snapshot_id} is not valid")
    
    @property
    def present_day_density(self) -> float:
        values = set()
        final_mass_defs = self.get_snapshot_instance(max(self.snapshot_ids))
        for mass_def_name in self.contained_definitions:
            fit = getattr(final_mass_defs, mass_def_name).mass_function.fit
            values.add(fit.present_day_mean_density)

        if len(values) != 1: 
            raise ValueError("Present Day Densities do not match")
        
        return values.pop()

    @property
    def threshold_density_values(self) -> dict[str, np.ndarray]:
        density_evo_values = {}

        for i, mass_def_name in enumerate(self.contained_definitions):
            key = self.contained_keys[i]
            mass_def_evo = getattr(self, mass_def_name)
            values = []
            for snapshot_id in mass_def_evo.snapshots_with_mf:
                scale_factor = mass_def_evo.moments.map_by_attribute(
                    key_attr="snapshot_id", 
                    attr_value=snapshot_id,
                    return_attr="scale_factor"
                )
                mass_def = mass_def_evo.data[snapshot_id]
                if (density := mass_def.density) is not None:
                    values.append((scale_factor, density))

            if values:
                density_evo_values[key] = np.asarray(values)
        
        return density_evo_values


    @property
    def mass_bins(self) -> np.ndarray:
        return np.unique(np.concatenate([
            getattr(self, mass_def).mass_bins
            for mass_def in self.contained_definitions
        ]))
    
    @property
    def mass_bins_by_mass_def(self) -> dict[str, np.ndarray]:
        return {
            mass_def: getattr(self, mass_def).mass_bins
            for mass_def in self 
            if getattr(self, mass_def) is not None
        }
    
    @property
    def num_halos_in_mass_bins_by_mass_def_evo(self) -> dict[str, dict[int, Counter]]:
        return {
            self.contained_keys[i] : getattr(self, mass_def).num_halos_in_each_mass_bins
            for i, mass_def in enumerate(self.contained_definitions)
            if getattr(self, mass_def) is not None
        }

    def snapshots_with_mass_def_data(self, mass_def: str) -> np.ndarray:
        return self[mass_def].snapshots_with_mf

    @property
    def halo_mass_function_evo(self) -> HaloMassFunctionEvoData:
        return HaloMassFunctionEvoData(
            **self.get_mass_def_attrs("mass_function_evo")
        )
    
    def get_halo_mass_function_evo(self, mass_defs: str | Collection[str]) -> HaloMassFunctionEvoData:
        if isinstance(mass_defs, str):
            mass_defs = [mass_defs]

        return HaloMassFunctionEvoData(
            **{
                mass_def: self.get_mass_def_attrs(mass_def)["mass_function_evo"]
                for mass_def in mass_defs
            }
        )

    @property
    def jackknife_mass_function_evo(self) -> HaloMassFunctionEvoData:
        return HaloMassFunctionEvoData(
            **self.get_mass_def_attrs("jackknifed_mass_function_evo")
        )
    
    @property
    def universal_mass_function_evo(self) -> HaloMassFunctionEvoData:
        return HaloMassFunctionEvoData(
            **self.get_mass_def_attrs("universal_mass_function_evo")
        )

    @property
    def abundance_accumuluation(self) -> PopulationAbundanceAccumulationHistories:
        accumulations = OrderedDict()
        for mass_bin in self.mass_bins:
            try:
                accumulations[mass_bin] = self.get_abundance_accumulation(mass_bin)
            except ValueError:
                continue
        return PopulationAbundanceAccumulationHistories(data=accumulations)
    
    @property
    def abundance_evo_model(self) -> AsymptoticAbundanceEvoModel:
        return self.halo_mass_function_evo.model

    def get_mass_function_evo(self, mass_def: str) -> MassFunctionEvoData:
        return self[mass_def].mass_function_evo

    
    def get_abundance_accumulation(self, mass_bin: int) -> AbundanceAccumulationHistories:
        histories = AbundanceAccumulationHistories(
            **self.get_mass_def_attrs_instance(
                attr_name="accumulation_histories", 
                instance_id=mass_bin
            )
        )

        if histories.contained_definitions == self.contained_definitions:
            return histories

        raise ValueError(f"Mass Definitions for M={mass_bin} accumulations do not match")
    
    def add_peak_heights(self, peaks_evo: DensityPeaksEvo) -> None:
       for mass_def in self.contained_definitions:
           getattr(self, mass_def).add_peak_heights(peaks_evo)

    def display_model_param_evos(
            self, 
            text_size: int = 12,
            plot_loglog: bool = False, 
            cutoff_scale_factor: float = 0.4,
        ) -> None:

        _, ax = plt.subplots(
            4, 1, figsize=(4, 6), sharex=True, gridspec_kw={"hspace": 0.0}
        )

        masked = lambda params: params[params[:, 0] > cutoff_scale_factor]
        
        evo_model = self.abundance_evo_model

        evo_params_A = evo_model.parameter_evos_A
        evo_params_a = evo_model.parameter_evos_a
        evo_params_b = evo_model.parameter_evos_b
        evo_params_c = evo_model.parameter_evos_c

        for mdef in self.contained_keys:

            evoA = masked(evo_params_A[mdef])
            evoa = masked(evo_params_a[mdef])
            evoB = masked(evo_params_b[mdef])
            evoC = masked(evo_params_c[mdef])

            ax[0].plot(evoA[:, 0], evoA[:, 1], 'o')
            ax[1].plot(evoa[:, 0], evoa[:, 1], 'o')
            ax[2].plot(evoB[:, 0], evoB[:, 1], 'o')
            ax[3].plot(evoC[:, 0], evoC[:, 1], 'o')


        for i in range(4):
            ax[i].yaxis.set_tick_params(labelsize=text_size)
            ax[i].set_xscale("log")

            if plot_loglog:
                ax[i].set_yscale("log")


        ax[0].set_ylabel(r"$A$")
        ax[1].set_ylabel(r"$\alpha$")
        ax[2].set_ylabel(r"$\beta$")
        ax[3].set_ylabel(r"$\gamma$")

        ax[3].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$")

        ax[3].legend(
            ax[0].lines, 
            labels=self.contained_display_names,
            loc="center", ncol=len(self.threshold_density_values),
            bbox_to_anchor=(0.5, -0.65),
            fontsize=text_size
        )

        ax[3].set_xlim(left=cutoff_scale_factor)