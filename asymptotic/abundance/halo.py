from __future__ import annotations

import contextlib
import numpy as np, pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py
import pickle

from tqdm.auto import tqdm
from pathlib import Path
from scipy import integrate
from collections import defaultdict
from functools import cached_property

from matplotlib.lines import Line2D
from collections.abc import Collection
from scipy.interpolate import interp1d
from attrs import define, field, fields
from collections import defaultdict, OrderedDict


from ..fields.peaks import DensityPeaks, DensityPeaksEvo
from .triangle import CoverageTriangle
from ..simulation.evo import EvolutionData
from .type_defs import AbundanceType, RelationType
from ..simulation.moments import Moment, MomentsInTime
from .accumulation import (
    AbundanceAccumulationHistories,
    PopulationAbundanceAccumulationHistories,
    MassBinKey,
)
from ..mass_def.base import (
    MassDefinitionDependentType, 
    MassDefinitionDependentEvoType,
    MASS_DEF_PLOT_LINES,
    MASS_DEF_MARKER_STYLES
)
from .mass_function import (
    MassFunction, 
    MassFunctionEvo,
    MassFunctionData, 
    MassFunctionEvoData,
    MassFunctionValue,
    is_int_value
)
from .data import AbundanceRelation, Abundances
from ..model.mass_function import (
    AsymptoticAbundanceFit, 
    AsymptoticAbundanceFitEvo,
    AsymptoticAbundanceModel,
    AsymptoticAbundanceEvoModel
)
from ..utils.freeze_out import compute_peak_and_freeze_times

@define(slots=True)
class HaloMassFunction(MassDefinitionDependentType):
    # Order matches global_mass_def_mapping.fields so that **kwargs
    # construction from get_mass_def_attrs lines up cleanly across all
    # mass defs the full_catalog loader populates.
    fof: MassFunctionData | None = field(default=None, repr=False)
    subfind: MassFunctionData | None = field(default=None, repr=False)
    crit200: MassFunctionData | None = field(default=None, repr=False)
    crit500: MassFunctionData | None = field(default=None, repr=False)
    mean200: MassFunctionData | None = field(default=None, repr=False)
    virial: MassFunctionData | None = field(default=None, repr=False)
    scale: MassFunctionData | None = field(default=None, repr=False)
    four_scale: MassFunctionData | None = field(default=None, repr=False)
    max_circ: MassFunctionData | None = field(default=None, repr=False)
    bound: MassFunctionData | None = field(default=None, repr=False)
    splashback: MassFunctionData | None = field(default=None, repr=False)
    hydrostatic: MassFunctionData | None = field(default=None, repr=False)
    turnaround: MassFunctionData | None = field(default=None, repr=False)
    infall: MassFunctionData | None = field(default=None, repr=False)
    marginally_bound: MassFunctionData | None = field(default=None, repr=False)
    asymptotic: MassFunctionData | None = field(default=None, repr=False)

    peaks: DensityPeaks | None = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        # Randomly choose a DensityPeaks object from one of the MassFunctionData 
        # objects that are not None and set the peaks attribute
        if self.peaks is None:
            with contextlib.suppress(AttributeError):
                for mass_def in self.contained_definitions:
                    if getattr(self, mass_def) is not None:
                        self.peaks = getattr(self, mass_def).peaks
                        break

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def model(self) -> AsymptoticAbundanceModel:
        # Add post-init functionality to fit overdensity parameter evolution
        return AsymptoticAbundanceModel(**self.get_mass_def_attrs("fit"))
    
    @property
    def raw(self) -> HaloMassFunction:
        return HaloMassFunction(**self.get_mass_def_attrs("raw"))

    @property
    def population_masses(self) -> np.ndarray:
        masses = set()
        for mass_def in self.contained_definitions:
            masses.update(getattr(self, mass_def).population_masses)
        return np.asarray(sorted(list(masses)))


    @property
    def round_one_population_masses(self) -> np.ndarray:
        masses = set()
        for mass_def in self.contained_definitions:
            masses.update(getattr(self, mass_def).round_one_population_masses)
        return np.asarray(sorted(list(masses)))
    

    def get_masked_version(self, mass_def_masks: dict[str, np.ndarray]) -> HaloMassFunction:

        masked_versions = {}
        for mass_def, mask in mass_def_masks.items():
            try:
                masked_versions[mass_def] = getattr(self, mass_def).get_masked_version(mask)
            except IndexError as error:
                print(f"Skipping {mass_def} due to {error}")
                continue

        full_version = masked_versions | {
            mass_def: getattr(self, mass_def)
            for mass_def in self.contained_definitions
            if ((mass_def not in mass_def_masks) or (mass_def_masks[mass_def] is None))
        }

        return HaloMassFunction(**full_version)


    @property
    def fold_instances(self) -> dict[int, HaloMassFunction]:
        ''' 
        Return a dictionary of HaloMassFunction instances, one for each fold.
        '''
        folds = {}
        for mass_def in self.contained_definitions:
            if getattr(self, mass_def) is not None:
                folds[mass_def] = getattr(self, mass_def).fold_instances

        fold_ids = set().union(*(fold_data.keys() for fold_data in folds.values()))

        return {
            fold_id: HaloMassFunction(**{
                mass_def: fold_data
                for mass_def, fold_data in folds.items()
                if (fold_id in fold_data)
            })
            for fold_id in fold_ids
        }
    

    @classmethod
    def join(
            cls, mass_functions: dict[int, HaloMassFunction] | Collection[HaloMassFunction]
        ) -> HaloMassFunction:

        if isinstance(mass_functions, dict):
            return join_hmf_by_dict(mass_functions)
        else:
            return join_hmf_by_collection(mass_functions)

    @classmethod
    def join_with_provenance(
            cls, mass_functions: dict[int, HaloMassFunction]
        ) -> tuple[HaloMassFunction, dict[str, np.ndarray]]:
        """Join per-box HMFs like :meth:`join`, and additionally return, per mass
        definition, the source-box label of every stitched point.

        ``join`` concatenates the per-box relations in dict-iteration order but
        discards which box each retained mass bin came from. The standalone
        splashback-HMF diagnostic keeps that per-point provenance (``box_mpc_h``);
        the study stitching path (``convergence.get_box_study_joint_hmf``) loses it.
        This method recovers it without changing ``join``'s behaviour.

        Only a ``{box_label: HaloMassFunction}`` dict carries box labels, so a dict
        is required. ``provenance[mass_def]`` is aligned 1:1 with
        ``getattr(joint, mass_def).masses`` -- same concatenation order as
        :func:`join_hmf_by_dict` -- and each entry is the box key that point
        entered from. No mass is double counted when the caller applies the
        half-open own-floor mask (adjacent boxes cover disjoint mass ranges), so a
        given stitched mass carries exactly one box label. A size-mismatch between
        the labels and the joined masses raises rather than silently mislabelling.
        """
        if not isinstance(mass_functions, dict):
            raise TypeError(
                "join_with_provenance requires a {box: HaloMassFunction} dict to "
                "label each stitched point with its source box; got "
                f"{type(mass_functions).__name__}."
            )

        joint = cls.join(mass_functions)

        # Rebuild the per-mass_def source-box labels in the SAME order
        # join_hmf_by_dict concatenates them: iterate the dict's values and, for
        # each box, append one label per retained mass of every mass_def it holds.
        labels: dict[str, list[np.ndarray]] = defaultdict(list)
        for box_label, hmf in mass_functions.items():
            for mass_def in hmf.contained_definitions:
                mass_func = getattr(hmf, mass_def, None)
                if mass_func is None:
                    continue
                labels[mass_def].append(
                    np.full(np.asarray(mass_func.masses).size, box_label)
                )

        provenance: dict[str, np.ndarray] = {}
        for mass_def, chunks in labels.items():
            merged = np.concatenate(chunks) if chunks else np.asarray([])
            joined_mf = getattr(joint, mass_def, None)
            n_joined = 0 if joined_mf is None else np.asarray(joined_mf.masses).size
            if merged.size != n_joined:
                raise ValueError(
                    f"Provenance/join misalignment for mass_def '{mass_def}': "
                    f"{merged.size} source-box labels vs {n_joined} joined points. "
                    "The joined masses must be the ordered concatenation of the "
                    "per-box masses for the labels to align."
                )
            provenance[mass_def] = merged

        return joint, provenance

    @property
    def data_total_trapped_masses(self) -> dict[str | int, dict[str, float]]:
        return self.get_mass_def_attrs("data_total_trapped_mass")
    @property
    def data_trapped_masses_by_bin(self) -> dict[str, dict[str | int, dict[str, float]]]:
        return self.get_mass_def_attrs("data_trapped_mass_by_bin")
    
    @property
    def fitted_total_trapped_masses(self) -> dict[str | int, dict[str, float]]:
        return self.get_mass_def_attrs("fitted_total_trapped_mass")
    @property
    def fitted_trapped_masses_by_bin(self) -> dict[str, dict[str | int, dict[str, float]]]:
        return self.get_mass_def_attrs("fitted_trapped_mass_by_bin")
    
    @property
    def raw_total_trapped_masses(self) -> dict[str | int, dict[str, float]]:
        return self.get_mass_def_attrs("raw_total_trapped_mass")
    @property
    def raw_trapped_masses_by_bin(self) -> dict[str, dict[str | int, dict[str, float]]]:
        return self.get_mass_def_attrs("raw_trapped_mass_by_bin")

    # When time permits, refactor this by adding a generalized function that 
    # that can be used for display, display_fits, display_fits_evo, etc. They
    # all have the same structure, just different arguments and return types.
    def display_fits(
            self,
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            y_tick_text_size: int = 10,
            return_fig: bool = False,
            color_palette: str = "viridis",
            mass_defs_to_display: list[str] | None = None
        ) -> plt.Axes | None:

        use_subset = mass_defs_to_display is not None

        shared_plot_args = {
            "mf_type": mf_type,
            "ax_main": ax_main,
            "x_label_text_size": x_label_text_size,
            "y_label_text_size": y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "y_tick_text_size": y_tick_text_size
        }

        if use_subset and len(mass_defs_to_display) == 1:
            return self[mass_defs_to_display[0]].display_fit(
                mass_def_key=mass_defs_to_display[0],
                return_fig=return_fig,
                **shared_plot_args
            )

        colormap = plt.cm.get_cmap(color_palette, len(self.contained_keys))
        handles = []

        definitions = (
            [
                self.contained_definitions[self.contained_keys.index(key)]
                for key in mass_defs_to_display
            ] if use_subset else self.contained_definitions
        )


        for i, mass_def in enumerate(definitions):

            key = self.contained_keys[self.contained_definitions.index(mass_def)]

            mass_func = getattr(self, mass_def)
            color = colormap(i)
            linestyle = MASS_DEF_PLOT_LINES[mass_def]
            marker = MASS_DEF_MARKER_STYLES[mass_def]
            handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color=color, 
                    linestyle=linestyle,
                    marker=marker,
                    label=self.contained_mass_eqn[i]
                )
            )

            ax_main = mass_func.display_fit(
                mass_def_key=key,
                color = colormap(i),
                plot_linestyle=linestyle,
                marker_style=marker,
                return_fig=True,
                **shared_plot_args
            )

        ax_main.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")

        ax_main.legend(handles=handles, fontsize=legend_text_size)

        if return_fig:
            return ax_main
    
    def display(
            self, 
            mf_type: str = "differential",
            with_fits: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
            show_raw: bool = False,
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
            color_palette: str = "viridis",
            mass_defs_to_display: list[str] | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        use_subset = mass_defs_to_display is not None

        plot_args = {
            "mf_type": mf_type,
            "with_fits": with_fits,
            "show_raw": show_raw,
            "resid_min": resid_min,
            "resid_max": resid_max,
            # "text_size": text_size,
            "show_legend" : False,
            "x_label_text_size": x_label_text_size,
            "main_y_label_text_size": main_y_label_text_size,
            "resid_y_label_text_size": resid_y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "main_y_tick_text_size": main_y_tick_text_size,
            "resid_y_tick_text_size": resid_y_tick_text_size
        }

        if use_subset and len(mass_defs_to_display) == 1:
            return self[mass_defs_to_display[0]].display(
                mass_def_key=mass_defs_to_display[0],
                ax_main=ax_main,
                ax_resid=ax_resid,
                return_fig=return_fig,
                **plot_args
            )

        # Only treat axes as "incomplete" when the panel we actually
        # need is missing: with_fits requires both ax_main and ax_resid,
        # but without_fits only needs ax_main. Previously the without_fits
        # branch checked ax_resid is None too, which silently replaced a
        # user-supplied ax_main with a fresh figure.
        if with_fits and (ax_main is None or ax_resid is None):
            _, (ax_main, ax_resid) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True, sharey=False,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
            )

        if not with_fits and ax_main is None:
            _, ax_main = plt.subplots()
            ax_resid = None

        colormap = plt.cm.get_cmap(color_palette, len(self.contained_keys))
        handles = []

        definitions = (
            [
                self.contained_definitions[self.contained_keys.index(key)]
                for key in mass_defs_to_display
            ] if use_subset else self.contained_definitions
        )

        for i, mass_def in enumerate(definitions):

            full_idx = self.contained_definitions.index(mass_def)
            key = self.contained_keys[full_idx]

            mass_func = getattr(self, mass_def)
            color = colormap(i)
            linestyle = MASS_DEF_PLOT_LINES[mass_def]
            marker = MASS_DEF_MARKER_STYLES[mass_def]
            handles.append(
                plt.Line2D(
                    xdata=[0],
                    ydata=[0],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    # Index into contained_mass_eqn using the full-list
                    # position of this mass_def. Using `i` (the subset
                    # iteration index) would mis-label every entry as
                    # the first N items of the full list.
                    label=self.contained_mass_eqn[full_idx],
                )
            )

            ax_main, ax_resid = mass_func.display(
                mass_def_key=key, 
                color = colormap(i),
                ax_main = ax_main,
                ax_resid = ax_resid,
                plot_linestyle=linestyle,
                marker_style=marker,
                return_fig=True,
                **plot_args
            )

        if ax_resid is not None:
            ax_resid.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")
        else:
            ax_main.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")

        ax_main.legend(handles=handles, fontsize=legend_text_size)

        if return_fig:
            return ax_main, ax_resid



@define(slots=True) 
class HaloMassFunctionEvoData(MassDefinitionDependentEvoType):
    fof: MassFunctionEvoData | None = field(default=None, repr=False)
    subfind: MassFunctionEvoData | None = field(default=None, repr=False)
    crit200: MassFunctionEvoData | None = field(default=None, repr=False)
    crit500: MassFunctionEvoData | None = field(default=None, repr=False)
    mean200: MassFunctionEvoData | None = field(default=None, repr=False)
    virial: MassFunctionEvoData | None = field(default=None, repr=False)
    splashback: MassFunctionEvoData | None = field(default=None, repr=False)
    hydrostatic: MassFunctionEvoData | None = field(default=None, repr=False)
    turnaround: MassFunctionEvoData | None = field(default=None, repr=False)
    infall: MassFunctionEvoData | None = field(default=None, repr=False)
    marginally_bound: MassFunctionEvoData | None = field(default=None, repr=False)
    asymptotic: MassFunctionEvoData | None = field(default=None, repr=False)

    # peaks: DensityPeaksEvoData | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return super().__repr__()


    @cached_property
    def final_hmf(self) -> HaloMassFunction:
        return self.get_snapshot_instance(max(self.snapshot_ids))

    @property
    def population_masses(self) -> np.ndarray:
        return self.final_hmf.population_masses

    @property
    def round_one_population_masses(self) -> np.ndarray:
        return self.final_hmf.round_one_population_masses

    @cached_property
    def fold_instances(self) -> dict[int, HaloMassFunctionEvoData]:
        ''' 
        Return a dictionary of HaloMassFunctionEvoData instances, one for each fold.
        '''
        folds = {} # mass_def -> fold_id -> MassFunctionEvoData
        for mass_def in self.contained_definitions:
            if getattr(self, mass_def) is not None:
                folds[mass_def] = getattr(self, mass_def).fold_instances

        fold_ids = set().union(*(fold_data.keys() for fold_data in folds.values()))

        instances = defaultdict(dict) # fold_id -> mass_def -> MassFunctionEvoData
        for fold_id in fold_ids:
            for mass_def in self.contained_definitions:
                instances[fold_id][mass_def] = folds[mass_def][fold_id]
            
        return {
            fold_id: HaloMassFunctionEvoData(**fold_mf_evos)
            for fold_id, fold_mf_evos in instances.items()
        }

    @property
    def model(self) -> AsymptoticAbundanceEvoModel:
        return AsymptoticAbundanceEvoModel(**self.get_mass_def_attrs("fits"))
    
    @cached_property
    def raw_evo(self) -> HaloMassFunctionEvoData:
        return HaloMassFunctionEvoData(**self.get_mass_def_attrs("raw_data_evo"))
    
    @cached_property
    def data_trapped_masses_evo(self) -> dict[str, np.ndarray]:
        return self.get_mass_def_attrs("data_trapped_mass_evo")
    
    @cached_property
    def data_trapped_masses_by_bin_evo(self) -> dict[str, dict[str | int, np.ndarray]]:
        return self.get_mass_def_attrs("data_trapped_mass_by_bin_evo")
    
    @cached_property
    def fitted_trapped_masses_evo(self) -> dict[str, np.ndarray]:
        return self.get_mass_def_attrs("fitted_trapped_mass_evo")
    
    @cached_property
    def fitted_trapped_masses_by_bin_evo(self) -> dict[str, dict[str | int, np.ndarray]]:
        return self.get_mass_def_attrs("fitted_trapped_mass_by_bin_evo")
    
    @cached_property
    def raw_trapped_masses_evo(self) -> dict[str, np.ndarray]:
        return self.get_mass_def_attrs("raw_trapped_mass_evo")
    
    @cached_property
    def raw_trapped_masses_by_bin_evo(self) -> dict[str, dict[str | int, np.ndarray]]:
        return self.get_mass_def_attrs("raw_trapped_mass_by_bin_evo")

    def get_snapshot_instance(self, snapshot_id: int) -> HaloMassFunction:
        return HaloMassFunction(
            **self.get_mass_def_attrs_instance("data", snapshot_id)
        )

    def get_abundance_accumulation(
            self,
            mass_bin: MassBinKey,
            use_round_one: bool = False,
        ) -> AbundanceAccumulationHistories:
        """
        Retrieve abundance accumulation histories for a single mass bin across
        all mass definitions.

        Parameters
        ----------
        mass_bin : MassBinKey
            Mass-bin identifier (int, float, or string label).
        use_round_one : bool, optional
            If True (default), pull from the round-one accumulation histories
            on each MassFunctionEvoData (higher-resolution in log10 M).
            If False, fall back to the original coarser accumulation_histories.
        """
        attr_name = (
            "round_one_accumulation_histories"
            if use_round_one else 
            "accumulation_histories"
        )

        histories = AbundanceAccumulationHistories(
            **self.get_mass_def_attrs_instance(
                attr_name=attr_name,
                instance_id=mass_bin,
            )
        )

        if histories.contained_definitions == self.contained_definitions:
            return histories

        raise ValueError(
            f"Mass Definitions for M={mass_bin} accumulations do not match"
        )
    

    def get_fold_abundance_accumulation(
            self,
            mass_bin: MassBinKey,
            use_round_one: bool = False,
        ) -> dict[int, AbundanceAccumulationHistories]:
        """
        Retrieve abundance accumulation histories for a single mass bin across
        all folds and mass definitions.

        Parameters
        ----------
        mass_bin : MassBinKey
            Mass-bin identifier (int, float, or string label).
        use_round_one : bool, optional
            If True (default), pull from round-one fold accumulation histories
            on each MassFunctionEvoData. If False, use the original coarser
            fold_accumulation_histories.
        """
        attr_name = (
            "round_one_fold_accumulation_histories"
            if use_round_one else 
            "fold_accumulation_histories"
        )

        # mass_def -> fold_id -> PopulationAccumulationHistories
        mdef_histories = self.get_mass_def_attrs(attr_name)
        fold_ids = {fid for histories in mdef_histories.values() for fid in histories}
        mass_bin_key = mass_bin if is_int_value(mass_bin) else f"{mass_bin:.1f}"

        return {
            fold_id: AbundanceAccumulationHistories(
                **{
                    mass_def: mdef_histories[mass_def][fold_id].get(mass_bin_key)
                    for mass_def in self.contained_definitions
                    if fold_id in mdef_histories[mass_def]
                }
            )
            for fold_id in fold_ids
        }
        
    @property
    def peaks_evo(self) -> DensityPeaksEvo | None:
        for mdef in self.contained_definitions:
            if (peak_evo := getattr(self, mdef).peaks_evo) is not None:
                return peak_evo

        return None

    def get_population_accumulation_histories(
            self, 
            masses: np.ndarray, 
            include_folds: bool = False,
            new_key_type: str = "integer", # integer | str | float
            str_key_format: callable = lambda x: f"{x:.1f}",
            use_round_one: bool = False,
        ) -> PopulationAbundanceAccumulationHistories:

        accumulations = OrderedDict()
        fold_accumulations = defaultdict(dict) # fold_id -> mass_bin -> AccumulationHistory

        for mass_bin in tqdm(
            masses,
            desc="Computing population accumulation histories",
            total=masses.size
        ):
        
            mass_bin_key = get_new_key(
                key=mass_bin, 
                new_key_type=new_key_type, 
                str_key_format=str_key_format
            )

            try:
                accumulations[mass_bin_key] = self.get_abundance_accumulation(
                    mass_bin=mass_bin, use_round_one=use_round_one
                )
                if include_folds:
                    fold_histories = self.get_fold_abundance_accumulation(
                        mass_bin=mass_bin, use_round_one=use_round_one
                    )
                    for fold_id, history in fold_histories.items():
                        fold_accumulations[fold_id][mass_bin_key] = history
            except (ValueError, IndexError):
                continue

        return PopulationAbundanceAccumulationHistories(
            data=accumulations, 
            folds={
                fold_id : PopulationAbundanceAccumulationHistories(
                    data=OrderedDict(
                        (mass_bin_key, accum)
                        for mass_bin_key, accum in accumulations.items()
                    )
                )
                for fold_id, accumulations in fold_accumulations.items()
            }
            # folds=fold_accumulations if include_folds else {}
        )

    @cached_property
    def accumulation_histories(self) -> PopulationAbundanceAccumulationHistories:
        return self.get_population_accumulation_histories(
            masses=self.final_hmf.population_masses,
            include_folds=False,
            new_key_type="integer",
            use_round_one=False
        )

    @cached_property
    def round_one_accumulation_histories(self) -> PopulationAbundanceAccumulationHistories:
        return self.get_population_accumulation_histories(
            masses=self.final_hmf.round_one_population_masses,
            include_folds=False,
            new_key_type="string",
            str_key_format=lambda x: f"{x:.1f}",
            use_round_one=True
        )
    
    @cached_property
    def fold_accumulation_histories(self) -> dict[int, PopulationAbundanceAccumulationHistories]:
        final_hmf = self.final_hmf

        fold_accumulations = defaultdict(dict) # fold_id -> mass_bin -> AccumulationHistory
        for mass_bin in tqdm(
            final_hmf.population_masses, 
            desc="Computing fold accumulation histories",
            total=final_hmf.population_masses.size
        ):
            try:
                fold_histories = self.get_fold_abundance_accumulation(mass_bin)
                for fold_id, history in fold_histories.items():
                    fold_accumulations[fold_id][mass_bin] = history
            except (ValueError, IndexError):
                continue

        return {
            fold_id: PopulationAbundanceAccumulationHistories(data=mass_bin_histories)
            for fold_id, mass_bin_histories in fold_accumulations.items()
        }

    @cached_property
    def round_one_fold_accumulation_histories(self) -> dict[int, PopulationAbundanceAccumulationHistories]:
        final_hmf = self.final_hmf

        fold_accumulations = defaultdict(dict) # fold_id -> mass_bin -> AccumulationHistory
        for mass_bin in tqdm(
            final_hmf.round_one_population_masses, 
            desc="Computing round-one fold accumulation histories",
            total=final_hmf.round_one_population_masses.size
        ):
            mass_bin_key = f"{mass_bin:.1f}"
            try:
                fold_histories = self.get_fold_abundance_accumulation(
                    mass_bin, use_round_one=True
                )
                for fold_id, history in fold_histories.items():
                    fold_accumulations[fold_id][mass_bin_key] = history
            except (ValueError, IndexError):
                continue

        return {
            fold_id: PopulationAbundanceAccumulationHistories(data=mass_bin_histories)
            for fold_id, mass_bin_histories in fold_accumulations.items()
        }


    @cached_property
    def full_accumulation_histories(self) -> PopulationAbundanceAccumulationHistories:
        return self.get_population_accumulation_histories(
            masses=self.final_hmf.population_masses,
            include_folds=True,
            new_key_type="integer",
            use_round_one=False
        )

    @cached_property
    def round_one_full_accumulation_histories(self) -> PopulationAbundanceAccumulationHistories:
        return self.get_population_accumulation_histories(
            masses=self.final_hmf.round_one_population_masses,
            include_folds=True,
            new_key_type="string",
            str_key_format=lambda x: f"{x:.1f}",
            use_round_one=True
        )
     
    def compute_accumulation_peak_and_freeze_times(
            self,
            abundance_idx: int = 4,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 10_000,
            return_full_grid: bool = True,
            method: str = "hybrid_window",
            verbose: bool = False,
            use_round_one: bool = False,
            return_stats: bool = False,
        ) -> dict | tuple[dict, dict]:
        """
        Compute peak and freeze-out times for all mass bins of a given mass
        definition using the accumulation histories and their folds.

        This method mirrors the external snippet that operates on
        `composite_hmf_evo.accumulation_histories` and
        `composite_hmf_evo.fold_accumulation_histories`, but is now attached
        directly to the `HaloMassFunctionEvoData` class.

        Parameters
        ----------
        mass_def : str
            Mass-definition key (e.g., "fof", "virial", "200c", ...).
        abundance_idx : int, optional
            Index of the abundance track in the 6×N evolution array
            (1 → number density, 2 → differential, 3 → normalized,
             4 → cumulative, 5 → multiplicity). Default is 4.
        tol, rel_tol_no_folds, n_hi_res, return_full_grid, method :
            Passed through to `compute_peak_and_freeze_times`.
        verbose : bool, optional
            If True, prints information about skipped mass bins / folds.

        Returns
        -------
        accum_samples : dict[int, dict[str, np.ndarray]]
            For each mass bin, stores the stacked jackknife mean and std
            curves for the chosen abundance index:
                accum_samples[mass_bin] = {"mean": mean, "std": std}
        time_results : dict[str, dict[str, np.ndarray]]
            For each mass definition, a dictionary of time results are stored
            in arrays. These arrays have shape (N_mass_bins,). For the i^th mass 
            bin, the i^th entry in each array corresponds to that mass bin.
        """
        
        if use_round_one:
            full_accum_histories = self.round_one_full_accumulation_histories
        else:
            full_accum_histories = self.full_accumulation_histories

        return full_accum_histories.compute_accumulation_peak_and_freeze_times(
            abundance_idx=abundance_idx,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            return_stats=return_stats,
        )


    def save(self, file_path: Path) -> None:
        save_halo_mass_function_evo_data(self, file_path)

    
    @classmethod
    def load(cls, file_path: Path) -> HaloMassFunctionEvoData:
        return load_halo_mass_function_evo_data(file_path)
    
    # Display Accumulation Histories for different mass definitions potentially    
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

        self.accumulation_histories.display_population_accumulation_histories(
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
            mf_type: str = "cumulative",
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

        self.accumulation_histories.display(
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


    def display_fits_evo(
            self,
            target_scale_factors: Collection[float],
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            y_tick_text_size: int = 10,
            return_fig: bool = False,
            color_palette: str = "viridis",
            mass_defs_to_display: list[str] | None = None,
            bottom_legend_xloc: float = 0.5,
            bottom_legend_yloc: float = 0.15,
        ) -> plt.Axes | None:

        use_subset = mass_defs_to_display is not None

        if use_subset and len(mass_defs_to_display) == 1:
            return self[mass_defs_to_display[0]].display_fit_evolution(
                target_scale_factors=target_scale_factors,
                mass_def_key=mass_defs_to_display[0],
                mf_type=mf_type,
                ax_main=ax_main,
                x_label_text_size=x_label_text_size,
                y_label_text_size=y_label_text_size,
                x_tick_text_size=x_tick_text_size,
                y_tick_text_size=y_tick_text_size,
                return_fig=return_fig
            )
        
        target_snapshot_ids = self.moments.map_by_attribute(
            key_attr="scale_factor",
            attr_value=target_scale_factors,
            return_attr="snapshot_id"
        )

        actual_scale_factors = self.moments.map_by_attribute(
            key_attr="snapshot_id",
            attr_value=target_snapshot_ids,
            return_attr="scale_factor"
        )

        if isinstance(target_snapshot_ids, (int, float)):
            target_snapshot_ids = np.asarray(target_snapshot_ids)

        if ax_main is None:
            _, ax_main = plt.subplots()

        use_subset = mass_defs_to_display is not None

        colormap = plt.cm.get_cmap(color_palette, len(target_scale_factors))
        mass_def_handles, scale_factor_handles = [], []

        definitions = (
            [
                self.contained_definitions[self.contained_keys.index(key)]
                for key in mass_defs_to_display
            ] if use_subset else self.contained_definitions
        )

        for i, mass_def in enumerate(definitions):

            key = self.contained_keys[self.contained_definitions.index(mass_def)]

            mass_func = getattr(self, mass_def)
            linestyle = MASS_DEF_PLOT_LINES[mass_def]
            marker = MASS_DEF_MARKER_STYLES[mass_def]
            mass_def_handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color="k", 
                    linestyle=linestyle,
                    label=self.contained_mass_eqn[i]
                )
            )

            for j, snap_id in enumerate(target_snapshot_ids):

                color = colormap(j)

                if (i == 0):

                    scale_factor_handles.append(
                        plt.Line2D(
                            xdata=[0], 
                            ydata=[0], 
                            color=color, 
                            label=rf"$a={actual_scale_factors[j]:.2f}$"
                        )
                    )

                ax_main = mass_func.data[snap_id].display_fit(
                    mass_def_key=key, 
                    mf_type=mf_type,
                    ax_main = ax_main,
                    return_fig=True,    
                    plot_linestyle=linestyle,
                    marker_style=marker,
                    color = color,
                    x_label_text_size=x_label_text_size,
                    y_label_text_size=y_label_text_size,
                    x_tick_text_size=x_tick_text_size,
                    y_tick_text_size=y_tick_text_size
                )

        ax_main.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")

        mass_def_legend = ax_main.legend(
            handles=mass_def_handles, fontsize=legend_text_size,
            bbox_to_anchor=(bottom_legend_xloc, bottom_legend_yloc),
            ncols = len(mass_def_handles),
        )

        ax_main.legend(handles=scale_factor_handles, fontsize=legend_text_size)

        ax_main.add_artist(mass_def_legend)


        if return_fig:
            return ax_main
        
    def display_model_param_evos(
            self, 
            num_legend_cols: int = 1,
            text_size: int = 12,
            plot_loglog: bool = False, 
            cutoff_scale_factor: float = 0.05,
        ) -> None:

        _, ax = plt.subplots(
            4, 1, figsize=(4, 6), sharex=True, gridspec_kw={"hspace": 0.0}
        )

        masked = lambda params: params[params[:, 0] > cutoff_scale_factor]

        evo_model = self.model

        evo_params_A = evo_model.parameter_evos_A
        evo_params_a = evo_model.parameter_evos_a
        evo_params_b = evo_model.parameter_evos_b
        evo_params_c = evo_model.parameter_evos_c

        marker_style = "."

        for mdef in self.contained_keys:

            evo_A = masked(evo_params_A[mdef])
            evo_a = masked(evo_params_a[mdef])
            evo_b = masked(evo_params_b[mdef])
            evo_c = masked(evo_params_c[mdef])

            ax[0].semilogx(evo_A[:, 0], evo_A[:, 1], marker_style)
            ax[1].semilogx(evo_a[:, 0], evo_a[:, 1], marker_style)
            ax[2].semilogx(evo_b[:, 0], evo_b[:, 1], marker_style)
            ax[3].semilogx(evo_c[:, 0], evo_c[:, 1], marker_style)


        for i in range(4):
            ax[i].yaxis.set_tick_params(labelsize=text_size)

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
            loc="center", ncol=num_legend_cols,
            bbox_to_anchor=(0.5, -0.65),
            fontsize=text_size
        )

        ax[3].set_xlim(left=cutoff_scale_factor)
        
    def display(
            self,
            target_scale_factors: Collection[float] | None = None,
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
            color_palette: str = "viridis",
            mass_defs_to_display: list[str] | None = None,
            bottom_legend_xloc: float = 0.5,
            bottom_legend_yloc: float = 0.15,
            main_legend_loc: str = "lower right",
            display_legend: bool = True,
        ) -> tuple[plt.Figure | None, plt.Axes | None, plt.Axes | None] | None:

        use_subset = mass_defs_to_display is not None

        if target_scale_factors is None:
            target_scale_factors = self.moments.scale_factors

        if use_subset and len(mass_defs_to_display) == 1:
            return self[mass_defs_to_display[0]].display(
                target_scale_factors=target_scale_factors,
                mass_def_key=mass_defs_to_display[0],
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
                plot_linestyle=(0, ()),
                return_fig=return_fig,
                resid_min=resid_min,
                resid_max=resid_max,
                color_palette=color_palette,
            )

        target_snapshot_ids = self.moments.map_by_attribute(
            key_attr="scale_factor",
            attr_value=target_scale_factors,
            return_attr="snapshot_id"
        )

        actual_scale_factors = self.moments.map_by_attribute(
            key_attr="snapshot_id",
            attr_value=target_snapshot_ids,
            return_attr="scale_factor"
        )

        if isinstance(target_snapshot_ids, (int, float)):
            target_snapshot_ids = np.asarray(target_snapshot_ids)

        colormap = plt.cm.get_cmap(color_palette, len(target_snapshot_ids))
        mass_def_handles, scale_factor_handles = [], []

        axes_incomplete = any((ax_main is None, ax_resid is None))

        if with_fits and axes_incomplete:
            fig, (ax_main, ax_resid) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True, sharey=False,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
            )

        if not with_fits and axes_incomplete:
            fig, ax_main = plt.subplots()

        definitions = (
            [
                self.contained_definitions[self.contained_keys.index(key)]
                for key in mass_defs_to_display
            ] if use_subset else self.contained_definitions
        )

        for i, mass_def in enumerate(definitions):

            key = self.contained_keys[self.contained_definitions.index(mass_def)]

            mass_func = getattr(self, mass_def)
            linestyle = MASS_DEF_PLOT_LINES[mass_def]
            marker = MASS_DEF_MARKER_STYLES[mass_def]
            mass_def_handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color="k", 
                    linestyle=linestyle,
                    marker=marker,
                    label=self.contained_mass_eqn[i]
                )
            )

            for j, snap_id in enumerate(target_snapshot_ids):

                color = colormap(j)

                if (i == 0):

                    scale_factor_handles.append(
                        plt.Line2D(
                            xdata=[0], 
                            ydata=[0], 
                            color=color, 
                            label=rf"$a={actual_scale_factors[j]:.2f}$"
                        )
                    )

                if ((hmf := mass_func.data.get(snap_id)) is not None):

                    try:

                        ax_main, ax_resid = hmf.display(
                            mass_def_key=key,
                            mf_type=mf_type,
                            with_fits=with_fits,
                            ax_main=ax_main,
                            ax_resid=ax_resid,
                            show_raw=show_raw,
                            show_legend=show_legend,
                            return_fig=True,
                            resid_min=resid_min,
                            resid_max=resid_max,
                            plot_linestyle=linestyle,
                            marker_style=marker,
                            color=colormap(j),
                            x_label_text_size=x_label_text_size,
                            main_y_label_text_size=main_y_label_text_size,
                            resid_y_label_text_size=resid_y_label_text_size,
                            x_tick_text_size=x_tick_text_size,
                            main_y_tick_text_size=main_y_tick_text_size,
                            resid_y_tick_text_size=resid_y_tick_text_size
                        )

                    except TypeError as err:
                        print(f"Skipping {mass_def} due to {err}")
                        pdb.set_trace()

        if ax_resid is not None:
            ax_resid.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")
        else:
            ax_main.set_xlabel(r"$M_{\Delta} \, [h^{-1} M_{\odot}]$")

        if display_legend:
            mass_def_legend = ax_main.legend(
                handles=mass_def_handles, fontsize=legend_text_size,
                bbox_to_anchor=(bottom_legend_xloc, bottom_legend_yloc),
                ncols = len(mass_def_handles),
                facecolor="white", 
                framealpha=1
            )

            ax_main.add_artist(mass_def_legend)

            ax_main.legend(
                handles=scale_factor_handles, 
                fontsize=legend_text_size,
                loc=main_legend_loc,
                facecolor="white", 
                framealpha=1
            )

        if return_fig:
            return fig, ax_main, ax_resid
        

def get_new_key(
        key: MassBinKey, new_key_type: str, str_key_format: callable
    ) -> MassBinKey:

    match new_key_type:
        case "integer": 
            return key if isinstance(key, int) else int(key)
        case "string":
            return key if isinstance(key, str) else str_key_format(float(key))
        case "float":
            return key if isinstance(key, float) else float(key)
        case _:
            raise ValueError(f"Unrecognized key_type: {new_key_type}")  

def join_hmf_by_dict(mass_functions: dict[int, HaloMassFunction]) -> HaloMassFunction:
    hmfs_to_join = defaultdict(list)

    for hmf in mass_functions.values():
        for mass_def in hmf.contained_definitions:
            hmfs_to_join[mass_def].append(getattr(hmf, mass_def))

    return HaloMassFunction(
        **{
            mass_def: MassFunctionData.join(hmfs) 
            for mass_def, hmfs in hmfs_to_join.items()
        }
    )

def join_hmf_by_collection(mass_functions: Collection[HaloMassFunction]) -> HaloMassFunction:
    hmfs_to_join = defaultdict(list)

    for hmf in mass_functions:
        for mass_def in hmf.contained_definitions:
            hmfs_to_join[mass_def].append(getattr(hmf, mass_def))

    return HaloMassFunction(
        **{
            mass_def: MassFunctionData.join(hmfs) 
            for mass_def, hmfs in hmfs_to_join.items()
        }
    )



# ---------------------------------------------------------------------------
# HDF5 serialization helpers for abundance / mass-function objects
# ---------------------------------------------------------------------------

def _write_abundances_to_hdf5_group(grp: h5py.Group, abundances: Abundances) -> None:
    """
    Store an Abundances instance into an HDF5 group using only plain arrays.

    Layout inside `grp`:
        histogram
        histogram_density
        number_density
        differential
        normalized
        cumulative
        multiplicity
    """
    grp.create_dataset("histogram",          data=abundances.histogram)
    grp.create_dataset("histogram_density",  data=abundances.histogram_density)
    grp.create_dataset("number_density",     data=abundances.number_density)
    grp.create_dataset("differential",       data=abundances.differential)
    grp.create_dataset("normalized",         data=abundances.normalized)
    grp.create_dataset("cumulative",         data=abundances.cumulative)
    grp.create_dataset("multiplicity",       data=abundances.multiplicity)


def _read_abundances_from_hdf5_group(grp: h5py.Group) -> Abundances:
    """Reconstruct an Abundances object from an HDF5 subgroup."""
    def get(name: str) -> np.ndarray:
        return grp[name][()] if name in grp else np.empty(0)

    return Abundances(
        histogram=get("histogram"),
        histogram_density=get("histogram_density"),
        number_density=get("number_density"),
        differential=get("differential"),
        normalized=get("normalized"),
        cumulative=get("cumulative"),
        multiplicity=get("multiplicity"),
    )

def _write_abundance_relation_to_hdf5_group(grp: h5py.Group, relation: AbundanceRelation) -> None:
    """
    Store an AbundanceRelation instance into an HDF5 group using only
    basic, language-agnostic types (floats, arrays, etc.).

    Layout inside `grp`:
        /sample_bins/edges
        /sample_bins/centers
        /estimate/<fields>
        /errors/<fields>          (optional)
        /peaks/edges              (optional)
        /peaks/centers            (optional)
        /peaks/jacobian           (optional)
        /folds/<fold_id>/<name>   (optional; includes fold-level peaks)
    """
    # sample bins
    sample_grp = grp.create_group("sample_bins")
    sample_grp.create_dataset("edges",   data=relation.sample_bins["edges"])
    sample_grp.create_dataset("centers", data=relation.sample_bins["centers"])

    # estimate
    est_grp = grp.create_group("estimate")
    for name, arr in relation.estimate.as_dict.items():
        est_grp.create_dataset(name, data=arr)

    # errors (optional)
    if relation.errors is not None:
        err_grp = grp.create_group("errors")
        for name, arr in relation.errors.as_dict.items():
            err_grp.create_dataset(name, data=arr)

    # peaks (optional)
    if relation.peak_bins is not None:
        peaks_grp = grp.create_group("peaks")
        peaks_grp.create_dataset("edges",   data=relation.peak_bins["edges"])
        peaks_grp.create_dataset("centers", data=relation.peak_bins["centers"])
        if relation.peak_jacobian is not None:
            peaks_grp.create_dataset("jacobian", data=relation.peak_jacobian)

    # folds (optional)
    if relation.folds_info is not None:
        folds_grp = grp.create_group("folds")
        for fold_id, fold_entry in relation.folds_info.items():
            fgrp = folds_grp.create_group(str(fold_id))

            for key, value in fold_entry.items():
                if isinstance(value, np.ndarray):
                    fgrp.create_dataset(key, data=value)
                elif isinstance(value, dict):
                    subgrp = fgrp.create_group(key)
                    for sk, sv in value.items():
                        subgrp.create_dataset(sk, data=sv)


def _read_abundance_relation_from_hdf5_group(
        grp: h5py.Group,
        relation_cls: type[AbundanceRelation],
    ) -> AbundanceRelation:

    # sample bins
    sb = grp["sample_bins"]
    sample_bins = {
        "edges": sb["edges"][()],
        "centers": sb["centers"][()],
    }

    # estimate
    est_grp = grp["estimate"]
    estimate = _read_abundances_from_hdf5_group(est_grp)

    # errors
    errors = None
    if "errors" in grp:
        errors = _read_abundances_from_hdf5_group(grp["errors"])

    # peaks
    peak_bins = None
    peak_jacobian = None
    if "peaks" in grp:
        p = grp["peaks"]
        peak_bins = {
            "edges": p["edges"][()],
            "centers": p["centers"][()],
        }
        if "jacobian" in p:
            peak_jacobian = p["jacobian"][()]

    # folds
    folds_info = None
    if "folds" in grp:
        folds_info = {}
        fgrp = grp["folds"]
        for fold_name in fgrp.keys():
            fold_id = int(fold_name)
            fold_entry = {}

            subgrp = fgrp[fold_name]
            for key in subgrp.keys():
                obj = subgrp[key]
                if isinstance(obj, h5py.Dataset):
                    fold_entry[key] = obj[()]
                else:  # group
                    fold_entry[key] = {sk: obj[sk][()] for sk in obj.keys()}

            folds_info[fold_id] = fold_entry

    return relation_cls(
        sample_bins=sample_bins,
        estimate=estimate,
        errors=errors,
        peak_bins=peak_bins,
        peak_jacobian=peak_jacobian,
        folds_info=folds_info,
    )




def _write_triangle_to_hdf5_group(grp: h5py.Group, triangle: CoverageTriangle) -> None:
    """
    Store a CoverageTriangle instance into an HDF5 group.
    """
    grp.attrs["min_count"] = triangle.min_count
    grp.attrs["min_density"] = triangle.min_density
    grp.attrs["min_sample_value"] = triangle.min_sample_value
    grp.attrs["max_sample_value"] = triangle.max_sample_value
    grp.attrs["min_peak_height"] = triangle.min_peak_height
    grp.attrs["max_peak_height"] = triangle.max_peak_height
    grp.create_dataset("abundance_bounds", data=triangle.abundance_bounds)


def _write_peaks_to_hdf5_group(grp: h5py.Group, peaks: MassFunctionPeaks) -> None:
    """
    Store a MassFunctionPeaks instance into an HDF5 group.
    """
    grp.attrs["redshift"] = peaks.redshift
    grp.attrs["present_day_density"] = peaks.present_day_density
    grp.attrs["transfer"] = peaks.transfer
    grp.attrs["corrections"] = peaks.corrections
    grp.attrs["use_ps_file"] = peaks.use_ps_file

    if peaks.ps_file_path:
        grp.attrs["ps_file_path"] = str(peaks.ps_file_path)


def _write_mass_function_data_to_hdf5_group(sgrp: h5py.Group, mf_data: MassFunctionData) -> None:
    """
    Store a MassFunctionData instance for a single snapshot into an HDF5 group.

    Layout inside `sgrp`:
        attrs:
            comoving_volume
            relation_type
        /raw/...
        /cleaned/...    (if available)
        /fitted/...     (if available)
        /residuals/...  (if available)
        /triangle       (if available)
    """
    # Basic metadata
    sgrp.attrs["comoving_volume"] = mf_data.comoving_volume
    sgrp.attrs["relation_type"] = mf_data.relation_type.value

    if mf_data.peaks is not None:
        peaks_grp = sgrp.create_group("peaks")
        _write_peaks_to_hdf5_group(peaks_grp, mf_data.peaks)

    # Raw relation
    raw_grp = sgrp.create_group("raw")
    _write_abundance_relation_to_hdf5_group(raw_grp, mf_data.raw)

    # Cleaned relation, if present
    if mf_data.cleaned is not None:
        cleaned_grp = sgrp.create_group("cleaned")
        _write_abundance_relation_to_hdf5_group(cleaned_grp, mf_data.cleaned)

    # Fitted abundances, if present
    if hasattr(mf_data, "fitted") and mf_data.fitted is not None:
        fitted_grp = sgrp.create_group("fitted")
        _write_abundances_to_hdf5_group(fitted_grp, mf_data.fitted)

    # Residuals, if present
    if hasattr(mf_data, "residuals") and mf_data.residuals is not None:
        resid_grp = sgrp.create_group("residuals")
        _write_abundances_to_hdf5_group(resid_grp, mf_data.residuals)

    # Coverage triangle, if present
    if hasattr(mf_data, "triangle") and mf_data.triangle is not None:
        tri_grp = sgrp.create_group("triangle")
        _write_triangle_to_hdf5_group(tri_grp, mf_data.triangle)


# ---------------------------------------------------------------------------
# HDF5 I/O for HaloMassFunctionEvoData
# ---------------------------------------------------------------------------
def save_halo_mass_function_evo_data(
        hmf_evo_data: HaloMassFunctionEvoData,
        file_path: Path,
    ) -> None:
    """
    Save a HaloMassFunctionEvoData instance to an HDF5 file.

    This function writes two complementary representations:

    1) A compact, pickle-based payload for each mass-definition group
       (used by our internal loader to faithfully reconstruct the
       full Python object, including fits, peaks objects, etc.).

    2) A fully non-pickled, language-agnostic export under each
       mass-definition group, storing all mass-function data and
       jackknife information as basic arrays and scalars. This is
       intended for external users who do not have access to the
       analysis pipeline but still want to inspect or analyze the
       results.

    Layout
    ------
    /
      attrs:
        class          = "HaloMassFunctionEvoData"
        schema_version = 1
      /<mass_def>/
        attrs:
          class = "MassFunctionEvoData"
        /payload                 (uint8, pickled MassFunctionEvoData)
        /export/
          /moments/
            snapshot_id
            scale_factor
            redshift
            ages
            lookback_times
            hubble_epochs
          /snapshots/
            /<snapshot_id>/
              attrs:
                comoving_volume
                relation_type
              /raw/...
              /cleaned/...    (if available)
              /fitted/...     (if available)
              /residuals/...  (if available)
              /triangle       (if available)
    """

    file_path = Path(file_path)

    with h5py.File(file_path, "w") as h5f:
        h5f.attrs["class"] = "HaloMassFunctionEvoData"
        h5f.attrs["schema_version"] = 1

        for mass_def in hmf_evo_data.contained_definitions:
            mf_evo = getattr(hmf_evo_data, mass_def, None)
            if mf_evo is None:
                continue

            grp = h5f.create_group(mass_def)
            grp.attrs["class"] = mf_evo.__class__.__name__

            export_grp = grp.create_group("export")

            # ----- moments -----
            m = mf_evo.moments
            mgrp = export_grp.create_group("moments")

            for name in [
                "snapshot_ids", "scale_factors", "redshifts",
                "ages", "lookback_times", "hubble_epochs"
            ]:
                arr = getattr(m, name)
                if arr is not None and len(arr) > 0:
                    mgrp.create_dataset(name, data=arr)

            # ----- snapshots -----
            snaps_grp = export_grp.create_group("snapshots")

            for snap_id, mf_data in mf_evo.data.items():
                sgrp = snaps_grp.create_group(str(snap_id))
                _write_mass_function_data_to_hdf5_group(sgrp, mf_data)


def load_halo_mass_function_evo_data(file_path: Path) -> HaloMassFunctionEvoData:
    """
    Load a HaloMassFunctionEvoData instance from an HDF5 file created by
    `save_halo_mass_function_evo_data`.

    This loader currently relies on the compact pickle payload stored
    for each mass definition, which guarantees faithful reconstruction
    of the internal Python objects (MassFunctionEvoData instances) including
    fits, peaks, and any other nested state.

    External users who do not have access to the analysis pipeline are
    expected to read the `/export` subtree directly via their own HDF5
    tooling, without using this function.
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5f:
        if h5f.attrs.get("class", "") != "HaloMassFunctionEvoData":
            raise ValueError(f"{file_path} is not a HaloMassFunctionEvoData file")

        init_kwargs = {}

        for mass_def in HaloMassFunctionEvoData.__annotations__.keys():
            if mass_def not in h5f:
                continue

            grp = h5f[mass_def]["export"]

            # --- load moments ---
            mgrp = grp["moments"]

            def _maybe(name): 
                return mgrp[name][()] if name in mgrp else np.empty(0)

            moments = MomentsInTime(
                snapshot_ids=_maybe("snapshot_ids"),
                scale_factors=_maybe("scale_factors"),
                redshifts=_maybe("redshifts"),
                ages=_maybe("ages"),
                lookback_times=_maybe("lookback_times"),
                hubble_epochs=_maybe("hubble_epochs"),
            )

            # --- load snapshot mass functions ---
            data = OrderedDict()
            sgrp = grp["snapshots"]

            for snap_name in sorted(sgrp.keys(), key=int):
                snap_id = int(snap_name)
                sg = sgrp[snap_name]

                comoving_volume = float(sg.attrs["comoving_volume"])
                relation_type = RelationType(sg.attrs["relation_type"])

                raw = _read_abundance_relation_from_hdf5_group(
                    sg["raw"], MassFunction
                )

                cleaned = None
                if "cleaned" in sg:
                    cleaned = _read_abundance_relation_from_hdf5_group(
                        sg["cleaned"], MassFunction
                    )

                triangle = None
                if "triangle" in sg:
                    tg = sg["triangle"]
                    triangle = CoverageTriangle(
                        min_count=int(tg.attrs["min_count"]),
                        min_density=float(tg.attrs["min_density"]),
                        min_sample_value=float(tg.attrs["min_sample_value"]),
                        max_sample_value=float(tg.attrs["max_sample_value"]),
                        abundance_bounds=tg["abundance_bounds"][()],
                        min_peak_height=float(tg.attrs["min_peak_height"]),
                        max_peak_height=float(tg.attrs["max_peak_height"]),
                    )

                if "peaks" in sg:
                    pg = sg["peaks"]
                    peaks = DensityPeaks(
                        redshift=float(pg.attrs["redshift"]),
                        present_day_density=float(pg.attrs["present_day_density"]),
                        transfer=str(pg.attrs["transfer"]),
                        corrections=bool(pg.attrs["corrections"]),
                        use_ps_file=bool(pg.attrs["use_ps_file"]),
                        ps_file_path=Path(pg.attrs["ps_file_path"]) if "ps_file_path" in pg.attrs else None,
                    )
                else:
                    peaks = None

                mf_data = MassFunctionData(
                    comoving_volume=comoving_volume,
                    relation_type=relation_type,
                    raw=raw,
                    cleaned=cleaned,
                    triangle=triangle,
                    peaks=peaks,
                    run_fit=False,
                )

                if "fitted" in sg:
                    mf_data.fitted = _read_abundances_from_hdf5_group(sg["fitted"])

                if "residuals" in sg:
                    mf_data.residuals = _read_abundances_from_hdf5_group(sg["residuals"])

                data[snap_id] = mf_data

            mf_evo = MassFunctionEvoData(moments=moments, data=data)
            init_kwargs[mass_def] = mf_evo

    return HaloMassFunctionEvoData(**init_kwargs)

