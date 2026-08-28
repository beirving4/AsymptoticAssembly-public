from __future__ import annotations

import warnings
import numpy as np, pdb
import matplotlib.pyplot as plt

# numpy 2 renamed trapz to trapezoid; support both lines
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

from pathlib import Path
from typing import Type, ClassVar
from attrs import define, field
from collections import OrderedDict, defaultdict
from matplotlib.lines import Line2D
from collections.abc import Collection
from scipy.interpolate import interp1d


from .type_defs import AbundanceType
from ..fields.peaks import DensityPeaks, DensityPeaksEvo
from .data import (
    AbundanceValue, 
    Abundances,
    AbundanceRelation,
    AbundanceRelationEvo,
    AbundanceDataEvo,
    AbundanceData
)
from .accumulation import (
    AccumulationHistory, 
    AccumulationHistoryData, 
    PopulationAccumulationHistories
)
from ..model.mass_function import (
    AsymptoticAbundanceFit, AsymptoticAbundanceFitEvo
)
from .viz import (
    display_mass_function, 
    display_halo_multiplicity, 
    display_mass_function_data, 
    display_halo_multiplicity_fit,
)


FloatType = float | np.floating
IntType = int | np.integer
MassBinKey = str | IntType | FloatType

# May move this to bounded folder 

MIN_RESOLVED_HALO_MASS = 1E8

@define(slots=True)
class MassFunctionValue:
    mass: float
    abundance: AbundanceValue

    @property
    def nu(self) -> float:
        return self.abundance.peak_height
    
    @property
    def dndM(self) -> float:
        return self.abundance.number_density
    
    @property
    def dndlnM(self) -> float:
        return self.abundance.differential
    
    @property
    def M2_rho_dndM(self) -> float:
        return self.abundance.normalized
    
    @property
    def n_gtm(self) -> float:
        return self.abundance.cumulative

    @property
    def f(self) -> float:
        return self.abundance.multiplicity    

@define(slots=True)
class MassFunction(AbundanceRelation):
    def __repr__(self) -> str:
        return (
            f"MassFunction(M={self.masses.min():.2e}-"
            f"{self.masses.max():.2e})"
        )
    
    @property
    def mass_bins(self) -> dict[str, np.ndarray]:
        return self.sample_bins
    
    @property
    def masses(self) -> np.ndarray:
        return self.samples
    
    # @classmethod
    # def from_data(
    #         cls, data: dict[str, dict[str, np.ndarray]]
    #     ) -> MassFunction:

    #     # this super() call points to AbundanceRelation.from_data,
    #     # but binds its `cls` to MassFunctions

    #     return super(MassFunction, cls).from_data(data, bin_name="mass")
    
    @property
    def population_masses(self) -> np.ndarray:
        return np.unique(np.asarray(np.log10(self.masses), dtype=int))

    @property
    def round_one_population_masses(self) -> np.ndarray:
        return np.unique(np.round(np.log10(self.masses), 1))
    
    @property
    def total_trapped_mass(self) -> dict[str, float]:
        masses = np.logspace(
            # np.log10(MIN_RESOLVED_HALO_MASS), 
            np.log10(self.masses).min(),
            np.log10(self.masses).max(), 
            num=1000
        )
        estimate_integrand = interp1d(
            np.log10(self.masses),
            np.log10(self.estimate.multiplicity * self.peak_jacobian),
            fill_value="extrapolate"
        )
        error_integrand = interp1d(
            np.log10(self.masses),
            np.log10(self.errors.multiplicity * self.peak_jacobian),
            fill_value="extrapolate"
        )
        return {
            "estimate" : _trapezoid(
                10.0**estimate_integrand(np.log10(masses)), 
                np.log(masses)
            ), 
            "error" : _trapezoid(
                10.0**error_integrand(np.log10(masses)), np.log(masses)
            )
        }
    

    
    @property
    def trapped_mass_by_bin(self) -> dict[str | int, dict[str, float]]:
        trapped_masses = {"all" : self.total_trapped_mass}
        estimate_integrand = interp1d(
            np.log10(self.masses),
            np.log10(self.estimate.multiplicity * self.peak_jacobian),
            fill_value="extrapolate"
        )

        if self.errors is not None: 
            error_integrand = interp1d(
                np.log10(self.masses),
                np.log10(self.errors.multiplicity * self.peak_jacobian),
                fill_value="extrapolate"
            )
        else:
            error_integrand = None

        # if self.population_masses.min() > np.log10(MIN_RESOLVED_HALO_MASS):
        #     lower_mass_range = np.arange(
        #         np.log10(MIN_RESOLVED_HALO_MASS),
        #         self.population_masses.min()
        #     )
        #     population_masses = np.concatenate([
        #         lower_mass_range,
        #         self.population_masses
        #     ])
        # else:
        #     population_masses = self.population_masses

        population_masses = self.population_masses


        for mass in population_masses:
            mass_bin_range = np.logspace(mass, mass + 1.0, num=1000)
            
            trapped_masses[mass] = {
                "estimate" : _trapezoid(
                    10.0**estimate_integrand(np.log10(mass_bin_range)), 
                    np.log(mass_bin_range)
                )
            }

            if error_integrand is None:
                trapped_masses[mass]["error"] = 0.0
            else:
                trapped_masses[mass]["error"] = _trapezoid(
                    10.0**error_integrand(np.log10(mass_bin_range)), 
                    np.log(mass_bin_range)
                )

        return trapped_masses

    def display(
            self, 
            mass_def_key: str, 
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            x_label_text_size: int = 10,
            main_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            main_y_tick_text_size: int = 10,
            return_fig: bool = False,
            show_log_x: bool = True,
            show_log_y: bool = True, 
            plot_linestyle: tuple | None = (0, ()),
            marker_style: str | None = '.',
            color: tuple[float, float, float, float] | str | None = None
        ) -> plt.Axes | None: 

        valid_estimate_mask = np.isfinite(getattr(self.estimate, mf_type))
        if not np.any(valid_estimate_mask):
            warnings.warn(
                f"Mass function {mf_type} is empty for {mass_def_key}"
            )
            return None

        plot_args = {
            "mass_def_key" : mass_def_key,
            "estimates" : getattr(self.estimate, mf_type)[valid_estimate_mask],
            "errors" : (
                getattr(self.errors, mf_type)[valid_estimate_mask]
                if self.errors is not None else 
                None
            ),
            "ax_main" : ax_main,
            "return_fig" : True, 
            "show_log_x" : show_log_x,
            "show_log_y" : show_log_y, 
            "plot_linestyle" : plot_linestyle, 
            "marker_style" : marker_style,
            "color" : color, 
            "x_label_text_size" : x_label_text_size,
            "y_label_text_size" : main_y_label_text_size,
            "x_tick_text_size" : x_tick_text_size,
            "y_tick_text_size" : main_y_tick_text_size
        }

        if mf_type == AbundanceType.MULTIPLICITY.value:

            ax_main = display_halo_multiplicity(
                peak_heights=self.peak_heights[valid_estimate_mask], **plot_args
            )

        else:
            
            ax_main = display_mass_function(
                masses = self.masses[valid_estimate_mask],
                mf_type=mf_type, 
                **plot_args,
            )

        return ax_main if return_fig else None
    

    

@define(slots=True)
class MassFunctionEvo(AbundanceRelationEvo):

    data: OrderedDict[int, MassFunction] = field(factory=OrderedDict)
    relation_cls = MassFunction

    def __repr__(self) -> str:
        return super().__repr__()
    
    @property
    def trapped_mass_evo(self) -> np.ndarray:
        scale_factors, fractions, errors = [], [], []

        for snapshot_id, mass_function in self.data.items():
            if mass_function is None: continue
            scale_factors.append(
                self.moments.map_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snapshot_id,
                    return_attr="scale_factor"
                )
            )
            fraction = mass_function.total_trapped_mass
            fractions.append(fraction["estimate"])
            errors.append(fraction["error"])

        scale_factors = np.asarray(scale_factors)
        fractions = np.asarray(fractions)
        errors = np.asarray(errors)

        if fractions.size == 0: return np.empty((0, 3))

        if any(np.isnan(fractions)):

            nan_idx = np.where(np.isnan(fractions))[0] 
            fractions[nan_idx] = np.interp(
                scale_factors[nan_idx], 
                scale_factors[~np.isnan(fractions)], 
                fractions[~np.isnan(fractions)]
            )

        if any(np.isnan(errors)):

            nan_idx = np.where(np.isnan(errors))[0]
            errors[nan_idx] = np.nanmedian(errors[~np.isnan(errors)])

        return np.column_stack([scale_factors, fractions, errors])

    @property
    def trapped_mass_by_bin_evo(self) -> dict[str | int, np.ndarray]:

        trapped_mass_dict = {}
        trapped_mass_error_dict = {}
        scale_factors = defaultdict(list)

        for snapshot_id, mass_function in self.data.items():
            if mass_function is None:
                continue
            
            scale_factor = self.moments.map_by_attribute(
                key_attr="snapshot_id",
                attr_value=snapshot_id,
                return_attr="scale_factor"
            )
        
            for bin_key, mass_dict in mass_function.trapped_mass_by_bin.items():
                if bin_key not in trapped_mass_dict:
                    trapped_mass_dict[bin_key] = []
                    trapped_mass_error_dict[bin_key] = []
                
                trapped_mass_dict[bin_key].append(mass_dict["estimate"])
                trapped_mass_error_dict[bin_key].append(mass_dict["error"])
                scale_factors[bin_key].append(scale_factor)

        for bin_key in trapped_mass_dict:

            # Interpolate missing values
            # if any(np.isnan(trapped_mass_dict[bin_key])):

            #     nan_idx = np.where(np.isnan(trapped_mass_dict[bin_key]))[0]
            #     trapped_mass_dict[bin_key][nan_idx] = np.interp(
            #         scale_factors[bin_key][nan_idx], 
            #         scale_factors[bin_key][~np.isnan(trapped_mass_dict[bin_key])], 
            #         trapped_mass_dict[bin_key][~np.isnan(trapped_mass_dict[bin_key])]
            #     )

            # if any(np.isnan(trapped_mass_error_dict[bin_key])):

            #     nan_idx = np.where(np.isnan(trapped_mass_error_dict[bin_key]))[0]
            #     trapped_mass_error_dict[bin_key][nan_idx] = np.nanmedian(
            #         trapped_mass_error_dict[bin_key][~np.isnan(trapped_mass_error_dict[bin_key])]
                # )

            trapped_mass_dict[bin_key] = np.column_stack([
                scale_factors[bin_key], 
                trapped_mass_dict[bin_key],
                trapped_mass_error_dict[bin_key]
            ])

        return trapped_mass_dict 

    
    def display(
            self, 
            mass_def_key: str, 
            target_scale_factors: Collection[float],
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            x_label_text_size: int = 10,
            y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            y_tick_text_size: int = 10,
            return_fig: bool = False,
            show_log_x: bool = True,
            show_log_y: bool = True, 
            plot_linestyle: tuple | None = (0, ()),
            marker_style: str = '.',  
            colormap_name: str = "viridis",  
        ) -> plt.Axes | None: 

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

        colormap = plt.cm.get_cmap(colormap_name, len(target_snapshot_ids))

        for i, snap_id in enumerate(target_snapshot_ids): 

            ax_main = self.data[snap_id].display(
                mass_def_key=mass_def_key,
                mf_type=mf_type,
                ax_main=ax_main,
                x_label_text_size=x_label_text_size,
                main_y_label_text_size=y_label_text_size,
                x_tick_text_size=x_tick_text_size,
                main_y_tick_text_size=y_tick_text_size,
                return_fig=True,
                plot_linestyle=plot_linestyle,
                marker_style=marker_style,
                show_log_x=show_log_x,
                show_log_y=show_log_y,
                color=colormap(i),
            )

        ax_main.legend(
            handles=[
                Line2D([0], [0], color=colormap(i), label=rf"$a={a:.2f}$")
                for i, a in enumerate(actual_scale_factors)
            ],
            fontsize=text_size,
        )


        return ax_main if return_fig else None
    
    
@define
class MassFunctionData(AbundanceData):

    relation_cls = MassFunction

    def __repr__(self) -> str:
        return (
            f"MassFunctionData("
            f"L={np.cbrt(self.comoving_volume):.0f} cMpc / h, "
            f"log10(M)={self.population_masses.min():.1f}-"
            f"{self.population_masses.max():.1f}, "
            f"is_cleaned={not self.not_cleaned}, "
            f"has_optimized_fit={self.has_optimized_fit})"
        )
    

    def initialize_fit(self) -> None:
        """
        Initialize the mass-function fit model.

        For newly generated MassFunctionData instances, `self.peaks` should
        be a valid DensityPeaks object and we use its present_day_density.
        For objects reconstructed from disk (e.g., via HDF5) we may not
        have peaks available; in that case we still construct a fit object
        with a placeholder present-day density so that attributes that
        expect `self.fit` to exist continue to work, but we do not attempt
        to run or interpret any optimized fit.
        """
        if self.peaks is None:
            # Use a benign placeholder; this fit will not be optimized and
            # is only present so that attributes relying on `self.fit`
            # (e.g., has_fit / has_optimized_fit) do not fail.
            self.fit = AsymptoticAbundanceFit(0.0)
        else:
            self.fit = AsymptoticAbundanceFit(self.peaks.present_day_density)
    
    @property
    def masses(self) -> np.ndarray: 
        return self.samples
    @property
    def mass_bins(self) -> dict[str, np.ndarray]:
        return self.sample_bins

    @property
    def population_masses(self) -> np.ndarray:
        return (
            self.raw.population_masses 
            if self.not_cleaned else 
            self.cleaned.population_masses # type: ignore
        ) 
    
    @property
    def round_one_population_masses(self) -> np.ndarray:
        return (
            self.raw.round_one_population_masses
            if self.not_cleaned else 
            self.cleaned.round_one_population_masses # type: ignore
        )

    @property
    def mass_variances(self) -> np.ndarray:
        return np.empty(0) if self.no_peaks else self.peaks.mass_variance(self.masses)

    
    @property
    def fitted_mf(self) -> MassFunction:
        return self.fitted_abundance_relation
    

    def get_data_value(self, mass: float, is_logged: bool = True) -> MassFunctionValue:
        return MassFunctionValue(
            mass=(10.0**mass if is_logged else mass),
            abundance=self.get_interped_abundance_value(
                value=mass, 
                wrt_mass=True,
                from_masses=True,
                is_logged=is_logged
            )
        )
    
    def get_fitted_value(self, mass: float, is_logged: bool = True) -> MassFunctionValue:
        return MassFunctionValue(
            mass=(10.0**mass if is_logged else mass),
            abundance=self.get_fitted_abundance_value(
                value=mass, 
                relation_type="mass", 
                wrt_mass=True,
                is_logged=is_logged
            )
        )

    def get_value(
            self, mass: float, is_logged: bool = True, for_data: bool = True
        ) -> MassFunctionValue:

        return MassFunctionValue(
            mass=(10.0**mass if is_logged else mass),
            abundance=self.get_abundance_value(
                value=mass, 
                relation_type="mass",
                is_logged=is_logged, 
                for_data=for_data
            )
        )
    
    @property
    def data_total_trapped_mass(self) -> dict[str, float]:
        return self.cleaned.total_trapped_mass

    @property
    def data_trapped_mass_by_bin(self) -> dict[str | int, dict[str, float]]:
        return self.cleaned.trapped_mass_by_bin
    
    @property
    def fitted_total_trapped_mass(self) -> dict[str, dict[str, float]]:
        return self.fitted.total_trapped_mass
    
    @property
    def fitted_trapped_mass_by_bin(self) -> dict[str | int, dict[str, float]]:
        return self.fitted.trapped_mass_by_bin
    
    @property
    def raw_total_trapped_mass(self) -> dict[str, float]:
        return self.raw.total_trapped_mass
    
    @property
    def raw_trapped_mass_by_bin(self) -> dict[str | int, dict[str, float]]:
        return self.raw.trapped_mass_by_bin
    
    def save_to_file(self, save_dir: Path, mass_def: str) -> None:
        scale_factor = 1.0 / (1.0 + self.peaks.redshift)
        filename = f"mass_function_a{scale_factor:.3f}_{mass_def}"
        np.savez(Path(save_dir, filename), **self.as_dict)

    def display_fit(
            self,
            mass_def_key: str,
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            x_label_text_size: int = 10,
            y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            y_tick_text_size: int = 10,
            return_fig: bool = False,
            show_log_x: bool = True,
            show_log_y: bool = True,
            plot_linestyle: tuple | None = (0, ()),
            marker_style: str | None = '.',
            color: tuple[float, float, float, float] | str | None = None
        ) -> plt.Axes | None:

        if not self.has_fit:
            raise ValueError("Need to fit mass function to display fit!")
        
        if not self.has_optimized_fit:
            warnings.warn("Fit has not been optimized, displaying initial fit!")

        if ax_main is None:
            _, ax_main = plt.subplots()


        ax_main = self.fitted_mf.display(
            mass_def_key=mass_def_key,
            mf_type=mf_type,
            ax_main=ax_main,
            x_label_text_size=x_label_text_size,
            main_y_label_text_size=y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            main_y_tick_text_size=y_tick_text_size,
            show_log_x=show_log_x,
            show_log_y=show_log_y,
            return_fig=True,
            plot_linestyle=plot_linestyle,
            marker_style=marker_style,
            color=color
        )

        return ax_main if return_fig else None
        


    def display(
            self, 
            mass_def_key: str,
            mf_type: str = "differential",
            with_fits: bool = False,
            ax_main: plt.Axes | None = None,
            ax_resid: plt.Axes | None = None,
            show_raw: bool = False,
            show_legend: bool = True,
            x_label_text_size: int = 10,
            main_y_label_text_size: int = 10,
            resid_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            main_y_tick_text_size: int = 10,
            resid_y_tick_text_size: int = 10,
            return_fig: bool = False,
            show_log_x: bool = True,
            show_log_y: bool = True,
            resid_min: float | None = None,
            resid_max: float | None = None,
            plot_linestyle: tuple | None = (0, ()),
            marker_style: str | None = '.',
            color: tuple[float, float, float, float] | str | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        plot_args = dict( 
            mass_def_key=mass_def_key,
            mf_type=mf_type,
            ax_main=ax_main,
            return_fig=True,
            color=color,
            show_log_x=show_log_x,
            show_log_y=show_log_y,
            plot_linestyle=plot_linestyle,
            marker_style=marker_style,
            x_label_text_size=x_label_text_size,
            main_y_label_text_size=main_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            main_y_tick_text_size=main_y_tick_text_size,
        )

        if (self.cleaned is None) and not show_raw: return None

        # use_raw = show_raw or (self.cleaned is None)

        if not with_fits:
            ax_main = (
                self.raw.display(**plot_args)
                if show_raw else
                self.cleaned.display(**plot_args)
            )
            return (ax_main, ax_resid) if return_fig else None

        masses = self.raw.masses if show_raw else self.cleaned.masses
        peak_heights = (
            self.raw.peak_heights if show_raw else self.cleaned.peak_heights
        )
        
        plot_args |= {
            "estimates" : getattr(
                self.raw.estimate if show_raw else self.cleaned.estimate, 
                mf_type
            ),
            "errors" : (
                getattr(
                    self.raw.errors if show_raw else self.cleaned.errors, 
                    mf_type
                )
                if self.errors is not None else
                None
            ),
            "theoretical" : self.fit(peak_heights, masses, mf_type=mf_type),
            "residuals" : (
                getattr(self.residuals, mf_type) 
                if self.residuals is not None else None
            ),
            "ax_resid" : ax_resid,
            "resid_min" : resid_min,
            "resid_max" : resid_max,
            "show_legend" : show_legend,
            "plot_linestyle": plot_linestyle,
            "resid_y_label_text_size": resid_y_label_text_size,
            "resid_y_tick_text_size": resid_y_tick_text_size
        }

        ax_main, ax_resid = (
            display_halo_multiplicity_fit(peak_heights=peak_heights, **plot_args)
            if mf_type == AbundanceType.MULTIPLICITY.value else
            display_mass_function_data(masses=masses, **plot_args)
        )

        return (ax_main, ax_resid) if return_fig else None



@define
class MassFunctionEvoData(AbundanceDataEvo):

    data: OrderedDict[int, MassFunctionData] = field(factory=OrderedDict)

    relation_cls = MassFunction
    solo_evo_class = MassFunctionEvo

    def __repr__(self) -> str:
        return super().__repr__()
        
    @property
    def _final_instance(self) -> MassFunctionData:
        return self.data[max(self.data)]
    @property
    def population_masses(self) -> np.ndarray:
        return self._final_instance.population_masses

    @property
    def round_one_population_masses(self) -> np.ndarray:
        return self._final_instance.round_one_population_masses
    
    @property
    def data_trapped_mass_evo(self) -> np.ndarray:
        return self.clean_data_evo.trapped_mass_evo
    
    @property
    def data_trapped_mass_by_bin_evo(self) -> dict[str | int, np.ndarray]:
        return self.clean_data_evo.trapped_mass_by_bin_evo
    
    @property
    def fitted_trapped_mass_evo(self) -> np.ndarray:
        return self.fitted_evo.trapped_mass_evo
    
    @property
    def fitted_trapped_mass_by_bin_evo(self) -> dict[str | int, np.ndarray]:
        return self.fitted_evo.trapped_mass_by_bin_evo
    

    @property
    def raw_trapped_mass_evo(self) -> np.ndarray:
        return self.raw_data_evo.trapped_mass_evo

    @property
    def raw_trapped_mass_by_bin_evo(self) -> dict[str | int, np.ndarray]:
        return self.raw_data_evo.trapped_mass_by_bin_evo


    def get_accumulation_history(
            self, 
            mass: float, 
            for_data: bool = True, 
            is_logged: bool = True, 
            verbose: bool = True
        ) -> AccumulationHistory | None:

        data = OrderedDict()
        for snap_id, mf in self.data.items():
            if mf is None: continue
            try:
                mf_value = mf.get_value(mass, is_logged, for_data=for_data)
                data[snap_id] = mf_value.abundance
            except Exception as error:
                if verbose:
                    scale_factor = self.moments.map_by_attribute(
                        key_attr="snapshot_id",
                        attr_value=snap_id,
                        return_attr="scale_factor"
                    )
                    history_type = "data" if for_data else "fitted" 
                    print(
                        f"log10M={mass} {history_type} value not"
                        f"found for {scale_factor = :.3f} due to {error}"
                    )
                continue

        return AccumulationHistory(moments=self.moments, data=data) if data else None
    
        
    def get_accumulation_history_data(
            self, mass: float, is_logged: bool = True, verbose: bool = False
        ) -> AccumulationHistoryData | None:

        data = self.get_accumulation_history(
            mass=mass, for_data=True, is_logged=is_logged, verbose=verbose
        )
        fitted = self.get_accumulation_history(
            mass=mass, for_data=False, is_logged=is_logged, verbose=verbose
        )

        if ((data is None) and (fitted is None)):
            print(f"Could not get accumulation history data for M={mass:.1f}")
            return None
        
        return AccumulationHistoryData(final_mass=mass, data=data, fitted=fitted)
    

    def get_population_accumulation_history_data(
            self, masses: np.ndarray, is_logged: bool = True, verbose: bool = False
        ) -> PopulationAccumulationHistories:
        
        accumulation_data = OrderedDict()
    
        for mass_bin in masses: 
            if (
                (
                    mass_bin_data := self.get_accumulation_history_data(
                        mass=mass_bin, is_logged=is_logged, verbose=verbose
                    )
                )
                is not None
            ):
                
                mass_bin_key = mass_bin if is_int_value(mass_bin) else f"{mass_bin:.1f}"
                accumulation_data[mass_bin_key] = mass_bin_data

        return PopulationAccumulationHistories(data=accumulation_data)
        
        
    @property
    def accumulation_histories(self) -> PopulationAccumulationHistories:
        return self.get_population_accumulation_history_data(self.population_masses)

    @property
    def round_one_accumulation_histories(self) -> PopulationAccumulationHistories:
        return self.get_population_accumulation_history_data(
            self.round_one_population_masses
        )
    

    def get_fold_accumulation_history(
            self, mass: MassBinKey, 
            is_logged: bool = True, 
            verbose: bool = True
        ) -> dict[int, AccumulationHistory] | None:

        fold_instances = self.fold_instances
        results = {}

        for fold_id, instance in fold_instances.items():
            results[fold_id] = instance.get_accumulation_history(
                mass=float(mass), 
                for_data=True, 
                is_logged=is_logged, 
                verbose=verbose
            )

        return results or None
        

    def get_fold_accumulation_history_data(
            self, 
            mass: MassBinKey, 
            is_logged: bool = True, 
            verbose: bool = False
        ) -> dict[int, AccumulationHistoryData] | None:

        float_mass = float(mass)

        fold_data = self.get_fold_accumulation_history(
            mass=float_mass, 
            is_logged=is_logged, 
            verbose=verbose
        )

        if (fold_data is None):
            print(f"Could not get accumulation history data for M={float_mass:.1f}")
            return None
        
        return {
            fold_id: AccumulationHistoryData(final_mass=float_mass, data=data, fitted=None)
            for fold_id, data in fold_data.items() if data is not None
        } or None

    
    def get_population_fold_accumulation_history_data(
            self, masses: np.ndarray, is_logged: bool = True, verbose: bool = False
        ) -> dict[int, PopulationAccumulationHistories]:

        accumulation_data = OrderedDict()
    
        for mass_bin in masses: 
            if (
                (
                    mass_bin_data := self.get_fold_accumulation_history_data(
                        mass=mass_bin, is_logged=is_logged, verbose=verbose
                    )
                )
                is not None
            ):
                mass_bin_key = mass_bin if is_int_value(mass_bin) else f"{mass_bin:.1f}"
                accumulation_data[mass_bin_key] = mass_bin_data

        output = {}
        fold_ids = self.fold_instances.keys()
        for fold_id in fold_ids:
            output[fold_id] = OrderedDict()
            for mass_bin_key, accum_data in accumulation_data.items():
                if fold_id in accum_data:
                    output[fold_id][mass_bin_key] = accum_data[fold_id]

        return {
            fold_id : PopulationAccumulationHistories(data=data)
            for fold_id, data in output.items()
        }
    
    @property
    def fold_accumulation_histories(self) -> dict[int, PopulationAccumulationHistories]:
        return self.get_population_fold_accumulation_history_data(self.population_masses)
    

    @property
    def round_one_fold_accumulation_histories(self) -> dict[int, PopulationAccumulationHistories]:
        return self.get_population_fold_accumulation_history_data(self.round_one_population_masses)
    
    
    def display_accumulation_history(
            self,
            mass_def: str, 
            mass_eqn: str,
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
            top_legend_xloc: float = 1.0,
            top_legend_yloc: float = 1.0,
            color_palette: str = "viridis",
            populations_to_display: list[int] | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:

        return self.accumulation_histories.display(
            mass_def=mass_def, 
            mass_eqn=mass_eqn,
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
            top_legend_xloc=top_legend_xloc,
            top_legend_yloc=top_legend_yloc,
            color_palette=color_palette,
            populations_to_display=populations_to_display
        )


    def display_fit_evolution(  
            self, 
            target_scale_factors: Collection[float],
            mass_def_key: str,
            mf_type: str = "differential",
            ax_main: plt.Axes | None = None,
            legend_text_size: int = 10,
            x_label_text_size: int = 10,
            y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            y_tick_text_size: int = 10,
            return_fig: bool = False,
            color: tuple[float, float, float, float] | str | None = None,
            plot_linestyle: tuple | None = (0, ()),
            marker_style: str | None = '.',
        ) -> plt.Axes | None:

        for_multiplicity = mf_type == AbundanceType.MULTIPLICITY.value

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

        if color is None:

            colormap = plt.cm.get_cmap(self.color_palette, len(target_snapshot_ids))
        

        for i, snap_id in enumerate(target_snapshot_ids):

            ax_main = self.data[snap_id].display_fit(
                mass_def_key=mass_def_key,
                mf_type=mf_type,
                ax_main=ax_main,
                x_label_text_size=x_label_text_size,
                y_label_text_size=y_label_text_size,
                x_tick_text_size=x_tick_text_size,
                y_tick_text_size=y_tick_text_size,
                show_log_x=False,
                show_log_y=False,
                return_fig=True,
                plot_linestyle=plot_linestyle,
                marker_style=marker_style,
                color=colormap(i) if color is None else color
            )

        ax_main.set_xscale("log", base=2 if for_multiplicity else 10)
        ax_main.set_yscale("log", base=10)

        ax_main.legend(
            handles=[
                Line2D(
                    [0], [0], 
                    color=colormap(i) if color is None else color, 
                    label=rf"$a={a:.2f}$"
                )
                for i, a in enumerate(actual_scale_factors)
            ],
            fontsize=legend_text_size,
        )

        return ax_main if return_fig else None

    def display(
            self, 
            target_scale_factors: Collection[float],
            mass_def_key: str,
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
            color_palette: str | None = None,
            # color: tuple[float, float, float, float] | str | None = None
            # plot_backgroud_alpha: float = 0.2,
            # error_scale: float = 0.0,
            # save_fig: bool = False,
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        for_multiplicity = mf_type == AbundanceType.MULTIPLICITY.value
        
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

        palette = self.color_palette if color_palette is None else color_palette
        colormap = plt.cm.get_cmap(palette, len(target_snapshot_ids))

        axes_incomplete = any((ax_main is None, ax_resid is None))

        if with_fits and axes_incomplete:
            _, (ax_main, ax_resid) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True, sharey=False,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0}
            )

        if not with_fits and axes_incomplete:
            _, ax_main = plt.subplots()

        for i, snap_id in enumerate(target_snapshot_ids):

            if (snap_mf := self.data.get(snap_id)) is None: continue 

            if show_raw or snap_mf.cleaned is not None:
                ax_main, ax_resid = snap_mf.display(
                    mass_def_key=mass_def_key,
                    mf_type=mf_type,
                    with_fits=with_fits,
                    ax_main=ax_main,
                    ax_resid=ax_resid,
                    show_raw=show_raw,
                    show_log_x=False,
                    show_log_y=False,
                    show_legend=show_legend,
                    x_label_text_size=x_label_text_size,
                    main_y_label_text_size=main_y_label_text_size,
                    resid_y_label_text_size=resid_y_label_text_size,
                    x_tick_text_size=x_tick_text_size,
                    main_y_tick_text_size=main_y_tick_text_size,
                    resid_y_tick_text_size=resid_y_tick_text_size,
                    return_fig=True,
                    resid_min=resid_min,
                    resid_max=resid_max,
                    plot_linestyle=plot_linestyle,
                    marker_style=marker_style,
                    color=colormap(i)
                )

        ax_main.set_xscale("log", base=2 if for_multiplicity else 10)
        ax_main.set_yscale("log", base=10)

        if ax_resid is not None:
            ax_resid.set_xscale("log", base=2 if for_multiplicity else 10)

        ax_main.legend(
            handles=[
                Line2D([0], [0], color=colormap(i), label=rf"$a={a:.2f}$")
                for i, a in enumerate(actual_scale_factors)
            ],
            fontsize=legend_text_size
        )

        return (ax_main, ax_resid) if return_fig else None
    

def is_int_value(x: MassBinKey) -> bool:
    return (
        isinstance(x, IntType) or (isinstance(x, FloatType)) and x.is_integer()
    )