from __future__ import annotations

import h5py
import itertools
import numpy as np, pdb 
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Iterator
from attrs import define, field
from functools import cached_property
from scipy.interpolate import interp1d
from collections import OrderedDict, defaultdict


from ..simulation.evo import EvolutionData
from ..simulation.moments import MomentsInTime
from ..model.mass_function import AsymptoticAbundanceFit
from ..utils.freeze_out import compute_peak_and_freeze_times
from ..mass_def.base import (
    MassDefinitionDependentType, 
    MASS_DEF_MARKER_STYLES,
    MASS_DEF_PLOT_LINES
)
from .data import (
    AbundanceValue, 
    get_full_nan_array,
    get_abundance_data_idx
)
from .viz import (
    plot_single_accumulation_history,
    plot_accumulation_history_with_fits,
)

MassBinKey = int | float | str

def is_int_value(x: MassBinKey) -> bool:
    if isinstance(x, int): return True
    return bool(isinstance(x, float) and x.is_integer())

def mask_histories_for_residuals(
        data_scale_factors: np.ndarray,
        data_values: np.ndarray,
        fitted_scale_factors: np.ndarray,
        fitted_values: np.ndarray
    ) -> dict[str, np.ndarray]:

    common_scale_factors = np.intersect1d(data_scale_factors, fitted_scale_factors)
    data_mask = np.isin(data_scale_factors, common_scale_factors)
    fitted_mask = np.isin(fitted_scale_factors, common_scale_factors)

    return {
        "data" : data_values[:, data_mask],
        "fitted" : fitted_values[:, fitted_mask],
        "scale_factors" : common_scale_factors
    }


@define(slots=True)
class AccumulationHistory(EvolutionData):
    data: OrderedDict[int, AbundanceValue]

    normalizing_scale_factor: float = field(default=10.0)
    normalizing_abundance: AbundanceValue = field(init=False)

    folds: dict[int, AccumulationHistory] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        self.update_normalizing_abundance(self.normalizing_scale_factor)

    def __repr__(self) -> str:
        return super().__repr__() 

    @property
    def to_numpy(self) -> np.ndarray:

        number_densities = []
        differentials = []
        normalizeds = []
        cumulatives = []
        multiplicities = []

        for _, value in sorted(self.data.items(), key=lambda x: x[0]):
            number_densities.append(value.number_density)
            differentials.append(value.differential)
            normalizeds.append(value.normalized)
            cumulatives.append(value.cumulative)
            multiplicities.append(value.multiplicity)

        scale_factors = self.moments.map_by_attribute(
            key_attr="snapshot_id",
            attr_value=sorted(self.data.keys()),
            return_attr="scale_factor"
        )

        return np.array([
            scale_factors,
            number_densities, # dn(M, a)/dM
            differentials, # dn(M, a)/dlnM
            normalizeds, # (M^2 / rho_m) dn(M, a)/dM
            cumulatives, # n(>M, a)
            multiplicities, # f(nu, a)
        ])
    
    @property
    def rates(self) -> np.ndarray:
        ''' Calculate the rates of change of the mass functions '''
        values = self.to_numpy
        def rate(x: np.ndarray) -> np.ndarray:
            if not np.isfinite(x).all():
                return get_full_nan_array(x.shape)
            return np.gradient(np.log(x), np.log(values[0]))

        return np.array([
            values[0],
            rate(values[1]),
            rate(values[2]),
            rate(values[3]),
            rate(values[4]),
            rate(values[5])
        ])
    
    @property
    def accelerations(self) -> np.ndarray:
        rates = self.rates

        def accel(x: np.ndarray) -> np.ndarray:
            if not np.isfinite(x).all():
                return get_full_nan_array(x.shape)
            return np.gradient(np.log(x), np.log(rates[0]))

        return np.array([
            rates[0],
            accel(rates[1]),
            accel(rates[2]),
            accel(rates[3]),
            accel(rates[4]),
            accel(rates[5])
        ])
    
    @property
    def normalized(self) -> np.ndarray:
        values = self.to_numpy
        return np.array([
            values[0],
            values[1] / self.normalizing_abundance.number_density,
            values[2] / self.normalizing_abundance.differential,
            values[3] / self.normalizing_abundance.normalized,
            values[4] / self.normalizing_abundance.cumulative,
            values[5] / self.normalizing_abundance.multiplicity
        ])
        

    @property
    def freeze_out_times(self) -> np.ndarray:
        values = self.to_numpy  # shape (6, N)
        scale_factors = values[0, :]

        # Build fold evolutions once per abundance index
        has_folds = len(self.folds) > 0

        freeze_out_times = [np.inf] * (values.shape[0] - 1)

        for i in range(1, values.shape[0]):
            track = values[i, :]
            if ~np.isfinite(track).all() or track[-1] == 0.0:
                continue

            full_evolution = track / track[-1]

            if has_folds:
                fold_id_to_evolution: dict[int, np.ndarray] = {}
                for fold_id, fold_data in self.folds.items():
                    fold_vals = fold_data.to_numpy
                    fold_track = fold_vals[i, :]
                    if ~np.isfinite(fold_track).all() or fold_track[-1] == 0.0:
                        continue
                    fold_evo = fold_track / fold_track[-1]
                    fold_id_to_evolution[fold_id] = fold_evo
                fold_arg = fold_id_to_evolution or None
            else:
                fold_arg = None

            times = compute_peak_and_freeze_times(
                scale_factors=scale_factors,
                full_evolution=full_evolution,
                fold_id_to_evolution=fold_arg,
                tol=2.0,
                rel_tol_no_folds=0.02,
                n_hi_res=10_000,
                method="hybrid_window",
            )
            freeze_out_times[i - 1] = times["a_frz"]

        return np.array(freeze_out_times)

    

    @property
    def freeze_out_times_with_errors(self) -> dict[str, np.ndarray]:
        values = self.to_numpy
        scale_factors = values[0, :]

        has_folds = len(self.folds) > 0

        freeze_out_times = [np.inf] * (values.shape[0] - 1)
        freeze_out_errors = [np.inf] * (values.shape[0] - 1)

        for i in range(1, values.shape[0]):
            track = values[i, :]
            if ~np.isfinite(track).all() or track[-1] == 0.0:
                continue

            full_evolution = track / track[-1]

            if has_folds:
                fold_id_to_evolution: dict[int, np.ndarray] = {}
                for fold_id, fold_data in self.folds.items():
                    fold_vals = fold_data.to_numpy
                    fold_track = fold_vals[i, :]
                    if ~np.isfinite(fold_track).all() or fold_track[-1] == 0.0:
                        continue
                    fold_evo = fold_track / fold_track[-1]
                    fold_id_to_evolution[fold_id] = fold_evo
                fold_arg = fold_id_to_evolution or None
            else:
                fold_arg = None

            times = compute_peak_and_freeze_times(
                scale_factors=scale_factors,
                full_evolution=full_evolution,
                fold_id_to_evolution=fold_arg,
                tol=2.0,
                rel_tol_no_folds=0.02,
                n_hi_res=10_000,
                method="hybrid_window",
            )
            freeze_out_times[i - 1] = times["a_frz"]
            freeze_out_errors[i - 1] = times["a_frz_err"]

        return {
            "times": np.array(freeze_out_times),
            "errors": np.array(freeze_out_errors),
        }
                

        
    
    def update_normalizing_abundance(self, cutoff_scale_factor: float) -> None:
        final_peak_height = self.data[max(self.data)].peak_height
        self.normalizing_scale_factor = cutoff_scale_factor
        self.normalizing_abundance = AbundanceValue.for_normalizing(
            peak_height=final_peak_height,
            cutoff_scale_factor=self.normalizing_scale_factor,
            accumulation_history_array=self.to_numpy
        )

    def get_freeze_out_time(
            self,
            abundance_type: str = "differential",
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 10_000,
            return_error: bool = False,
            method: str = "hybrid_window",
        ) -> float | tuple[float, float]:

        abundance_idx = get_abundance_data_idx(abundance_type)
        values = self.to_numpy
        scale_factors = values[0, :]
        track = values[abundance_idx, :]

        full_evolution = track / track[-1]

        if len(self.folds) > 0:
            fold_id_to_evolution: dict[int, np.ndarray] = {}
            for fold_id, fold_data in self.folds.items():
                fold_vals = fold_data.to_numpy
                fold_track = fold_vals[abundance_idx, :]
                fold_evo = fold_track / fold_track[-1]
                fold_id_to_evolution[fold_id] = fold_evo
            fold_arg = fold_id_to_evolution or None
        else:
            fold_arg = None

        times = compute_peak_and_freeze_times(
            scale_factors=scale_factors,
            full_evolution=full_evolution,
            fold_id_to_evolution=fold_arg,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            method=method,
        )

        return (
            (times["a_frz"], times["a_frz_err"])
            if return_error else times["a_frz"]
        )

    def get_peak_time(
            self, 
            abundance_type: str = "differential",
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 10_000,
            return_error: bool = False,
            method: str = "hybrid_window",
        ) -> float | tuple[float | float]:

        abundance_idx = get_abundance_data_idx(abundance_type)
        array = self.to_numpy
        full_evolution = np.empty_like(array)
        full_evolution[0, :] = array[0, :]
        full_evolution[1:, :] = array[1:, :] / array[1:, -1][:, np.newaxis]

        if len(self.folds) > 0:
            fold_id_to_evolution = {}
            for fold_id, fold_data in self.folds.items():
                fold_array = fold_data.to_numpy
                fold_full_evolution = np.empty_like(fold_array)
                fold_full_evolution[0, :] = fold_array[0, :]
                fold_full_evolution[1:, :] = (
                    fold_array[1:, :] / fold_array[1:, -1][:, np.newaxis]
                )
                fold_id_to_evolution[fold_id] = fold_full_evolution
        else:
            fold_id_to_evolution = None

        times = compute_peak_and_freeze_times(
            full_evolution=full_evolution, 
            fold_id_to_evolution=fold_id_to_evolution,
            abundance_type=abundance_idx,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            method=method,
        )

        return (
            (times["a_peak"], times["a_peak_err"]) if return_error else times["a_peak"]
        )

     # Make different functions for different abundance types, and use this as a wrapper    
    
    def display( 
            self, 
            mass_def: str | None = None,
            mf_type: str = "differential",
            normalize: bool = True, 
            normalizing_scale_factor: float = 10.0,
            return_fig: bool = False,
            x_label_text_size: int = 10,
            top_y_label_text_size: int = 10,
            bottom_y_label_text_size: int = 10,
            x_tick_text_size: int = 10,
            top_y_tick_text_size: int = 10,
            bottom_y_tick_text_size: int = 10,
            plot_linestyle: tuple = (0, ()),
            show_top_log_y: bool = False,
            show_bottom_log_y: bool = False,
            ax_top: plt.Axes | None = None,
            ax_bottom: plt.Axes | None = None,
            color: tuple[float, float, float, float] | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        self.update_normalizing_abundance(normalizing_scale_factor)
        
        values = self.to_numpy

        data_idx = get_abundance_data_idx(mf_type)

        ax_top, ax_bottom = plot_single_accumulation_history(
            scale_factors=values[0],
            accumulation_history=values[data_idx],
            accumulation_rate=self.rates[data_idx],
            mf_type=mf_type,
            normalizing_factor = (
                getattr(self.normalizing_abundance, mf_type) if normalize else 1.0
            ),
            return_fig=True,
            x_label_text_size=x_label_text_size,
            top_y_label_text_size=top_y_label_text_size,
            bottom_y_label_text_size=bottom_y_label_text_size,
            x_tick_text_size=x_tick_text_size,
            top_y_tick_text_size=top_y_tick_text_size,
            bottom_y_tick_text_size=bottom_y_tick_text_size,
            show_top_log_y=show_top_log_y,
            show_bottom_log_y=show_bottom_log_y,
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            color=color,
            plot_linestyle=plot_linestyle
        )

        if mass_def is not None:

            ax_bottom.text(
                0.9, 0.9, 
                mass_def,
                transform=ax_bottom.transAxes, fontsize=text_size
            )

        if return_fig:
            return ax_top, ax_bottom




@define(slots=True)
class AccumulationHistoryData:
    final_mass: float
    data: AccumulationHistory | None # change name to sim
    fitted: AccumulationHistory | None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(M={self.final_mass})"
    
    # def update_

    @property
    def history_residuals(self) -> np.ndarray:

        if self.data is None or self.fitted is None:
            raise ValueError(
                "Need data and fitted mass function to set residuals!"
            )
        
        data_values = self.data.to_numpy
        fitted_values = self.fitted.to_numpy

        values = mask_histories_for_residuals(
            data_scale_factors=data_values[0], 
            data_values=data_values[1:], 
            fitted_scale_factors=fitted_values[0], 
            fitted_values=fitted_values[1:]
        )

        log_diff = (np.log10(values["data"]) - np.log10(values["fitted"]))

        return np.vstack([values["scale_factors"], 10**log_diff - 1.0])
    
    @property
    def rate_residuals(self) -> np.ndarray:

        if self.data is None or self.fitted is None:
            raise ValueError(
                "Need data and fitted mass function to set residuals!"
            )

        data_values = self.data.rates
        fitted_values = self.fitted.rates

        values = mask_histories_for_residuals(
            data_scale_factors=data_values[0], 
            data_values=data_values[1:], 
            fitted_scale_factors=fitted_values[0], 
            fitted_values=fitted_values[1:]
        )

        ratio = values["data"] / values["fitted"]

        return np.vstack([values["scale_factors"], ratio - 1.0])

    @property
    def freeze_out_times(self) -> float:
        return self.data.freeze_out_times

    @property
    def freeze_out_times_with_errors(self) -> dict[str, np.ndarray]:
        return self.data.freeze_out_times_with_errors

    def update_normalizing_abundance(self, cutoff_scale_factor: float) -> None:
        self.data.update_normalizing_abundance(cutoff_scale_factor)
        self.fitted.update_normalizing_abundance(cutoff_scale_factor)

    def display(
            self, 
            mass_def: str,
            display_def: bool = True,
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
            show_legend: bool = True,
            show_top_log_y: bool = False,
            show_bottom_log_y: bool = False,
            return_fig: bool = False,
            plot_linestyle: tuple = (0, ()),
            plot_marker_style: str = ".",
            ax_main_top: plt.Axes | None = None,
            ax_resid_top: plt.Axes | None = None,
            ax_main_bottom: plt.Axes | None = None,
            ax_resid_bottom: plt.Axes | None = None,
            color: tuple[float, float, float, float] | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:
        
        plt_args = {
            "mf_type": mf_type,
            "return_fig": True,
            "normalize": normalize,
            "show_top_log_y": show_top_log_y,
            "show_bottom_log_y": show_bottom_log_y,
            "color": color, 
            "plot_linestyle": plot_linestyle,
        }

        if not with_fits:

            ax_main_top, ax_main_bottom = self.data.display(
                ax_top=ax_main_top,
                ax_bottom=ax_main_bottom,
                normalizing_scale_factor=(
                    normalizing_scale_factor if normalize else 1.0
                ),
                x_label_text_size=x_label_text_size,
                top_y_label_text_size=top_main_y_label_text_size,
                bottom_y_label_text_size=bottom_main_y_label_text_size,
                x_tick_text_size=x_tick_text_size,
                top_y_tick_text_size=top_main_y_tick_text_size,
                bottom_y_tick_text_size=bottom_main_y_tick_text_size,
                **plt_args
            )

            ax_resid_top, ax_resid_bottom = None, None

        else:

            data_history, data_rates = self.data.to_numpy, self.data.rates
            data_idx = get_abundance_data_idx(mf_type)
            
            fitted_history = self.fitted.to_numpy
            fitted_rates = self.fitted.rates
            resid_history = self.history_residuals
            resid_rates = self.rate_residuals


            self.update_normalizing_abundance(normalizing_scale_factor)    

            _, _, ax_main_bottom, _ = plot_accumulation_history_with_fits(
                data_scale_factors=data_history[0],
                data_history=data_history[data_idx],
                data_rate=data_rates[data_idx],
                fitted_scale_factors=fitted_history[0],
                fitted_history=fitted_history[data_idx],
                fitted_rate=fitted_rates[data_idx],
                history_residual_scale_factors=resid_history[0],
                history_residuals=resid_history[1],
                rate_residual_scale_factors=resid_rates[0],
                rate_residuals=resid_rates[1],
                ax_main_top=ax_main_top,
                ax_resid_top=ax_resid_top,
                ax_main_bottom=ax_main_bottom,
                ax_resid_bottom=ax_resid_bottom,
                show_legend=show_legend,
                plot_marker_style=plot_marker_style,
                top_resid_min=top_resid_min,
                top_resid_max=top_resid_max,
                bottom_resid_min=bottom_resid_min,
                bottom_resid_max=bottom_resid_max,
                data_normalizing_factor = (
                    getattr(self.data.normalizing_abundance, mf_type)
                    if normalize else 1.0
                 ),
                fitted_normalizing_factor = (
                    getattr(self.fitted.normalizing_abundance, mf_type)
                    if normalize else 1.0
                ),
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
                **plt_args
            )

        if display_def:

            ax_main_bottom.text(
                x=0.8, 
                y=0.9, 
                s=rf"{mass_def}=$10^{{{self.final_mass:.0f}}}$ $M_{{\odot}}$", 
                transform=ax_main_bottom.transAxes, fontsize=text_size
            )

        if return_fig:
            return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom
        

@define(slots=True)
class PopulationAccumulationHistories: # This can be generalized for mass or size bins.
    data: OrderedDict[MassBinKey, AccumulationHistoryData]

    folds: dict[int, PopulationAccumulationHistories] = field(factory=dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\t" + ",\n\t".join([f"{k}: {v}" for k, v in self.data.items()]) +
            f"\n)"
        )

    def __getitem__(self, key: MassBinKey) -> AccumulationHistoryData:
        return self.data[key]
    
    def __iter__(self) -> Iterator[MassBinKey]:
        return iter(self.data)
    
    def get(self, key: MassBinKey) -> AccumulationHistoryData | None:
        return self.data.get(key)
    
    @property
    def mass_bins(self) -> np.ndarray:
        """
        Return the mass-bin keys as a sorted numpy array.

        If the keys are numeric (ints / floats / numeric strings), they are
        cast to floats and sorted numerically. If casting fails (e.g. truly
        non-numeric labels), the keys are returned in their insertion order.
        """
        if not self.data:
            return np.array([])

        keys = list(self.data.keys())
        try:
            numeric = np.array([float(k) for k in keys], dtype=float)
            order = np.argsort(numeric)
            return numeric[order]
        except (TypeError, ValueError):
            # Heterogeneous or non-numeric keys; preserve insertion order
            return np.array(keys, dtype=object)
    
    def display(
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
        
        use_subset = populations_to_display is not None
        
        plt_args = {
            "with_fits": with_fits,
            "mf_type": mf_type,
            "normalize": normalize,
            "normalizing_scale_factor": normalizing_scale_factor,
            "return_fig": True,
            "show_top_log_y": show_top_log_y,
            "show_bottom_log_y": show_bottom_log_y,
            "display_def": False,
            "show_legend" : False,
            "legend_text_size": legend_text_size,
            "x_label_text_size": x_label_text_size,
            "top_main_y_label_text_size": top_main_y_label_text_size,
            "top_resid_y_label_text_size": top_resid_y_label_text_size,
            "bottom_main_y_label_text_size": bottom_main_y_label_text_size,
            "bottom_resid_y_label_text_size": bottom_resid_y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "top_main_y_tick_text_size": top_main_y_tick_text_size,
            "top_resid_y_tick_text_size": top_resid_y_tick_text_size,
            "bottom_main_y_tick_text_size": bottom_main_y_tick_text_size,
            "bottom_resid_y_tick_text_size": bottom_resid_y_tick_text_size,
        }

        axes_incomplete = any((
            (
                ax_main_top is None,
                ax_resid_top is None,
                ax_main_bottom is None,
                ax_resid_bottom is None
            ) if with_fits else (
                ax_main_top is None,
                ax_main_bottom is None
            )
        ))


        if with_fits and axes_incomplete:
            _, axes = plt.subplots(
                4, 1, figsize=(6, 12), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0, "height_ratios" : [3, 1, 3, 1]}
            )
            (ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom) = axes
        
        if not with_fits and axes_incomplete: 
            _, (ax_main_top,  ax_main_bottom) = plt.subplots(
                2, 1, figsize=(6, 8), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0}
            )
        
        colormap = plt.cm.get_cmap(color_palette, len(self.data))
        handles = []

        for i, (mass_bin, history_data) in enumerate(self.data.items()):
            
            if use_subset and mass_bin not in populations_to_display:
                continue

            color = colormap(i)
            handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color=color, 
                    label=rf"{mass_eqn}=$10^{{{mass_bin}}}$ $M_{{\odot}}$"
                )
            )

            if with_fits:

                (
                    ax_main_top, ax_resid_top, 
                    ax_main_bottom, ax_resid_bottom
                ) = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_resid_top=ax_resid_top,
                    ax_main_bottom=ax_main_bottom,
                    ax_resid_bottom=ax_resid_bottom,
                    plot_linestyle='-',
                    plot_marker_style='o',
                    color=color,
                    top_resid_min=top_resid_min,
                    top_resid_max=top_resid_max,
                    bottom_resid_min=bottom_resid_min,
                    bottom_resid_max=bottom_resid_max,
                    **plt_args
                )

            else:

                ax_main_top, _, ax_main_bottom, _ = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_main_bottom=ax_main_bottom,
                    color=color,
                    **plt_args
                )

        if len(handles) < 5:
            ax_main_top.legend(handles=handles, fontsize=legend_text_size)
        else:
            ax_main_top.legend(
                handles=handles, 
                fontsize=legend_text_size, 
                bbox_to_anchor=(top_legend_xloc, top_legend_yloc),
            )

        if return_fig:
            return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom
        


@define(slots=True)  # This will get moved to halo.py
class AbundanceAccumulationHistories(MassDefinitionDependentType):
    fof: AccumulationHistoryData | None = field(default=None)
    subfind: AccumulationHistoryData | None = field(default=None)
    crit200: AccumulationHistoryData | None = field(default=None)
    crit500: AccumulationHistoryData | None = field(default=None)
    mean200: AccumulationHistoryData | None = field(default=None)
    virial: AccumulationHistoryData | None = field(default=None)
    splashback: AccumulationHistoryData | None = field(default=None)
    asymptotic: AccumulationHistoryData | None = field(default=None)

    def __attr_post_init__(self) -> None:
        # Assert that all of the final_masses of non-None instances are the same
        final_masses = {
            getattr(self, mass_def).final_mass
            for mass_def in self.contained_keys
            if getattr(self, mass_def) is not None
        }

        assert len(final_masses) == 1, (
            "Final masses must be the same for all mass definitions"
        )

    def __repr__(self) -> str:
        return super().__repr__()
    
    def display(
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
            mass_defs_to_display: list[str] | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:
        
        use_subset = mass_defs_to_display is not None
        
        plt_args = {
            "with_fits": with_fits,
            "mf_type": mf_type,
            "normalize": normalize,
            "normalizing_scale_factor": normalizing_scale_factor,
            "return_fig": True,
            "show_top_log_y": show_top_log_y,
            "show_bottom_log_y": show_bottom_log_y,
            "display_def": False,
            "show_legend" : False,
            "legend_text_size": legend_text_size,
            "x_label_text_size": x_label_text_size,
            "top_main_y_label_text_size": top_main_y_label_text_size,
            "top_resid_y_label_text_size": top_resid_y_label_text_size,
            "bottom_main_y_label_text_size": bottom_main_y_label_text_size,
            "bottom_resid_y_label_text_size": bottom_resid_y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "top_main_y_tick_text_size": top_main_y_tick_text_size,
            "top_resid_y_tick_text_size": top_resid_y_tick_text_size,
            "bottom_main_y_tick_text_size": bottom_main_y_tick_text_size,
            "bottom_resid_y_tick_text_size": bottom_resid_y_tick_text_size,
        }

        axes_incomplete = any((
            (
                ax_main_top is None,
                ax_resid_top is None,
                ax_main_bottom is None,
                ax_resid_bottom is None
            ) if with_fits else (
                ax_main_top is None,
                ax_main_bottom is None
            )
        ))

        if with_fits and axes_incomplete:
            _, axes = plt.subplots(
                4, 1, figsize=(6, 12), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0, "height_ratios" : [3, 1, 3, 1]}
            )
            (ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom) = axes
        
        if not with_fits and axes_incomplete: 
            _, (ax_main_top,  ax_main_bottom) = plt.subplots(
                2, 1, figsize=(6, 8), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0}
            )
        
        colormap = plt.cm.get_cmap(color_palette, len(self.contained_keys))
        handles = []
        
        for i, mass_def in enumerate(self.contained_definitions):

            key = self.contained_keys[i]
            if use_subset and (key not in mass_defs_to_display):
                continue
            
            history_data = getattr(self, mass_def)
            color = colormap(i)
            linestyle = MASS_DEF_PLOT_LINES[mass_def]
            marker = MASS_DEF_MARKER_STYLES[mass_def]
            handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color=color, 
                    label=self.contained_mass_eqn[i]
                )
            )

            if with_fits:

                (
                    ax_main_top, ax_resid_top, 
                    ax_main_bottom, ax_resid_bottom
                ) = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_resid_top=ax_resid_top,
                    ax_main_bottom=ax_main_bottom,
                    ax_resid_bottom=ax_resid_bottom,
                    plot_linestyle=linestyle,
                    plot_marker_style=marker,
                    top_resid_min=top_resid_min,
                    top_resid_max=top_resid_max,
                    bottom_resid_min=bottom_resid_min,
                    bottom_resid_max=bottom_resid_max,
                    color=color,
                    **plt_args
                )

            else:

                ax_main_top, _, ax_main_bottom, _ = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_main_bottom=ax_main_bottom,
                    color=color,
                    plot_linestyle=linestyle,
                    **plt_args
                )

        ax_main_bottom.legend(handles=handles, fontsize=legend_text_size)

        if return_fig:
            return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom


@define(slots=True)
class PopulationAbundanceAccumulationHistories:
    data: OrderedDict[MassBinKey, AbundanceAccumulationHistories]

    folds: dict[int, PopulationAbundanceAccumulationHistories] = field(factory=dict)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\t" + ",\n\t".join([f"{k}: {v}" for k, v in self.data.items()]) +
            f"\n)"
        )

    def __getitem__(self, population: MassBinKey) -> AbundanceAccumulationHistories:
        return self.data[population]

    @property
    def populations(self) -> list[MassBinKey]:
        return list(self.data.keys()) 
    
    def get_mass_def_instance(self, mass_def_key: str) -> PopulationAccumulationHistories:
        return PopulationAccumulationHistories(
            data=OrderedDict(
                (mass_bin, histories[mass_def_key] )
                for mass_bin, histories in self.data.items()
            )
        )


    def get_fold_evolutions(self, mass_bin: MassBinKey, mass_def_key: str) -> dict[int, AccumulationHistory]: 

        fold_evolution = {} 
        for fold_id, fold_data in self.folds.items():
            if mass_bin not in fold_data.data:
                raise ValueError(f"Mass bin {mass_bin} not found in fold data.")
            fold_evolution[fold_id] = fold_data[mass_bin][mass_def_key]

        return fold_evolution

    def get_fold_abundance_accumulations(self, mass_bin: MassBinKey) -> dict[int, AbundanceAccumulationHistories]:

        fold_abundance_accumulations = {}
        for fold_id, fold_data in self.folds.items():
            if mass_bin not in fold_data.data:
                raise ValueError(f"Mass bin {mass_bin} not found in fold data.")
            fold_abundance_accumulations[fold_id] = fold_data[mass_bin]

        return fold_abundance_accumulations

    def join(
            self,
            other: PopulationAbundanceAccumulationHistories,
            new_key_type: str = "integer", # integer | str | float
            str_key_format: callable = lambda x: f"{x:.1f}"
        ) -> PopulationAbundanceAccumulationHistories:

        return join_pop_abundance_accumulation_histories(
            main=self,
            other=other,
            new_key_type=new_key_type,
            str_key_format=str_key_format
        )

    def save(self, filepath: Path) -> None:
        save_pop_abundance_accumulation_histories(self, filepath=filepath)

    @classmethod 
    def load(cls, filepath: Path) -> PopulationAbundanceAccumulationHistories:
        return load_pop_abundance_accumulation_histories(filepath=filepath)

    # @property
    # def fold_evolutions(self) -> dict[int, dict[int, dict]]


    # Method to display the accumulation histories for each population within
    # a particular mass definition. I.e., FOF at M=10^12, FOF at M=10^13, etc.
    def display_population_accumulation_histories(
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
            top_legend_xloc: float = 1.0,
            top_legend_yloc: float = 1.0,
            color_palette: str = "viridis",
            populations_to_display: list[int] | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:
        
        use_subset = populations_to_display is not None
        
        plt_args = {
            "with_fits": with_fits,
            "mf_type": mf_type,
            "normalize": normalize,
            "normalizing_scale_factor": normalizing_scale_factor,
            "return_fig": True,
            "show_top_log_y": show_top_log_y,
            "show_bottom_log_y": show_bottom_log_y,
            "display_def": False,
            "show_legend" : False,
            "legend_text_size": legend_text_size,
            "x_label_text_size": x_label_text_size,
            "top_main_y_label_text_size": top_main_y_label_text_size,
            "top_resid_y_label_text_size": top_resid_y_label_text_size,
            "bottom_main_y_label_text_size": bottom_main_y_label_text_size,
            "bottom_resid_y_label_text_size": bottom_resid_y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "top_main_y_tick_text_size": top_main_y_tick_text_size,
            "top_resid_y_tick_text_size": top_resid_y_tick_text_size,
            "bottom_main_y_tick_text_size": bottom_main_y_tick_text_size,
            "bottom_resid_y_tick_text_size": bottom_resid_y_tick_text_size,
        }

        if use_subset and (len(populations_to_display) == 1):
            pop_accum_instance = self.get_mass_def_instance(mass_def_key)
            histories = self[populations_to_display[0]]
            key_idx = histories.contained_keys.index(mass_def_key)
            pop_accum_instance.display(
                mass_def=histories.contained_definitions[key_idx],
                mass_eqn=histories.contained_mass_eqns[key_idx],
                **plt_args
            )

        axes_incomplete = any((
            (
                ax_main_top is None,
                ax_resid_top is None,
                ax_main_bottom is None,
                ax_resid_bottom is None
            ) if with_fits else (
                ax_main_top is None,
                ax_main_bottom is None
            )
        ))


        if with_fits and axes_incomplete:
            _, axes = plt.subplots(
                4, 1, figsize=(6, 12), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0, "height_ratios" : [3, 1, 3, 1]}
            )
            (ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom) = axes
        
        if not with_fits and axes_incomplete: 
            _, (ax_main_top,  ax_main_bottom) = plt.subplots(
                2, 1, figsize=(6, 8), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0}
            )
        
        colormap = plt.cm.get_cmap(color_palette, len(self.data))
        handles = []

        for i, (mass_bin, histories) in enumerate(self.data.items()):

            if use_subset and mass_bin not in populations_to_display: 
                continue

            if mass_def_key not in histories.contained_keys:
                raise ValueError(f"Invalid mass definition: {mass_def_key}")
        
            key_idx = histories.contained_keys.index(mass_def_key)
            eqn = histories.contained_mass_eqn[key_idx]
            mass_def = histories.contained_definitions[key_idx]

            history_data = getattr(histories, mass_def)

            color = colormap(i)
            handles.append(
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color=color, 
                    label=rf"{eqn}=$10^{{{mass_bin}}}$ $M_{{\odot}}$"
                )
            )

            if with_fits:

                (
                    ax_main_top, ax_resid_top, 
                    ax_main_bottom, ax_resid_bottom
                ) = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_resid_top=ax_resid_top,
                    ax_main_bottom=ax_main_bottom,
                    ax_resid_bottom=ax_resid_bottom,
                    plot_linestyle='-',
                    plot_marker_style='o',
                    color=color,
                    top_resid_min=top_resid_min,
                    top_resid_max=top_resid_max,
                    bottom_resid_min=bottom_resid_min,
                    bottom_resid_max=bottom_resid_max,
                    **plt_args
                )

            else:

                ax_main_top, _, ax_main_bottom, _ = history_data.display(
                    mass_def=mass_def,
                    ax_main_top=ax_main_top,
                    ax_main_bottom=ax_main_bottom,
                    color=color,
                    **plt_args
                )

        if len(handles) < 5:
            ax_main_top.legend(handles=handles, fontsize=legend_text_size)
        else:
            ax_main_top.legend(
                handles=handles, 
                fontsize=legend_text_size, 
                bbox_to_anchor=(top_legend_xloc, top_legend_yloc),
            )

        if return_fig:
            return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom

    @cached_property
    def contained_keys(self) -> list[str]:
        if not self.data:
            return []
        # Make a set of all mass definition keys present in all of the 
        # AbundanceAccumulationHistories instances
        keys_set = set()
        for histories in self.data.values():
            keys_set.update(histories.contained_keys)
        return sorted(keys_set)

    @cached_property
    def contained_definitions(self) -> list[str]:
        if not self.data:
            return []
        # Make a set of all mass definition keys present in all of the 
        # AbundanceAccumulationHistories instances
        keys_set = set()
        for histories in self.data.values():
            keys_set.update(histories.contained_definitions)
        return sorted(keys_set)
            

    def compute_accumulation_peak_and_freeze_times(
            self,
            abundance_idx: int = 4,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = True,
            method: str = "hybrid_window",
            verbose: bool = False,
            return_stats: bool = False,
        ) -> tuple[dict, dict] | dict[MassBinKey, dict[str, np.ndarray]]:


        time_results_dict, accum_stats = get_accumulation_peak_and_freeze_times(
            accumulation_histories=self,
            fold_accumulation_histories=self.folds,
            mass_defs=self.contained_keys,
            abundance_idx=abundance_idx,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            return_stats=True,
        )

        time_results = convert_time_results_to_arrays(time_results_dict)

        return (time_results, accum_stats) if return_stats else time_results


    # Use this to compare the accumulation histories of different populations
    # with respect to different mass definitions. (i.e., FOF at M=10^12, 10^13, 
    # etc. vs 200c at M=10^12, 10^13, etc. vs 200m at M=10^12, 10^13, etc. ...)
    def display(
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
            top_legend_xloc: float = 1.0,
            top_legend_yloc: float = 1.0,
            bottom_legend_xloc: float = 1.0,
            bottom_legend_yloc: float = -0.25,
            save_fig: bool = False,
        ) -> tuple[plt.Axes | None, plt.Axes | None, plt.Axes | None, plt.Axes | None] | None:

        use_mass_def_subset = mass_defs_to_display is not None
        use_population_subset = populations_to_display is not None
        
        plt_args = {
            "with_fits": with_fits,
            "mf_type": mf_type,
            "normalize": normalize,
            "normalizing_scale_factor": normalizing_scale_factor,
            "legend_text_size": legend_text_size,
            "x_label_text_size": x_label_text_size,
            "top_main_y_label_text_size": top_main_y_label_text_size,
            "top_resid_y_label_text_size": top_resid_y_label_text_size,
            "bottom_main_y_label_text_size": bottom_main_y_label_text_size,
            "bottom_resid_y_label_text_size": bottom_resid_y_label_text_size,
            "x_tick_text_size": x_tick_text_size,
            "top_main_y_tick_text_size": top_main_y_tick_text_size,
            "top_resid_y_tick_text_size": top_resid_y_tick_text_size,
            "bottom_main_y_tick_text_size": bottom_main_y_tick_text_size,
            "bottom_resid_y_tick_text_size": bottom_resid_y_tick_text_size,
        }

        if use_mass_def_subset and len(mass_defs_to_display) == 1:
            return self.display_population_accumulation_histories( 
                mass_def_key=mass_defs_to_display[0],
                return_fig=return_fig,
                top_legend_xloc=top_legend_xloc,
                top_legend_yloc=top_legend_yloc,
                **plt_args
            )

        plt_args |= {
            "return_fig": True,
            "display_def": False, 
            "show_legend" : False,
            "show_top_log_y": show_top_log_y,
            "show_bottom_log_y": show_bottom_log_y,
        }

        axes_incomplete = any((
            (
                ax_main_top is None,
                ax_resid_top is None,
                ax_main_bottom is None,
                ax_resid_bottom is None
            ) if with_fits else (
                ax_main_top is None,
                ax_main_bottom is None
            )
        ))

        if with_fits and axes_incomplete:
            _, axes = plt.subplots(
                4, 1, figsize=(6, 12), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0, "height_ratios" : [3, 1, 3, 1]}
            )
            (ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom) = axes
        
        if not with_fits and axes_incomplete: 
            _, (ax_main_top,  ax_main_bottom) = plt.subplots(
                2, 1, figsize=(6, 8), sharex=True, sharey=False,
                gridspec_kw={"hspace" : 0.0}
            )

        used_definitions = set()

        colormap = plt.cm.get_cmap(color_palette, len(self.data))

        for i, (mass_bin, histories) in enumerate(self.data.items()):

            if use_population_subset and (mass_bin not in populations_to_display):
                continue

            for mass_def_key in histories.contained_keys:

                if use_mass_def_subset and (mass_def_key not in mass_defs_to_display):
                    continue

                used_definitions.add(mass_def_key)

                key_idx = histories.contained_keys.index(mass_def_key)
                eqn = histories.contained_mass_eqn[key_idx]
                mass_def = histories.contained_definitions[key_idx]

                history_data = getattr(histories, mass_def)

                color = colormap(i)
                linestyle = MASS_DEF_PLOT_LINES[mass_def]

                if with_fits:

                    (
                        ax_main_top, ax_resid_top, 
                        ax_main_bottom, ax_resid_bottom
                    ) = history_data.display(
                        mass_def=mass_def,
                        ax_main_top=ax_main_top,
                        ax_resid_top=ax_resid_top,
                        ax_main_bottom=ax_main_bottom,
                        ax_resid_bottom=ax_resid_bottom,
                        plot_linestyle=linestyle,
                        plot_marker_style=MASS_DEF_MARKER_STYLES[mass_def],
                        color=color,
                        top_resid_min=top_resid_min,
                        top_resid_max=top_resid_max,
                        bottom_resid_min=bottom_resid_min,
                        bottom_resid_max=bottom_resid_max,
                        **plt_args
                    )

                else:

                    ax_main_top, _, ax_main_bottom, _ = history_data.display(
                        mass_def=mass_def,
                        ax_main_top=ax_main_top,
                        ax_main_bottom=ax_main_bottom,
                        color=color,
                        plot_linestyle=linestyle,
                        **plt_args    
                    )

        mass_handles = [
            plt.Line2D(
                xdata=[0], 
                ydata=[0], 
                color=colormap(i), 
                label=rf"$M_{{\Delta}}=10^{{{mass_bin}}} M_{{\odot}}$"
            )
            for i, mass_bin in enumerate(self.data)
        ]

        if len(mass_handles) < 5:
            ax_main_top.legend(handles=mass_handles, fontsize=legend_text_size)
        else:
            ax_main_top.legend(
                handles=mass_handles, 
                fontsize=legend_text_size, 
                bbox_to_anchor=(top_legend_xloc, top_legend_yloc),
            )

        # pdb.set_trace()

        heaviest_pop = self.data[max(self.data)]
        ax_main_bottom.legend(
            handles=[
                plt.Line2D(
                    xdata=[0], 
                    ydata=[0], 
                    color="k", 
                    label=mass_eqn,
                    linestyle=MASS_DEF_PLOT_LINES[
                        heaviest_pop.contained_definitions[i]
                    ],
                    marker=(
                        MASS_DEF_MARKER_STYLES[
                            heaviest_pop.contained_definitions[i]
                        ] if with_fits else None    
                    )   
                )
                for i, mass_eqn in enumerate(heaviest_pop.contained_mass_eqn)
                if heaviest_pop.contained_keys[i] in used_definitions
            ],
            fontsize=legend_text_size,
            bbox_to_anchor=(bottom_legend_xloc, bottom_legend_yloc),
            ncols=len(used_definitions) // 2,
        )

        if save_fig:
            plt.savefig("pop_abundance_accumulation.pdf", dpi=600, bbox_inches="tight")


        if return_fig:
            return ax_main_top, ax_resid_top, ax_main_bottom, ax_resid_bottom


def _get_accum_bin_samples(
        fold_accum: dict[int, PopulationAbundanceAccumulationHistories],
        mass_defs: list[str],
        abundance_idx: int,
        verbose: bool = False,
    ) -> dict[str, dict[MassBinKey, list[np.ndarray]]]:
    # ------------------------------------------------------
    # 1) Build per-mass-bin samples across folds
    # ------------------------------------------------------
    # mass_def -> mass_bin -> list of np.ndarray samples
    accum_mass_bin_samples = defaultdict(lambda: defaultdict(list))

    for mass_def in mass_defs:
        for fold_id, hmf_pop_accum in fold_accum.items():
            for mass_bin, pop_accum in hmf_pop_accum.data.items():
                # pdb.set_trace()
                try:
                    pop_accum_mdef = pop_accum[mass_def]
                    accum_arr = pop_accum_mdef.data.to_numpy
                except (KeyError, ValueError):
                    if verbose:
                        print(
                            f"Skipping fold {fold_id}, mass bin {mass_bin} "
                            f"for mass def {mass_def}"
                        )
                    continue

                # Normalize by final-time value of the chosen abundance track
                track = accum_arr[abundance_idx]
                if not np.isfinite(track[-1]) or track[-1] == 0:
                    if verbose:
                        print(
                            f"Skipping fold {fold_id}, mass bin {mass_bin} "
                            f"for mass def {mass_def} due to non-finite "
                            f"final value"
                        )
                    continue

                accum_mass_bin_samples[mass_def][mass_bin].append(track / track[-1])

    return accum_mass_bin_samples


# ------------------------------------------------------
# 2) Compute log-mean and std for each mass bin
# ------------------------------------------------------
def _get_accum_bin_stats(
        accum_mass_bin_samples: dict[str, dict[MassBinKey, list[np.ndarray]]]
    ) -> dict[str, dict[MassBinKey, dict[str, np.ndarray]]]:
    """
    Compute log-mean and log-std statistics for accumulation histories.

    Parameters
    ----------
    accum_mass_bin_samples : dict
        mass_def -> mass_bin -> list of normalized accumulation tracks

    Returns
    -------
    dict
        mass_def -> mass_bin -> {"mean": np.ndarray, "std": np.ndarray}
    """
    accum_stats = defaultdict(lambda: defaultdict(dict))

    for mass_def, bins in accum_mass_bin_samples.items():
        for mass_bin, samples in bins.items():

            if not samples: continue

            stacked = np.log10(np.stack(samples))
            accum_stats[mass_def][mass_bin] = {
                "mean": 10.0 ** np.nanmean(stacked, axis=0),
                "std":  10.0 ** np.nanstd(stacked, axis=0),
            }

    return accum_stats

def _fill_peak_and_freeze_times_dict(
        accum_samples: dict[str, dict[MassBinKey, dict[str, np.ndarray]]],
        accum_histories: PopulationAbundanceAccumulationHistories,
        fold_accum: dict[int, PopulationAbundanceAccumulationHistories],
        mass_defs: list[str],
        abundance_idx: int = 4,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = True,
        method: str = "hybrid_window",
        verbose: bool = False,
    ) -> dict:
    """
    Compute peak and freeze-out times for each mass definition and mass bin.

    This function works identically with or without fold information.
    """

    # mass_def -> mass_bin -> results dict
    time_results = defaultdict(lambda: defaultdict(dict))

    has_folds = bool(fold_accum)
    any_fold = next(iter(fold_accum.values())) if has_folds else None

    for mass_def in mass_defs:

        # Ground truth: which mass bins exist
        for mass_bin_key in accum_histories.data.keys():

            # Optional fold membership check
            if has_folds and mass_bin_key not in any_fold.data:
                if verbose:
                    print(
                        f"Skipping mass bin {mass_bin_key} for {mass_def} "
                        "(not present in folds)"
                    )
                continue

            # --- Full evolution ---
            try:
                array = accum_histories[mass_bin_key][mass_def].data.to_numpy
            except (KeyError, AttributeError):
                continue

            scale_factors = array[0, :]
            track = array[abundance_idx, :]

            if not np.isfinite(track[-1]) or track[-1] == 0.0:
                if verbose:
                    print(
                        f"Skipping mass bin {mass_bin_key} for {mass_def} "
                        "due to non-finite final value"
                    )
                continue

            full_evolution = track / track[-1]

            # --- Fold evolutions (optional) ---
            fold_id_to_evolution: dict[int, np.ndarray] = {}
            if has_folds:
                for f_id, fold_data in fold_accum.items():
                    if mass_bin_key not in fold_data.data:
                        continue

                    try:
                        fold_array = fold_data[mass_bin_key][mass_def].data.to_numpy
                    except (KeyError, AttributeError):
                        continue

                    fold_track = fold_array[abundance_idx, :]
                    if not np.isfinite(fold_track[-1]) or fold_track[-1] == 0.0:
                        continue

                    fold_id_to_evolution[f_id] = fold_track / fold_track[-1]

            fold_id_to_evolution_arg = fold_id_to_evolution or None

            # --- Compute times ---
            time_results[mass_def][mass_bin_key] = compute_peak_and_freeze_times(
                scale_factors=scale_factors,
                full_evolution=full_evolution,
                fold_id_to_evolution=fold_id_to_evolution_arg,
                tol=tol,
                rel_tol_no_folds=rel_tol_no_folds,
                n_hi_res=n_hi_res,
                return_full_grid=return_full_grid,
                method=method,
            )

            if verbose:
                res = time_results[mass_def][mass_bin_key]
                print(
                    f"Mass def {mass_def} : Mass bin {mass_bin_key} -> "
                    f"a_peak={res.get('a_peak')}, "
                    f"a_frz={res.get('a_frz')} "
                    f"(method={method})"
                )

    return time_results


def get_accumulation_peak_and_freeze_times(
        accumulation_histories: PopulationAbundanceAccumulationHistories,
        fold_accumulation_histories: dict[int, PopulationAbundanceAccumulationHistories],
        mass_defs: list[str] | str,
        abundance_idx: int = 4,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = True,
        method: str = "hybrid_window",
        verbose: bool = False,
        return_stats: bool = False,
    ) -> dict | tuple[dict, dict]:

    if isinstance(mass_defs, str):
        mass_defs = [mass_defs]

    accum_mass_bin_samples = _get_accum_bin_samples(
        fold_accum=fold_accumulation_histories,
        mass_defs=mass_defs,
        abundance_idx=abundance_idx,
        verbose=verbose,
    )

    accum_mass_bin_stats = _get_accum_bin_stats(accum_mass_bin_samples)

    time_results = _fill_peak_and_freeze_times_dict(
        accum_samples=accum_mass_bin_stats,
        accum_histories=accumulation_histories,
        fold_accum=fold_accumulation_histories,
        mass_defs=mass_defs,
        abundance_idx=abundance_idx,
        verbose=verbose,
        tol=tol,
        rel_tol_no_folds=rel_tol_no_folds,
        n_hi_res=n_hi_res,
        return_full_grid=return_full_grid,
        method=method,
    )

    return (time_results, accum_mass_bin_stats) if return_stats else time_results


def convert_time_results_to_arrays(time_results: dict) -> dict:
    """
    Convert dictionary:
        dict[str, dict[str|int, dict[str, float]]]
    into:
        dict[str, dict[str, np.ndarray]]
    
    Example: time_results["fof"]["8.7"]["a_frz"] -> float
    
    Output:
      time_results_array["fof"]["masses"] -> np.ndarray
      time_results_array["fof"]["a_frz"]  -> np.ndarray
      time_results_array["fof"]["a_peak"] -> np.ndarray
      ...
    """

    final = {}

    for mass_def, mass_dict in time_results.items():
        # Sort mass keys numerically regardless of being str/int
        mass_keys = sorted(mass_dict.keys(), key=lambda x: float(x))

        # Identify all inner fields by looking at the first valid entry
        example_mass = next(iter(mass_keys))
        example_fields = mass_dict[example_mass].keys()

        # Build container for this mass definition
        out = {"masses": np.array([10.0 ** float(m) for m in mass_keys])}

        # For each field: collect all values in mass-key order
        for field in example_fields:
            arr = []
            for m in mass_keys:
                val = mass_dict[m].get(field, np.nan)  # Handle None or missing fields
                arr.append(val if val is not None else np.nan)
            out[field] = np.array(arr)

        final[mass_def] = out

    return final

def _merge_data_arrays(
        main_data: np.ndarray | None, 
        other_data: np.ndarray | None
    ) -> np.ndarray | None:
    """Merge data arrays from two sources, handling None values."""
    if main_data is not None and other_data is not None:
        return np.concatenate([main_data, other_data], axis=0)
    elif main_data is not None:
        return main_data
    elif other_data is not None:
        return other_data
    else:
        raise ValueError("Key missing from both histories")


def _merge_fold_data_for_key(
        main_folds: dict[int, dict[MassBinKey, np.ndarray]] | None, 
        other_folds: dict[int, dict[MassBinKey, np.ndarray]] | None, 
        fold_id: int, key: MassBinKey
    ) -> np.ndarray | None:
    """Merge fold data for a specific fold_id and key."""
    main_fold_value = main_folds[fold_id].get(key) if main_folds else None
    other_fold_value = other_folds[fold_id].get(key) if other_folds else None
    
    # Skip if both are None
    if main_fold_value is None and other_fold_value is None:
        return None
    
    # Return the merge result
    return _merge_data_arrays(main_fold_value, other_fold_value)


def _get_all_fold_ids(
        main: PopulationAbundanceAccumulationHistories, 
        other: PopulationAbundanceAccumulationHistories
    ) -> set[int]:
    """Get all unique fold IDs from both histories."""
    fold_ids = set()
    if main.folds is not None:
        fold_ids.update(main.folds.keys())
    if other.folds is not None:
        fold_ids.update(other.folds.keys())
    return fold_ids


def _merge_folds_for_all_keys(
        # main: PopulationAbundanceAccumulationHistories, 
        other: PopulationAbundanceAccumulationHistories, 
        key_mapping: dict[MassBinKey, MassBinKey]
    ) -> dict[int, dict[MassBinKey, np.ndarray]]:
    """
    Merge fold data for all keys.
    
    Parameters
    ----------
    main, other : PopulationAbundanceAccumulationHistories
        The two histories to merge
    key_mapping : dict
        Maps original keys to new keys
        
    Returns
    -------
    dict
        Merged fold data with structure {fold_id: {new_key: data}}
    """
    # Handle case where only one has folds
    if main.folds is None:
        return {} if other.folds is None else other.folds
    if other.folds is None:
        return main.folds
    
    # Both have folds - merge them
    new_fold_data = defaultdict(dict)
    all_fold_ids = _get_all_fold_ids(main, other)
    
    for key, new_key in key_mapping.items():
        for fold_id in all_fold_ids:
            merged_value = _merge_fold_data_for_key(main.folds, other.folds, fold_id, key)
            if merged_value is not None:
                new_fold_data[fold_id][new_key] = merged_value
    
    return new_fold_data


def join_pop_abundance_accumulation_histories(
        main: PopulationAbundanceAccumulationHistories,
        other: PopulationAbundanceAccumulationHistories,
        new_key_type: str = "integer",  # integer | string | float
        str_key_format: callable = lambda x: f"{x:.1f}"
    ) -> PopulationAbundanceAccumulationHistories:
    
    # Get all unique keys and build mapping to new keys
    all_keys = set(main.data.keys()) | set(other.data.keys())
    key_mapping = {
        key: get_new_key(key=key, new_key_type=new_key_type, str_key_format=str_key_format)
        for key in all_keys
    }
    
    # Merge main data
    new_data = {}
    for key, new_key in key_mapping.items():
        main_data = main.data.get(key)
        other_data = other.data.get(key)
        new_data[new_key] = _merge_data_arrays(main_data, other_data)
    
    # Merge fold data
    new_fold_data = _merge_folds_for_all_keys(main, other, key_mapping)
    
    # Return with new_data and each fold's data sorted by new keys
    return PopulationAbundanceAccumulationHistories(
        data=OrderedDict(sorted(new_data.items(), key=lambda item: float(item[0]))),
        folds={
            fold_id: PopulationAbundanceAccumulationHistories(
                data=OrderedDict(sorted(fold_data.items(), key=lambda item: float(item[0])))
            ) 
            for fold_id, fold_data in new_fold_data.items() 
            if fold_data
        }
    )    


# -----------------------------------------------------------------------------
# HDF5 IO helpers for PopulationAbundanceAccumulationHistories
# -----------------------------------------------------------------------------

_H5_SCHEMA_VERSION_POP_ABUND_ACCUM = 1


def _infer_moments_from_histories(histories: PopulationAbundanceAccumulationHistories) -> MomentsInTime | None:
    """Best-effort extraction of MomentsInTime from a histories container."""
    if hasattr(histories, "moments") and isinstance(getattr(histories, "moments"), MomentsInTime):
        return getattr(histories, "moments")

    data_map = getattr(histories, "data", None)
    if isinstance(data_map, dict) and len(data_map) > 0:
        for _, v in data_map.items():
            for md_name in ("fof", "subfind", "crit200", "crit500", "mean200", "virial", "splashback"):
                if not hasattr(v, md_name):
                    continue
                ahd = getattr(v, md_name)
                if ahd is None:
                    continue
                for h in (getattr(ahd, "data", None), getattr(ahd, "fitted", None)):
                    if h is not None and hasattr(h, "moments") and isinstance(h.moments, MomentsInTime):
                        return h.moments

    return None


def _encode_mass_bin_key(key: MassBinKey) -> tuple[str, str]:
    """Return (key_type, key_string) for stable round-tripping."""
    if isinstance(key, bool):
        return ("str", str(key))
    if isinstance(key, int):
        return ("int", str(int(key)))
    if isinstance(key, float):
        return ("float", repr(float(key)))
    return ("str", str(key))


def _decode_mass_bin_key(key_type: str, key_string: str) -> MassBinKey:
    if key_type == "int":
        return int(key_string)
    return float(key_string) if key_type == "float" else key_string


def _write_accumulation_history(hg: h5py.Group, hist: AccumulationHistory) -> None:
    """Write an AccumulationHistory (and its folds) into an HDF5 group."""
    hg.attrs["normalizing_scale_factor"] = float(getattr(hist, "normalizing_scale_factor", 10.0))

    snap_ids = np.asarray(sorted(hist.data.keys()), dtype=int)
    vals = np.empty((6, snap_ids.size), dtype=float)

    for j, sid in enumerate(snap_ids.tolist()):
        v = hist.data[int(sid)]
        vals[0, j] = float(v.peak_height)
        vals[1, j] = float(v.number_density)
        vals[2, j] = float(v.differential)
        vals[3, j] = float(v.normalized)
        vals[4, j] = float(v.cumulative)
        vals[5, j] = float(v.multiplicity)

    for name in ("snapshot_ids", "values"):
        if name in hg:
            del hg[name]

    hg.create_dataset("snapshot_ids", data=snap_ids)
    hg.create_dataset("values", data=vals)
    hg.attrs["values_order"] = np.array(
        ["peak_height", "number_density", "differential", "normalized", "cumulative", "multiplicity"],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    folds = getattr(hist, "folds", {})
    if isinstance(folds, dict) and len(folds) > 0:
        fg = hg.require_group("folds")
        for k in list(fg.keys()):
            del fg[k]
        for fold_id, fold_hist in folds.items():
            fgid = fg.create_group(str(int(fold_id)))
            _write_accumulation_history(fgid, fold_hist)


def _read_accumulation_history(hg: h5py.Group, moments: MomentsInTime) -> AccumulationHistory:
    snap_ids = np.asarray(hg["snapshot_ids"][()], dtype=int)
    vals = np.asarray(hg["values"][()], dtype=float)

    data: OrderedDict[int, AbundanceValue] = OrderedDict()
    for j, sid in enumerate(snap_ids.tolist()):
        data[int(sid)] = AbundanceValue(
            peak_height=float(vals[0, j]),
            number_density=float(vals[1, j]),
            differential=float(vals[2, j]),
            normalized=float(vals[3, j]),
            cumulative=float(vals[4, j]),
            multiplicity=float(vals[5, j]),
        )

    hist = AccumulationHistory(
        data=data,
        moments=moments,
        normalizing_scale_factor=float(hg.attrs.get("normalizing_scale_factor", 10.0)),
    )

    if "folds" in hg:
        folds: dict[int, AccumulationHistory] = {}
        for fold_id_str, fg in hg["folds"].items():
            try:
                fold_id = int(fold_id_str)
            except ValueError:
                continue
            folds[fold_id] = _read_accumulation_history(fg, moments)
        hist.folds = folds

    return hist


def _write_accumulation_history_data(hg: h5py.Group, ahd: AccumulationHistoryData) -> None:
    hg.attrs["final_mass"] = float(ahd.final_mass)

    if "data" in hg:
        del hg["data"]
    if ahd.data is not None:
        dg = hg.create_group("data")
        _write_accumulation_history(dg, ahd.data)

    if "fitted" in hg:
        del hg["fitted"]
    if ahd.fitted is not None:
        fg = hg.create_group("fitted")
        _write_accumulation_history(fg, ahd.fitted)


def _read_accumulation_history_data(hg: h5py.Group, moments: MomentsInTime) -> AccumulationHistoryData:
    final_mass = float(hg.attrs.get("final_mass", np.nan))
    data = _read_accumulation_history(hg["data"], moments) if "data" in hg else None
    fitted = _read_accumulation_history(hg["fitted"], moments) if "fitted" in hg else None
    return AccumulationHistoryData(final_mass=final_mass, data=data, fitted=fitted)


def save_pop_abundance_accumulation_histories(
    histories: "PopulationAbundanceAccumulationHistories",
    filepath: Path,
) -> None:
    """Save PopulationAbundanceAccumulationHistories to HDF5, including MomentsInTime."""
    filepath = Path(filepath)

    moments = _infer_moments_from_histories(histories)
    if moments is None:
        raise ValueError(
            "Could not infer MomentsInTime from the provided histories. "
            "Ensure at least one AccumulationHistory exists and has a .moments attribute."
        )

    with h5py.File(filepath, "w") as f:
        root = f.create_group("population_abundance_accumulation_histories")
        root.attrs["class"] = "PopulationAbundanceAccumulationHistories"
        root.attrs["schema_version"] = _H5_SCHEMA_VERSION_POP_ABUND_ACCUM

        mg = root.create_group("moments_in_time")
        mg.attrs["class"] = "MomentsInTime"
        mg.attrs["schema_version"] = 1
        moments.to_hdf5(mg)

        data_map = getattr(histories, "data", None)
        if not isinstance(data_map, dict):
            raise ValueError("histories.data must be a mapping")

        dg = root.create_group("data")
        for i, (mass_bin, abund_histories) in enumerate(data_map.items()):
            mbg = dg.create_group(f"mass_bin_{i:04d}")
            ktype, kstr = _encode_mass_bin_key(mass_bin)
            mbg.attrs["mass_bin_key_type"] = ktype
            mbg.attrs["mass_bin_key"] = kstr

            for md_name in ("fof", "subfind", "crit200", "crit500", "mean200", "virial", "splashback"):
                if not hasattr(abund_histories, md_name):
                    continue
                ahd = getattr(abund_histories, md_name)
                if ahd is None:
                    continue
                mdg = mbg.create_group(md_name)
                _write_accumulation_history_data(mdg, ahd)

        folds = getattr(histories, "folds", None)
        if isinstance(folds, dict) and len(folds) > 0:
            fg = root.create_group("folds")
            for fold_id, fold_histories in folds.items():
                fold_grp = fg.create_group(str(int(fold_id)))
                f_dg = fold_grp.create_group("data")
                f_map = getattr(fold_histories, "data", fold_histories)
                if not isinstance(f_map, dict):
                    continue
                for i, (mass_bin, abund_histories) in enumerate(f_map.items()):
                    mbg = f_dg.create_group(f"mass_bin_{i:04d}")
                    ktype, kstr = _encode_mass_bin_key(mass_bin)
                    mbg.attrs["mass_bin_key_type"] = ktype
                    mbg.attrs["mass_bin_key"] = kstr
                    for md_name in ("fof", "subfind", "crit200", "crit500", "mean200", "virial", "splashback"):
                        if not hasattr(abund_histories, md_name):
                            continue
                        ahd = getattr(abund_histories, md_name)
                        if ahd is None:
                            continue
                        mdg = mbg.create_group(md_name)
                        _write_accumulation_history_data(mdg, ahd)


def load_pop_abundance_accumulation_histories(
        filepath: Path,
    ) -> PopulationAbundanceAccumulationHistories:
    """Load PopulationAbundanceAccumulationHistories from HDF5 (including MomentsInTime)."""
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        if "population_abundance_accumulation_histories" not in f:
            raise KeyError(
                "Group 'population_abundance_accumulation_histories' not found. "
                "Did you save with save_pop_abundance_accumulation_histories(...)?"
            )
        root = f["population_abundance_accumulation_histories"]

        if "moments_in_time" not in root:
            raise KeyError("Missing 'moments_in_time' group in file")
        moments = MomentsInTime.from_hdf5(root["moments_in_time"])

        dg = root.get("data", None)
        if dg is None:
            raise KeyError("Missing 'data' group in file")

        data = OrderedDict()
        for mb_name in sorted(dg.keys()):
            mbg = dg[mb_name]
            ktype = str(mbg.attrs.get("mass_bin_key_type", "str"))
            kstr = str(mbg.attrs.get("mass_bin_key", mb_name))
            mass_bin = _decode_mass_bin_key(ktype, kstr)

            md_kwargs = {}
            for md_name, mdg in mbg.items():
                md_kwargs[md_name] = _read_accumulation_history_data(mdg, moments)

            data[mass_bin] = AbundanceAccumulationHistories(**md_kwargs)

        folds_out = {}
        if "folds" in root:
            for fold_id_str, fold_grp in root["folds"].items():
                try:
                    fold_id = int(fold_id_str)
                except ValueError:
                    continue

                fdg = fold_grp.get("data", None)
                if fdg is None:
                    continue

                fold_data: OrderedDict[MassBinKey, Any] = OrderedDict()
                for mb_name in sorted(fdg.keys()):
                    mbg = fdg[mb_name]
                    ktype = str(mbg.attrs.get("mass_bin_key_type", "str"))
                    kstr = str(mbg.attrs.get("mass_bin_key", mb_name))
                    mass_bin = _decode_mass_bin_key(ktype, kstr)

                    md_kwargs: dict[str, Any] = {}
                    for md_name, mdg in mbg.items():
                        md_kwargs[md_name] = _read_accumulation_history_data(mdg, moments)
                    fold_data[mass_bin] = AbundanceAccumulationHistories(**md_kwargs)

                folds_out[fold_id] = PopulationAbundanceAccumulationHistories(data=fold_data, folds={})

        return PopulationAbundanceAccumulationHistories(data=data, folds=folds_out)
