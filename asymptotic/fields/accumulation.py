from __future__ import annotations

import h5py
import itertools
import numpy as np, pdb 
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Iterator
from attrs import define, field
from functools import cached_property
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from collections import OrderedDict, defaultdict


from ..simulation.evo import EvolutionData
from ..simulation.moments import MomentsInTime
from ..utils.freeze_out import compute_peak_and_freeze_times


KeyType = str | int | float | np.integer | np.floating

@define(slots=True)
class AccumulationHistory:
    time_steps: np.ndarray
    accumulation: np.ndarray
    weighted: np.ndarray
    normalized: np.ndarray
    rate: np.ndarray
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @classmethod
    def null_initialize(cls) -> AccumulationHistory:
        return cls(
            time_steps=np.array([]),
            accumulation=np.array([]),
            weighted=np.array([]),
            normalized=np.array([]),
            rate=np.array([]),
            in_comoving=False,
            time_metric="scale_factor"
        )
    
    @property
    def is_null(self) -> bool:
        return self.time_steps.size == 0
    
    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:
        
        # Handle null/empty accumulation history
        if self.is_null:
            return {
                "a_peak": np.nan,
                "a_frz": np.nan,
                "a_frz_err_lo": np.nan,
                "a_frz_err_hi": np.nan,
            }

        return compute_peak_and_freeze_times(
            scale_factors=self.time_steps,
            full_evolution=self.normalized,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )

    

@define(slots=True)
class AccumulationHistories:
    accumulations: OrderedDict[KeyType, AccumulationHistory]
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")
    is_linear: bool = field(default=False)

    def __getitem__(self, key: KeyType) -> AccumulationHistory:
        return self.accumulations[key]

    @classmethod
    def null_initialize(cls) -> AccumulationHistories:
        return cls(
            accumulations=OrderedDict(), 
            in_comoving=False, 
            time_metric="scale_factor"
        )
    
    @property
    def is_null(self) -> bool:
        return len(self.accumulations) == 0
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            for_wavenumber: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        times_dict = {
            key: accumulation.get_peak_and_freeze_out_times(
                tol=tol,
                rel_tol_no_folds=rel_tol_no_folds,
                n_hi_res=n_hi_res,
                return_full_grid=return_full_grid,
                method=method,
                verbose=verbose,
                **kwargs,
            )
            for key, accumulation in self.accumulations.items()
        }

        return convert_time_results_to_array(
            time_results=times_dict, 
            result_sep_key="wavenumber" if for_wavenumber else "separation"
        )
    
    


@define(slots=True)
class MatterPowerSpectrumAccumulation:
    linear: AccumulationHistory
    nonlinear: AccumulationHistory
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @classmethod
    def from_matter_ps_evo_outputs(
            cls, 
            evo_array: np.ndarray,
            evo_rate: np.ndarray,
            evo_variance: np.ndarray,
            in_comoving: bool = False,
            time_metric: str = "scale_factor",
        ) -> MatterPowerSpectrumAccumulation:
            
        a_f = evo_array[-1, 0]
        lin_Pk_final, nonlin_Pk_final = evo_array[-1, 1], evo_array[-1, 2]

        lin_scaling = lin_Pk_final
        nonlin_scaling = (
            nonlin_Pk_final 
            if in_comoving else
            nonlin_Pk_final * (evo_array[:, 0] / a_f) ** 3
        )

        linear = AccumulationHistory(
            time_steps=evo_array[:, 0],
            accumulation=evo_array[:, 1],
            weighted=evo_variance[:, 1],
            normalized=evo_array[:, 1] / lin_scaling,
            rate=evo_rate[:, 1],
            in_comoving=in_comoving,
            time_metric=time_metric
        )
        nonlinear = AccumulationHistory(
            time_steps=evo_array[:, 0],
            accumulation=evo_array[:, 2],
            weighted=evo_variance[:, 2],
            normalized=evo_array[:, 2] / nonlin_scaling,
            rate=evo_rate[:, 2],
            in_comoving=in_comoving,
            time_metric=time_metric
        )
        return cls(
            linear=linear, 
            nonlinear=nonlinear,
            in_comoving=in_comoving,
            time_metric=time_metric
        )
    
    @classmethod
    def null_initialize(cls) -> MatterPowerSpectrumAccumulation:
        return cls(
            linear=AccumulationHistory.null_initialize(),
            nonlinear=AccumulationHistory.null_initialize(),
            in_comoving=False,
            time_metric="scale_factor"
        ) 
    
    @property
    def is_null(self) -> bool:
        return self.linear.is_null and self.nonlinear.is_null
    
    @property
    def linear_variance(self) -> np.ndarray:
        return self.linear.weighted
    
    @property
    def nonlinear_variance(self) -> np.ndarray:
        return self.nonlinear.weighted
    
    @cached_property
    def boost(self) -> AccumulationHistory:
        """
        Compute the boost factor (nonlinear / linear) as an AccumulationHistory.
        
        The boost measures the departure from linear theory at each time step.
        For the rate, we use a smoothed derivative via Savitzky-Golay filter
        to reduce noise from numerical differentiation.
        
        If linear and nonlinear have different time steps, interpolate to a 
        common grid (the intersection of valid time ranges).
        """
        lin_times = self.linear.time_steps
        nonlin_times = self.nonlinear.time_steps
        
        # Check for empty arrays - return null if either is empty
        if len(lin_times) == 0 or len(nonlin_times) == 0:
            return AccumulationHistory.null_initialize()
        
        # Check if time steps match
        if len(lin_times) == len(nonlin_times) and np.allclose(lin_times, nonlin_times):
            # Time steps match - use directly
            common_times = lin_times
            lin_acc = self.linear.accumulation
            lin_norm = self.linear.normalized
            lin_weighted = self.linear.weighted
            nonlin_acc = self.nonlinear.accumulation
            nonlin_norm = self.nonlinear.normalized
            nonlin_weighted = self.nonlinear.weighted
        else:
            # Time steps differ - interpolate to common grid
            # Use the intersection of time ranges
            t_min = max(lin_times.min(), nonlin_times.min())
            t_max = min(lin_times.max(), nonlin_times.max())
            
            if t_min >= t_max:
                # No overlap - return null
                return AccumulationHistory.null_initialize()
            
            # Use the denser sampling as the common grid
            n_common = max(len(lin_times), len(nonlin_times))
            common_times = np.linspace(t_min, t_max, n_common)
            
            # Interpolate linear values using interp1d
            lin_acc_interp = interp1d(lin_times, self.linear.accumulation, kind='linear', fill_value='extrapolate')
            lin_norm_interp = interp1d(lin_times, self.linear.normalized, kind='linear', fill_value='extrapolate')
            lin_weighted_interp = interp1d(lin_times, self.linear.weighted, kind='linear', fill_value='extrapolate')
            
            lin_acc = lin_acc_interp(common_times)
            lin_norm = lin_norm_interp(common_times)
            lin_weighted = lin_weighted_interp(common_times)
            
            # Interpolate nonlinear values using interp1d
            nonlin_acc_interp = interp1d(nonlin_times, self.nonlinear.accumulation, kind='linear', fill_value='extrapolate')
            nonlin_norm_interp = interp1d(nonlin_times, self.nonlinear.normalized, kind='linear', fill_value='extrapolate')
            nonlin_weighted_interp = interp1d(nonlin_times, self.nonlinear.weighted, kind='linear', fill_value='extrapolate')
            
            nonlin_acc = nonlin_acc_interp(common_times)
            nonlin_norm = nonlin_norm_interp(common_times)
            nonlin_weighted = nonlin_weighted_interp(common_times)
        
        # Avoid division by zero
        lin_acc_safe = np.where(lin_acc != 0, lin_acc, np.nan)
        lin_norm_safe = np.where(lin_norm != 0, lin_norm, np.nan)
        lin_weighted_safe = np.where(lin_weighted != 0, lin_weighted, np.nan)
        
        boost_accumulation = nonlin_acc / lin_acc_safe
        boost_normalized = nonlin_norm / lin_norm_safe
        boost_weighted = nonlin_weighted / lin_weighted_safe
        
        # Compute smoothed derivative using Savitzky-Golay filter
        # Window length must be odd and <= data length
        n_pts = len(common_times)
        if n_pts >= 5:
            # Use window of ~10% of data or 5 points, whichever is larger
            window_length = min(max(5, n_pts // 10 | 1), n_pts)  # Ensure odd
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(3, window_length - 1)
            
            # Savitzky-Golay derivative (deriv=1) with delta = mean spacing
            delta = np.mean(np.diff(common_times)).astype(float)
            boost_rate = savgol_filter(
                boost_accumulation, 
                window_length=window_length, 
                polyorder=polyorder, 
                deriv=1, 
                delta=delta
            )
        else:
            # Fall back to simple gradient for very short arrays
            boost_rate = np.gradient(boost_accumulation, common_times)
        
        return AccumulationHistory(
            time_steps=common_times,
            accumulation=boost_accumulation,
            weighted=boost_weighted,
            normalized=boost_normalized,
            rate=boost_rate,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    

    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_boost_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.boost.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            include_boost: bool = True,
            **extra_kwargs,
        ) -> dict[str, dict[str, float | np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose,
            **extra_kwargs,
        }

        results = {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }
        
        if include_boost:
            results["boost"] = self.get_boost_peak_and_freeze_out_times(**kwargs)
        
        return results
    
@define(slots=True)
class MatterPowerSpectraAccumulation:
    accumulations: OrderedDict[KeyType, MatterPowerSpectrumAccumulation]
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    def __getitem__(self, key: KeyType) -> MatterPowerSpectrumAccumulation:
        return self.accumulations[key]
    
    def __iter__(self) -> Iterator[KeyType]:
        return iter(self.accumulations)
    
    def __len__(self) -> int:
        return len(self.accumulations)
    
    @classmethod
    def null_initialize(cls) -> MatterPowerSpectraAccumulation:
        return cls(
            accumulations=OrderedDict(),
            in_comoving=False,
            time_metric="scale_factor"
        )
    
    @property
    def is_null(self) -> bool:
        return len(self.accumulations) == 0

    @property
    def linear(self) -> AccumulationHistories:
        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value.linear) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric,
            is_linear=True
        )
    
    @property
    def nonlinear(self) -> AccumulationHistories:
        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value.nonlinear) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric,
            is_linear=False
        )
    
    @property
    def boost(self) -> AccumulationHistories:
        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value.boost) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric,
            is_linear=False
        )
    
    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=True,
            **kwargs,
        )
    
    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=True,
            **kwargs,
        )
    
    def get_boost_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.boost.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=True,
            **kwargs,
        )
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            include_boost: bool = True,
            **extra_kwargs,
        ) -> dict[str, dict[str, np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose,
            **extra_kwargs,
        }

        results = {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }
        
        if include_boost:
            results["boost"] = self.get_boost_peak_and_freeze_out_times(**kwargs)
        
        return results
    



@define(slots=True)
class TwoPointCorrelationAccumulation:
    main: AccumulationHistory
    folds: dict[int, AccumulationHistory] = field(factory=dict)
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @classmethod
    def null_initialize(cls) -> TwoPointCorrelationAccumulation:
        return cls(
            main=AccumulationHistory.null_initialize(),
            folds={},
            in_comoving=False,
            time_metric="scale_factor"
        )
    
    @property
    def is_null(self) -> bool:
        return self.main.is_null
    
    @classmethod
    def from_tpcf_evo_outputs(
            cls, 
            time_steps: np.ndarray,
            xi_vals: np.ndarray,
            weighted_xi_vals: np.ndarray,
            xi_fold_vals: dict[int, np.ndarray] | None = None,
            weighted_xi_fold_vals: dict[int, np.ndarray] | None = None,
            in_comoving: bool = False,
            time_metric: str = "scale_factor",
            is_linear: bool = False,
        ) -> TwoPointCorrelationAccumulation:
            
        # Build output array, filtering invalid entries
        valid_mask = np.isfinite(xi_vals)

        valid_time_steps = time_steps[valid_mask]
        accumulation = xi_vals[valid_mask]
        weighted = weighted_xi_vals[valid_mask]

        if accumulation.size == 0:
            return cls.null_initialize()

        # Update for linear scaling 
        a_f, xi_f = valid_time_steps[-1], accumulation[-1]

        if is_linear or in_comoving:
            scaling = xi_f
        else:
            scaling = xi_f * (valid_time_steps / a_f) ** 3

        rate = np.gradient(
            np.log(accumulation), 
            np.log(valid_time_steps), 
            edge_order=2
        )

        # If everything is invalid, return empty
        main_accumulation = AccumulationHistory(
            time_steps=valid_time_steps,
            accumulation=accumulation,
            weighted=weighted,
            normalized=accumulation / scaling,
            rate=rate,
            in_comoving=in_comoving,
            time_metric=time_metric
        )

        # Build fold accumulations if provided
        fold_accumulations: dict[int, AccumulationHistory] = {}
        if (xi_fold_vals is not None) and (weighted_xi_fold_vals is not None): 
            for fold_idx in xi_fold_vals.keys():

                fold_accumulation = xi_fold_vals[fold_idx][valid_mask]

                if fold_accumulation.size == 0:
                    fold_accumulations[fold_idx] = AccumulationHistory.null_initialize()
                    continue

                # Check out what is causing the error here...
                fold_weighted = weighted_xi_fold_vals[fold_idx][valid_mask]
                fold_scaling = ( 
                    fold_accumulation[-1] 
                    if in_comoving else 
                    fold_accumulation[-1] * (valid_time_steps / a_f) ** 3
                )

                fold_rate = np.gradient(
                    np.log(fold_accumulation), 
                    np.log(valid_time_steps), 
                    edge_order=2
                )

                fold_accumulations[fold_idx] = AccumulationHistory(
                    time_steps=valid_time_steps,
                    accumulation=fold_accumulation,
                    weighted=fold_weighted,
                    normalized=fold_accumulation / fold_scaling,
                    rate=fold_rate, 
                    in_comoving=in_comoving,
                    time_metric=time_metric
                )
        
        return cls(
            main=main_accumulation,
            folds=fold_accumulations,
            in_comoving=in_comoving,
            time_metric=time_metric
        )
    
    @property
    def fold_xi_stack(self) -> np.ndarray:
        ''' Shape: (n_valid_times, n_folds) '''
        return np.vstack([
            fold.accumulation 
            for fold in self.folds.values() 
            if not fold.is_null
        ]).T
    
    @property
    def fold_weighted_xi_stack(self) -> np.ndarray:
        ''' Shape: (n_valid_times, n_folds) '''
        return np.vstack([
            fold.weighted 
            for fold in self.folds.values() 
            if not fold.is_null
        ]).T
    
    @property
    def fold_normalized_xi_stack(self) -> np.ndarray:
        ''' Shape: (n_valid_times, n_folds) '''
        return np.vstack([
            fold.normalized 
            for fold in self.folds.values() 
            if not fold.is_null
        ]).T
    
    @property
    def fold_rate_stack(self) -> np.ndarray:
        ''' Shape: (n_valid_times, n_folds) '''
        return np.vstack([
            fold.rate 
            for fold in self.folds.values() 
            if not fold.is_null
        ]).T

    @property
    def std_dev(self) -> np.ndarray:
        return np.nanstd(self.fold_xi_stack, axis=1)

    @property
    def accumulation_bounds(self) -> np.ndarray:
        return make_bound_array(self.main.accumulation, self.std_dev)

    @property
    def weighted_std_dev(self) -> np.ndarray: 
        return np.nanstd(self.fold_weighted_xi_stack, axis=1)

    @property
    def weighted_bounds(self) -> np.ndarray:
        return make_bound_array(self.main.weighted, self.weighted_std_dev)
    
    @property
    def rate_std_dev(self) -> np.ndarray:
        return np.nanstd(self.fold_rate_stack, axis=1)
    
    @property
    def rate_bounds(self) -> np.ndarray:
        return make_bound_array(self.main.rate, self.rate_std_dev)
    
    @property
    def normalized_std_dev(self) -> np.ndarray:
        return np.nanstd(self.fold_normalized_xi_stack, axis=1)
    
    @property
    def normalized_bounds(self) -> np.ndarray:
        return make_bound_array(self.main.normalized, self.normalized_std_dev)
    
    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def n_non_null_folds(self) -> int:
        return sum(not fold.is_null for fold in self.folds.values())
    
    @property
    def has_folds(self) -> bool:
        return self.n_folds > 0
    
    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return compute_peak_and_freeze_times(
            scale_factors=self.main.time_steps,
            full_evolution=self.main.normalized,
            fold_id_to_evolution={
                fold_id: fold.normalized
                for fold_id, fold in self.folds.items()
                if not fold.is_null
            } if self.has_folds else None,
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )


@define(slots=True)
class TwoPointCorrelationAccumulations:
    accumulations: OrderedDict[KeyType, TwoPointCorrelationAccumulation]
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")
    is_linear: bool = field(default=False)


    def __getitem__(self, key: KeyType) -> TwoPointCorrelationAccumulation:
        return self.accumulations[key]

    def __iter__(self) -> Iterator[KeyType]:
        return iter(self.accumulations)
    
    def __len__(self) -> int:
        return len(self.accumulations)
    

    @classmethod
    def null_initialize(cls) -> TwoPointCorrelationAccumulations:
        return cls(
            accumulations=OrderedDict(),
            in_comoving=False,
            time_metric="scale_factor",
            is_linear=False
        )
    
    @property
    def is_null(self) -> bool:
        return len(self.accumulations) == 0
    
    @property
    def main_accumulations(self) -> AccumulationHistories:
        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value.main) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric,
            is_linear=self.is_linear
        )
    
    @property
    def fold_accumulations(self) -> dict[int, AccumulationHistories]:
        fold_dict: dict[int, AccumulationHistories] = defaultdict(
            lambda: AccumulationHistories.null_initialize()
        )

        for key, accumulation in sorted(
            self.accumulations.items(), key=lambda item: float(item[0])
        ):
            for fold_id, fold_accumulation in accumulation.folds.items():
                fold_dict[fold_id].accumulations[key] = fold_accumulation
        
        for fold_id in fold_dict:
            fold_dict[fold_id].in_comoving = self.in_comoving
            fold_dict[fold_id].time_metric = self.time_metric
            fold_dict[fold_id].is_linear = self.is_linear

        return dict(fold_dict)
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        times_dict = {
            key : accumulation.get_peak_and_freeze_out_times(
                tol=tol,
                rel_tol_no_folds=rel_tol_no_folds,
                n_hi_res=n_hi_res,
                return_full_grid=return_full_grid,
                method=method,
                verbose=verbose,
                **kwargs,
            )
            for key, accumulation in self.accumulations.items()
            if not accumulation.is_null
        }

        return convert_time_results_to_array(
            time_results=times_dict, 
            result_sep_key="separation"  # TPCF is always in terms of separation
        )



@define(slots=True)
class MatterTwoPointCorrelationAccumulation:
    linear: TwoPointCorrelationAccumulation
    nonlinear: TwoPointCorrelationAccumulation
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @property
    def is_null(self) -> bool:
        return self.linear.is_null and self.nonlinear.is_null

    @cached_property
    def boost(self) -> AccumulationHistory:
        """
        Compute the boost factor (nonlinear / linear) as an AccumulationHistory.
        
        The boost measures the departure from linear theory at each time step
        for the two-point correlation function.
        
        If linear and nonlinear have different time steps, interpolate to a 
        common grid (the intersection of valid time ranges).
        """
        lin_times = self.linear.main.time_steps
        nonlin_times = self.nonlinear.main.time_steps
        
        # Check for empty arrays - return null if either is empty
        if len(lin_times) == 0 or len(nonlin_times) == 0:
            return AccumulationHistory.null_initialize()
        
        # Check if time steps match
        if len(lin_times) == len(nonlin_times) and np.allclose(lin_times, nonlin_times):
            # Time steps match - use directly
            common_times = lin_times
            lin_acc = self.linear.main.accumulation
            lin_norm = self.linear.main.normalized
            lin_weighted = self.linear.main.weighted
            nonlin_acc = self.nonlinear.main.accumulation
            nonlin_norm = self.nonlinear.main.normalized
            nonlin_weighted = self.nonlinear.main.weighted
        else:
            # Time steps differ - interpolate to common grid
            # Use the intersection of time ranges
            t_min = max(lin_times.min(), nonlin_times.min())
            t_max = min(lin_times.max(), nonlin_times.max())
            
            if t_min >= t_max:
                # No overlap - return null
                return AccumulationHistory.null_initialize()
            
            # Use the denser sampling as the common grid
            n_common = max(len(lin_times), len(nonlin_times))
            common_times = np.linspace(t_min, t_max, n_common)
            
            # Interpolate linear values
            lin_acc = np.interp(common_times, lin_times, self.linear.main.accumulation)
            lin_norm = np.interp(common_times, lin_times, self.linear.main.normalized)
            lin_weighted = np.interp(common_times, lin_times, self.linear.main.weighted)
            
            # Interpolate nonlinear values
            nonlin_acc = np.interp(common_times, nonlin_times, self.nonlinear.main.accumulation)
            nonlin_norm = np.interp(common_times, nonlin_times, self.nonlinear.main.normalized)
            nonlin_weighted = np.interp(common_times, nonlin_times, self.nonlinear.main.weighted)
        
        # Avoid division by zero
        lin_acc_safe = np.where(lin_acc != 0, lin_acc, np.nan)
        lin_norm_safe = np.where(lin_norm != 0, lin_norm, np.nan)
        lin_weighted_safe = np.where(lin_weighted != 0, lin_weighted, np.nan)
        
        boost_accumulation = nonlin_acc / lin_acc_safe
        boost_normalized = nonlin_norm / lin_norm_safe
        boost_weighted = nonlin_weighted / lin_weighted_safe
        
        # Compute smoothed derivative using Savitzky-Golay filter
        n_pts = len(common_times)
        if n_pts >= 5:
            window_length = min(max(5, n_pts // 10 | 1), n_pts)
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(3, window_length - 1)
            delta = np.mean(np.diff(common_times)).astype(float)
            boost_rate = savgol_filter(
                boost_accumulation, 
                window_length=window_length, 
                polyorder=polyorder, 
                deriv=1, 
                delta=delta
            )
        else:
            boost_rate = np.gradient(boost_accumulation, common_times)
        
        return AccumulationHistory(
            time_steps=common_times,
            accumulation=boost_accumulation,
            weighted=boost_weighted,
            normalized=boost_normalized,
            rate=boost_rate,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )

    @classmethod
    def null_initialize(cls) -> MatterTwoPointCorrelationAccumulation:
        return cls(
            linear=TwoPointCorrelationAccumulation.null_initialize(),
            nonlinear=TwoPointCorrelationAccumulation.null_initialize(),
            in_comoving=False,
            time_metric="scale_factor"
        )
    
    @classmethod
    def from_matter_tpcf_evo_outputs(
            cls, 
            time_steps: np.ndarray,
            lin_xi_vals: np.ndarray,
            nonlin_xi_vals: np.ndarray,
            lin_weighted_xi_vals: np.ndarray,
            nonlin_weighted_xi_vals: np.ndarray,
            nonlin_xi_fold_vals: dict[int, np.ndarray] | None = None,
            nonlin_weighted_xi_fold_vals: dict[int, np.ndarray] | None = None,
            in_comoving: bool = False,
            time_metric: str = "scale_factor",
        ) -> MatterTwoPointCorrelationAccumulation:
            
        linear = TwoPointCorrelationAccumulation.from_tpcf_evo_outputs(
            time_steps=time_steps,
            xi_vals=lin_xi_vals,
            weighted_xi_vals=lin_weighted_xi_vals,
            in_comoving=in_comoving,
            time_metric=time_metric,
            is_linear=True
        )
        nonlinear = TwoPointCorrelationAccumulation.from_tpcf_evo_outputs(
            time_steps=time_steps,
            xi_vals=nonlin_xi_vals,
            weighted_xi_vals=nonlin_weighted_xi_vals,
            xi_fold_vals=nonlin_xi_fold_vals,
            weighted_xi_fold_vals=nonlin_weighted_xi_fold_vals,
            in_comoving=in_comoving,
            time_metric=time_metric,
            is_linear=False
        )
        return cls(
            linear=linear, 
            nonlinear=nonlinear,
            in_comoving=in_comoving,
            time_metric=time_metric
        )
    
    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_boost_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, float | np.ndarray]:

        return self.boost.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            include_boost: bool = True,
            **extra_kwargs,
        ) -> dict[str, dict[str, float | np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose,
            **extra_kwargs,
        }

        results = {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }
        
        if include_boost:
            results["boost"] = self.get_boost_peak_and_freeze_out_times(**kwargs)
        
        return results


@define(slots=True)
class MatterTwoPointCorrelationAccumulations:
    accumulations: OrderedDict[KeyType, MatterTwoPointCorrelationAccumulation]
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")


    def __getitem__(self, key: KeyType) -> MatterTwoPointCorrelationAccumulation:
        return self.accumulations[key]
    
    def __iter__(self) -> Iterator[KeyType]:
        return iter(self.accumulations)
    
    def __len__(self) -> int:
        return len(self.accumulations)

    @property
    def linear(self) -> TwoPointCorrelationAccumulations:
        return TwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (key, value.linear) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    
    @property
    def nonlinear(self) -> TwoPointCorrelationAccumulations:
        return TwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (key, value.nonlinear) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    
    @property
    def boost(self) -> AccumulationHistories:
        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value.boost) 
                for key, value in sorted(
                    self.accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=self.time_metric,
            is_linear=False
        )
    
    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = True,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            **kwargs,
        )
    
    def get_boost_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = True,
            method: str = "hybrid_window",
            verbose: bool = False,
            **kwargs,
        ) -> dict[str, np.ndarray]:

        return self.boost.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=False,
            **kwargs,
        )
        
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = True,
            method: str = "hybrid_window",
            verbose: bool = False,
            include_boost: bool = True,
            **extra_kwargs,
        ) -> dict[str, dict[str, np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose,
            **extra_kwargs,
        }

        results = {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }
        
        if include_boost:
            results["boost"] = self.get_boost_peak_and_freeze_out_times(**kwargs)
        
        return results
    
@define(slots=True)
class FieldAccumulation:
    tpcf: AccumulationHistory
    spectrum: AccumulationHistory
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")


    def get_tpcf_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, float | np.ndarray]:

        return self.tpcf.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )
    

    def get_spectrum_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, float | np.ndarray]:

        return self.spectrum.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )


    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, float | np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose
        }

        return {
            "tpcf": self.get_tpcf_peak_and_freeze_out_times(**kwargs),
            "spectrum": self.get_spectrum_peak_and_freeze_out_times(**kwargs)
        }
    
@define(slots=True)
class FieldAccumulations:
    tpcf: AccumulationHistories
    spectrum: AccumulationHistories
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")


    def get_tpcf_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, np.ndarray]:

        return self.tpcf.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=False
        )
    

    def get_spectrum_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, np.ndarray]:

        return self.spectrum.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose,
            for_wavenumber=True
        )


    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, np.ndarray]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose
        }

        return {
            "tpcf": self.get_tpcf_peak_and_freeze_out_times(**kwargs),
            "spectrum": self.get_spectrum_peak_and_freeze_out_times(**kwargs)
        }

@define(slots=True)
class FieldMatterAccumulation:
    spectrum: MatterPowerSpectrumAccumulation
    tpcf: MatterTwoPointCorrelationAccumulation
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @property
    def linear(self) -> FieldAccumulation:
        return FieldAccumulation(
            tpcf=self.tpcf.linear.main,
            spectrum=self.spectrum.linear,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    
    @property
    def nonlinear(self) -> FieldAccumulation:
        return FieldAccumulation(
            tpcf=self.tpcf.nonlinear.main,
            spectrum=self.spectrum.nonlinear,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    

    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, float | np.ndarray]]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )


    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, float | np.ndarray]]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )
    

    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, dict[str, float | np.ndarray]]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose
        }

        return {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }


@define(slots=True)
class FieldMatterAccumulations:
    tpcf: MatterTwoPointCorrelationAccumulations
    spectrum: MatterPowerSpectraAccumulation
    in_comoving: bool = field(default=False)
    time_metric: str = field(default="scale_factor")

    @property
    def linear(self) -> FieldAccumulations:
        return FieldAccumulations(
            tpcf=self.tpcf.linear,
            spectrum=self.spectrum.linear,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    
    @property
    def nonlinear(self) -> FieldAccumulations:
        return FieldAccumulations(
            tpcf=self.tpcf.nonlinear,
            spectrum=self.spectrum.nonlinear,
            in_comoving=self.in_comoving,
            time_metric=self.time_metric
        )
    
    def get_linear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, np.ndarray]]:

        return self.linear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )
    
    def get_nonlinear_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, np.ndarray]]:

        return self.nonlinear.get_peak_and_freeze_out_times(
            tol=tol,
            rel_tol_no_folds=rel_tol_no_folds,
            n_hi_res=n_hi_res,
            return_full_grid=return_full_grid,
            method=method,
            verbose=verbose
        )
    
    def get_peak_and_freeze_out_times(
            self,
            tol: float = 2.0,
            rel_tol_no_folds: float = 0.02,
            n_hi_res: int = 1000,
            return_full_grid: bool = False,
            method: str = "hybrid_window",
            verbose: bool = False,
        ) -> dict[str, dict[str, dict[str, np.ndarray]]]:

        kwargs = {
            "tol": tol,
            "rel_tol_no_folds": rel_tol_no_folds,
            "n_hi_res": n_hi_res,
            "return_full_grid": return_full_grid,
            "method": method,
            "verbose": verbose
        }

        return {
            "linear": self.get_linear_peak_and_freeze_out_times(**kwargs),
            "nonlinear": self.get_nonlinear_peak_and_freeze_out_times(**kwargs)
        }



def make_bound_array(main: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.array([main - std, main + std])


def convert_time_results_to_array(
        time_results: dict[KeyType, dict[str, float | np.ndarray]], 
        result_sep_key: str
    ) -> dict[str, np.ndarray]:
    """
    Convert separation/wavenumber results to arrays.
    
    Parameters
    ----------
    time_results : dict[KeyType, dict[str, float]]
        The time results dictionary:
            sep_keys -> fields -> values
        where KeyType can be str, int, float, or numpy numeric types
    result_sep_key : str
        Key name for output ("wavenumber" or "separation")
    
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with separation/wavenumber array and field arrays
    """
    # Sort keys numerically regardless of type
    sep_keys = sorted(time_results.keys(), key=lambda x: float(x))
    
    # Identify all inner fields by looking at the first valid entry
    example_sep = sep_keys[0]
    example_fields = time_results[example_sep].keys()
    
    # Build container with separation/wavenumber values
    out = {result_sep_key: np.array([10.0 ** float(s) for s in sep_keys])}
    
    # For each field: collect all values in separation-key order
    for field in example_fields:
        arr = []
        for s in sep_keys:
            val = time_results[s].get(field, np.nan)
            if isinstance(val, np.ndarray): continue
            arr.append(val if val is not None else np.nan)
        out[field] = np.array(arr)
    
    return out