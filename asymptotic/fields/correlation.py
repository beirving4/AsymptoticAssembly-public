from __future__ import annotations

import mcfit, h5py
import treecorr
import numpy as np, pdb
import matplotlib.pyplot as plt

from pathlib import Path
from attrs import define, field
from collections import OrderedDict
from collections.abc import Collection
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, UnivariateSpline



from .two_point import get_tpcf, DEFAULT_NUM_TARGET_PAIRS
from ..cosmo.model import Cosmology
from ..simulation.moments import Moment, MomentsInTime, get_correct_snapshot_mapping
from ..simulation.evo import EvolutionData
from ..utils.jackknife import get_jackknife_results
from .accumulation import (
    TwoPointCorrelationAccumulation,
    TwoPointCorrelationAccumulations,
    MatterTwoPointCorrelationAccumulation,
    MatterTwoPointCorrelationAccumulations
)
from .power import (
    PowerSpectrumData,
    PowerSpectrumDataEvo,
    MatterPowerSpectrum, 
    MatterPowerSpectrumData, 
    MatterPowerSpectrumEvo,
    MatterPowerSpectrumEvoData
)
from .viz import (
    display_correlation,
    display_matter_correlation
)
from .correlation_freeze import (
    FreezeOutMetrics,
    compute_freezeout_metrics,
    compute_freezeout_metrics_treecorr,
)

@define(slots=True)
class TwoPointCorrelationValue:
    radius: float
    correlation: float
    is_linear: bool = field(default=False)
    in_comoving: bool = field(default=False)

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        self.radius = self.radius / scale_factor
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        self.radius = self.radius * scale_factor
        self.in_comoving = False


@define(slots=True)
class MatterTwoPointCorrelationValue:
    linear: TwoPointCorrelationValue
    nonlinear: TwoPointCorrelationValue
    in_comoving: bool = field(default=False)

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        self.linear.convert_to_comoving(scale_factor)
        self.nonlinear.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        self.linear.convert_to_physical(scale_factor)
        self.nonlinear.convert_to_physical(scale_factor)
        self.in_comoving = False

@define(slots=True)
class TwoPointCorrelation: 
    radii: np.ndarray # when time permits change variable name to separations
    correlation: np.ndarray
    is_linear: bool = field(default=False)
    in_comoving: bool = field(default=False)

    _spline: interp1d | None = field(init=False)

    def __attrs_post_init__(self) -> None:
        self._rebuild_spline()

    def __repr__(self) -> str:
        return (
            f"TwoPointCorrelation("
            f"r=({self.radii.min():.3e}, "
            f"{self.radii.max():.3e}), "
            f"xi=({self.correlation.min():.3e}, "
            f"{self.correlation.max():.3e}), "
            f"is_linear={self.is_linear})"
        )
    
    def _rebuild_spline(self) -> None:
        try:
            self._spline = interp1d(
                np.log10(self.radii), 
                np.log10(self.correlation)
            )
        except Exception:
            self._spline = None

    @property
    def separation_scales(self) -> np.ndarray:
        return 10 ** np.unique(np.log10(self.radii).astype(int)).astype(float)


    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return 10.0 ** np.unique(np.round(np.log10(self.radii), 1))

    
    @property
    def radial_range(self) -> tuple[float, float]:
        return (self.radii.min(), self.radii.max())

    
    def in_range(self, radius: float) -> bool:
        return self.radii.min() <= radius <= self.radii.max()
    
    @property
    def as_dict(self) -> dict[str, np.ndarray | bool]:
        return {
            "radii": self.radii,
            "correlation": self.correlation,
            "is_linear": self.is_linear,
            "in_comoving": self.in_comoving,
        }

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        self.radii = self.radii / scale_factor
        self.in_comoving = True
        self._rebuild_spline()

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        self.radii = self.radii * scale_factor
        self.in_comoving = False
        self._rebuild_spline()
    
    @classmethod
    def null_initialize(cls, arr_size: int = 0, is_linear: bool = False) -> TwoPointCorrelation:
        return cls(
            radii=np.full(arr_size, np.nan), 
            correlation=np.full(arr_size, np.nan), 
            is_linear=is_linear
        )

    @property
    def is_null(self) -> bool:
        # A null instance has empty arrays or arrays containing only uninitialized (NaN) values.
        if self.radii.size == 0 and self.correlation.size == 0:
            return True
        return np.logical_and(
            np.all(np.isnan(self.radii)), np.all(np.isnan(self.correlation))
        )

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology, 
            separations: np.ndarray,
            redshift: float = 0.0 
        ) -> TwoPointCorrelation:

        """Build a two-point correlation from a Cosmology instance."""

        return cls(
            radii=separations,
            correlation=cosmo.linear_matter_correlation(r=separations, z=redshift),
            is_linear=True,
            in_comoving=True
        )

    @classmethod
    def from_data(
            cls,
            comoving_coordinates: np.ndarray,
            rmin: float,
            rmax: float,
            nbins: int,
            boxsize: float,
            weights: np.ndarray | None = None,
            other_coordinates: np.ndarray | None = None,
            weights2: np.ndarray | None = None,
            eps: float = 1e-12,
            use_natural: bool = True,
            # Subsampling controls:
            use_recommended_subsample: bool = False,
            num_target_pairs: int = 100000,
            subsample_secondary: bool = False,
            subsample_rng_seed: int | None = None,
        ) -> "TwoPointCorrelation":
        """Build a single two-point correlation (no jackknife).

        Parameters mirror `get_tpcf` for the non-jackknife path.
        """
        out = get_tpcf(
            comoving_coordinates=comoving_coordinates,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            boxsize=boxsize,
            weights=weights,
            other_coordinates=other_coordinates,
            weights2=weights2,
            eps=eps,
            use_natural=use_natural,
            run_jackknife=False,
            use_recommended_subsample=use_recommended_subsample,
            num_target_pairs=num_target_pairs,
            subsample_secondary=subsample_secondary,
            subsample_rng_seed=subsample_rng_seed,
        )

        # `get_tpcf` returns a dict; take the estimate branch
        est = out["estimate"] if isinstance(out, dict) else out
        return cls(radii=est[:, 0], correlation=est[:, 1], is_linear=False)

    @classmethod
    def from_power_spectrum(
            cls, 
            wavenumbers: np.ndarray, 
            amplitudes: np.ndarray, 
            ell: int = 0,
            extrapolate: bool = True,
            is_linear: bool = False
        ) -> TwoPointCorrelation:
        return cls(
            **powerspec_to_correlation(
                wavenumbers=wavenumbers, 
                amplitudes=amplitudes, 
                ell=ell, 
                extrapolate=extrapolate
            ), 
            is_linear=is_linear
        )

    @property
    def scaled(self) -> np.ndarray:
        return self.correlation * self.radii**2
    
    @property
    def slope(self) -> np.ndarray:
        ''' 
        Return the logarithmic slope dln(xi)/dln(r) 
        using the savitzky-golay filter for smoothing.
        '''
        d_log_r = np.diff(np.log10(self.radii)).mean()
        
        return savgol_filter(
            np.log10(self.correlation),
            window_length=11, 
            polyorder=2, # polyorder=window_length - 1 to fit exactly
            deriv=1,
            delta=d_log_r
        )
    
    def get_interpolated_xi(self, radii: float | np.ndarray) -> float | np.ndarray:
        if self._spline is None:
            raise ValueError("Cannot interpolate correlation function")
        return 10**self._spline(np.log10(radii))


    def get_interpolated_instance(
            self, radii: float | np.ndarray
        ) -> TwoPointCorrelationValue | TwoPointCorrelation:
        
        if isinstance(radii, (float, int, np.floating, np.integer)):
            return TwoPointCorrelationValue(
                radius=float(radii),
                correlation=float(self.get_interpolated_xi(radii)),
                is_linear=self.is_linear,
                in_comoving=self.in_comoving
            )
        else:
            return TwoPointCorrelation(
                radii=np.asarray(radii),
                correlation=np.asarray(self.get_interpolated_xi(radii)),
                is_linear=self.is_linear,
                in_comoving=self.in_comoving
            )
    
    def display(
            self, text_size: int = 16,
            use_scaled: bool = False,
            ax: plt.Axes | None = None,
            return_fig: bool = False,
            color: str | tuple | None = None
        ) -> plt.Axes | None:

        ax = display_correlation(
            radii=self.radii,
            correlations=(self.scaled if use_scaled else self.correlation),
            is_scaled=use_scaled,
            is_linear=self.is_linear,
            text_size=text_size,
            ax=ax,
            color=color,
            return_fig=True
        )

        return ax if return_fig else None
    

@define(slots=True)
class TwoPointCorrelationData: 
    estimate: TwoPointCorrelation
    error: TwoPointCorrelation
    correlation_matrix: np.ndarray = field(repr=False)
    in_comoving: bool = field(default=False)
    is_linear: bool = field(default=False)
    folds: np.ndarray | None = field(default=None, repr=False)
    treecorr_nn: treecorr.NNCorrelation | None = field(default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        # Sync child in_comoving states with parent
        self.estimate.in_comoving = self.in_comoving
        self.error.in_comoving = self.in_comoving

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual underlying TwoPointCorrelation state."""
        self.in_comoving = self.estimate.in_comoving
        self.error.in_comoving = self.in_comoving

    @property
    def radii(self) -> np.ndarray:
        return self.estimate.radii
    
    @property
    def xi(self) -> np.ndarray:
        return self.estimate.correlation
    
    @property
    def sigma(self) -> np.ndarray:
        return self.error.correlation

    @property
    def covariance_matrix(self) -> np.ndarray:
        return correlation_to_covariance(self.sigma, self.correlation_matrix)

    @property
    def separation_scales(self) -> np.ndarray:
        return self.estimate.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.estimate.round_one_separation_scales

    @property
    def as_dict(self) -> dict[str, np.ndarray | dict[str, np.ndarray | bool]]:
        return {
            "estimate": self.estimate.as_dict,
            "error": self.error.as_dict,
            "correlation_matrix": self.correlation_matrix,
            "in_comoving": self.in_comoving,
        }
    

    @property
    def folds_as_dict(self) -> dict[str, np.ndarray | dict[int, np.ndarray]] | None:
        if self.folds is None:
            return None
        return {
            "radii": self.folds[:, 0],
            "folds" : { 
                i: self.folds[:, i + 1] for i in range(self.folds.shape[1] - 1) 
            }
        }

    @property
    def fold_instances(self) -> dict[int, TwoPointCorrelation]:
        
        if ((folds_dict := self.folds_as_dict) is None) or self.is_linear:
            raise ValueError("No folds data available")

        return {
            i: TwoPointCorrelation(
                    radii=folds_dict["radii"],
                    correlation=fold,
                    is_linear=False,
                    in_comoving=self.in_comoving
                ) 
            for i, fold in folds_dict["folds"].items()
        }


    @property
    def radial_range(self) -> tuple[float, float]:
        return self.estimate.radial_range
    
    @property
    def _fold_slopes_array(self) -> np.ndarray:
        if (self.folds is None) or self.is_linear:
            raise ValueError("No folds data available for slope estimation")
        return np.column_stack([
            fold.slope for fold in self.fold_instances.values()
        ])
    
    @property
    def slope(self) -> np.ndarray:
        return self.estimate.slope

    @property
    def slope_error(self) -> np.ndarray:
        fold_slopes = self._fold_slopes_array  # Shape: (N, 8)
        return np.std(fold_slopes, axis=1)
    
    @property
    def has_null_errors(self) -> bool:
        return self.error.is_null

    @property
    def is_null(self) -> bool:
        return self.estimate.is_null

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            separations: np.ndarray, 
            redshift: float = 0.0
        ) -> TwoPointCorrelationData:

        return cls(
            estimate=TwoPointCorrelation.from_cosmology(
                cosmo=cosmo, 
                separations=separations,
                redshift=redshift
            ),
            error=TwoPointCorrelation.null_initialize(
                arr_size=separations.size,
                is_linear=True
            ),
            correlation_matrix=np.eye(separations.size),
            is_linear=True,
            in_comoving=True
        )

    @classmethod
    def from_data(
            cls,
            comoving_coordinates: np.ndarray,
            rmin: float,
            rmax: float,
            nbins: int,
            boxsize: float,
            weights: np.ndarray | None = None,
            other_coordinates: np.ndarray | None = None,
            weights2: np.ndarray | None = None,
            sub_box_info: dict[int, list[tuple[float, float]]] | None = None,
            eps: float = 1e-12,
            use_natural: bool = True,
            run_jackknife: bool = True,
            # jackknife controls (optional):
            random_multiplier: int | None = None,
            rng_seed: int | None = None,
            return_folds: bool = False,
            # Subsampling controls (optional):
            use_recommended_subsample: bool = False,
            num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
            subsample_secondary: bool = False,
            subsample_rng_seed: int | None = None,
            # Adaptive randoms controls (jackknife only):
            adaptive_randoms: bool = False,
            max_randoms_per_fold: int | None = None,
            min_randoms_per_fold: int = 1024,
        ) -> TwoPointCorrelationData:
        """Build a jackknife-estimated two-point correlation with errors & correlation matrix."""
        
        if (sub_box_info is None) and run_jackknife:
            raise ValueError("sub_box_info must be provided when building jackknifed TPCF data")

        out = get_tpcf(
            comoving_coordinates=comoving_coordinates,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            boxsize=boxsize,
            weights=weights,
            other_coordinates=other_coordinates,
            weights2=weights2,
            sub_box_info=sub_box_info,
            eps=eps,
            use_natural=use_natural,
            run_jackknife=run_jackknife,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
            return_folds=return_folds,
            use_recommended_subsample=use_recommended_subsample,
            num_target_pairs=num_target_pairs,
            subsample_secondary=subsample_secondary,
            subsample_rng_seed=subsample_rng_seed,
            adaptive_randoms=adaptive_randoms,
            max_randoms_per_fold=max_randoms_per_fold,
            min_randoms_per_fold=min_randoms_per_fold
        )

        return cls(
            estimate=TwoPointCorrelation(
                radii=out["estimate"][:, 0],
                correlation=out["estimate"][:, 1],
            ),
            error=TwoPointCorrelation(
                radii=out["errors"][:, 0],
                correlation=out["errors"][:, 1],
            ),
            correlation_matrix=out["correlation_matrix"],
            folds=out.get("folds", None),
            is_linear=False,
            in_comoving=True
        )

    def get_interpolated_xi(self, radii: float | np.ndarray) -> float | np.ndarray:
        return self.estimate.get_interpolated_xi(radii)

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        # Force sync child states before conversion
        self.estimate.in_comoving = False
        self.error.in_comoving = False
        self.estimate.convert_to_comoving(scale_factor)
        self.error.convert_to_comoving(scale_factor)
        if self.folds is not None:
            self.folds[:, 0] = self.folds[:, 0] / scale_factor
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        # Force sync child states before conversion
        self.estimate.in_comoving = True
        self.error.in_comoving = True
        self.estimate.convert_to_physical(scale_factor)
        self.error.convert_to_physical(scale_factor)
        if self.folds is not None:
            self.folds[:, 0] = self.folds[:, 0] * scale_factor
        self.in_comoving = False

    def save(self, filepath: Path) -> None: 
        payload = self.as_dict
        # include folds when present
        if self.folds is not None:
            payload["folds"] = self.folds
        elif self.folds_as_dict is not None:
            payload["folds_as_dict"] = self.folds_as_dict
        save_two_point_correlation(filepath, payload)

    @classmethod
    def load(cls, filepath: Path, from_treecorr: bool = False) -> TwoPointCorrelationData:
        data = load_two_point_correlation(filepath, from_treecorr=from_treecorr)
        return cls(
            estimate=TwoPointCorrelation(
                radii=data["estimate"]["radii"],
                correlation=data["estimate"]["correlation"],
            ),
            error=TwoPointCorrelation(
                radii=data["error"]["radii"],
                correlation=data["error"]["correlation"],
            ),
            correlation_matrix=data["correlation_matrix"],
            in_comoving=data["in_comoving"],
            folds=data.get("folds"),
            treecorr_nn=data.get("treecorr_nn"),
            # is_linear=data["estimate"].get("is_linear", False)
        )


    @classmethod
    def from_treecorr_nn(cls, treecorr_nn: NNCorrelation) -> TwoPointCorrelationData:
        radii = np.exp(treecorr_nn.meanlogr)
        return cls(
            estimate=TwoPointCorrelation(
                radii=radii,
                correlation=treecorr_nn.xi,
            ),
            error=TwoPointCorrelation(
                radii=radii,
                correlation=treecorr_nn.varxi**0.5,
            ),
            correlation_matrix=treecorr_nn.cov,
            in_comoving=True,
            folds=extract_treecorr_fold_data(treecorr_nn, radii),
            treecorr_nn=treecorr_nn
        )
    


    
@define(slots=True)
class TwoPointCorrelationEvo(EvolutionData):
    data: OrderedDict[int, TwoPointCorrelation]
    in_comoving: bool = field(default=False)
    is_linear: bool = field(default=False)

    def __repr__(self) -> str:  
        return super().__repr__()

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual child data state."""
        if non_null_data := [v for v in self.data.values() if not v.is_null]:
            # Use the state of the first non-null child as ground truth
            self.in_comoving = non_null_data[0].in_comoving

    def convert_to_comoving(self) -> None:
        # Sync state from children first to handle cases where data was
        # converted via a different accessor path
        self._sync_comoving_state()
        if self.in_comoving:
            return  # Already in comoving coordinates
        for key, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[key].scale_factor
            # Force sync child state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        # Sync state from children first
        self._sync_comoving_state()
        if not self.in_comoving:
            return  # Already in physical coordinates
        for key, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[key].scale_factor
            # Force sync child state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            moments: MomentsInTime,
            separations: np.ndarray,
        ) -> TwoPointCorrelationEvo:
        """
        Create a TwoPointCorrelationEvo from a cosmology model at the redshifts
        specified by the MomentsInTime object.
        
        Parameters
        ----------
        cosmo : Cosmology
            The cosmology model to use for computing correlation functions.
        moments : MomentsInTime
            The time steps (snapshot IDs, scale factors, redshifts) at which
            to compute the correlation functions.
        separations : np.ndarray
            The separation array [Mpc/h] at which to evaluate the correlations.
        
        Returns
        -------
        TwoPointCorrelationEvo
            A TwoPointCorrelationEvo instance containing linear correlation 
            functions at each snapshot specified in moments.
        """
        return cls(
            moments=moments,
            data=OrderedDict(
                (snap_id, TwoPointCorrelation.from_cosmology(
                    cosmo=cosmo,
                    separations=separations,
                    redshift=moments[snap_id].redshift
                ))
                for snap_id in moments.snapshot_ids
            ),
            in_comoving=True,
            is_linear=True
        )
    
    @property
    def scaled_evo(self) -> OrderedDict[int, np.ndarray]:
        return OrderedDict(
            (key, value.scaled) for key, value in self.data.items()
            if not value.is_null
        )

    @property
    def final_correlation(self) -> TwoPointCorrelation:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null correlation data available")

    def get_interpolated_xi_evo(self, radii: float) -> OrderedDict[int, float]:
        evo_dict = OrderedDict()
        for key, value in self.data.items():
            if value.is_null:
                continue
            try:
                evo_dict[key] = float(
                    value.get_interpolated_xi(radii)
                )
            except ValueError:
                evo_dict[key] = np.nan

        return evo_dict

    def get_interpolated_weighted_xi_evo(self, radii: float) -> OrderedDict[int, float]:
        evo_dict = OrderedDict()
        for key, value in self.data.items():
            if value.is_null:
                continue
            try:
                evo_dict[key] = float(
                    value.get_interpolated_xi(radii) * radii**2
                )
            except ValueError:
                evo_dict[key] = np.nan

        return evo_dict
    
    # Add the ability to return the evolution of xi or r^2 * xi at a given radius
    def get_xi_evo_array(self, radii: float, wrt: str = "scale_factor") -> np.ndarray:
        ''' (N,2) array where coloumn 0 is the time variable and column 1 is xi at that time'''
        evo_vals = self.get_interpolated_xi_evo(radii)
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.column_stack([
            (time_map[key], val) for key, val in evo_vals.items()
        ]).T

    def get_xi_evo_rate(self, radii: float, wrt: str = "scale_factor") -> np.ndarray:
        ''' (N, 2) array that is the dln(xi)/dln(time) '''
        xi_evo = self.get_xi_evo_array(radii, wrt=wrt)
        rate = np.gradient(np.log(xi_evo[:, 1]), np.log(xi_evo[:, 0]), edge_order=2)
        return np.column_stack([xi_evo[:, 0], rate]).T

    
    def get_weighted_evo_array(self, radii: float, wrt: str = "scale_factor") -> np.ndarray:
        ''' (N,2) array where coloumn 0 is the time variable and column 1 is r^2 * xi at that time'''
        evo_vals = self.get_interpolated_weighted_xi_evo(radii)
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.column_stack([
            (time_map[key], val) for key, val in evo_vals.items()
        ]).T

    def get_weighted_evo_rate(self, radii: float, wrt: str = "scale_factor") -> np.ndarray:
        ''' (N, 2) array that is the dln(r^2 * xi)/dln(time) '''
        weighted_evo = self.get_weighted_evo_array(radii, wrt=wrt)
        rate = np.gradient(np.log(weighted_evo[:, 1]), np.log(weighted_evo[:, 0]), edge_order=2)
        return np.column_stack([weighted_evo[:, 0], rate]).T
    
    def display(
            self, 
            target_scale_factors: Collection[float] | None = None,
            use_scaled: bool = False,
            text_size: int = 16,
            return_fig: bool = False,
            ax: plt.Axes | None = None,
            legend_xloc: float = 1.3,
            legend_yloc: float = 1.0
        ) -> plt.Axes | None:

        if ax is None:
            _, ax = plt.subplots()

        if target_scale_factors is None:
            target_scale_factors = self.moments.scale_factors

        colormap = plt.cm.get_cmap("viridis", len(target_scale_factors))

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

        handles = []

        for i, snap_id in enumerate(target_snapshot_ids):
            ax = self[snap_id].display(
                text_size=text_size,
                use_scaled=use_scaled,
                ax=ax,
                return_fig=True,
                color=colormap(i)
            )

            handles.append(
                plt.Line2D(
                    [], [], 
                    color=colormap(i), 
                    label=f"$a={actual_scale_factors[i]:.2f}$"
                )
            )

        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(legend_xloc, legend_yloc),
            fontsize=text_size
        )

        if return_fig:
            return ax
        
        plt.show()

@define(slots=True)
class TwoPointCorrelationEvoData(EvolutionData):
    data: OrderedDict[int, TwoPointCorrelationData]
    in_comoving: bool = field(default=False)
    is_linear: bool = field(default=False)


    def __attrs_post_init__(self) -> None:
        # Sync child in_comoving states to match parent's state
        # This ensures that when the parent is constructed with a specific
        # in_comoving value, all children are updated to match
        for v in self.data.values():
            if not v.is_null:
                v.in_comoving = self.in_comoving
                # Also sync the underlying TwoPointCorrelation objects
                v.estimate.in_comoving = self.in_comoving
                v.error.in_comoving = self.in_comoving

    @property
    def final_tpcf_data(self) -> TwoPointCorrelationData:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null correlation data available")

    @property
    def separation_scales(self) -> np.ndarray:
        return self.final_tpcf_data.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.final_tpcf_data.round_one_separation_scales
    @property
    def folds_array_evo(self) -> OrderedDict[int, np.ndarray] | None:
        folds_evo = OrderedDict()

        for snap_id, tpcf_data in self.data.items():
            if tpcf_data.folds is None:
                continue
            folds_evo[snap_id] = tpcf_data.folds

        return folds_evo or None
    
    @property
    def fold_instances_evo(self) -> OrderedDict[int, dict[int, TwoPointCorrelation]] | None:
        folds_evo = OrderedDict()

        for snap_id, tpcf_data in self.data.items():
            if tpcf_data.folds is None:
                continue
            folds_evo[snap_id] = tpcf_data.fold_instances

        return folds_evo or None

    @property
    def treecorr_nn_evo(self) -> OrderedDict[int, treecorr.NNCorrelation] | None:
        treecorr_evo = OrderedDict()

        for snap_id, tpcf_data in self.data.items():
            if tpcf_data.treecorr_nn is None:
                continue
            treecorr_evo[snap_id] = tpcf_data.treecorr_nn

        return treecorr_evo or None


    @classmethod 
    def load(
            cls, 
            directory: Path, 
            sim_cosmo: Cosmology, 
            moments: MomentsInTime,
            for_jackknife: bool = False,
            from_treecorr: bool = False
        ) -> TwoPointCorrelationEvoData:

        if not directory.is_dir():
            raise ValueError(f"Directory {directory} does not exist")

        data = {}
        new_moments = MomentsInTime()

        for filepath in directory.glob("*.hdf5"):

            if from_treecorr and ("treecorr" not in filepath.stem):
                continue

            if (
                for_jackknife 
                and (not from_treecorr)
                and ("jackknife" not in filepath.stem)
            ):
                continue

            snapshot_id = int(filepath.stem.split("_")[-1])

            try:
                new_moments.add_moment(moments[snapshot_id])
            except IndexError:
                print(f"Warning: Snapshot ID {snapshot_id} not found in provided moments.")
                continue


            try:
                data[snapshot_id] = TwoPointCorrelationData.load(
                    filepath=filepath, 
                    from_treecorr=from_treecorr
                )
            except Exception as e:
                print(f"Warning: Failed to load data from {filepath}: {e}")
                continue

        new_moments.add_times(sim_cosmo)

        return cls(moments=new_moments, data=OrderedDict(sorted(data.items())))

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual child data state."""
        if non_null_data := [v for v in self.data.values() if not v.is_null]:
            # First sync each child's state from underlying TwoPointCorrelation
            for v in non_null_data:
                v._sync_comoving_state()
            # Then use the first non-null child's state as ground truth
            self.in_comoving = non_null_data[0].in_comoving

    def convert_to_comoving(self) -> None:
        # Sync state from children first to handle cases where data was
        # converted via a different accessor path
        self._sync_comoving_state()
        if self.in_comoving:
            return  # Already in comoving coordinates
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force sync child state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        # Sync state from children first
        self._sync_comoving_state()
        if not self.in_comoving:
            return  # Already in physical coordinates
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force sync child state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            moments: MomentsInTime,
            separations: np.ndarray,
        ) -> TwoPointCorrelationEvoData:
        """
        Create a TwoPointCorrelationEvoData from a cosmology model at the redshifts
        specified by the MomentsInTime object.
        
        Parameters
        ----------
        cosmo : Cosmology
            The cosmology model to use for computing correlation functions.
        moments : MomentsInTime
            The time steps (snapshot IDs, scale factors, redshifts) at which
            to compute the correlation functions.
        separations : np.ndarray
            The separation array [Mpc/h] at which to evaluate the correlations.
        
        Returns
        -------
        TwoPointCorrelationEvoData
            A TwoPointCorrelationEvoData instance containing linear correlation 
            functions (with null errors) at each snapshot specified in moments.
        """
        return cls(
            moments=moments,
            data=OrderedDict(
                (snap_id, TwoPointCorrelationData.from_cosmology(
                    cosmo=cosmo,
                    separations=separations,
                    redshift=moments[snap_id].redshift
                ))
                for snap_id in moments.snapshot_ids
            ),
            in_comoving=True,
            is_linear=True
        )

    def get_radial_bin_evo(
            self,
            comoving_radial_bin: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0
        ) -> TwoPointCorrelationAccumulation:
        """Return the evolution of xi at a given comoving radius.

        Parameters
        ----------
        comoving_radial_bin : float
            Comoving radius at which to evaluate the correlation function.
        return_in_physical : bool, optional
            Whether to return radii in physical units (default: False).
        return_normalized : bool, optional
            Whether to normalize xi values by the final value (default: False).

        Returns
        -------
        np.ndarray or dict[str, np.ndarray]
            If no fold information: array of shape (N, 2) with columns [a, xi(a; r)].
            If fold information exists: dict with "mean" and "std_dev" arrays.
            NaN is used where the radius is outside the available range.
        """
        # Validate input
        if not np.isfinite(comoving_radial_bin) or comoving_radial_bin <= 0:
            return TwoPointCorrelationAccumulation.null_initialize()

        if not (
            snap_ids := [
                k for k, v in sorted(self.data.items(), key=lambda x: x[0])
                if not v.is_null    
            ]
        ):
            return TwoPointCorrelationAccumulation.null_initialize()
        
        # Save original coordinate state to restore after computation
        original_in_comoving = self.in_comoving
        
        # Ensure correct coordinate system for computation
        if return_in_physical and self.in_comoving:
            self.convert_to_physical()
        elif not return_in_physical and not self.in_comoving:
            self.convert_to_comoving()

        time_steps = self._get_time_steps(snap_ids, wrt=wrt)
        time_step_mask = np.logical_and(
            time_steps >= min_time_step_value,
            time_steps <= max_time_step_value
        )
        time_steps = time_steps[time_step_mask]
        snap_ids = np.asarray(snap_ids)[time_step_mask].tolist()
        xi_vals, xi_fold_vals = self._interpolate_xi_values(
            snap_ids=snap_ids, 
            radial_bin=comoving_radial_bin, 
            return_weighted=False
        )

        weighted_xi_vals, weighted_xi_fold_vals = self._interpolate_xi_values(
            snap_ids=snap_ids, 
            radial_bin=comoving_radial_bin, 
            return_weighted=True
        )

        result = TwoPointCorrelationAccumulation.from_tpcf_evo_outputs(
            time_steps=time_steps,
            xi_vals=xi_vals,
            xi_fold_vals=xi_fold_vals,
            weighted_xi_vals=weighted_xi_vals,
            weighted_xi_fold_vals=weighted_xi_fold_vals,
            in_comoving=self.in_comoving,
            time_metric=wrt,
            is_linear=self.is_linear
        )
        
        # Restore original coordinate state so this method has no side effects
        if original_in_comoving and not self.in_comoving:
            self.convert_to_comoving()
        elif not original_in_comoving and self.in_comoving:
            self.convert_to_physical()
        
        return result


    def _interpolate_xi_values(
            self,
            snap_ids: list,
            radial_bin: float,
            return_weighted: bool = False
        ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Interpolate xi values at the given radius for all snapshots."""
        
        xi_vals = np.full(len(snap_ids), np.nan)
        xi_fold_vals: dict[int, np.ndarray] = {}

        for i, snap_id in enumerate(snap_ids):
            tpcf_data = self.data[snap_id]
            xi_vals[i] = _safe_interpolate(tpcf_data, radial_bin)

            if return_weighted:
                xi_vals[i] = xi_vals[i] * radial_bin**2

            try:
                fold_instances = tpcf_data.fold_instances
            except ValueError:
                continue

            for fold_idx, fold_tpcf in fold_instances.items():
                if fold_idx not in xi_fold_vals:
                    xi_fold_vals[fold_idx] = np.full(len(snap_ids), np.nan)
                xi_fold_vals[fold_idx][i] = _safe_interpolate(fold_tpcf, radial_bin)

                if return_weighted:
                    xi_fold_vals[fold_idx][i] = xi_fold_vals[fold_idx][i] * radial_bin**2

        return xi_vals, xi_fold_vals
    


    def get_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> TwoPointCorrelationAccumulations:

        ''' Get xi evolution at all separation scales. ''' 
        
        # Save original coordinate state to restore after computation
        original_in_comoving = self.in_comoving
        
        # Work in comoving coordinates for consistent separation scale lookup
        if not self.in_comoving:
            self.convert_to_comoving()

        separation_scales = self.separation_scales
    
        accumulations = {}
        for scale in separation_scales:
            sep_key = key_format.format(np.log10(scale))
            accumulations[sep_key] = self.get_radial_bin_evo(
                comoving_radial_bin=scale,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        result = TwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (key, accumulation) for key, accumulation in sorted(
                    accumulations.items(), key=lambda x: float(x[0])
                )
            ),
            in_comoving=not return_in_physical,
            time_metric=wrt
        )
        
        # Restore original coordinate state so this method has no side effects
        if not original_in_comoving and self.in_comoving:
            self.convert_to_physical()
        
        return result

    def get_round_one_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> TwoPointCorrelationAccumulations:

        ''' Get xi evolution at all separation scales. ''' 
        
        # Save original coordinate state to restore after computation
        original_in_comoving = self.in_comoving
        
        # Work in comoving coordinates for consistent separation scale lookup
        if not self.in_comoving:
            self.convert_to_comoving()

        separation_scales = self.round_one_separation_scales
    
        accumulations = {}
        for scale in separation_scales:
            sep_key = key_format.format(np.log10(scale))
            accumulations[sep_key] = self.get_radial_bin_evo(
                comoving_radial_bin=scale,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        result = TwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (key, accumulation) for key, accumulation in sorted(
                    accumulations.items(), key=lambda x: float(x[0])
                )
            ),
            in_comoving=not return_in_physical,
            time_metric=wrt
        )
        
        # Restore original coordinate state so this method has no side effects
        if not original_in_comoving and self.in_comoving:
            self.convert_to_physical()
        
        return result

    def compute_freezeout_metrics(
            self,
            r_phys_grid: np.ndarray | None = None,
            r_phys_min: float = 0.2,
            r_phys_max: float = 50.0,
            n_phys_bins: int = 30,
            final_snap_id: int | None = None,
            scale_bands: dict[str, tuple[float, float]] | None = None,
            n_bands: int | None = None,
            min_scale_factor: float = 0.0,
            max_scale_factor: float = 100.0,
            use_treecorr: bool = False,
            treecorr_method: str = "jackknife",
            cross_patch_weight: str = "match",
            use_fixed_mask: bool = False,
            fixed_mask_min_scale_factor: float | None = None,
        ) -> FreezeOutMetrics:
        """Compute freeze-out distance metrics for correlation matrices.
        
        This method computes the evolution of the correlation matrix relative
        to its final state, on a fixed physical separation grid. By comparing
        at fixed physical (not comoving) scales, we isolate true dynamical
        freeze-out from the trivial effect of scale-factor drift.
        
        Parameters
        ----------
        r_phys_grid : np.ndarray | None
            Fixed physical separation grid (Mpc/h). If None, creates log-spaced
            grid from r_phys_min to r_phys_max with n_phys_bins bins.
        r_phys_min : float
            Minimum physical scale if creating grid. Default 0.2 Mpc/h.
        r_phys_max : float
            Maximum physical scale if creating grid. Default 50.0 Mpc/h.
        n_phys_bins : int
            Number of bins if creating grid. Default 30.
        final_snap_id : int | None
            Snapshot ID to use as the final reference. If None, uses the
            latest snapshot with valid folds data.
        scale_bands : dict[str, tuple[float, float]] | None
            Explicit band boundaries for hierarchical analysis.
            E.g. {"small": (0.2, 2.0), "mid": (2.0, 10.0), "large": (10.0, 50.0)}.
        n_bands : int | None
            If provided and scale_bands is None, create this many log-spaced bands.
        min_scale_factor : float
            Minimum scale factor to include. Default 0.0.
        max_scale_factor : float
            Maximum scale factor to include. Default 100.0.
        use_treecorr : bool
            If True and TreeCorr NNCorrelation objects are available, use
            TreeCorr's covariance estimation with cross-patch weighting.
            Default False (use folds-based estimation).
        treecorr_method : str
            TreeCorr variance method. Default 'jackknife'.
        cross_patch_weight : str
            TreeCorr cross-patch weighting scheme. Default 'match'
            (recommended by Mohammad & Percival 2022).
        use_fixed_mask : bool
            If True, use a fixed bin mask (intersection of valid bins across
            all epochs) to ensure Frobenius distances reflect only physical
            evolution, not changing bin validity. Default False.
        fixed_mask_min_scale_factor : float | None
            Minimum scale factor for epochs included in fixed mask computation.
            Useful for excluding early epochs with sparse coverage. Default None.
            
        Returns
        -------
        FreezeOutMetrics
            Container with freeze-out distance metrics including:
            - scale_factors: scale factor at each epoch
            - frobenius_global: normalized Frobenius distance to final
            - frobenius_bands: per-band distances (if bands specified)
            - fixed_mask_global: boolean mask used (if use_fixed_mask=True)
            - fixed_masks_bands: per-band masks (if use_fixed_mask=True)
            
        Raises
        ------
        ValueError
            If no valid folds data is available.
            
        Notes
        -----
        The key metric is the normalized Frobenius distance:
            D_F(a) = ||rho(a) - rho(a_f)||_F / ||rho(a_f)||_F
        
        For hierarchical freeze-out analysis, small physical scales should
        show D_F dropping/plateauing earlier than large scales.
        
        When ``use_treecorr=True``, this method uses TreeCorr's
        ``build_cov_design_matrix`` with cross-patch weighting for more
        accurate covariance estimation. This is recommended when TreeCorr
        data is available.
        
        When ``use_fixed_mask=True``, only bins that are valid across all
        epochs (or epochs above fixed_mask_min_scale_factor) are used.
        This removes artifacts from changing bin validity and provides
        a cleaner test of the freeze-out hypothesis.
        """
        scale_factors_dict = OrderedDict(
            (snap_id, self.moments[snap_id].scale_factor)
            for snap_id in self.data.keys()
        )
        
        # Check if TreeCorr is requested and available
        if use_treecorr:
            treecorr_nn_evo = self.treecorr_nn_evo
            if treecorr_nn_evo is None or all(nn is None for nn in treecorr_nn_evo.values()):
                import warnings
                warnings.warn(
                    "TreeCorr NNCorrelation objects not available, "
                    "falling back to folds-based estimation."
                )
            else:
                return compute_freezeout_metrics_treecorr(
                    treecorr_nn_evo=treecorr_nn_evo,
                    scale_factors=scale_factors_dict,
                    r_phys_grid=r_phys_grid,
                    r_phys_min=r_phys_min,
                    r_phys_max=r_phys_max,
                    n_phys_bins=n_phys_bins,
                    final_snap_id=final_snap_id,
                    scale_bands=scale_bands,
                    n_bands=n_bands,
                    min_scale_factor=min_scale_factor,
                    max_scale_factor=max_scale_factor,
                    method=treecorr_method,
                    cross_patch_weight=cross_patch_weight,
                    use_fixed_mask=use_fixed_mask,
                    fixed_mask_min_scale_factor=fixed_mask_min_scale_factor,
                )
        
        # Default: use folds-based estimation
        folds_evo = self.folds_array_evo
        if folds_evo is None:
            raise ValueError("No jackknife folds data available for freeze-out computation")
        
        return compute_freezeout_metrics(
            folds_evo=folds_evo,
            scale_factors=scale_factors_dict,
            r_phys_grid=r_phys_grid,
            r_phys_min=r_phys_min,
            r_phys_max=r_phys_max,
            n_phys_bins=n_phys_bins,
            final_snap_id=final_snap_id,
            scale_bands=scale_bands,
            n_bands=n_bands,
            min_scale_factor=min_scale_factor,
            max_scale_factor=max_scale_factor,
        )
        

@define(slots=True)
class MatterTwoPointCorrelation: 
    linear: TwoPointCorrelation 
    nonlinear: TwoPointCorrelation 
    in_comoving: bool = field(default=False)

    def __attrs_post_init__(self) -> None:
        # Sync child in_comoving states with parent
        self.linear.in_comoving = self.in_comoving
        self.nonlinear.in_comoving = self.in_comoving

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual underlying TwoPointCorrelation state."""
        self.in_comoving = self.nonlinear.in_comoving
        self.linear.in_comoving = self.in_comoving

    def __repr__(self) -> str:
        return (
            f"MatterTwoPointCorrelation(linear=xi("
            f"r={self.linear.radii.min():.3e}-{self.linear.radii.max():.3e}),"
            f" nonlinear=xi("
            f"r={self.nonlinear.radii.min():.3e}-{self.nonlinear.radii.max():.3e}"
            f"))"
        )
    
    @property
    def separation_scales(self) -> np.ndarray:
        return self.nonlinear.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.nonlinear.round_one_separation_scales
    
    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        # Force sync child states before conversion
        self.linear.in_comoving = False
        self.nonlinear.in_comoving = False
        self.linear.convert_to_comoving(scale_factor)
        self.nonlinear.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        # Force sync child states before conversion
        self.linear.in_comoving = True
        self.nonlinear.in_comoving = True
        self.linear.convert_to_physical(scale_factor)
        self.nonlinear.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def null_initialize(cls, arr_size: int = 0) -> MatterTwoPointCorrelation:
        return cls(
            linear=TwoPointCorrelation.null_initialize(arr_size, is_linear=True),
            nonlinear=TwoPointCorrelation.null_initialize(arr_size, is_linear=False)
        )

    @property
    def is_null(self) -> bool:
        return self.linear.is_null and self.nonlinear.is_null

    @property
    def nonlinear_to_linear(self) -> np.ndarray:
        """
        Compute the ratio of nonlinear to linear two-point correlation.
        
        Returns an array of shape (N, 2) with columns [separation, ratio].
        Returns array of NaNs if data is insufficient for interpolation.
        """
        # Check for null/empty data
        if self.is_null:
            return np.array([]).reshape(0, 2)
        
        lin_r = self.linear.radii
        lin_xi = self.linear.correlation
        nonlin_r = self.nonlinear.radii
        nonlin_xi = self.nonlinear.correlation
        
        # Need at least 2 points for interpolation
        if len(lin_r) < 2 or len(nonlin_r) < 2:
            return np.column_stack((nonlin_r, np.full_like(nonlin_r, np.nan)))
        
        # Filter out non-positive radii (can't take log)
        # For correlation, we allow negative values but need positive radii
        lin_valid = lin_r > 0
        if np.sum(lin_valid) < 2:
            return np.column_stack((nonlin_r, np.full_like(nonlin_r, np.nan)))
        
        lin_r_valid = lin_r[lin_valid]
        lin_xi_valid = lin_xi[lin_valid]
        
        # Build interpolator in log-r space (correlation can be negative, so don't log xi)
        log_interp_linear_xi = interp1d(
            np.log(lin_r_valid),
            lin_xi_valid,
            bounds_error=False,
            fill_value=np.nan  # Safer than extrapolate for edge cases
        )
        
        # Only interpolate where nonlinear r is positive
        nonlin_valid = nonlin_r > 0
        interp_lin_xi = np.full_like(nonlin_xi, np.nan)
        
        if np.any(nonlin_valid):
            interp_lin_xi[nonlin_valid] = log_interp_linear_xi(np.log(nonlin_r[nonlin_valid]))
        
        # Compute ratio, handling division by zero/nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = nonlin_xi / interp_lin_xi
            # Replace inf with nan for cleaner downstream handling
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)

        return np.column_stack((nonlin_r, ratio))
    
    @property
    def nonlinear_separation(self) -> float:
        """
        Use linear interpolation to get the maximum separation where 
        nonlinear to linear ratio is greater than or equal to 1.
        
        This is the "nonlinearity scale" for the correlation function - 
        below this scale, clustering is enhanced relative to linear theory.
        """
        ratio = self.nonlinear_to_linear
        
        # Check for empty or all-nan ratio
        if ratio.size == 0 or np.all(np.isnan(ratio[:, 1])):
            return np.nan
        
        # Filter valid entries
        valid = np.isfinite(ratio[:, 1]) & (ratio[:, 0] > 0)
        if np.sum(valid) < 2:
            return np.nan
        
        valid_r = ratio[valid, 0]
        valid_ratio = ratio[valid, 1]
        
        hi_res_r = np.logspace(
            np.log10(valid_r.min()),
            np.log10(valid_r.max()),
            1000
        )
        interp = interp1d(
            np.log10(valid_r), 
            valid_ratio, 
            bounds_error=False,
            fill_value=np.nan
        )
        hi_res_vals = np.asarray(interp(np.log10(hi_res_r)))
        
        # Find where ratio >= 1 (nonlinear enhancement)
        above_one = hi_res_vals >= 1.0
        if not np.any(above_one):
            return np.nan
        
        # Return the maximum separation where ratio >= 1
        return hi_res_r[above_one][-1]

    def deviation_separation(self, threshold: float = 1.1) -> float:
        """
        Find the separation where the boost factor xi_NL(r)/xi_lin(r)
        first drops below the given threshold (going from small to large r).
        
        This marks the onset of nonlinear deviations from linear theory,
        which occurs at smaller scales (smaller r) than the standard 
        nonlinear_separation definition where ratio = 1.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation from linear).
            Common choices:
            - 1.01: 1% deviation (very conservative)
            - 1.05: 5% deviation
            - 1.1: 10% deviation (default, typical choice)
            - 1.2: 20% deviation
        
        Returns
        -------
        float
            The separation r_dev where boost first drops below threshold
            (going from small to large r).
        """
        ratio = self.nonlinear_to_linear

        # Filter out NaN values before interpolation
        valid = np.isfinite(ratio[:, 0]) & np.isfinite(ratio[:, 1]) & (ratio[:, 0] > 0)
        if np.sum(valid) < 2:
            return np.nan

        r_valid = ratio[valid, 0]
        ratio_valid = ratio[valid, 1]

        # Interpolate in log-r space
        hi_res_r = np.logspace(
            np.log10(r_valid.min()),
            np.log10(r_valid.max()),
            1000
        )
        interp = interp1d(
            np.log10(r_valid), 
            ratio_valid, 
            bounds_error=False,
            fill_value=np.nan
        )
        hi_res_vals = np.asarray(interp(np.log10(hi_res_r)))

        # Find where ratio crosses the threshold from above to below
        # (going from small r to large r)
        above_threshold = hi_res_vals >= threshold
        below_threshold = hi_res_vals < threshold
        
        # Find crossing: above at index i and below at index i+1
        crossing = above_threshold[:-1] & below_threshold[1:]
        
        if np.any(crossing):
            cross_idx = np.where(crossing)[0][-1] + 1  # Take the largest r crossing
            return hi_res_r[cross_idx]
        elif np.all(above_threshold):
            # Always above threshold (fully nonlinear at all scales)
            return hi_res_r[-1]
        elif np.all(below_threshold):
            # Always below threshold (still linear at all scales)
            return np.nan
        else:
            return hi_res_r[below_threshold][-1] if np.any(below_threshold) else np.nan

    @classmethod
    def from_power_spectrum_data(
            cls,
            power_data: MatterPowerSpectrum,
            ell: int = 0,
            extrapolate: bool = True
        ) -> MatterTwoPointCorrelation:
    
        
        return cls(
            linear=TwoPointCorrelation.from_power_spectrum(
                wavenumbers=power_data.linear.wavenumbers,
                amplitudes=power_data.linear.amplitudes,
                ell=ell,
                extrapolate=extrapolate,
                is_linear=True
            ),
            nonlinear=TwoPointCorrelation.from_power_spectrum(
                wavenumbers=power_data.nonlinear.wavenumbers,
                amplitudes=power_data.nonlinear.amplitudes,
                ell=ell,
                extrapolate=extrapolate,
                is_linear=False
            )
        )
    
    def display(
            self, text_size: int = 16,
            use_scaled: bool = False,
            show_nonlinear: bool = True,
            show_linear: bool = True,
            ax: plt.Axes | None = None,
            return_fig: bool = False,
            color: str | tuple | None = None,
            show_legend: bool = True
        ) -> plt.Axes | None:

        attr_value = "scaled" if use_scaled else "correlation"

        ax = display_matter_correlation(
            linear_radii=self.linear.radii if show_linear else None,
            linear_correlations=getattr(self.linear, attr_value) if show_linear else None,
            nonlinear_radii=self.nonlinear.radii if show_nonlinear else None,
            nonlinear_correlations=getattr(self.nonlinear, attr_value) if show_nonlinear else None,
            is_scaled=use_scaled,
            text_size=text_size,
            ax=ax,
            return_fig=True,
            color=color,
            show_legend=show_legend
        )

        if return_fig:
            return ax
        
        plt.show()

@define(slots=True)
class MatterTwoPointCorrelationData:
    estimate: MatterTwoPointCorrelation
    error: MatterTwoPointCorrelation
    correlation_matrix: np.ndarray = field(repr=False)
    in_comoving: bool = field(default=False)

    folds: np.ndarray | None = field(default=None, repr=False)
    treecorr_nn: treecorr.NNCorrelation | None = field(default=None, repr=False)
    
    def __attrs_post_init__(self) -> None:
        # Sync child in_comoving states with parent
        self.estimate.in_comoving = self.in_comoving
        self.error.in_comoving = self.in_comoving

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual underlying TwoPointCorrelation state."""
        # The estimate.nonlinear is a TwoPointCorrelation - check its actual state
        self.in_comoving = self.estimate.nonlinear.in_comoving
        # Also sync the MatterTwoPointCorrelation wrappers
        self.estimate.in_comoving = self.in_comoving
        self.error.in_comoving = self.in_comoving

    @property
    def linear(self) -> TwoPointCorrelationData:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return TwoPointCorrelationData(
            estimate=self.estimate.linear,
            error=self.error.linear,
            correlation_matrix=np.eye(self.estimate.linear.radii.size),
            in_comoving=self.in_comoving,
            is_linear=True,
        )
    
    @property
    def nonlinear(self) -> TwoPointCorrelationData:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return TwoPointCorrelationData(
            estimate=self.estimate.nonlinear,
            error=self.error.nonlinear,
            correlation_matrix=self.correlation_matrix,
            in_comoving=self.in_comoving,
            folds=self.folds,
            is_linear=False,
            treecorr_nn=self.treecorr_nn
        )
    
    @property
    def separation_scales(self) -> np.ndarray:
        return self.estimate.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.estimate.round_one_separation_scales

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        # Force sync child states before conversion
        self.estimate.in_comoving = False
        self.error.in_comoving = False
        self.estimate.convert_to_comoving(scale_factor)
        self.error.convert_to_comoving(scale_factor)
        if self.folds is not None:
            self.folds[:, 0] = self.folds[:, 0] / scale_factor
        self.in_comoving = True
    
    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        # Force sync child states before conversion
        self.estimate.in_comoving = True
        self.error.in_comoving = True
        self.estimate.convert_to_physical(scale_factor)
        self.error.convert_to_physical(scale_factor)
        if self.folds is not None:
            self.folds[:, 0] = self.folds[:, 0] * scale_factor
        self.in_comoving = False

    @classmethod
    def from_power_spectrum_data(
            cls,
            power_data: MatterPowerSpectrumData,
            ell: int = 0,
            extrapolate: bool = True,
            in_comoving: bool = False
        ) -> MatterTwoPointCorrelationData:

        num_vals = power_data.estimate.nonlinear.wavenumbers.size
        return cls(
            estimate=MatterTwoPointCorrelation.from_power_spectrum_data(
                power_data=power_data.estimate,
                ell=ell,
                extrapolate=extrapolate
            ),
            error=MatterTwoPointCorrelation.null_initialize(num_vals), 
            correlation_matrix=np.eye(num_vals),
            in_comoving=in_comoving,
        )


    @classmethod
    def null_initialize(cls, arr_size: int = 0) -> MatterTwoPointCorrelationData:
        return cls(
            estimate=MatterTwoPointCorrelation.null_initialize(arr_size),
            error=MatterTwoPointCorrelation.null_initialize(arr_size),
            correlation_matrix=np.eye(arr_size)
        )

    @property
    def has_null_errors(self) -> bool:
        return self.error.is_null

    @property
    def is_null(self) -> bool:
        return self.estimate.is_null

    @property
    def nonlinear_to_linear(self) -> np.ndarray:
        """
        Compute the ratio of nonlinear to linear two-point correlation (from estimate).
        
        Delegates to the estimate's `nonlinear_to_linear` property.
        """
        return self.estimate.nonlinear_to_linear
    
    @property
    def nonlinear_separation(self) -> float:
        """
        Get the nonlinearity scale from the estimate.
        
        Delegates to the estimate's `nonlinear_separation` property.
        """
        return self.estimate.nonlinear_separation

    def deviation_separation(self, threshold: float = 1.1) -> float:
        """
        Find the separation where the boost factor first drops below threshold.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        float
            The separation r_dev where boost first drops below threshold.
        """
        return self.estimate.deviation_separation(threshold=threshold)
    
    @classmethod
    def from_tpcf_data(
            cls,
            tpcf_data: TwoPointCorrelationData,
            linear_ps_data: PowerSpectrumData
        ) -> MatterTwoPointCorrelationData:

    
        linear_tpcf = TwoPointCorrelation.from_power_spectrum(
            wavenumbers=linear_ps_data.estimate.wavenumbers,
            amplitudes=linear_ps_data.estimate.amplitudes,
            ell=0,
            extrapolate=True,
            is_linear=True
        )

        return cls(
            estimate=MatterTwoPointCorrelation(
                linear=linear_tpcf,
                nonlinear=tpcf_data.estimate
            ),
            error=MatterTwoPointCorrelation(
                linear=TwoPointCorrelation.null_initialize(
                    arr_size=linear_tpcf.radii.size, 
                    is_linear=True
                ),
                nonlinear=tpcf_data.error
            ),
            correlation_matrix=tpcf_data.correlation_matrix,
            in_comoving=tpcf_data.in_comoving,
            folds=tpcf_data.folds,
            treecorr_nn=tpcf_data.treecorr_nn
        )


@define(slots=True)
class MatterTwoPointCorrelationEvo(EvolutionData):
    data: OrderedDict[int, MatterTwoPointCorrelation]
    in_comoving: bool = field(default=False)

    colormap_name: str = field(default="viridis", repr=False)
    colormap: plt.cm.ColorMap = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.colormap = plt.cm.get_cmap(self.colormap_name, len(self.data))

    def __repr__(self) -> str:  
        return super().__repr__()

    @property
    def final_matter_correlation(self) -> MatterTwoPointCorrelation:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null correlation data available")

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return  # Already in comoving coordinates
        for v in self.data.values():
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return  # Already in physical coordinates
        for v in self.data.values():
            v.convert_to_physical(scale_factor)
        self.in_comoving = False


    @classmethod
    def from_power_spectrum_evo_data(
            cls,
            power_data: MatterPowerSpectrumEvo,
            ell: int = 0,
            extrapolate: bool = True
        ) -> MatterTwoPointCorrelationEvo:
        
        if len(power_data.data) == 0:
            raise ValueError("No data found")
        
        data = {}
        for snap_id, power_spec_data in power_data.data.items():
            try:
                data[snap_id] = MatterTwoPointCorrelation.from_power_spectrum_data(
                    power_data=power_spec_data, 
                    ell=ell, 
                    extrapolate=extrapolate
                )
            except Exception as e:
                print(f"Excluding snap {snap_id}: {e}")
            # except ValueError:
                continue

        if not data: raise ValueError("No valid data found")

        return cls(
            moments = power_data.moments.get_subset_by_attribute(
                key_attr="snapshot_id",
                attr_value=sorted(data.keys())
            ), 
            data = OrderedDict(sorted(data.items())),
            in_comoving = power_data.in_comoving
        )
    

    @property
    def linear_correlation_evo(self) -> TwoPointCorrelationEvo:
        return TwoPointCorrelationEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.linear) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving,
            is_linear=True
        )
    
    @property
    def nonlinear_correlation_evo(self) -> TwoPointCorrelationEvo:
        return TwoPointCorrelationEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.nonlinear) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving,
            is_linear=False
        )
    
    def get_interpolated_xi_evo(
            self,
            radii: float,
            component: str = "nonlinear"
        ) -> OrderedDict[int, float]:

        if component not in {"linear", "nonlinear"}:
            raise ValueError("component must be 'linear' or 'nonlinear'")

        evo_dict = OrderedDict()
        for key, value in self.data.items():
            if value.is_null:
                continue

            try:
                evo_dict[key] = float(
                    getattr(value, component).get_interpolated_xi(radii)
                )
            except ValueError:
                evo_dict[key] = np.nan

        return evo_dict
    

    def get_matter_tpcf_evo_array(
            self,
            radii: float,
            wrt: str = "scale_factor"
        ) -> np.ndarray:
        ''' (N,2) array where coloumn 0 is the time variable and column 1 is xi at that time'''
        
        linear_evo_vals = self.get_interpolated_xi_evo(radii, component="linear")
        nonlinear_evo_vals = self.get_interpolated_xi_evo(radii, component="nonlinear")
        evo_vals = {
            key : (linear_evo_vals[key], nonlinear_evo_vals[key]) 
            for key in linear_evo_vals.keys()
            if key in nonlinear_evo_vals
        }
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.column_stack([
            (time_map[key], val[0], val[1]) for key, val in evo_vals.items()
        ]).T 


    def get_matter_weighted_tpcf_evo_array(
            self,
            radii: float,
            wrt: str = "scale_factor"
        ) -> np.ndarray:
        ''' (N,2) array where coloumn 0 is the time variable and column 1 is r^2 * xi at that time'''
        
        linear_evo_vals = self.get_interpolated_xi_evo(radii, component="linear")
        nonlinear_evo_vals = self.get_interpolated_xi_evo(radii, component="nonlinear")
        evo_vals = {
            key : (linear_evo_vals[key] * radii**2, nonlinear_evo_vals[key] * radii**2) 
            for key in linear_evo_vals.keys()
            if key in nonlinear_evo_vals
        }
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.column_stack([
            (time_map[key], val[0], val[1]) for key, val in evo_vals.items()
        ]).T
    

    def get_matter_tpcf_evo_rate(
            self,
            radii: float,
            wrt: str = "scale_factor"
        ) -> np.ndarray:
        ''' (N, 3) array that is the dln(xi)/dln(time) for linear and nonlinear '''
        
        tpcf_evo = self.get_matter_tpcf_evo_array(radii, wrt=wrt)
        linear_rate = np.gradient(
            np.log(tpcf_evo[:, 1]), 
            np.log(tpcf_evo[:, 0]), 
            edge_order=2
        )
        nonlinear_rate = np.gradient(
            np.log(tpcf_evo[:, 2]), 
            np.log(tpcf_evo[:, 0]), 
            edge_order=2
        )
        return np.column_stack([tpcf_evo[:, 0], linear_rate, nonlinear_rate]).T 
    
    def display(
            self,
            target_scale_factors: Collection[float] | None = None,
            text_size: int = 16,
            use_scaled: bool = False,
            show_linear: bool = True,
            show_nonlinear: bool = True,
            return_fig: bool = False,
            ax_main: plt.Axes | None = None,
            legend_xloc: float = 1.05,
            legend_yloc: float = 0.92,
            colormap: plt.cm.ColorMap | None = None
        ) -> plt.Axes | None:

        if ax_main is None:
            _, ax_main = plt.subplots()

        if all([not show_linear, not show_nonlinear]):
            raise ValueError(
                'At least one of show_linear or show_nonlinear must be True'
            )

        if target_scale_factors is None:
            target_scale_factors = self.moments.scale_factors
            colormap = self.colormap if colormap is None else colormap
        else:
            colormap = (
                plt.cm.get_cmap("viridis", len(target_scale_factors))
                if colormap is None
                else colormap
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

        handles = []


        for i, snap_id  in enumerate(target_snapshot_ids):
            ax_main = self[snap_id].display(
                text_size=text_size,
                use_scaled=use_scaled,
                show_linear=show_linear,
                show_nonlinear=show_nonlinear,
                ax=ax_main,
                return_fig=True,
                color=colormap(i),
                show_legend=False
            )

            handles.append(
                plt.Line2D(
                    [], [], 
                    color=colormap(i),
                    label=f"$a={actual_scale_factors[i]:.2f}$"
                )
            )

        ax_main.legend(
            handles=handles,
            bbox_to_anchor=(legend_xloc, legend_yloc), 
            loc='upper left', fontsize=text_size
        )

        return ax_main if return_fig else None

    @property
    def nonlinear_separation_evo(self) -> OrderedDict[int, float]:
        """
        Get the nonlinearity scale at each snapshot.
        
        Returns an OrderedDict mapping snapshot_id to nonlinear_separation.
        """
        return OrderedDict(
            (key, value.nonlinear_separation) 
            for key, value in self.data.items() if not value.is_null
        )

    def deviation_separation_evo(self, threshold: float = 1.1) -> OrderedDict[int, float]:
        """
        Get the deviation separation (where boost first drops below threshold)
        for each snapshot.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        OrderedDict[int, float]
            Mapping from snapshot index to deviation separation.
        """
        return OrderedDict(
            (key, value.deviation_separation(threshold=threshold)) 
            for key, value in self.data.items() if not value.is_null
        )
    
    @property
    def nonlinear_to_linear_evo(self) -> OrderedDict[int, np.ndarray]:
        """
        Get the nonlinear-to-linear ratio at each snapshot.
        
        Returns an OrderedDict mapping snapshot_id to the ratio array.
        """
        return OrderedDict(
            (key, value.nonlinear_to_linear) 
            for key, value in self.data.items() if not value.is_null
        )
    
@define(slots=True)
class MatterTwoPointCorrelationEvoData(EvolutionData):
    data: OrderedDict[int, MatterTwoPointCorrelationData]
    in_comoving: bool = field(default=False)

    colormap_name: str = field(default="viridis", repr=False)
    colormap: plt.cm.ColorMap = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        # Sync child in_comoving states to match parent's state
        # This ensures that when the parent is constructed with a specific
        # in_comoving value, all children are updated to match
        for v in self.data.values():
            if not v.is_null:
                v.in_comoving = self.in_comoving
                # Also sync the underlying TwoPointCorrelation objects
                v.estimate.in_comoving = self.in_comoving
                v.error.in_comoving = self.in_comoving
                v.estimate.linear.in_comoving = self.in_comoving
                v.estimate.nonlinear.in_comoving = self.in_comoving
                v.error.linear.in_comoving = self.in_comoving
                v.error.nonlinear.in_comoving = self.in_comoving

    def __repr__(self) -> str:  
        return super().__repr__()
    
    @property
    def final_matter_tpcf_data(self) -> MatterTwoPointCorrelationData:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null correlation data available")

    def _sync_comoving_state(self) -> None:
        """Sync in_comoving flag from actual child data state."""
        if (non_null_data := [v for v in self.data.values() if not v.is_null]):
            # First sync each child's state from underlying TwoPointCorrelation
            for v in non_null_data:
                v._sync_comoving_state()
            # Then use the first non-null child's state as ground truth
            self.in_comoving = non_null_data[0].in_comoving

    def convert_to_comoving(self) -> None:
        # Sync state from children first to handle cases where data was
        # converted via a different accessor path
        self._sync_comoving_state()
        if self.in_comoving:
            return  # Already in comoving coordinates
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force sync child state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        # Sync state from children first
        self._sync_comoving_state()
        if not self.in_comoving:
            return  # Already in physical coordinates
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force sync child state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False
        
    @property
    def separation_scales(self) -> np.ndarray:
        return self.final_matter_tpcf_data.estimate.separation_scales
    

    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.final_matter_tpcf_data.estimate.round_one_separation_scales

    @property
    def nonlinear_separation_evo(self) -> OrderedDict[int, float]:
        """
        Get the nonlinearity scale at each snapshot.
        
        Returns an OrderedDict mapping snapshot_id to nonlinear_separation.
        """
        return OrderedDict(
            (key, value.nonlinear_separation) 
            for key, value in self.data.items() if not value.is_null
        )

    def deviation_separation_evo(self, threshold: float = 1.1) -> OrderedDict[int, float]:
        """
        Get the deviation separation (where boost first drops below threshold)
        for each snapshot.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        OrderedDict[int, float]
            Mapping from snapshot index to deviation separation.
        """
        return OrderedDict(
            (key, value.deviation_separation(threshold=threshold)) 
            for key, value in self.data.items() if not value.is_null
        )
    
    @property
    def nonlinear_to_linear_evo(self) -> OrderedDict[int, np.ndarray]:
        """
        Get the nonlinear-to-linear ratio at each snapshot.
        
        Returns an OrderedDict mapping snapshot_id to the ratio array.
        """
        return OrderedDict(
            (key, value.nonlinear_to_linear) 
            for key, value in self.data.items() if not value.is_null
        )

    @property
    def linear_evo(self) -> TwoPointCorrelationEvoData:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return TwoPointCorrelationEvoData(
            moments=self.moments,
            data=OrderedDict(
                (key, value.linear) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving,
            is_linear=True
        )
    
    @property
    def nonlinear_evo(self) -> TwoPointCorrelationEvoData:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return TwoPointCorrelationEvoData(
            moments=self.moments,
            data=OrderedDict(
                (key, value.nonlinear) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving,
            is_linear=False
        )
    
    @property
    def estimate(self) -> MatterTwoPointCorrelationEvo:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return MatterTwoPointCorrelationEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.estimate) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def error(self) -> MatterTwoPointCorrelationEvo:
        # Sync state before creating wrapper to ensure correct in_comoving flag
        self._sync_comoving_state()
        return MatterTwoPointCorrelationEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.error) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def correlation_matrices(self) -> OrderedDict[int, np.ndarray]:
        return OrderedDict(
            (key, value.correlation_matrix) for key, value in self.data.items()
        )


    @property
    def folds_evo(self) -> OrderedDict[int, np.ndarray | None]:
        return OrderedDict(
            (key, value.folds) for key, value in self.data.items()
        )

    @property
    def fold_instances_evo(self) -> OrderedDict[int, TwoPointCorrelationEvo]:
        nonlinear_evo = self.nonlinear_evo
        output: OrderedDict[int, TwoPointCorrelationEvo] = OrderedDict()
        for key, value in nonlinear_evo.data.items():
            if value.folds is None:
                continue
            for fold_idx, fold_tpcf in value.fold_instances.items():
                if fold_idx not in output:
                    output[fold_idx] = TwoPointCorrelationEvo(
                        moments=self.moments,
                        data=OrderedDict(),
                        in_comoving=self.in_comoving
                    )
                output[fold_idx].data[key] = fold_tpcf
        return output

    @property
    def treecorr_nn_evo(self) -> OrderedDict[int, treecorr.NNCorrelation] | None:
        treecorr_evo = OrderedDict()

        for snap_id, tpcf_data in self.data.items():
            if tpcf_data.treecorr_nn is None:
                continue
            treecorr_evo[snap_id] = tpcf_data.treecorr_nn

        return treecorr_evo or None
    
    @classmethod
    def from_power_spectrum_evo_data(
            cls,
            power_data: MatterPowerSpectrumEvoData,
            ell: int = 0,
            extrapolate: bool = True,
            in_comoving: bool = True
        ) -> MatterTwoPointCorrelationEvoData:

        return cls(
            moments=power_data.moments,
            data=OrderedDict(
                (
                    key, 
                    MatterTwoPointCorrelationData.from_power_spectrum_data(
                        value, 
                        ell=ell, 
                        extrapolate=extrapolate
                    )
                )
                for key, value in power_data.data.items()
            ),
            in_comoving=in_comoving
        )
    
    @classmethod
    def from_tpcf_evo_data(
            cls,
            tpcf_evo_data: TwoPointCorrelationEvoData,
            linear_ps_evo_data: PowerSpectrumDataEvo
        ) -> MatterTwoPointCorrelationEvoData:
        
        return cls(
            moments=tpcf_evo_data.moments,
            data=OrderedDict(
                (
                    key, 
                    MatterTwoPointCorrelationData.from_tpcf_data(
                        tpcf_data=value, 
                        linear_ps_data=linear_ps_evo_data[key]
                    )
                )
                for key, value in tpcf_evo_data.data.items()
                if key in linear_ps_evo_data.data
            ),
            in_comoving=tpcf_evo_data.in_comoving
        )

    def get_radial_bin_evo(
            self,
            comoving_radial_bin: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0
        ) -> MatterTwoPointCorrelationAccumulation:
        """Return the evolution of xi at a given comoving radius for both linear and nonlinear.

        This is a thin wrapper around the corresponding `TwoPointCorrelationEvoData` method for
        the linear and nonlinear components.
        """

        # Build the component evo objects ONCE to avoid repeatedly constructing new wrappers
        # that can trigger repeated in-place conversions of shared underlying data.
        linear_evo = self.linear_evo
        nonlinear_evo = self.nonlinear_evo

        linear_accum = linear_evo.get_radial_bin_evo(
            comoving_radial_bin=comoving_radial_bin,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
        )

        nonlinear_accum = nonlinear_evo.get_radial_bin_evo(
            comoving_radial_bin=comoving_radial_bin,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
        )

        return MatterTwoPointCorrelationAccumulation(
            linear=linear_accum,
            nonlinear=nonlinear_accum,
            in_comoving=nonlinear_accum.in_comoving,
            time_metric=wrt,
        )


    def get_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> MatterTwoPointCorrelationAccumulations:
        """Get xi evolution at all separation scales for both linear and nonlinear.

        This method delegates the heavy lifting to the component `TwoPointCorrelationEvoData`
        implementations (which already handle fold logic, filtering, and unit conversion).

        Notes
        -----
        We intentionally *do not* iterate over scales and call `self.get_radial_bin_evo(...)`
        repeatedly, because that would repeatedly construct component evo wrappers and can
        cause repeated in-place unit conversions of shared underlying data.
        """

        linear_evo = self.linear_evo
        nonlinear_evo = self.nonlinear_evo

        linear_accums = linear_evo.get_separation_scale_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format,
        )

        nonlinear_accums = nonlinear_evo.get_separation_scale_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format,
        )

        return MatterTwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (
                    bin_key, MatterTwoPointCorrelationAccumulation(
                        linear=linear_accums.accumulations.get(
                            bin_key, 
                            TwoPointCorrelationAccumulation.null_initialize()
                        ),
                        nonlinear=nonlinear, 
                        in_comoving=not return_in_physical,
                        time_metric=wrt
                    )
                )
                for bin_key, nonlinear in nonlinear_accums.accumulations.items()
            ),
            in_comoving=not return_in_physical,
            time_metric=wrt
        )


    def get_round_one_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> MatterTwoPointCorrelationAccumulations:
        """Get xi evolution at all round-one separation scales for both linear and nonlinear.

        See `get_separation_scale_accumulations` for implementation notes.
        """

        linear_evo = self.linear_evo
        nonlinear_evo = self.nonlinear_evo

        linear_accums = linear_evo.get_round_one_separation_scale_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format,
        )

        nonlinear_accums = nonlinear_evo.get_round_one_separation_scale_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format,
        )

        return MatterTwoPointCorrelationAccumulations(
            accumulations=OrderedDict(
                (
                    bin_key, MatterTwoPointCorrelationAccumulation(
                        linear=linear_accums.accumulations.get(
                            bin_key, 
                            TwoPointCorrelationAccumulation.null_initialize()
                        ),
                        nonlinear=nonlinear, 
                        in_comoving=not return_in_physical,
                        time_metric=wrt
                    )
                )
                for bin_key, nonlinear in nonlinear_accums.accumulations.items()
            ),
            in_comoving=not return_in_physical,
            time_metric=wrt
        )

    def compute_freezeout_metrics(
            self,
            r_phys_grid: np.ndarray | None = None,
            r_phys_min: float = 0.2,
            r_phys_max: float = 50.0,
            n_phys_bins: int = 30,
            final_snap_id: int | None = None,
            scale_bands: dict[str, tuple[float, float]] | None = None,
            n_bands: int | None = None,
            min_scale_factor: float = 0.0,
            max_scale_factor: float = 100.0,
            use_treecorr: bool = False,
            treecorr_method: str = "jackknife",
            cross_patch_weight: str = "match",
            use_fixed_mask: bool = False,
            fixed_mask_min_scale_factor: float | None = None,
        ) -> FreezeOutMetrics:
        """Compute freeze-out distance metrics for the nonlinear correlation matrix.
        
        This is a convenience wrapper that delegates to the nonlinear component's
        freeze-out computation. Since freeze-out is a nonlinear phenomenon,
        we use the nonlinear correlation matrices.
        
        Parameters
        ----------
        r_phys_grid : np.ndarray | None
            Fixed physical separation grid (Mpc/h). If None, creates log-spaced
            grid from r_phys_min to r_phys_max with n_phys_bins bins.
        r_phys_min : float
            Minimum physical scale if creating grid. Default 0.2 Mpc/h.
        r_phys_max : float
            Maximum physical scale if creating grid. Default 50.0 Mpc/h.
        n_phys_bins : int
            Number of bins if creating grid. Default 30.
        final_snap_id : int | None
            Snapshot ID to use as the final reference. If None, uses the
            latest snapshot with valid folds data.
        scale_bands : dict[str, tuple[float, float]] | None
            Explicit band boundaries for hierarchical analysis.
            E.g. {"small": (0.2, 2.0), "mid": (2.0, 10.0), "large": (10.0, 50.0)}.
        n_bands : int | None
            If provided and scale_bands is None, create this many log-spaced bands.
        min_scale_factor : float
            Minimum scale factor to include. Default 0.0.
        max_scale_factor : float
            Maximum scale factor to include. Default 100.0.
        use_treecorr : bool
            If True and TreeCorr NNCorrelation objects are available, use
            TreeCorr's covariance estimation with cross-patch weighting.
            Default False (use folds-based estimation).
        treecorr_method : str
            TreeCorr variance method. Default 'jackknife'.
        cross_patch_weight : str
            TreeCorr cross-patch weighting scheme. Default 'match'
            (recommended by Mohammad & Percival 2022).
        use_fixed_mask : bool
            If True, use a fixed bin mask (intersection of valid bins across
            all epochs) for consistent comparison. Default False.
        fixed_mask_min_scale_factor : float | None
            Minimum scale factor for epochs included in fixed mask computation.
            
        Returns
        -------
        FreezeOutMetrics
            Container with freeze-out distance metrics.
            
        See Also
        --------
        TwoPointCorrelationEvoData.compute_freezeout_metrics : 
            Full documentation of the underlying computation.
        """
        return self.nonlinear_evo.compute_freezeout_metrics(
            r_phys_grid=r_phys_grid,
            r_phys_min=r_phys_min,
            r_phys_max=r_phys_max,
            n_phys_bins=n_phys_bins,
            final_snap_id=final_snap_id,
            scale_bands=scale_bands,
            n_bands=n_bands,
            min_scale_factor=min_scale_factor,
            max_scale_factor=max_scale_factor,
            use_treecorr=use_treecorr,
            treecorr_method=treecorr_method,
            cross_patch_weight=cross_patch_weight,
            use_fixed_mask=use_fixed_mask,
            fixed_mask_min_scale_factor=fixed_mask_min_scale_factor,
        )
    

def group_halo_halo_correlations_by_property(
        comoving_positions: np.ndarray,
        prop_values: np.ndarray,
        resolved_mask: np.ndarray,
        value_bins: dict[int | str, tuple[float, float]],
        r_min: float,
        r_max: float,
        num_radial_bins: int,
        comoving_box_size: float,
        scale_factor: float,
        sub_box_info: dict | None = None,
        use_natural: bool = False,
        use_jackknifed: bool = False,
        random_multiplier: int | None = None,
        rng_seed: int | None = None,
        eps: float = 1e-5,
        return_in_comoving: bool = False,
        return_folds: bool = False,
    ) -> dict[int | str, TwoPointCorrelationData]:

    correlations: dict[int | str, TwoPointCorrelationData] = {}
    for prop_value_bin, (prop_value_min, prop_value_max) in value_bins.items():

        prop_bin_mask = np.logical_and(
            prop_values >= prop_value_min,
            prop_values < prop_value_max
        )
        position_mask = np.logical_and(resolved_mask, prop_bin_mask)

        target_positions = comoving_positions[position_mask]
        if target_positions.shape[0] < 2:
            print(f"Skipping {prop_value_bin = :.1f} due to insufficient sample size")
            continue

        print(
            f"Computing correlations for prop value bin {prop_value_bin = :.1f} with "
            f"{target_positions.shape[0]} objects"
        )

        # Build a TwoPointCorrelationData object for this prop value bin
        tpcf_data = TwoPointCorrelationData.from_data(
            comoving_coordinates=target_positions,
            rmin=r_min,
            rmax=r_max,
            nbins=num_radial_bins,
            boxsize=comoving_box_size,
            sub_box_info=(sub_box_info if use_jackknifed else None),
            use_natural=use_natural,
            run_jackknife=use_jackknifed,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
            eps=eps,
            return_folds=return_folds
        )

        if not return_in_comoving:
            tpcf_data.convert_to_physical(scale_factor)

        correlations[prop_value_bin] = tpcf_data

    return correlations


def save_two_point_correlation(
        filepath: Path, 
        data: dict[str, np.ndarray | dict[str, np.ndarray | bool]]
    ) -> None:
    """
    Save a TwoPointCorrelationData-like dictionary to HDF5.

    Required keys in `data`:
      - "estimate": {"radii", "correlation", ["is_linear"], ["in_comoving"]}
      - "error":    {"radii", "correlation", ["is_linear"], ["in_comoving"]}
      - "correlation_matrix": (nbins, nbins) array
      - "in_comoving": bool

    Optional (either one is fine):
      - "folds": (nbins, 1+Nfolds) array where col0 is radii, remaining columns are per-fold xi
      - "folds_as_dict": {"radii": (nbins,), "folds": {int -> (nbins,) array}}
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(filepath), "w") as f:
        # convenience top-level flag
        f.attrs["in_comoving"] = bool(data.get("in_comoving", False))

        def _write_group(gname: str, block: dict[str, np.ndarray | bool]) -> None:
            g = f.create_group(gname)
            g.create_dataset("radii", data=np.asarray(block["radii"]))
            g.create_dataset("correlation", data=np.asarray(block["correlation"]))
            if "is_linear" in block:
                g.attrs["is_linear"] = bool(block["is_linear"])
            if "in_comoving" in block:
                g.attrs["in_comoving"] = bool(block["in_comoving"])

        _write_group("estimate", data["estimate"])  # type: ignore[index]
        _write_group("error",    data["error"])     # type: ignore[index]

        f.create_dataset("correlation_matrix", data=np.asarray(data["correlation_matrix"]))  # type: ignore[index]

        # Optional: compact matrix form
        if "folds" in data and data["folds"] is not None:  # type: ignore[index]
            f.create_dataset("folds", data=np.asarray(data["folds"]))  # type: ignore[index]

        # Optional: structured dict form
        elif "folds_as_dict" in data and data["folds_as_dict"] is not None:  # type: ignore[index]
            fad = data["folds_as_dict"]  # type: ignore[index]
            g = f.create_group("folds_as_dict")
            g.create_dataset("radii", data=np.asarray(fad["radii"]))  # type: ignore[index]
            gfolds = g.create_group("folds")
            for k in sorted(fad["folds"].keys(), key=lambda x: int(x)):  # type: ignore[index]
                gfolds.create_dataset(str(k), data=np.asarray(fad["folds"][k]))  # type: ignore[index]


def _write_two_point_group(
        h5: h5py.File, 
        group_name: str, 
        branch: dict[str, np.ndarray]
    ) -> None:
    """Write a branch (with keys 'radii' and 'correlation') into an HDF5 group.

    Parameters
    ----------
    h5 : h5py.File or Group
        Open HDF5 file handle.
    group_name : str
        Name of the group to create/overwrite (e.g., 'estimate', 'error').
    branch : dict
        Dictionary expected to contain 'radii' and/or 'correlation' arrays.
    """
    grp = h5.require_group(group_name)
    if "radii" in branch:
        if "radii" in grp:
            del grp["radii"]
        grp.create_dataset("radii", data=np.asarray(branch["radii"]))
    if "correlation" in branch:
        if "correlation" in grp:
            del grp["correlation"]
        grp.create_dataset("correlation", data=np.asarray(branch["correlation"]))

def load_two_point_correlation(filepath: Path, from_treecorr: bool=False) -> dict[
    str, np.ndarray | dict[str, np.ndarray | bool] | treecorr.NNCorrelation
]: 
    return (
        load_treecorr_tpcf(filepath) 
        if from_treecorr else 
        load_corrfunc_tpcf(filepath)
    )


def load_corrfunc_tpcf(filepath: Path) -> dict[str, np.ndarray | dict[str, np.ndarray | bool]]: 
    """
    Load from HDF5. Returns a dict suitable for TwoPointCorrelationData.load.
    If folds are present, returns:
      - "folds": (nbins, 1+Nfolds) array
      - and also "folds_as_dict" when the file stores the structured form.
    """
    out: dict[str, np.ndarray | dict[str, np.ndarray | bool]] = {}

    with h5py.File(str(filepath), "r") as f:
        def _read_group(gname: str) -> dict[str, np.ndarray | bool]:
            g = f[gname]
            d: dict[str, np.ndarray | bool] = {
                "radii":       np.array(g["radii"]),
                "correlation": np.array(g["correlation"]),
            }
            if "is_linear"   in g.attrs: d["is_linear"]   = bool(g.attrs["is_linear"])
            if "in_comoving" in g.attrs: d["in_comoving"] = bool(g.attrs["in_comoving"])
            return d

        out["estimate"] = _read_group("estimate")
        out["error"]    = _read_group("error")
        out["correlation_matrix"] = np.array(f["correlation_matrix"])
        out["in_comoving"] = bool(f.attrs.get("in_comoving", False))

        # Prefer compact 2D dataset if present
        if "folds" in f:
            out["folds"] = np.array(f["folds"])

        # Otherwise, reconstruct from structured form if available
        elif "folds_as_dict" in f:
            g = f["folds_as_dict"]
            radii = np.array(g["radii"])
            gfolds = g["folds"]
            keys = sorted(gfolds.keys(), key=lambda x: int(x))
            fold_cols = [np.array(gfolds[k]) for k in keys]
            if fold_cols:
                out["folds"] = np.column_stack([radii] + fold_cols)
            out["folds_as_dict"] = {
                "radii": radii,
                "folds": {int(k): np.array(gfolds[k]) for k in keys},
            }

    return out

def extract_from_treecorr_file(
        filepath: Path
    ) -> dict[str, np.ndarray | dict[str, np.ndarray | bool] | treecorr.NNCorrelation]: 
    
    out: dict[str, np.ndarray | dict[str, np.ndarray | bool]] = {}

    with h5py.File(str(filepath), "r") as f:

        radii = np.exp(f['main']['meanlogr'][:])
        xi_err = f['main']['sigma_xi'][:]

        out['estimate'] = {'radii' : radii, 'correlation' : f['main']['xi'][:]}
        out['error'] = {'radii' : radii, 'correlation' : xi_err}
        covariance = f['cov']['data'][:]
        C_ii = np.diag(covariance)
        normalization = np.sqrt(np.outer(C_ii, C_ii))
        out['correlation_matrix'] = np.divide(covariance, normalization)
        out["in_comoving"] = True

        # No folds for this, but reload treecorr object if needed
        out['treecorr_nn'] = treecorr.NNCorrelation.from_file(filepath)

    return out

def extract_treecorr_fold_data(
        treecorr_nn: treecorr.NNCorrelation,
        radii: np.ndarray,
    ) -> np.ndarray | None:

    # Build folds if available
    if len(treecorr_nn.results) <= 1: return None
    
    folds_list = []
    for key, dd in treecorr_nn.results.items():
        if key[0] != key[1]: continue
        dd._ro.var_method = 'shot'
        xi = dd.calculateXi(
            dr=treecorr_nn._dr.results[key], 
            rr=treecorr_nn._rr.results[key],
        )
        folds_list.append(xi[0])
    return np.column_stack([radii] + folds_list)

def load_treecorr_tpcf(filepath: Path) -> dict[
    str, np.ndarray | dict[str, np.ndarray | bool] | treecorr.NNCorrelation
]: 

    out = extract_from_treecorr_file(filepath)
    out['folds'] = extract_treecorr_fold_data(
        treecorr_nn=out['treecorr_nn'], 
        radii=out['estimate']['radii']
    )
        
    return out


def correlation_to_covariance(stdev: np.ndarray, corr_matrix: np.ndarray) -> np.ndarray:
    return np.diag(stdev) @ corr_matrix @ np.diag(stdev)


# Just try UnivariateSpline
def powerspec_to_correlation(
        wavenumbers: np.ndarray, 
        amplitudes: np.ndarray, 
        ell: int = 0,
        extrapolate: bool = True
    ) -> dict[str, np.ndarray]:
    ''' Modified from nbodykit implementation '''
    xi = mcfit.P2xi(wavenumbers, l=ell, lowring=True)
    rr, CF = xi(amplitudes, extrap=extrapolate)
    spline = UnivariateSpline(rr, CF)
    r_range = np.logspace(
        np.log10(rr.min()), np.log10(rr.max()), wavenumbers.size
    )

    return {"radii": r_range, "correlation": spline(r_range)}

def _safe_interpolate(tpcf_data: TwoPointCorrelationData, radial_bin: float) -> float:
    """Safely interpolate xi, returning NaN on failure."""
    try:
        return float(tpcf_data.get_interpolated_xi(radial_bin))
    except ValueError as err:
        return np.nan