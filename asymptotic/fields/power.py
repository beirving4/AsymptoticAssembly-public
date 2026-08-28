from __future__ import annotations

import mcfit
import numpy as np, pdb
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from attrs import define, field
from collections import OrderedDict
from collections.abc import Collection
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from ..cosmo.model import Cosmology
from ..simulation.evo import EvolutionData
from ..simulation.moments import Moment, MomentsInTime, get_correct_snapshot_mapping
from ..utils.get_data import (
    load_linear_ps, 
    get_power_spec, 
    get_power_spec_evo,
)
from .variance import get_mass_variance
from .accumulation import (
    AccumulationHistory,
    AccumulationHistories,
    MatterPowerSpectrumAccumulation,
    MatterPowerSpectraAccumulation
)
from .viz import (
    display_power_spectrum,
    display_matter_power_spectrum
)


@define(slots=True)
class PowerSpectrumValue:
    wavenumber: float
    amplitude: float
    normalized: float
    is_linear: bool = field(default=False)
    in_comoving: bool = field(default=True)

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        self.wavenumber *= scale_factor
        self.amplitude *= scale_factor**3
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        self.wavenumber /= scale_factor
        self.amplitude /= scale_factor**3
        self.in_comoving = False

@define(slots=True)
class MatterPowerSpectrumValue:
    linear: PowerSpectrumValue
    nonlinear: PowerSpectrumValue
    shot_noise: PowerSpectrumValue | None = field(default=None)
    in_comoving: bool = field(default=True)

    def __attrs_post_init__(self) -> None:
        # Sync children with parent's in_comoving state
        self.linear.in_comoving = self.in_comoving
        self.nonlinear.in_comoving = self.in_comoving
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = self.in_comoving

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.linear.in_comoving = False
        self.nonlinear.in_comoving = False
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = False
        self.linear.convert_to_comoving(scale_factor)
        self.nonlinear.convert_to_comoving(scale_factor)
        if self.shot_noise is not None:
            self.shot_noise.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.linear.in_comoving = True
        self.nonlinear.in_comoving = True
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = True
        self.linear.convert_to_physical(scale_factor)
        self.nonlinear.convert_to_physical(scale_factor)
        if self.shot_noise is not None:
            self.shot_noise.convert_to_physical(scale_factor)
        self.in_comoving = False


@define(slots=True)
class PowerSpectrum:  # add moment? 
    wavenumbers: np.ndarray
    amplitudes: np.ndarray
    normalized: np.ndarray
    is_linear: bool = field(default=False)
    in_comoving: bool = field(default=True)

    spline: interp1d | None = field(init=False)

    growth_factor: float = field(default=1.0)

    def __attrs_post_init__(self) -> None:
        # Remove duplicate wavenumbers
        unique_indices = np.unique(self.wavenumbers, return_index=True)[1]
        self.wavenumbers = self.wavenumbers[unique_indices]
        self.amplitudes = self.amplitudes[unique_indices]
        self.normalized = self.normalized[unique_indices]

        self._rebuild_spline()

    def __repr__(self) -> str:
        return (
            f"PowerSpectrum("
            f"k=({self.wavenumbers.min():.3e}, "
            f"{self.wavenumbers.max():.3e}), "
            f"Pk=({self.amplitudes.min():.3e}, "
            f"{self.amplitudes.max():.3e}), "
            f"is_linear={self.is_linear})"
        )
    
    def _rebuild_spline(self) -> None:
        try:
            self.spline = interp1d(
                np.log10(self.wavenumbers), 
                np.log10(self.normalized),
                fill_value="extrapolate",
                bounds_error=False
            )
        except ValueError:
            self.spline = None

    @property
    def wavenumber_range(self) -> tuple[float, float]:
        return (self.wavenumbers.min(), self.wavenumbers.max())
    
    def in_range(self, wavenumber: float) -> bool:
        return self.wavenumbers.min() <= wavenumber <= self.wavenumbers.max()

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        self.wavenumbers = self.wavenumbers * scale_factor
        self.amplitudes = self.amplitudes * scale_factor**3
        self.in_comoving = True
        self._rebuild_spline()

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        self.wavenumbers = self.wavenumbers / scale_factor
        self.amplitudes = self.amplitudes / scale_factor**3
        self.in_comoving = False
        self._rebuild_spline()

    @classmethod
    def from_linear_ps_file(
            cls, file_path: Path, growth_factor: float, use_music: bool = False
        ) -> PowerSpectrum:
        return cls(
            # Need to change this ... 
            **load_linear_ps(file_path, growth_factor, use_music=use_music),
            is_linear=True,
            in_comoving=True
        )

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            wavenumbers: np.ndarray,
            redshift: float = 0.0,
        ) -> PowerSpectrum:

        growth_factor = cosmo.linear_growth_factor(z=redshift, normalize=True)
        Pk = cosmo.linear_matter_ps(k=wavenumbers, redshift=redshift)
        return cls(
            wavenumbers=wavenumbers,
            amplitudes=Pk,
            normalized=Pk * wavenumbers**3 / (2 * np.pi**2),
            growth_factor=float(growth_factor),
            is_linear=True,
            in_comoving=True
        )
    
    @classmethod
    def null_initialize(
            cls, arr_size: int = 0, 
            is_linear: bool = False, 
            in_comoving: bool = True
        ) -> PowerSpectrum:
        return cls(
            wavenumbers=np.full(arr_size, np.nan),
            amplitudes=np.full(arr_size, np.nan),
            normalized=np.full(arr_size, np.nan),
            is_linear=is_linear,
            in_comoving=in_comoving
        )
    
    @property
    def is_null(self) -> bool:
        if self.wavenumbers.size == 0 and self.amplitudes.size == 0:
            return True
        return np.logical_and(
            np.all(np.isnan(self.wavenumbers)), 
            np.all(np.isnan(self.amplitudes))
        )
    
    @classmethod
    def from_matter_ps_file(
            cls, file_path: Path, linear_file_path: Path
        ) -> PowerSpectrum:
        matter_ps = get_power_spec(file_path, linear_file_path)["nonlinear"]
        return cls(**matter_ps, is_linear=False, in_comoving=True)
    
    @classmethod
    def from_array(
            cls, 
            wavenumbers: np.ndarray,
            amplitudes: np.ndarray,
            is_linear: bool = False,
            in_comoving: bool = True
        ) -> PowerSpectrum:
        
        return cls(
            wavenumbers=wavenumbers,
            amplitudes=amplitudes,
            normalized=amplitudes * wavenumbers**3 / (2 * np.pi**2),
            is_linear=is_linear,
            in_comoving=in_comoving
        )

    @classmethod
    def from_correlation(
            cls, 
            radii: np.ndarray, 
            correlation: np.ndarray, 
            ell: int = 0,
            extrapolate: bool = False,
            is_linear: bool = False,
            in_comoving: bool = True
        ) -> PowerSpectrum:
        return cls(
            **correlation_to_powerspec(
                radii=radii, 
                correlation=correlation, 
                ell=ell,
                extrapolate=extrapolate
            ),
            is_linear=is_linear,
            in_comoving=in_comoving
        )

    @property
    def to_numpy(self) -> np.ndarray:
        return np.column_stack((self.wavenumbers, self.amplitudes, self.normalized))

    @property
    def wavenumber_bins(self) -> np.ndarray:
        unique_log10_k_bins = np.unique(
            np.log10(self.wavenumbers).astype(int)
        )
        return 10.0**np.asarray(unique_log10_k_bins, dtype=float)

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        unique_log10_k_bins = np.unique(
            np.round(np.log10(self.wavenumbers), 1)
        )
        return 10.0**np.asarray(unique_log10_k_bins, dtype=float)
    
    @property
    def slope(self) -> np.ndarray:
        # Old way: np.gradient(np.log(self.amplitudes), np.log(self.wavenumbers))
        d_log_k = np.diff(np.log10(self.wavenumbers)).mean()

        return savgol_filter(
            np.log10(self.amplitudes),
            window_length=11,
            polyorder=3,
            deriv=1,
            delta=d_log_k
        )
    
    def get_interpolated_Delta_sq(self, wavenumbers: float | np.ndarray) -> float | np.ndarray:
        if self.spline is None:
            raise ValueError("Cannot interpolate without a spline")
        return 10.0**np.asarray(self.spline(np.log10(wavenumbers)))


    def get_normalization(
            self, 
            spectral_index: float, 
            min_length: int = 10
        ) -> float:

        """
        Return the normalization A of the primordial power spectrum A k^ns
        from the stored linear P(k).
        Requires that this PowerSpectrum instance represents the *linear* power spectrum.
        """
        if not self.is_linear:
            raise ValueError("Normalization is only defined for linear power spectra")

        # --- 1. Determine normalization A ---
        # On large (super-horizon) scales T(k) → 1, so A ~ P(k)/k^ns on small k.
        # Fit using the lowest-N k values.
        N_lowk = min(min_length, len(self.wavenumbers))
        ks = self.wavenumbers[:N_lowk]
        Ps = self.amplitudes[:N_lowk] / (self.growth_factor**2)

        # A ~ mean(P/k^ns) for small k
        return float(np.mean(Ps / (ks**spectral_index)))
    
    def get_initial_power_spectrum(
            self, 
            spectral_index: float, 
            min_length: int = 10
        ) -> np.ndarray:

        """
        Return the primordial power spectrum A k^ns from the stored linear P(k).
        Requires that this PowerSpectrum instance represents the *linear* power spectrum.
        """
        if not self.is_linear:
            raise ValueError("Initial power spectrum is only defined for linear power spectra")
        
        # --- 1. Determine normalization A ---
        A = self.get_normalization(
            spectral_index=spectral_index, 
            min_length=min_length
        )

        # --- 2. Compute primordial spectrum A k^ns ---
        return A * self.wavenumbers**spectral_index
    
    def get_transfer_fn(
            self, 
            spectral_index: float, 
            min_length: int = 10
        ) -> np.ndarray:
        """
        Return the transfer function T(k) from the stored linear P(k).
        Requires that this PowerSpectrum instance represents the *linear* power spectrum.
        """

        if not self.is_linear:
            raise ValueError("Transfer function is only defined for linear power spectra")

        Plin = self.amplitudes / (self.growth_factor**2)
        initPk = self.get_initial_power_spectrum(
            spectral_index=spectral_index, 
            min_length=min_length
        )

        # --- 3. Transfer function ---
        return np.sqrt(Plin / initPk)

    def get_interpolated_Pk(self, wavenumbers: float | np.ndarray) -> float | np.ndarray:

        if self.spline is None:
            raise ValueError("Cannot interpolate without a spline")

        Delta_sq = self.get_interpolated_Delta_sq(wavenumbers)
        return  Delta_sq * (2 * np.pi**2) / (wavenumbers**3)
    def get_mass_variance(
            self, 
            values: float | np.ndarray,
            filter_radius: float, # provide the option to use mass or radius
            filter_type: str = "tophat", # 'gaussian', 'sharp_k', or 'tophat'
            from_masses: bool = True,
            present_day_mean_density: float | None = None
        ) -> float | np.ndarray:

        if self.is_linear is False:
            raise ValueError("Mass variance can only be computed from linear power spectra")
        
        if from_masses and present_day_mean_density is None:
            raise ValueError("Present day mean density must be provided when computing from masses")

        interpolated_Pk = (
            None 
            if from_masses else 
            self.get_interpolated_Pk(wavenumbers=None if from_masses else values)
        )
        
        return get_mass_variance(
            values=np.asarray(values),
            power_spectrum=np.asarray(interpolated_Pk),
            filter_radius=filter_radius,
            present_day_mean_density=present_day_mean_density,
            filter_type=filter_type,
            from_masses=from_masses
        )
    
    
    
    def get_interpolated_instance(
            self, wavenumbers: float | np.ndarray,
        ) -> PowerSpectrumValue | PowerSpectrum:

        amplitudes = self.get_interpolated_Pk(wavenumbers)
        normalized = amplitudes * wavenumbers**3 / (2 * np.pi**2)
        if isinstance(wavenumbers, (float, np.floating, int, np.integer)):
            return PowerSpectrumValue(
                wavenumber=wavenumbers,
                amplitude=amplitudes,
                normalized=normalized,
                is_linear=self.is_linear,
                in_comoving=self.in_comoving
            )

        return PowerSpectrum(
            wavenumbers=wavenumbers,
            amplitudes=amplitudes,
            normalized=normalized,
            is_linear=self.is_linear,
            in_comoving=self.in_comoving
        )
    
    def display(
            self, text_size: int = 16,
            use_normalized: bool = False,
            ax: plt.Axes | None = None,
            return_fig: bool = False, 
            color: str | tuple | None = None,
            in_comoving: bool = True
        ) -> plt.Axes | None:

        ax = display_power_spectrum(
            wavenumbers=self.wavenumbers,
            amplitudes=self.normalized if use_normalized else self.amplitudes,
            text_size=text_size,
            is_linear=self.is_linear,
            is_normalized=use_normalized,
            ax=ax,
            return_fig=True,
            color=color,
            in_comoving=in_comoving
        )

        if return_fig:
            return ax
        
        plt.show()

@define(slots=True)
class PowerSpectrumData:
    estimate: PowerSpectrum
    errors: PowerSpectrum
    in_comoving: bool = field(default=True)

    def __attrs_post_init__(self) -> None:
        # Sync children with parent's in_comoving state
        self.estimate.in_comoving = self.in_comoving
        self.errors.in_comoving = self.in_comoving

    @classmethod
    def null_initialize(
            cls, arr_size: int = 0, is_linear: bool = False, in_comoving: bool = True
        ) -> PowerSpectrumData:
        return cls(
            estimate=PowerSpectrum.null_initialize(
                arr_size=arr_size, is_linear=is_linear, in_comoving=in_comoving
            ), 
            errors=PowerSpectrum.null_initialize(
                arr_size=arr_size, is_linear=is_linear, in_comoving=in_comoving
            ),
            in_comoving=in_comoving
        )
    
    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.estimate.in_comoving = False
        self.errors.in_comoving = False
        self.estimate.convert_to_comoving(scale_factor)
        self.errors.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.estimate.in_comoving = True
        self.errors.in_comoving = True
        self.estimate.convert_to_physical(scale_factor)
        self.errors.convert_to_physical(scale_factor)
        self.in_comoving = False


    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            wavenumbers: np.ndarray,
            redshift: float = 0.0,
        ) -> PowerSpectrumData:

        return cls(
            estimate=PowerSpectrum.from_cosmology(
                cosmo=cosmo,
                wavenumbers=wavenumbers,
                redshift=redshift
            ),
            errors=PowerSpectrum.null_initialize(
                arr_size=wavenumbers.size, 
                is_linear=True, 
                in_comoving=True
            ),
            in_comoving=True
        )

    @property
    def is_null(self) -> bool:
        return self.estimate.is_null

    @property
    def has_null_errors(self) -> bool:
        return self.errors.is_null

    @property
    def wavenumbers(self) -> np.ndarray:
        return self.estimate.wavenumbers
    
    @property
    def amplitudes(self) -> np.ndarray:
        return self.estimate.amplitudes

    @property
    def normalized(self) -> np.ndarray:
        return self.estimate.normalized

    @property
    def sigma(self) -> np.ndarray:
        return self.errors.amplitudes

    @property
    def slope(self) -> np.ndarray:
        return self.estimate.slope
    

@define(slots=True)
class PowerSpectrumEvo(EvolutionData):
    data: OrderedDict[int, PowerSpectrum]
    in_comoving: bool = field(default=True)

    def __repr__(self) -> str:
        return super().__repr__()

    def convert_to_comoving(self) -> None:
        if self.in_comoving:
            return
        for key, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[key].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        if not self.in_comoving:
            return
        for key, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[key].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            moments: MomentsInTime,
            wavenumbers: np.ndarray,
        ) -> PowerSpectrumEvo:
        """
        Create a PowerSpectrumEvo from a cosmology model at the redshifts
        specified by the MomentsInTime object.
        
        Parameters
        ----------
        cosmo : Cosmology
            The cosmology model to use for computing power spectra.
        moments : MomentsInTime
            The time steps (snapshot IDs, scale factors, redshifts) at which
            to compute the power spectra.
        wavenumbers : np.ndarray
            The wavenumber array [h/Mpc] at which to evaluate the power spectra.
        
        Returns
        -------
        PowerSpectrumEvo
            A PowerSpectrumEvo instance containing linear power spectra at each
            snapshot specified in moments.
        """
        
        return cls(
            moments=moments,
            data=OrderedDict(
                (snap_id, PowerSpectrum.from_cosmology(
                    cosmo=cosmo,
                    wavenumbers=wavenumbers,
                    redshift=moments[snap_id].redshift
                ))
                for snap_id in moments.snapshot_ids
            ),
            in_comoving=True
        )

    @property
    def final_spectrum(self) -> PowerSpectrum:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null power spectrum data available")
    
    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.wavenumber_bins
    
    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.round_one_wavenumber_bins

    @property
    def is_linear(self) -> bool:
        return all(
            value.is_linear 
            for value in self.data.values() 
            if not value.is_null
        )

    def get_interpolated_spectra_evo(self, wavenumber: float) -> OrderedDict[int, float]:
        return OrderedDict(
            (key, float(value.get_interpolated_Pk(wavenumber))) 
            for key, value in self.data.items()
            if not value.is_null
        )

    def get_interpolated_variance_evo(self, wavenumber: float) -> OrderedDict[int, float]:
        return OrderedDict(
            (key, float(value.get_interpolated_Delta_sq(wavenumber))) 
            for key, value in self.data.items()
            if not value.is_null
        )

    def get_ps_evo_array(self, wavenumber: float, wrt: str = "scale_factor") -> np.ndarray:
        evo_vals = self.get_interpolated_spectra_evo(wavenumber)
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.array([
            [time_map[key], val] for key, val in evo_vals.items()
        ])

    def get_variance_evo_array(self, wavenumber: float, wrt: str = "scale_factor") -> np.ndarray:
        evo_vals = self.get_interpolated_variance_evo(wavenumber)
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        return np.array([
            [time_map[key], val] for key, val in evo_vals.items()
        ])

    def get_ps_evo_rate(self, wavenumber: float, wrt: str = "scale_factor") -> np.ndarray:
        ps_evo = self.get_ps_evo_array(wavenumber, wrt=wrt)
        if len(ps_evo) < 3:
            return np.column_stack([ps_evo[:, 0], np.full(len(ps_evo), np.nan)])
        rate = np.gradient(np.log(ps_evo[:, 1]), np.log(ps_evo[:, 0]), edge_order=2)
        return np.column_stack([ps_evo[:, 0], rate])
    
    def get_variance_evo_rate(self, wavenumber: float, wrt: str = "scale_factor") -> np.ndarray:
        variance_evo = self.get_variance_evo_array(wavenumber, wrt=wrt)
        if len(variance_evo) < 3:
            return np.column_stack([variance_evo[:, 0], np.full(len(variance_evo), np.nan)])
        rate = np.gradient(np.log(variance_evo[:, 1]), np.log(variance_evo[:, 0]), edge_order=2)
        return np.column_stack([variance_evo[:, 0], rate])
    
    def display(
            self, 
            target_scale_factors: Collection[float] | None = None,
            use_normalized: bool = False,
            text_size: int = 16,
            return_fig: bool = False,
            ax: plt.Axes | None = None,
            legend_xloc: float = 1.3,
            legend_yloc: float = 1.0,
            in_comoving: bool = True
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
                use_normalized=use_normalized,
                ax=ax,
                return_fig=True,
                color=colormap(i),
                in_comoving=in_comoving
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

    def get_wavenumber_bin_evo(
            self, 
            comoving_wavenumber_bin: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0
        ) -> AccumulationHistory:
        """
        Get the accumulation history of P(k) at a specific wavenumber.
        
        Parameters
        ----------
        comoving_wavenumber_bin : float
            The comoving wavenumber [h/Mpc] at which to evaluate.
        return_in_physical : bool
            If True, convert to physical coordinates.
        wrt : str
            Time variable to use ('scale_factor', 'redshift', 'age', etc.).
        min_time_step_value : float
            Minimum time value to include.
        max_time_step_value : float
            Maximum time value to include.
        
        Returns
        -------
        AccumulationHistory
            The accumulation history at the given wavenumber.
        """
        # Validate input 
        if not np.isfinite(comoving_wavenumber_bin) or comoving_wavenumber_bin <= 0:
            return AccumulationHistory.null_initialize()

        if not [k for k, v in self.data.items() if not v.is_null]:
            return AccumulationHistory.null_initialize()
        
        # Ensure correct coordinate system
        if return_in_physical and self.in_comoving:
            self.convert_to_physical()
        elif not return_in_physical and not self.in_comoving:
            self.convert_to_comoving()

        evo_array = self.get_ps_evo_array(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt
        )
        
        valid_mask = np.logical_and.reduce([
            evo_array[:, 0] >= min_time_step_value,
            evo_array[:, 0] <= max_time_step_value, 
            ~np.isnan(evo_array).any(axis=1)
        ])
        
        evo_array = evo_array[valid_mask, :]
        
        if evo_array.size == 0 or len(evo_array) < 3:
            return AccumulationHistory.null_initialize()

        variance_array = self.get_variance_evo_array(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt
        )[valid_mask, :]
        
        # Get rate before applying mask since gradient needs full array
        full_rate = self.get_ps_evo_rate(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt
        )
        evo_rate = full_rate[valid_mask, :]

        time_steps = evo_array[:, 0]
        accumulation = evo_array[:, 1]
        weighted = variance_array[:, 1]
        
        # Normalize by final value
        a_f = time_steps[-1]
        Pk_final = accumulation[-1]
        scaling = (
            Pk_final
            if self.is_linear else 
            (Pk_final * (time_steps / a_f) ** 3) 
        )

        return AccumulationHistory(
            time_steps=time_steps,
            accumulation=accumulation,
            weighted=weighted,
            normalized=accumulation / scaling,
            rate=evo_rate[:, 1],
            in_comoving=self.in_comoving,
            time_metric=wrt
        )

    def get_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> AccumulationHistories:
        """
        Get accumulation histories at all wavenumber bins.
        
        Parameters
        ----------
        return_in_physical : bool
            If True, convert to physical coordinates.
        wrt : str
            Time variable to use.
        min_time_step_value : float
            Minimum time value to include.
        max_time_step_value : float
            Maximum time value to include.
        key_format : str
            Format string for wavenumber keys (applied to log10(k)).
        
        Returns
        -------
        AccumulationHistories
            Collection of accumulation histories at each wavenumber.
        """
        if not self.in_comoving:
            self.convert_to_comoving()

        wavenumbers = self.wavenumber_bins

        accumulations = {}
        for k in wavenumbers:
            wav_key = key_format.format(np.log10(k))
            accumulations[wav_key] = self.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=k,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value) 
                for key, value in sorted(
                    accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=wrt,
            is_linear=True  # PowerSpectrumEvo from_cosmology creates linear PS
        )

    def get_round_one_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> AccumulationHistories:
        """
        Get accumulation histories at round-one wavenumber bins.
        
        Parameters
        ----------
        return_in_physical : bool
            If True, convert to physical coordinates.
        wrt : str
            Time variable to use.
        min_time_step_value : float
            Minimum time value to include.
        max_time_step_value : float
            Maximum time value to include.
        key_format : str
            Format string for wavenumber keys (applied to log10(k)).
        
        Returns
        -------
        AccumulationHistories
            Collection of accumulation histories at each round-one wavenumber.
        """
        if not self.in_comoving:
            self.convert_to_comoving()

        wavenumbers = self.round_one_wavenumber_bins

        accumulations = {}
        for k in wavenumbers:
            wav_key = key_format.format(np.log10(k))
            accumulations[wav_key] = self.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=k,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        return AccumulationHistories(
            accumulations=OrderedDict(
                (key, value) 
                for key, value in sorted(
                    accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=wrt,
            is_linear=True
        )


@define(slots=True)
class PowerSpectrumDataEvo(EvolutionData):
    data: OrderedDict[int, PowerSpectrumData]
    in_comoving: bool = field(default=True)

    def __repr__(self) -> str:
        return super().__repr__()
    
    @property
    def estimate(self) -> PowerSpectrumEvo:
        return PowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.estimate) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def errors(self) -> PowerSpectrumEvo:
        return PowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.errors) for key, value in self.data.items()
            ),
            in_comoving=self.in_comoving
        )

    def convert_to_comoving(self) -> None:
        if self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        if not self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology,
            moments: MomentsInTime,
            wavenumbers: np.ndarray,
        ) -> PowerSpectrumDataEvo:
        """
        Create a PowerSpectrumDataEvo from a cosmology model at the redshifts
        specified by the MomentsInTime object.
        
        Parameters
        ----------
        cosmo : Cosmology
            The cosmology model to use for computing power spectra.
        moments : MomentsInTime
            The time steps (snapshot IDs, scale factors, redshifts) at which
            to compute the power spectra.
        wavenumbers : np.ndarray
            The wavenumber array [h/Mpc] at which to evaluate the power spectra.
        
        Returns
        -------
        PowerSpectrumDataEvo
            A PowerSpectrumDataEvo instance containing linear power spectra 
            (with null errors) at each snapshot specified in moments.
        """
        
        return cls(
            moments=moments,
            data = OrderedDict(
                (snap_id, PowerSpectrumData.from_cosmology(
                    cosmo=cosmo,
                    wavenumbers=wavenumbers,
                    redshift=moments[snap_id].redshift
                ))
                for snap_id in moments.snapshot_ids
            ),
            in_comoving=True
        )

    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.estimate.wavenumber_bins

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.estimate.round_one_wavenumber_bins

    def get_wavenumber_bin_evo(
            self, 
            comoving_wavenumber_bin: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0
        ) -> AccumulationHistory:
        """
        Get the accumulation history of P(k) at a specific wavenumber.
        Delegates to the estimate PowerSpectrumEvo.
        """
        return self.estimate.get_wavenumber_bin_evo(
            comoving_wavenumber_bin=comoving_wavenumber_bin,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value
        )

    def get_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> AccumulationHistories:
        """
        Get accumulation histories at all wavenumber bins.
        Delegates to the estimate PowerSpectrumEvo.
        """
        return self.estimate.get_wavenumber_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format
        )

    def get_round_one_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> AccumulationHistories:
        """
        Get accumulation histories at round-one wavenumber bins.
        Delegates to the estimate PowerSpectrumEvo.
        """
        return self.estimate.get_round_one_wavenumber_accumulations(
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            key_format=key_format
        )

@define(slots=True)
class MatterPowerSpectrum: 
    linear: PowerSpectrum
    nonlinear: PowerSpectrum
    shot_noise: PowerSpectrum | None
    in_comoving: bool = field(default=True)

    def __attrs_post_init__(self) -> None:
        # Sync children with parent's in_comoving state
        self.linear.in_comoving = self.in_comoving
        self.nonlinear.in_comoving = self.in_comoving
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = self.in_comoving

    def __repr__(self) -> str:
        return (
            f"MatterPowerSpectrum(linear=Pk("
            f"k={self.linear.wavenumbers.min():.3e}"
            f"-{self.linear.wavenumbers.max():.3e}),"
            f" nonlinear=Pk("
            f"k={self.nonlinear.wavenumbers.min():.3e}-"
            f"{self.nonlinear.wavenumbers.max():.3e}"
            f"))"
        )
    
    @classmethod
    def from_file(cls, filepath: Path, input_ps_path: Path) -> MatterPowerSpectrum:
        return cls.from_data(get_power_spec(filepath, input_ps_path))
    
    @classmethod
    def from_data(
            cls, ps_data: dict[str, float | dict[str, np.ndarray]]
        ) -> MatterPowerSpectrum:
        return cls(
            linear=PowerSpectrum(**ps_data["linear"], is_linear=True, in_comoving=True),
            nonlinear=PowerSpectrum(**ps_data["nonlinear"], is_linear=False, in_comoving=True),
            shot_noise=PowerSpectrum(**ps_data["shot_noise"], is_linear=False, in_comoving=True),
            in_comoving=True
        )
    
    @classmethod
    def null_initialize(cls, arr_size: int = 0) -> MatterPowerSpectrum:
        return cls(
            linear=PowerSpectrum.null_initialize(arr_size, is_linear=True, in_comoving=True),
            nonlinear=PowerSpectrum.null_initialize(arr_size, is_linear=False, in_comoving=True),
            shot_noise=PowerSpectrum.null_initialize(arr_size, is_linear=False, in_comoving=True),
            in_comoving=True
        )

    @property
    def is_null(self) -> bool:
        return self.nonlinear.is_null and self.linear.is_null

    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.nonlinear.wavenumber_bins
    
    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.nonlinear.round_one_wavenumber_bins
    
    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.linear.in_comoving = False
        self.nonlinear.in_comoving = False
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = False
        self.linear.convert_to_comoving(scale_factor)
        self.nonlinear.convert_to_comoving(scale_factor)
        if self.shot_noise is not None:
            self.shot_noise.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.linear.in_comoving = True
        self.nonlinear.in_comoving = True
        if self.shot_noise is not None:
            self.shot_noise.in_comoving = True
        self.linear.convert_to_physical(scale_factor)
        self.nonlinear.convert_to_physical(scale_factor)
        if self.shot_noise is not None:
            self.shot_noise.convert_to_physical(scale_factor)
        self.in_comoving = False
    
    @classmethod
    def from_powerspec_ratio(
            cls, 
            nonlinear_to_linear: np.ndarray,
            linearPk: np.ndarray,
            shot_noise_arr: np.ndarray | None = None,
            use_splined: bool = False,
            in_comoving: bool = True
        ) -> MatterPowerSpectrum:

        k = nonlinear_to_linear[:, 0]
        ratios = nonlinear_to_linear[:, 2 if use_splined else 1]
        interp_linearPk = interp1d(
            np.log10(linearPk[:, 0]),
            np.log10(linearPk[:, 1]),
            fill_value="extrapolate"
        )
        linearPk = 10**np.asarray(interp_linearPk(np.log10(k)))
        nonlinearPk = ratios * linearPk

        if shot_noise_arr is not None:
            interp_shot_noise = interp1d(
                np.log10(shot_noise_arr[:, 0]),
                np.log10(shot_noise_arr[:, 1]),
                fill_value="extrapolate"
            )
            
            shot_noise = PowerSpectrum.from_array(
                wavenumbers=k,
                amplitudes=10**np.asarray(interp_shot_noise(np.log10(k))),
                is_linear=False,
                in_comoving=in_comoving
            )

        return cls(
            linear=PowerSpectrum.from_array(
                wavenumbers=k, 
                amplitudes=linearPk, 
                is_linear=True,
                in_comoving=in_comoving
            ),
            nonlinear=PowerSpectrum.from_array(
                wavenumbers=k, 
                amplitudes=nonlinearPk,
                is_linear=False,
                in_comoving=in_comoving
            ),
            shot_noise=shot_noise if shot_noise_arr is not None else None,
            in_comoving=in_comoving
        )

    @property
    def to_colossus(self) -> np.ndarray:
        return np.column_stack(
            (np.log10(self.linear.wavenumbers), np.log10(self.linear.amplitudes))
        )
    
    @property
    def nonlinear_to_linear(self) -> np.ndarray:
        """
        Compute the ratio of nonlinear to linear power spectrum.
        
        Uses the NONLINEAR wavenumbers as the base grid and interpolates the
        linear P(k) onto that grid. This ensures the ratio covers the full
        k-range of the measured nonlinear spectrum, which is necessary for
        stitching multiple box sizes together.
        
        Returns an array of shape (N, 2) with columns [wavenumber, ratio].
        Returns array of NaNs if data is insufficient for interpolation.
        """
        # Check for null/empty data
        if self.is_null:
            return np.array([]).reshape(0, 2)
        
        lin_k = self.linear.wavenumbers
        lin_pk = self.linear.amplitudes
        nonlin_k = self.nonlinear.wavenumbers
        nonlin_pk = self.nonlinear.amplitudes
        
        # Need at least 2 points for interpolation
        if len(lin_k) < 2 or len(nonlin_k) < 2:
            return np.column_stack((nonlin_k, np.full_like(nonlin_k, np.nan)))
        
        # Filter out non-positive values in linear data (can't take log)
        lin_valid = (lin_k > 0) & (lin_pk > 0)
        if np.sum(lin_valid) < 2:
            return np.column_stack((nonlin_k, np.full_like(nonlin_k, np.nan)))
        
        lin_k_valid = lin_k[lin_valid]
        lin_pk_valid = lin_pk[lin_valid]
        
        # Build interpolator for linear P(k) in log-log space
        # Allow extrapolation at high-k (linear theory extends smoothly)
        log_interp_linearPk = interp1d(
            np.log(lin_k_valid),
            np.log(lin_pk_valid),
            bounds_error=False,
            fill_value="extrapolate"  # Extrapolate linear theory to high-k
        )
        
        # Only interpolate where nonlinear k is positive
        nonlin_valid = (nonlin_k > 0) & (nonlin_pk > 0)
        interp_linPk = np.full_like(nonlin_pk, np.nan)
        
        if np.any(nonlin_valid):
            log_interp_result = log_interp_linearPk(np.log(nonlin_k[nonlin_valid]))
            interp_linPk[nonlin_valid] = np.exp(log_interp_result)
        
        # Compute ratio: P_NL(k) / P_lin(k) at nonlinear k values
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = nonlin_pk / interp_linPk
            # Replace inf with nan for cleaner downstream handling
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)

        return np.column_stack((nonlin_k, ratio))
    
    @property
    def nonlinear_wavenumber(self) -> float:
        '''
        Find the nonlinear wavenumber k_NL where the dimensionless power 
        spectrum Delta^2(k) = k^3 P(k) / (2 pi^2) = 1.
        
        This is the standard definition of the nonlinear scale - where 
        fluctuations become order unity. Delta^2(k) is monotonically 
        increasing with k, so there's a unique crossing point.
        
        Returns the comoving wavenumber k_NL in h/cMpc.
        '''
        # Use the nonlinear power spectrum's normalized (Delta^2) values
        k = self.nonlinear.wavenumbers
        delta_sq = self.nonlinear.normalized  # This is Delta^2(k)

        # Filter out NaN/inf values
        valid = np.isfinite(k) & np.isfinite(delta_sq) & (delta_sq > 0) & (k > 0)
        if np.sum(valid) < 2:
            return np.nan

        k_valid = k[valid]
        delta_sq_valid = delta_sq[valid]

        # Interpolate in log-log space for smoothness
        hi_res_k = np.logspace(
            np.log10(k_valid.min()),
            np.log10(k_valid.max()),
            1000
        )
        interp = interp1d(
            np.log10(k_valid), 
            np.log10(delta_sq_valid), 
            fill_value="extrapolate"
        )
        hi_res_delta_sq = 10.0**np.asarray(interp(np.log10(hi_res_k)))

        # Find where Delta^2(k) crosses 1
        below_one = hi_res_delta_sq < 1.0
        above_one = hi_res_delta_sq >= 1.0
        
        # Find crossing: below at index i and above at index i+1
        crossing = below_one[:-1] & above_one[1:]
        
        if np.any(crossing):
            # Return the k value at the crossing point
            cross_idx = np.where(crossing)[0][0] + 1
            return hi_res_k[cross_idx]
        elif np.all(above_one):
            # Delta^2 >= 1 everywhere (fully nonlinear at all scales)
            return hi_res_k[0]
        elif np.all(below_one):
            # Delta^2 < 1 everywhere (still linear at all scales)
            return np.nan
        else:
            return hi_res_k[above_one][0] if np.any(above_one) else np.nan

    def deviation_wavenumber(self, threshold: float = 1.1) -> float:
        '''
        Find the wavenumber where the boost factor B(k) = P_NL(k)/P_lin(k)
        first exceeds the given threshold.
        
        This marks the onset of nonlinear deviations from linear theory,
        which occurs at larger scales (smaller k) than the standard k_NL
        definition where Delta^2(k) = 1.
        
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
            The comoving wavenumber k_dev where B(k) first exceeds threshold.
        '''
        ratio = self.nonlinear_to_linear

        # Filter out NaN values before interpolation
        valid = np.isfinite(ratio[:, 0]) & np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0)
        if np.sum(valid) < 2:
            return np.nan

        k_valid = ratio[valid, 0]
        ratio_valid = ratio[valid, 1]

        # Interpolate in log-log space
        hi_res_k = np.logspace(
            np.log10(k_valid.min()),
            np.log10(k_valid.max()),
            1000
        )
        interp = interp1d(
            np.log10(k_valid), 
            np.log10(ratio_valid), 
            fill_value="extrapolate"
        )
        hi_res_vals = 10.0**np.asarray(interp(np.log10(hi_res_k)))

        # Find where ratio crosses the threshold
        below_threshold = hi_res_vals < threshold
        above_threshold = hi_res_vals >= threshold
        
        # Find crossing: below at index i and above at index i+1
        crossing = below_threshold[:-1] & above_threshold[1:]
        
        if np.any(crossing):
            cross_idx = np.where(crossing)[0][0] + 1
            return hi_res_k[cross_idx]
        elif np.all(above_threshold):
            # Already above threshold at all k (fully nonlinear)
            return hi_res_k[0]
        elif np.all(below_threshold):
            # Still below threshold at all k (still linear)
            return np.nan
        else:
            return hi_res_k[above_threshold][0] if np.any(above_threshold) else np.nan

    
    def get_interpolated_value(self, wavenumber: float) -> MatterPowerSpectrumValue:
        return MatterPowerSpectrumValue(
            linear=self.linear.get_interpolated_instance(wavenumber),
            nonlinear=self.nonlinear.get_interpolated_instance(wavenumber),
            shot_noise=(
                self.shot_noise.get_interpolated_instance(wavenumber)
                if self.shot_noise is not None else 
                None
            )
        )
        
        
    def save_for_colossus(self, power_spec_path: Path) -> None:
        np.savetxt(power_spec_path, self.to_colossus)

    def display(
            self, text_size: int = 16,
            use_normalized: bool = False,
            show_nonlinear: bool = True,
            show_linear: bool = True,
            show_shot_noise: bool = True,
            ax: plt.Axes | None = None,
            return_fig: bool = False,
            color: str | tuple | None = None,
            show_legend: bool = True,
            in_comoving: bool = True
        ) -> plt.Axes | None:

        attr_value = "normalized" if use_normalized else "amplitudes"
    
        ax = display_matter_power_spectrum(
            linear_wavenumbers=self.linear.wavenumbers if show_linear else None,
            linear_amplitudes=getattr(self.linear, attr_value) if show_linear else None,
            nonlinear_wavenumbers=self.nonlinear.wavenumbers if show_nonlinear else None,
            nonlinear_amplitudes=getattr(self.nonlinear, attr_value) if show_nonlinear else None,
            shot_noise_wavenumbers=self.shot_noise.wavenumbers if show_shot_noise else None,
            shot_noise_amplitudes=getattr(self.shot_noise, attr_value) if show_shot_noise else None,
            is_normalized=use_normalized,
            text_size=text_size,
            ax=ax,
            return_fig=True,
            color=color,
            show_legend=show_legend,
            in_comoving=in_comoving
        )

        if return_fig:
            return ax
        
        plt.show()


@define(slots=True)
class MatterPowerSpectrumData:
    estimate: MatterPowerSpectrum
    errors: MatterPowerSpectrum
    in_comoving: bool = field(default=True) 

    def __attrs_post_init__(self) -> None:
        # Sync children with parent's in_comoving state
        self.estimate.in_comoving = self.in_comoving
        self.errors.in_comoving = self.in_comoving

    @property
    def linear(self) -> PowerSpectrumData:
        return PowerSpectrumData(
            estimate=self.estimate.linear,
            errors=self.errors.linear,
            in_comoving=self.in_comoving
        )
    
    @property
    def nonlinear(self) -> PowerSpectrumData:
        return PowerSpectrumData(
            estimate=self.estimate.nonlinear,
            errors=self.errors.nonlinear,
            in_comoving=self.in_comoving
        )

    @property
    def nonlinear_to_linear(self) -> np.ndarray:
        return self.estimate.nonlinear_to_linear
    
    @classmethod
    def null_initialize(cls, arr_size: int = 0) -> MatterPowerSpectrumData:
        return MatterPowerSpectrumData(
            estimate=MatterPowerSpectrum.null_initialize(arr_size),
            errors=MatterPowerSpectrum.null_initialize(arr_size),
            in_comoving=True
        )
    
    @property
    def has_null_errors(self) -> bool:
        return self.errors.is_null
    
    @property
    def is_null(self) -> bool:
        return self.estimate.is_null
    
    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.estimate.nonlinear.wavenumber_bins

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.estimate.nonlinear.round_one_wavenumber_bins

    @property
    def nonlinear_wavenumber(self) -> float:
        return self.estimate.nonlinear_wavenumber

    def deviation_wavenumber(self, threshold: float = 1.1) -> float:
        '''
        Find the wavenumber where the boost factor B(k) = P_NL(k)/P_lin(k)
        first exceeds the given threshold.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        float
            The comoving wavenumber k_dev where B(k) first exceeds threshold.
        '''
        return self.estimate.deviation_wavenumber(threshold=threshold)

    def convert_to_comoving(self, scale_factor: float) -> None:
        if self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.estimate.in_comoving = False
        self.errors.in_comoving = False
        self.estimate.convert_to_comoving(scale_factor)
        self.errors.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        if not self.in_comoving:
            return
        # Force-sync children to expected state before conversion
        self.estimate.in_comoving = True
        self.errors.in_comoving = True
        self.estimate.convert_to_physical(scale_factor)
        self.errors.convert_to_physical(scale_factor)
        self.in_comoving = False

    def get_interpolated_value(self, wavenumber: float) -> MatterPowerSpectrumValue:
        return self.estimate.get_interpolated_value(wavenumber)
    
    # def get_interpolated_instance(
    #         self, wavenumbers: np.ndarray
    #     ) -> MatterPowerSpectrumData:
    #     return MatterPowerSpectrumData(
    #         estimate=self.estimate.get_interpolated_instance(wavenumbers),
    #         errors=self.errors.get_interpolated_instance(wavenumbers),
    #         in_comoving=self.in_comoving
    #     )
    
    @classmethod
    def from_file(
            cls, 
            filepath: Path, 
            input_ps_path: Path,
            arr_size: int = 0
        ) -> MatterPowerSpectrumData:
        return cls(
            estimate=MatterPowerSpectrum.from_file(filepath, input_ps_path),
            errors=MatterPowerSpectrum.null_initialize(arr_size),
            in_comoving=True
        )
    
    @classmethod
    def from_data(
            cls, ps_data: dict[str, float | dict[str, np.ndarray]],
            in_comoving: bool = True
        ) -> MatterPowerSpectrumData:
        estimate = MatterPowerSpectrum.from_data(ps_data)
        num_vals = estimate.nonlinear.wavenumbers.size
        return cls(
            estimate=estimate,
            errors=MatterPowerSpectrum.null_initialize(num_vals),
            in_comoving=in_comoving
        )
    
    @classmethod
    def from_powerspec_ratio(
            cls, 
            nonlinear_to_linear: np.ndarray,
            linearPk: np.ndarray,
            shot_noise_arr: np.ndarray | None = None,
            use_splined: bool = False,
            in_comoving: bool = True
        ) -> MatterPowerSpectrumData:

        estimate=MatterPowerSpectrum.from_powerspec_ratio(
            nonlinear_to_linear=nonlinear_to_linear,
            linearPk=linearPk,
            shot_noise_arr=shot_noise_arr,
            use_splined=use_splined,
            in_comoving=in_comoving
        )
        arr_size = estimate.nonlinear.wavenumbers.size

        return cls(
            estimate=estimate,
            errors=MatterPowerSpectrum.null_initialize(arr_size=arr_size),
            in_comoving=in_comoving
        )


@define(slots=True)
class MatterPowerSpectrumEvo(EvolutionData):
    data: OrderedDict[int, MatterPowerSpectrum]
    in_comoving: bool = field(default=True)

    colormap_name: str = field(default="viridis", repr=False)
    colormap: plt.cm.ColorMap = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.colormap = plt.cm.get_cmap(self.colormap_name, len(self.data))
        # Sync children with parent's in_comoving state
        for v in self.data.values():
            if not v.is_null:
                v.in_comoving = self.in_comoving

    def __repr__(self) -> str:  
        return super().__repr__()

    @property
    def final_spectrum(self) -> MatterPowerSpectrum:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null matter power spectrum data available")

    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.wavenumber_bins

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.round_one_wavenumber_bins
    
    def convert_to_comoving(self) -> None:
        if self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        if not self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    
    @classmethod
    def from_directory(
            cls, directory: Path, 
            cosmo: Cosmology,
            use_music: bool = False, 
            use_colossus_linear_ps: bool = False
        ) -> MatterPowerSpectrumEvo:

        evo_data = get_power_spec_evo(directory, use_music_init_ps=use_music)

        moments, data = MomentsInTime(), OrderedDict()

        for snap_id, ps_data in evo_data.items():   
            
            if len(ps_data["nonlinear"]["wavenumbers"]) <= 1: continue

            moment = Moment(
                snapshot_id=snap_id,
                scale_factor=ps_data["scale_factor"],
            )

            if use_colossus_linear_ps:
                ps_data["linear"] = get_colossus_linear_ps(
                    cosmo=cosmo,
                    wavenumbers=ps_data["linear"]["wavenumbers"],
                    redshift=moment.redshift
                )

            data[snap_id] = MatterPowerSpectrum.from_data(ps_data)
            moments.add_moment(moment)

        moments.add_times(cosmo)

        return cls(moments=moments, data=data, in_comoving=True)


    def get_power_spectrum(
            self, snap_id: int, ps_type: str = "nonlinear"
        ) -> PowerSpectrum:
        return getattr(self[snap_id], ps_type)
    
    @property
    def linear_power_spectrum_evo(self) -> PowerSpectrumEvo:
        return PowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.linear) 
                for key, value in self.data.items() if not value.is_null
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def nonlinear_power_spectrum_evo(self) -> PowerSpectrumEvo:
        return PowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.nonlinear) 
                for key, value in self.data.items() if not value.is_null
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def nonlinear_wavenumber_evo(self) -> OrderedDict[int, float]:
        return OrderedDict(
            (key, value.nonlinear_wavenumber) 
            for key, value in self.data.items() if not value.is_null
        )

    def deviation_wavenumber_evo(self, threshold: float = 1.1) -> OrderedDict[int, float]:
        '''
        Get the deviation wavenumber (where B(k) first exceeds threshold)
        for each snapshot.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        OrderedDict[int, float]
            Mapping from snapshot index to deviation wavenumber.
        '''
        return OrderedDict(
            (key, value.deviation_wavenumber(threshold=threshold)) 
            for key, value in self.data.items() if not value.is_null
        )

    
    @property
    def nonlinear_to_linear_evo(self) -> OrderedDict[int, np.ndarray]:
        return OrderedDict(
            (key, value.nonlinear_to_linear) 
            for key, value in self.data.items() if not value.is_null
        )
    

    # @property
    # def shot_noise_power_spectrum_evo(self) -> PowerSpectrumEvo:
    #     return PowerSpectrumEvo(
    #         moments=self.moments,
    #         data=OrderedDict(
    #             (key, value.shot_noise) 
    #             for key, value in self.data.items()
                
    #         ),
    #         in_comoving=self.in_comoving
    #     )

    def get_interpolated_ps_evo(self, wavenumber: float)  -> OrderedDict[int, MatterPowerSpectrumValue]:
        return OrderedDict(
            (key, value.get_interpolated_value(wavenumber))
            for key, value in self.data.items()
            if not value.is_null
        )  

    def get_matter_ps_evo_array(
            self, 
            wavenumber: float, 
            wrt: str = "scale_factor",
            return_variance: bool = False
        ) -> np.ndarray:

        evo_vals = self.get_interpolated_ps_evo(wavenumber)
        time_map = get_correct_snapshot_mapping(self.moments, wrt)
        get_val = lambda val: (
            (val.linear.normalized, val.nonlinear.normalized)
            if return_variance else 
            (val.linear.amplitude, val.nonlinear.amplitude)
        )
        return np.column_stack([
            (time_map[key], *get_val(val)) for key, val in evo_vals.items()
        ]).T

    def get_matter_ps_evo_rate(
            self,
            wavenumber: float,
            wrt: str = "scale_factor",
            return_variance: bool = False
        ) -> np.ndarray:
        
        ps_evo = self.get_matter_ps_evo_array(
            wavenumber, wrt=wrt, return_variance=return_variance
        )
        rate_linear = np.gradient(
            np.log(ps_evo[:, 1]), np.log(ps_evo[:, 0]), edge_order=2
        )
        rate_nonlinear = np.gradient(
            np.log(ps_evo[:, 2]), np.log(ps_evo[:, 0]), edge_order=2
        )
        return np.column_stack([ps_evo[:, 0], rate_linear, rate_nonlinear])
    
    def save_for_colossus(self, power_spec_dir: Path) -> None:
        for snap_id, ps_data in self.data.items():
            filepath = Path(power_spec_dir, f"for_colossus_{snap_id:03d}.txt")
            ps_data.save_for_colossus(filepath)


    def display(
            self,
            target_scale_factors: Collection[float] | None = None,
            text_size: int = 16,
            use_normalized: bool = False, 
            show_linear: bool = True,
            show_nonlinear: bool = True,
            show_shot_noise: bool = False,
            return_fig: bool = False,
            ax_main: plt.Axes | None = None,
            legend_xloc: float = 1.05,
            legend_yloc: float = 0.92,
            show_legend: bool = True,
            colormap: plt.cm.ColorMap | None = None,
            in_comoving: bool = True
        ) -> plt.Axes | None:

        if ax_main is None:
            _, ax_main = plt.subplots()


        if all([not show_linear, not show_nonlinear, not show_shot_noise]):
            raise ValueError(
                'At least one of show_linear, show_nonlinear, or show_shot_noise must be True'
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
                use_normalized=use_normalized,
                show_linear=show_linear,
                show_nonlinear=show_nonlinear,
                show_shot_noise=(
                    (i == len(target_snapshot_ids) - 1) and show_shot_noise
                ),
                ax=ax_main,
                return_fig=True,
                color=colormap(i),
                show_legend=False,
                in_comoving=in_comoving
            )

            handles.append(
                plt.Line2D(
                    [], [], 
                    color=colormap(i),
                    label=f"$a={actual_scale_factors[i]:.2f}$"
                )
            )

        if show_legend:
            ax_main.legend(
                handles=handles,
                bbox_to_anchor=(legend_xloc, legend_yloc), 
                loc='upper left', fontsize=text_size
            )
        

        return ax_main if return_fig else None
    


@define(slots=True)
class MatterPowerSpectrumEvoData(EvolutionData):
    data: OrderedDict[int, MatterPowerSpectrumData]
    in_comoving: bool = field(default=True)

    colormap_name: str = field(default="viridis", repr=False)
    colormap: plt.cm.ColorMap = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.colormap = plt.cm.get_cmap(self.colormap_name, len(self.data))
        # Sync children with parent's in_comoving state
        for v in self.data.values():
            if not v.is_null:
                v.in_comoving = self.in_comoving

    def __repr__(self) -> str:  
        return super().__repr__()

    @property
    def final_spectrum(self) -> MatterPowerSpectrumData:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null matter power spectrum data available")
        
    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.estimate.wavenumber_bins

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.final_spectrum.estimate.round_one_wavenumber_bins

    @property
    def estimate(self) -> MatterPowerSpectrumEvo:
        return MatterPowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.estimate) for key, value in self.data.items()
            )
        )
    
    @property
    def errors(self) -> MatterPowerSpectrumEvo:
        return MatterPowerSpectrumEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.errors) for key, value in self.data.items()
            )
        )
    
    @property
    def linear_power_spectrum_evo(self) -> PowerSpectrumDataEvo:
        return PowerSpectrumDataEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.linear) for key, value in self.data.items()
            )
        )
    
    @property
    def nonlinear_power_spectrum_evo(self) -> PowerSpectrumDataEvo:
        return PowerSpectrumDataEvo(
            moments=self.moments,
            data=OrderedDict(
                (key, value.nonlinear) for key, value in self.data.items()
            )
        )
    
    @property
    def nonlinear_wavenumber_evo(self) -> OrderedDict[int, float]:
        return OrderedDict(
            (key, value.nonlinear_wavenumber) 
            for key, value in self.data.items() if not value.is_null
        )

    def deviation_wavenumber_evo(self, threshold: float = 1.1) -> OrderedDict[int, float]:
        '''
        Get the deviation wavenumber (where B(k) first exceeds threshold)
        for each snapshot.
        
        Parameters
        ----------
        threshold : float
            The boost factor threshold (default 1.1 = 10% deviation).
        
        Returns
        -------
        OrderedDict[int, float]
            Mapping from snapshot index to deviation wavenumber.
        '''
        return OrderedDict(
            (key, value.deviation_wavenumber(threshold=threshold)) 
            for key, value in self.data.items() if not value.is_null
        )

    @property
    def nonlinear_to_linear_evo(self) -> OrderedDict[int, np.ndarray]:
        return OrderedDict(
            (key, value.nonlinear_to_linear) 
            for key, value in self.data.items() if not value.is_null
        )

    def convert_to_comoving(self) -> None:
        if self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = False
            v.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        if not self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            # Force-sync child to expected state before conversion
            v.in_comoving = True
            v.convert_to_physical(scale_factor)
        self.in_comoving = False

    # ------------------------------------------------------------------
    # HDF5 I/O (implemented in fields/power_io.py; imported lazily so that
    # power_io can import the classes defined here without a cycle)
    # ------------------------------------------------------------------
    def save(
            self,
            file_path: Path | str,
            directory: Path | str | None = None
        ) -> Path:
        '''
        Save this instance to an HDF5 file.

        The whole nested structure (moments, per-snapshot estimate/errors,
        each linear / nonlinear / shot-noise spectrum) is written as plain
        arrays and scalar attributes, so the file is readable without this
        package. The derived `spline` and `colormap` fields are rebuilt on
        load rather than stored.

        Parameters
        ----------
        file_path : Path | str
            File name or full path. A `.h5` suffix is appended if missing.
        directory : Path | str | None
            Optional output directory. When given, only the *name* of
            `file_path` is used, so `save("ps.h5", directory=d)` and
            `save(d / "ps.h5")` are equivalent.

        Returns
        -------
        Path
            The path actually written.
        '''
        from .power_io import save_matter_power_spectrum_evo_data
        return save_matter_power_spectrum_evo_data(
            self, file_path, directory=directory
        )

    @classmethod
    def load(
            cls,
            file_path: Path | str,
            directory: Path | str | None = None
        ) -> MatterPowerSpectrumEvoData:
        '''
        Load an instance previously written with `save`.

        Parameters
        ----------
        file_path : Path | str
            File name or full path of the saved file.
        directory : Path | str | None
            Optional directory holding the file.
        '''
        from .power_io import load_matter_power_spectrum_evo_data
        return load_matter_power_spectrum_evo_data(
            file_path, directory=directory
        )

    @staticmethod
    def save_dict(
            evo_data_dict: dict[Any, MatterPowerSpectrumEvoData],
            file_path: Path | str,
            directory: Path | str | None = None
        ) -> Path:
        '''
        Save a `{key: MatterPowerSpectrumEvoData}` mapping to one HDF5 file.

        Intended for the resolution-keyed dicts returned by
        `get_joint_ps_evo_data`, e.g. `{512: ..., 1024: ...}`.
        '''
        from .power_io import save_matter_power_spectrum_evo_data_dict
        return save_matter_power_spectrum_evo_data_dict(
            evo_data_dict, file_path, directory=directory
        )

    @staticmethod
    def load_dict(
            file_path: Path | str,
            directory: Path | str | None = None
        ) -> dict[Any, MatterPowerSpectrumEvoData]:
        '''
        Load a mapping written with `save_dict`.

        A file holding a single instance comes back as a one-entry dict
        keyed by the file stem.
        '''
        from .power_io import load_matter_power_spectrum_evo_data_dict
        return load_matter_power_spectrum_evo_data_dict(
            file_path, directory=directory
        )

    @classmethod
    def from_directory(
            cls, directory: Path,
            cosmo: Cosmology,
            use_music: bool = False,
            use_colossus_linear_ps: bool = False
        ) -> MatterPowerSpectrumEvoData:

        evo_data = get_power_spec_evo(directory, use_music_init_ps=use_music)

        moments, data = MomentsInTime(), OrderedDict()

        for snap_id, ps_data in evo_data.items():

            if ps_data["nonlinear"]["wavenumbers"].size <= 1: continue

            moment = Moment(
                snapshot_id=snap_id,
                scale_factor=ps_data["scale_factor"],
            )

            if use_colossus_linear_ps:
                ps_data["linear"] = get_colossus_linear_ps(
                    cosmo=cosmo,
                    wavenumbers=ps_data["linear"]["wavenumbers"],
                    redshift=moment.redshift
                )

            data[snap_id] = MatterPowerSpectrumData.from_data(ps_data)
            moments.add_moment(moment)

        moments.add_times(cosmo)
        instance = cls(moments=moments, data=data)

        not_available = which_for_colossus_not_available(directory, data.keys())
        if len(not_available) == 0:
            return instance
        
        print(
            f"Saving missing Colossus files for snapshots: {not_available}"
        )

        for snap_id in not_available:
            filepath = Path(directory, f"for_colossus_{snap_id:03d}.txt")
            instance.data[snap_id].estimate.save_for_colossus(filepath)

        return instance
    
    def _get_scale_factors(self, snap_ids: list[int]) -> np.ndarray:
        """Retrieve scale factors for the given snapshot IDs."""
        try:
            return np.asarray(
                self.moments.map_by_attribute(
                    key_attr="snapshot_id",
                    attr_value=snap_ids,
                    return_attr="scale_factor",
                ),
                dtype=float,
            )
        except Exception:
            return np.array(
                [self.moments[sid].scale_factor for sid in snap_ids], 
                dtype=float
            )
        

    def get_wavenumber_bin_evo(
            self, 
            comoving_wavenumber_bin: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0
        ) -> MatterPowerSpectrumAccumulation:

        # Validate input 
        if not np.isfinite(comoving_wavenumber_bin) or comoving_wavenumber_bin <= 0:
            print("Invalid wavenumber bin provided.")
            return MatterPowerSpectrumAccumulation.null_initialize()

        if not [k for k, v in self.data.items() if not v.is_null]:
            print("No valid data available.")
            return MatterPowerSpectrumAccumulation.null_initialize()
        
        # Ensure correct coordinate system
        if return_in_physical and self.in_comoving:
            self.convert_to_physical()
        elif not return_in_physical and not self.in_comoving:
            self.convert_to_comoving()

        matter_ps_evo = self.estimate
        evo_array = matter_ps_evo.get_matter_ps_evo_array(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt,
            return_variance=False
        )
        valid_mask = np.logical_and.reduce([
            evo_array[:, 0] >= min_time_step_value,
            evo_array[:, 0] <= max_time_step_value, 
            ~np.isnan(evo_array).any(axis=1)
        ])
        
        variance_array = matter_ps_evo.get_matter_ps_evo_array(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt,
            return_variance=True
        )
        evo_rate = matter_ps_evo.get_matter_ps_evo_rate(
            wavenumber=comoving_wavenumber_bin,
            wrt=wrt,
            return_variance=False
        )

        return MatterPowerSpectrumAccumulation.from_matter_ps_evo_outputs(
            evo_array=evo_array[valid_mask, :], 
            evo_rate=evo_rate[valid_mask, :],
            evo_variance=variance_array[valid_mask, :],
            in_comoving=self.in_comoving,
            time_metric=wrt
        )

    
    def get_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> MatterPowerSpectraAccumulation:

        if not self.in_comoving:
            self.convert_to_comoving()

        wavenumbers = self.wavenumber_bins

        accumulations = {}
        for k in wavenumbers:
            wav_key = key_format.format(np.log10(k))
            accumulations[wav_key] = self.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=k,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        return MatterPowerSpectraAccumulation(
            accumulations=OrderedDict(
                (key, value) 
                for key, value in sorted(
                    accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=wrt
        )

    
    def get_round_one_wavenumber_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> MatterPowerSpectraAccumulation:

        if not self.in_comoving:
            self.convert_to_comoving()

        wavenumbers = self.round_one_wavenumber_bins

        accumulations = {}
        for k in wavenumbers:
            wav_key = key_format.format(np.log10(k))
            accumulations[wav_key] = self.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=k,
                return_in_physical=return_in_physical,
                wrt=wrt,
                min_time_step_value=min_time_step_value,
                max_time_step_value=max_time_step_value
            )

        return MatterPowerSpectraAccumulation(
            accumulations=OrderedDict(
                (key, value) 
                for key, value in sorted(
                    accumulations.items(), key=lambda item: float(item[0])
                )
            ),
            in_comoving=self.in_comoving,
            time_metric=wrt
        )



def is_for_colossus_available(directory: Path, snap_idx: int) -> bool:
    filepath = Path(directory, f"for_colossus_{snap_idx:03d}.txt")
    return filepath.exists()

def which_for_colossus_not_available(directory: Path, snap_ids: Collection[int]) -> list[int]:
    return [
        snap_id for snap_id in snap_ids 
        if not is_for_colossus_available(directory, snap_id)
    ]


def get_colossus_linear_ps(
        cosmo: Cosmology, 
        wavenumbers: np.ndarray, 
        redshift: float
    ) -> dict[str, np.ndarray]:

    Pk = cosmo.linear_matter_ps(k=wavenumbers, redshift=redshift)

    return {
        "wavenumbers": wavenumbers,
        "amplitudes": Pk,
        "normalized": Pk * wavenumbers**3 / (2 * np.pi**2)
    }

def correlation_to_powerspec(
        radii: np.ndarray, 
        correlation: np.ndarray, 
        ell: int = 0,
        extrapolate: bool = False
    ) -> dict[str, np.ndarray]:
    ''' Modified from nbodykit implementation '''
    P = mcfit.xi2P(radii, l=ell, lowring=True)
    k, Pk = P(correlation, extrap=extrapolate)
    return {
        "wavenumbers":k, 
        "amplitudes": Pk, 
        "normalized": Pk * k**3 / (2 * np.pi**2)
    }

# def overlapping_range(
#         lower_box_range: np.ndarray, 
#         upper_box_range: np.ndarray, 
#     ) -> tuple[np.ndarray, np.ndarray]:
    
#     start = max(lower_box_range[0], upper_box_range[0])
#     end = min(lower_box_range[-1], upper_box_range[-1])

#     where_overlapped = lambda x: np.where((x >= start) & (x <= end))[0]

#     return where_overlapped(lower_box_range), where_overlapped(upper_box_range)