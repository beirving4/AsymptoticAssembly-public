from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt, pdb

from typing import Any
from pathlib import Path
from attrs import define, field
from collections import OrderedDict
from collections.abc import Collection

from ..cosmo.model import Cosmology
from ..simulation.evo import EvolutionData
from ..simulation.moments import MomentsInTime
from ..utils.get_data import get_for_colossus_powerspec_paths
from .peaks import DensityPeaks, DensityPeaksEvo
from .knn import (
    kNNDistribution, 
    kNNDistributionData, 
    kNNDistributionEvoData,
    _coerce_knn,
    _coerce_knn_data
)
from .power import (
    PowerSpectrum, 
    PowerSpectrumData,
    MatterPowerSpectrum, 
    MatterPowerSpectrumData,
    MatterPowerSpectrumEvo,
    MatterPowerSpectrumEvoData
)
from .correlation import (
    FreezeOutMetrics,
    MatterTwoPointCorrelationData,
    TwoPointCorrelation, 
    TwoPointCorrelationData,
    TwoPointCorrelationEvoData,
    MatterTwoPointCorrelation, 
    MatterTwoPointCorrelationEvo,
    MatterTwoPointCorrelationEvoData,
)
from .viz import (
    display_clustering,
)
from .accumulation import (
    FieldAccumulation,
    FieldAccumulations,
    MatterTwoPointCorrelationAccumulation,
    MatterTwoPointCorrelationAccumulations,
    MatterPowerSpectrumAccumulation,
    MatterPowerSpectraAccumulation,
    FieldMatterAccumulation,
    FieldMatterAccumulations,
)

def safe_get(d: dict[int | str, Any], key: int | str) -> Any | None:
    return d.get(key) if (d is not None) else None

@define(slots=True)
class FieldContribution:
    power: PowerSpectrumData
    correlation: TwoPointCorrelationData
    knn: kNNDistribution = field(default=None, converter=_coerce_knn, repr=False)  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"FieldContribution(\n"
            f" power={self.power}, \n"
            f" correlation={self.correlation}\n"
            f")"    
        )

    @property
    def has_knn(self) -> bool:
        return self.knn is not None and not self.knn.is_null
    
    @property
    def is_null(self) -> bool:
        return self.power.is_null or self.correlation.is_null
    
    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology, 
            separations: np.ndarray | None = None,
            wavenumbers: np.ndarray | None = None,
        redshift: float = 0.0
        ) -> FieldContribution:

        # Assert that at least one of separations or wavenumbers is provided
        if (separations is None) and (wavenumbers is None):
            raise ValueError("At least one of 'separations' or 'wavenumbers' must be provided.")

        if (wavenumbers is not None) and (separations is None):
            separations = 2 * np.pi / wavenumbers

        if (separations is not None) and (wavenumbers is None):
            wavenumbers = 2 * np.pi / separations

        return cls(
            power = PowerSpectrumData.from_cosmology(
                cosmo=cosmo,
                wavenumbers=wavenumbers,
                redshift=redshift
            ),
            correlation = TwoPointCorrelationData.from_cosmology(
                cosmo=cosmo,
                separations=separations,
                redshift=redshift
            )
        )

    def display_clustering(
            self, 
            text_size: int = 16,
            use_normalized: bool = False,
            use_scaled: bool = False,
            ax_ps: plt.Axes | None = None,
            ax_corr: plt.Axes | None = None, 
            return_fig: bool = False,
            wspace: float = 0.3,
            color: str | tuple | None = None
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        if (ax_ps is None) or (ax_corr is None):
            _, (ax_ps, ax_corr) = plt.subplots(
                1, 2, figsize=(12, 6), gridspec_kw={"wspace" : wspace}
            )

        ax_ps = self.power.display(
            text_size=text_size,
            use_normalized=use_normalized,
            ax=ax_ps,
            return_fig=True,
            color=color
        )

        ax_corr = self.correlation.display(
            text_size=text_size,
            use_scaled=use_scaled,
            ax=ax_corr,
            return_fig=True,
            color=color
        )

        if return_fig:
            return ax_ps, ax_corr
        
        plt.show()

@define(slots=True)
class DensityField:
    correlation: MatterTwoPointCorrelationData
    power: MatterPowerSpectrumData
    peaks: DensityPeaks
    knn: kNNDistributionData = field(default=None, converter=_coerce_knn_data, repr=False)

    def __repr__(self) -> str:
        estimate = self.power.estimate
        linear_wavenumber_range = estimate.linear.wavenumber_range
        nonlinear_wavenumber_range = estimate.nonlinear.wavenumber_range
        return (
            f"DensityField("
            f"k_lin=({linear_wavenumber_range[0]:.3e}-{linear_wavenumber_range[1]:.3e}), "
            f"k=({nonlinear_wavenumber_range[0]:.3e}-{nonlinear_wavenumber_range[1]:.3e}), "
            f"peaks={self.peaks})"
        )
    
    @property
    def is_null(self) -> bool:
        return self.power.is_null or self.correlation.is_null

    @property
    def separation_scales(self) -> np.ndarray:
        return self.correlation.estimate.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.correlation.estimate.round_one_separation_scales
    
    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.power.estimate.wavenumber_bins
    

    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.power.estimate.round_one_wavenumber_bins


    @classmethod
    def from_ps_file(
            cls, file_path: Path, 
            linear_file_path: Path,
            redshift: float,
            present_day_density: float,
            transfer: str = "eisenstein98",
        ) -> DensityField:
        
        ps = MatterPowerSpectrumData.from_file(file_path, linear_file_path)
        return cls(
            power = ps,
            correlation = MatterTwoPointCorrelationData.from_power_spectrum_data(ps),
            peaks = DensityPeaks(
                redshift=redshift,
                present_day_density=present_day_density,
                transfer=transfer,
                use_ps_file=True,
                ps_file_path=file_path
            )
        )
    
    @property
    def linear(self) -> FieldContribution:
        return FieldContribution(
            power = self.power.linear,
            correlation = self.correlation.linear
        )
    
    @property
    def nonlinear(self) -> FieldContribution:
        return FieldContribution(
            power = self.power.nonlinear,
            correlation = self.correlation.nonlinear
        )
    
    def display_clustering(
            self,
            text_size: int = 16,
            use_normalized: bool = False,
            show_nonlinear: bool = True,
            show_linear: bool = True,
            show_shot_noise: bool = True,
            use_scaled: bool = False,
            return_fig: bool = False,
            wspace: float = 0.3,
            color: str | tuple | None = None,
            ax_ps: plt.Axes | None = None,
            ax_corr: plt.Axes | None = None,
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:
    
        if (ax_ps is None) or (ax_corr is None):
            _, (ax_ps, ax_corr) = plt.subplots(
                1, 2, figsize=(12, 6), gridspec_kw={"wspace" : wspace}
            )

        ax_ps = self.power.display(
            text_size=text_size,
            use_normalized=use_normalized,
            show_nonlinear=show_nonlinear,
            show_linear=show_linear,
            show_shot_noise=show_shot_noise,
            ax=ax_ps,
            return_fig=True,
            color=color, 
            show_legend=False
        )

        ax_corr = self.correlation.display(
            text_size=text_size,
            use_scaled=use_scaled,
            show_nonlinear=show_nonlinear,
            show_linear=show_linear,
            ax=ax_corr,
            return_fig=True,
            color=color
        )

        if return_fig:
            return ax_ps, ax_corr
        
        plt.show()


@define(slots=True)
class FieldContributionEvo(EvolutionData):
    data: OrderedDict[int, FieldContribution]

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def is_null(self) -> bool:
        return all(x.is_null for x in self.data.values())
    
    @property
    def final_contribution(self) -> FieldContribution:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null field contribution data available")
        
    
    @classmethod
    def from_cosmology(
            cls, 
            cosmo: Cosmology, 
            moments: MomentsInTime,
            separations: np.ndarray | None = None,
            wavenumbers: np.ndarray | None = None,
        ) -> FieldContributionEvo:
        """
        Create a FieldContributionEvo from a cosmology model at the redshifts
        specified by the MomentsInTime object.
        
        Parameters
        ----------
        cosmo : Cosmology
            The cosmology model to use for computing power spectra and correlations.
        moments : MomentsInTime
            The time steps (snapshot IDs, scale factors, redshifts) at which
            to compute the field contributions.
        separations : np.ndarray, optional
            The separation array [Mpc/h] at which to evaluate the correlations.
            If not provided, derived from wavenumbers as 2π/k.
        wavenumbers : np.ndarray, optional
            The wavenumber array [h/Mpc] at which to evaluate the power spectra.
            If not provided, derived from separations as 2π/r.
        
        Returns
        -------
        FieldContributionEvo
            A FieldContributionEvo instance containing linear field contributions
            at each snapshot specified in moments.
        """
        return cls(
            moments=moments,
            data=OrderedDict(
                (snap_id, FieldContribution.from_cosmology(
                    cosmo=cosmo,
                    separations=separations,
                    wavenumbers=wavenumbers,
                    redshift=moments[snap_id].redshift
                ))
                for snap_id in moments.snapshot_ids
            )
        )

@define(slots=True)
class DensityFieldEvo(EvolutionData):
    data: OrderedDict[int, DensityField]
    in_comoving: bool = field(default=True)

    def __repr__(self) -> str:
        return super().__repr__()

    def convert_to_comoving(self) -> None:
        if self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            v.power.convert_to_comoving(scale_factor)
            v.correlation.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self) -> None:
        if not self.in_comoving:
            return
        for snap_idx, v in self.data.items():
            if v.is_null: continue
            scale_factor = self.moments[snap_idx].scale_factor
            v.power.convert_to_physical(scale_factor)
            v.correlation.convert_to_physical(scale_factor)
        self.in_comoving = False

    @classmethod
    def from_relations(
            cls, 
            moments: MomentsInTime,
            correlation_evo: MatterTwoPointCorrelationEvoData | TwoPointCorrelationEvoData,
            peaks_evo: DensityPeaksEvo,
            spectra_evo: MatterPowerSpectrumEvoData,
            knn_evo: kNNDistributionEvoData | None = None
        ) -> DensityFieldEvo:

        if isinstance(correlation_evo, TwoPointCorrelationEvoData):
            correlation_evo = MatterTwoPointCorrelationEvoData.from_tpcf_evo_data(
                tpcf_evo_data=correlation_evo,
                linear_ps_evo_data=spectra_evo.linear_power_spectrum_evo
            )

            # pdb.set_trace()

        data = OrderedDict()
        for snap_idx, peaks in sorted(peaks_evo.data.items(), key=lambda x: x[0]):
            ps = spectra_evo.data.get(snap_idx, None)
            tpcf = correlation_evo.data.get(snap_idx, None)
            knn = safe_get(knn_evo.data, snap_idx)

            if (tpcf is None) and (ps is None) and (knn is None):
                continue
            elif (tpcf is None) and (ps is None):
                final_knn_data = knn_evo.data[max(knn_evo.data)]
                final_radii = final_knn_data.estimate.radii[max(final_knn_data.estimate.radii)]
                num_vals = final_radii.size
                tpcf = MatterTwoPointCorrelationData.null_initialize(num_vals)
                ps = MatterPowerSpectrumData.null_initialize(num_vals)
            elif tpcf is None:
                num_vals = ps.estimate.nonlinear.wavenumbers.size
                tpcf = MatterTwoPointCorrelationData.null_initialize(num_vals)
            elif ps is None:
                num_vals = tpcf.estimate.nonlinear.radii.size
                ps = MatterPowerSpectrumData.null_initialize(num_vals)

            data[snap_idx] = DensityField(
                correlation=tpcf,
                power=ps,
                peaks=peaks,
                knn=knn
            )


        return cls(moments=moments, data=OrderedDict(sorted(data.items())))
    

    @classmethod
    def from_directories(
            cls, 
            cosmo: Cosmology,
            ps_dir: Path,
            particle_tpcf_dir: Path,
            particle_knn_data_dir: Path | None = None,
            transfer: str = "eisenstein98",
            use_music: bool = False,
            use_colossus_linear_ps: bool = False,
            for_jackknife: bool = False,
            from_treecorr: bool = False,
            use_comoving: bool = True
        ) -> DensityFieldEvo:

        ps_evo = MatterPowerSpectrumEvoData.from_directory(
            directory=ps_dir, 
            cosmo=cosmo, 
            use_music=use_music,
            use_colossus_linear_ps=use_colossus_linear_ps
        )

        tpcf_evo = TwoPointCorrelationEvoData.load(
            directory=particle_tpcf_dir,
            sim_cosmo=cosmo,
            moments=ps_evo.moments,
            for_jackknife=for_jackknife,
            from_treecorr=from_treecorr
        )

        peaks_evo = DensityPeaksEvo.from_moments(
            moments=ps_evo.moments,
            present_day_density=cosmo.present_day_matter_density,
            transfer=transfer,
            use_ps_file=True,
            ps_file_paths=get_for_colossus_powerspec_paths(ps_dir)
        )

        if particle_knn_data_dir is not None:
            knn_evo = kNNDistributionEvoData.load(
                directory=particle_knn_data_dir,
                moments=ps_evo.moments, 
                sim_cosmo=cosmo
            )

        return cls.from_relations(
            moments=ps_evo.moments,
            correlation_evo=tpcf_evo,
            peaks_evo=peaks_evo,
            spectra_evo=ps_evo,
            knn_evo=knn_evo if (particle_knn_data_dir is not None) else None
        )

    @classmethod
    def from_ps_directory(
            cls, ps_dir: Path, 
            cosmo: Cosmology,
            ell: int = 0,
            extrapolate: bool = True,
            transfer: str = "eisenstein98",
            use_music: bool = False,
            use_colossus_linear_ps: bool = False,
            use_comoving: bool = True
        ) -> DensityFieldEvo:

        ps_evo = MatterPowerSpectrumEvoData.from_directory(
            directory=ps_dir, 
            cosmo=cosmo, 
            use_music=use_music,
            use_colossus_linear_ps=use_colossus_linear_ps
        )

        tpcf_evo = MatterTwoPointCorrelationEvoData.from_power_spectrum_evo_data(
            power_data=ps_evo, 
            ell=ell, 
            extrapolate=extrapolate,
            in_comoving=use_comoving
        )

        peaks_evo = DensityPeaksEvo.from_moments(
            moments=ps_evo.moments,
            present_day_density=cosmo.present_day_matter_density,
            transfer=transfer,
            use_ps_file=True,
            ps_file_paths=get_for_colossus_powerspec_paths(ps_dir)
        )

        return cls.from_relations(
                moments=ps_evo.moments,
                correlation_evo=tpcf_evo,
                peaks_evo=peaks_evo,
                spectra_evo=ps_evo
            )

    @property
    def spectra_evo(self) -> MatterPowerSpectrumEvoData:
        return MatterPowerSpectrumEvoData(
            moments=self.moments,
            data=OrderedDict(
                (k, v.power) for k, v in self.data.items() if not v.is_null
            ),
            in_comoving=self.in_comoving
        )
    
    @property
    def correlation_evo(self) -> MatterTwoPointCorrelationEvoData:
        return MatterTwoPointCorrelationEvoData(
            moments=self.moments,
            data=OrderedDict(
                (k, v.correlation) for k, v in self.data.items() if not v.is_null
            ),
            in_comoving=self.in_comoving,
            # must connect the TreeCorrNN object here, it may not attach automatically 
        )

    @property
    def peaks_evo(self) -> DensityPeaksEvo:
        return DensityPeaksEvo(
            moments=self.moments,
            data=OrderedDict((k, v.peaks) for k, v in self.data.items())
        )
    

    @property
    def knn_evo(self) -> kNNDistributionEvoData:
        return kNNDistributionEvoData(
            moments=self.moments,
            data=OrderedDict((k, v.knn) for k, v in self.data.items())
        )

    @property
    def linear_field_evo(self) -> FieldContributionEvo:
        return FieldContributionEvo(
            moments=self.moments,
            data=OrderedDict((k, v.linear) for k, v in self.data.items())
        )
    
    @property
    def nonlinear_field_evo(self) -> FieldContributionEvo:
        return FieldContributionEvo(
            moments=self.moments,
            data=OrderedDict((k, v.nonlinear) for k, v in self.data.items())
        )
    
    @property
    def final_field(self) -> DensityField:
        if (keys := {k for k, v in self.data.items() if not v.is_null}):
            return self.data[max(keys)]
        else:
            raise ValueError("No non-null field data available")

    @property
    def separation_scales(self) -> np.ndarray:
        return self.final_field.separation_scales
    
    @property
    def round_one_separation_scales(self) -> np.ndarray:
        return self.final_field.round_one_separation_scales
    
    @property
    def wavenumber_bins(self) -> np.ndarray:
        return self.final_field.wavenumber_bins
    
    @property
    def round_one_wavenumber_bins(self) -> np.ndarray:
        return self.final_field.round_one_wavenumber_bins  

    def get_radial_bin_evo(
            self,
            bin_value: float,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            is_wavenumber: bool = False
        ) -> FieldMatterAccumulation:

        spectra_evo = self.spectra_evo
        correlation_evo = self.correlation_evo

        kwargs = {
            "return_in_physical": return_in_physical,
            "wrt": wrt,
            "min_time_step_value": min_time_step_value,
            "max_time_step_value": max_time_step_value
        }

        if is_wavenumber:
            comoving_radial_bin = (2.0 * np.pi) / bin_value
            comoving_wavenumber_bin = bin_value
        else:
            comoving_radial_bin = bin_value
            comoving_wavenumber_bin = (2.0 * np.pi) / comoving_radial_bin


        return FieldMatterAccumulation(
            tpcf=correlation_evo.get_radial_bin_evo(
                comoving_radial_bin=comoving_radial_bin,
                **kwargs
            ),
            spectrum=spectra_evo.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=comoving_wavenumber_bin,
                **kwargs
            )
        )

    

    def get_accumulations(
            self,
            bin_values: Collection[float],
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            is_wavenumber: bool = False,
            key_format: str = "{:.1f}"
        ) -> FieldMatterAccumulations:

        kwargs = {
            "return_in_physical": return_in_physical,
            "wrt": wrt,
            "min_time_step_value": min_time_step_value,
            "max_time_step_value": max_time_step_value
        }

        if is_wavenumber:
            comoving_radial_bins = (2.0 * np.pi) / bin_values
            comoving_wavenumber_bins = bin_values
            iterator = sorted(
                zip(comoving_radial_bins, comoving_wavenumber_bins),
                key=lambda x: x[0]
            )
        else:
            comoving_radial_bins = bin_values
            comoving_wavenumber_bins = (2.0 * np.pi) / comoving_radial_bins
            iterator = sorted(
                zip(comoving_radial_bins, comoving_wavenumber_bins),
                key=lambda x: x[1]
            )

        spectra_evo = self.spectra_evo
        correlation_evo = self.correlation_evo

        separation_accumulations = OrderedDict()
        wavenumber_accumulations = OrderedDict()

        for (radial_bin, wavenumber_bin) in iterator:

            radial_bin_key = key_format.format(radial_bin)
            wavenumber_bin_key = key_format.format(wavenumber_bin)
            separation_accumulations[radial_bin_key] = correlation_evo.get_radial_bin_evo(
                comoving_radial_bin=radial_bin,
                **kwargs
            )
            wavenumber_accumulations[wavenumber_bin_key] = spectra_evo.get_wavenumber_bin_evo(
                comoving_wavenumber_bin=wavenumber_bin,
                **kwargs
            )

        return FieldMatterAccumulations(
            tpcf=MatterTwoPointCorrelationAccumulations(
                accumulations=separation_accumulations,
                in_comoving=not return_in_physical,
                time_metric=wrt
            ),
            spectrum=MatterPowerSpectraAccumulation(
                accumulations=wavenumber_accumulations,
                in_comoving=not return_in_physical,
                time_metric=wrt
            )
        )

    def get_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> FieldMatterAccumulations:

        if not self.in_comoving:
            self.convert_to_comoving()

        return self.get_accumulations(
            bin_values=self.separation_scales,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            is_wavenumber=False,
            key_format=key_format
        )
    
    def get_round_one_separation_scale_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> FieldMatterAccumulations:

        if not self.in_comoving:
            self.convert_to_comoving()

        return self.get_accumulations(
            bin_values=self.round_one_separation_scales,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            is_wavenumber=False,
            key_format=key_format
        )
    

    def get_wavenumber_bin_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> FieldMatterAccumulations:

        if not self.in_comoving:
            self.convert_to_comoving()

        return self.get_accumulations(
            bin_values=self.wavenumber_bins,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            is_wavenumber=True,
            key_format=key_format
        )
    
    def get_round_one_wavenumber_bin_accumulations(
            self,
            return_in_physical: bool = False,
            wrt: str = "scale_factor",
            min_time_step_value: float = 0.0,
            max_time_step_value: float = 100.0,
            key_format: str = "{:.1f}"
        ) -> FieldMatterAccumulations:

        if not self.in_comoving:
            self.convert_to_comoving()

        return self.get_accumulations(
            bin_values=self.round_one_wavenumber_bins,
            return_in_physical=return_in_physical,
            wrt=wrt,
            min_time_step_value=min_time_step_value,
            max_time_step_value=max_time_step_value,
            is_wavenumber=True,
            key_format=key_format
        )

    def compute_correlation_freezeout(
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
        """Compute freeze-out metrics for the matter correlation function.
        
        This method quantifies how the correlation matrix evolves toward its
        final state on a fixed physical separation grid. By comparing at fixed
        physical (not comoving) scales, we isolate true dynamical freeze-out
        from trivial scale-factor drift effects.
        
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
            If provided and scale_bands is None, create this many log-spaced bands
            spanning the physical grid range.
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
            all epochs) for consistent comparison. This removes artifacts from
            changing bin validity and provides a cleaner freeze-out test.
            Default False.
        fixed_mask_min_scale_factor : float | None
            Minimum scale factor for epochs included in fixed mask computation.
            Useful for excluding early epochs with very sparse coverage.
            
        Returns
        -------
        FreezeOutMetrics
            Container with freeze-out distance metrics including:
            - scale_factors: scale factor at each epoch
            - frobenius_global: normalized Frobenius distance to final
            - frobenius_bands: per-band distances (if bands specified)
            - fixed_mask_global: bins used if use_fixed_mask=True
            
        Examples
        --------
        >>> # Global freeze-out with default grid
        >>> metrics = field_evo.compute_correlation_freezeout()
        >>> plt.plot(metrics.scale_factors, metrics.frobenius_global)
        
        >>> # Hierarchical analysis with 3 log-spaced bands
        >>> metrics = field_evo.compute_correlation_freezeout(
        ...     r_phys_min=0.5, r_phys_max=40.0, n_bands=3
        ... )
        >>> for band, dist in metrics.frobenius_bands.items():
        ...     plt.plot(metrics.scale_factors, dist, label=band)
        
        >>> # Custom bands for specific physical scales
        >>> bands = {"1halo": (0.2, 2.0), "transition": (2.0, 10.0), "2halo": (10.0, 50.0)}
        >>> metrics = field_evo.compute_correlation_freezeout(scale_bands=bands)
        
        >>> # Use TreeCorr's cross-patch weighted covariance
        >>> metrics = field_evo.compute_correlation_freezeout(
        ...     use_treecorr=True, cross_patch_weight='match'
        ... )
        
        >>> # Fixed-mask robustness check (exclude early sparse epochs)
        >>> metrics = field_evo.compute_correlation_freezeout(
        ...     use_treecorr=True,
        ...     use_fixed_mask=True,
        ...     fixed_mask_min_scale_factor=0.3,
        ...     scale_bands={"1halo": (0.2, 2.0), "2halo": (10.0, 50.0)}
        ... )
        """
        return self.correlation_evo.compute_freezeout_metrics(
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
    


    def display_clustering(
            self,
            target_scale_factors: Collection[float] | None = None,
            text_size: int = 16,
            use_normalized: bool = False,
            show_nonlinear: bool = True,
            show_linear: bool = True,
            show_shot_noise: bool = True,
            use_scaled: bool = False,
            return_fig: bool = False,
            wspace: float = 0.3,
            color_palette: str = "viridis",
            ax_ps: plt.Axes | None = None,
            ax_corr: plt.Axes | None = None,
            legend_xloc: float = 1.05,
            legend_yloc: float = 0.92
        ) -> tuple[plt.Axes | None, plt.Axes | None] | None:

        if (ax_ps is None) or (ax_corr is None):
            _, (ax_ps, ax_corr) = plt.subplots(
                1, 2, figsize=(12, 6), gridspec_kw={"wspace" : wspace}
            )

        if target_scale_factors is None:
            target_scale_factors = self.moments.scale_factors
        
        colormap = plt.cm.get_cmap(color_palette, len(target_scale_factors))

        ax_ps = self.spectra_evo.display(
            target_scale_factors=target_scale_factors,
            text_size=text_size,
            use_normalized=use_normalized,
            show_nonlinear=show_nonlinear,
            show_linear=show_linear,
            show_shot_noise=show_shot_noise,
            ax_main=ax_ps,
            return_fig=True,
            colormap=colormap, 
            show_legend=False
        )

        ax_corr = self.correlation_evo.display(
            target_scale_factors=target_scale_factors,
            text_size=text_size,
            use_scaled=use_scaled,
            show_nonlinear=show_nonlinear,
            show_linear=show_linear,
            ax_main=ax_corr,
            return_fig=True,
            colormap=colormap,
            legend_xloc=legend_xloc,
            legend_yloc=legend_yloc
        )

        if return_fig:
            return ax_ps, ax_corr
        
        plt.show()

    
def get_colossus_powerspecs(ps_dir: Path, ps_evo: MatterPowerSpectrumEvo) -> OrderedDict[int, Path]:

    colossus_linear_ps_paths = get_for_colossus_powerspec_paths(ps_dir)

    if len(colossus_linear_ps_paths) == 0:
        ps_evo.save_for_colossus(ps_dir)
        colossus_linear_ps_paths = get_for_colossus_powerspec_paths(ps_dir)

    if any(not path.exists() for path in colossus_linear_ps_paths.values()):
        ps_evo.save_for_colossus(ps_dir)

    return get_for_colossus_powerspec_paths(ps_dir)