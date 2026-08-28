from __future__ import annotations

import colossus, pdb
import contextlib 
import numpy as np
import colossus.lss

from pathlib import Path
from attrs import define, field
from collections import OrderedDict
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology

from colossus.lss import peaks

# Cosmology for custom σ(R,z) support
from ..cosmo.model import Cosmology

from ..simulation.evo import EvolutionData
from ..simulation.moments import MomentsInTime

@define(slots=True)
class DensityPeaks:
    redshift: float
    present_day_density: float
    transfer: str = field(default="eisenstein98")
    corrections: bool = field(default=True)

    use_ps_file: bool = field(default=False)
    ps_file_path: Path | None = field(default=None)

    # Optional Cosmology for robust sigma/peak height at arbitrary z
    cosmo: Cosmology | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"DensityPeaks(a={1/(1.0+self.redshift):.3f}, T(k)={self.transfer})"
        )
    
    def __eq__(self, value: "DensityPeaks") -> bool:
        return (
            self.redshift == value.redshift
            and self.present_day_density == value.present_day_density
            and self.transfer == value.transfer
            and self.corrections == value.corrections
        )

    @property
    def scale_factor(self) -> float:
        return 1.0 / (1.0 + self.redshift)

    @property
    def _deltac_args(self) -> dict[str, bool]:
        return {"corrections" : self.corrections}
    
    @property
    def _ps_args(self) -> dict[str, str | Path]:
        if not self.use_ps_file:
            return {"model" : self.transfer}
        if self.ps_file_path is None:
            raise ValueError(
                "Must provide a path to the power spectrum filepath"
            )
        return {
            "model" : f"sim{self.transfer.capitalize()}",
            "path" : self.ps_file_path
        }

    @property
    def overdensity(self) -> float:
        return colossus.lss.peaks.collapseOverdensity(
            z=self.redshift, **self._deltac_args
        )
    
    def _heights_from_masses(self, masses: np.ndarray) -> np.ndarray:
        """
        Compute ν(M,z) = δ_c(z)/σ(R,z).
        If a Cosmology is attached (self.cosmo), use its sigma_linear (works for very negative z).
        Otherwise, fall back to Colossus' peaks.peakHeight (may fail for far-future redshifts).
        """
        # Preferred path: use our Cosmology if provided (robust for late times)
        if self.cosmo is not None:
            with contextlib.suppress(Exception):
                # Lagrangian radius corresponding to mass (uses present-day mean density convention).
                R = peaks.lagrangianR(masses)
                # σ(R,z) via our cosmology shim (handles z well beyond Colossus tables).
                sigma = self.cosmo.sigma_linear(R=R, z=self.redshift, ps_args=self._ps_args)
                # δ_c(z) from Colossus (still fine at these z; else could be replaced if needed)
                delta_c = colossus.lss.peaks.collapseOverdensity(z=self.redshift, **self._deltac_args)
                return delta_c / sigma

        # Fallback: Colossus implementation (fast, but limited z-range)
        try:
            return colossus.lss.peaks.peakHeight(
                M=masses,
                z=self.redshift,
                ps_args=self._ps_args,
                deltac_args=self._deltac_args
            )
        except Exception as e:
            raise RuntimeError(
                "Colossus peakHeight failed (likely due to far-future redshift). "
                "Attach a Cosmology instance to DensityPeaks.cosmo to enable the "
                "fallback computation via sigma_linear."
            ) from e

    def with_cosmology(self, cosmo: Cosmology) -> DensityPeaks:
        """Attach a Cosmology instance and return self (fluent helper)."""
        self.cosmo = cosmo
        return self

    def _heights_from_radii(self, radii: np.ndarray) -> np.ndarray:
        masses = get_masses(radii, from_masses=False)
        return self._heights_from_masses(masses)

    def heights(self, values: np.ndarray, from_masses: bool = True) -> np.ndarray: 
        return (
            self._heights_from_masses(values)
            if from_masses else
            self._heights_from_radii(values)
        )
    
    def mass_variance(self, values: np.ndarray, from_masses: bool = True) -> np.ndarray:
        return self.overdensity / self.heights(values, from_masses=from_masses)
    
    def sigma_jacobian(
            self, 
            values: np.ndarray, 
            wrt_mass: bool = True,
            from_masses: bool = True
        ) -> np.ndarray:

        masses = get_masses(values, from_masses=from_masses)

        # delta_c / (dln(sigma) / dln(R)) = delta_c * (dln(R) / dln(sigma)) 
        derivative = colossus.lss.peaks.peakHeight(
            M=masses, 
            z=self.redshift, 
            ps_args=self._ps_args,
            deltac_args=self._deltac_args,
            sigma_args = {"derivative" : True},
        )

        dln_sigma_dln_R = self.overdensity / (derivative)

        # dlnM = 3 * dln_R
        return (dln_sigma_dln_R / 3.0) if wrt_mass else dln_sigma_dln_R
    
    def jacobian(
            self, 
            values: np.ndarray, 
            wrt_mass: bool = True,
            from_masses: bool = True
        ) -> np.ndarray:
        ''' dln_nu = -dln_sigma '''
        return -1.0 * self.sigma_jacobian(
            values=values, 
            wrt_mass=wrt_mass, 
            from_masses=from_masses
        )
    
    def dln_nu(self, values: np.ndarray, from_masses: bool = True) -> np.ndarray:
        return np.gradient(np.log(self.heights(values, from_masses=from_masses)))
    
    def height_density(
            self, 
            number_density: np.ndarray, 
            values: np.ndarray, 
            from_masses: bool = True
        ) -> np.ndarray:

        return number_density / self.dln_nu(values, from_masses=from_masses)
    
    def multiplicity(
            self,
            number_density: np.ndarray,
            values: np.ndarray,
        ) -> np.ndarray:
        ''' Mass-based halo multiplicity: (M / rho) * dn/dln(nu) '''
        return (values / self.present_day_density) * self.height_density(
            number_density, values
        )

    
    def jacobian_interpolant(
            self, 
            values: np.ndarray, 
            wrt_mass: bool = True,
            from_masses: bool = True
        ) -> interp1d:
        masses = get_masses(values, from_masses=from_masses)
        return interp1d(
            np.log10(masses), 
            self.jacobian(
                values=masses, 
                wrt_mass=wrt_mass,
                from_masses=True
            )
        )
    
    def sigma_jacobian_interpolant(
            self, 
            values: np.ndarray, 
            wrt_mass: bool = True,
            from_masses: bool = True
        ) -> interp1d:
        masses = get_masses(values, from_masses=from_masses)
        return interp1d(
            np.log10(masses), 
            self.sigma_jacobian(
                values=masses, 
                wrt_mass=wrt_mass,
                from_masses=True
            )
        )
    
@define(slots=True)
class DensityPeaksEvo(EvolutionData):
    data: OrderedDict[int, DensityPeaks]

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_moments(
            cls, 
            moments: MomentsInTime, 
            present_day_density: float,
            transfer: str = "eisenstein98",
            use_ps_file: bool = False,
            ps_file_paths: OrderedDict[int, Path] | None = None,
            cosmo: Cosmology | None = None,
        ) -> DensityPeaksEvo:
        return cls(
            moments=moments,
            data=OrderedDict(
                (
                    snap_id, DensityPeaks(
                        redshift=moments.redshifts[i],
                        present_day_density=present_day_density,
                        transfer=transfer,
                        use_ps_file=use_ps_file,
                        ps_file_path=(
                            None if ps_file_paths is None
                            else ps_file_paths.get(snap_id)
                        ),
                        cosmo=cosmo,
                    )
                )
                for i, snap_id in enumerate(moments.snapshot_ids)
            )
        )

    def jacobian(
            self,
            values: np.ndarray, 
            wrt_mass: bool = True,
            from_masses: bool = True
        ) -> np.ndarray:
        return self.data[max(self.data)].jacobian(
            values=values, 
            wrt_mass=wrt_mass,
            from_masses=from_masses
        )
    
    # will probably put in a method to get the time evolution of a peak value 
    # for a given mass value 

def get_masses(values: np.ndarray, from_masses: bool = True) -> np.ndarray:
    return values if from_masses else colossus.lss.peaks.lagrangianM(values)
