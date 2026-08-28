from __future__ import annotations

import sys
import h5py
import pickle, pdb
import colossus
import numpy as np


from pathlib import Path
from astropy import units as u
from attrs import define, field
from scipy.integrate import quad
import astropy.cosmology.units as cu
from astropy import constants as const
from collections.abc import Collection
from functools import lru_cache
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology


from .config import CosmoType, CosmoConfig
from .parameters import get_cosmo_parameters
from ..utils.get_data import read_from_music, read_from_gadget

ColossusCosmology = colossus.cosmology.cosmology.Cosmology


type IntType = int | np.integer
type FloatType = float | np.floating
type Number = IntType | FloatType
type Numbers = Collection[Number] | np.ndarray

@define(slots=True)
class Cosmology:
    H0: float
    Omega_m: float
    Omega_b: float
    Omega_L: float
    sigma_8: float
    nspec: float
    w0: float
    wa: float
    cfg: CosmoConfig
    h: float = field(init=False)
    __cosmo: ColossusCosmology = field(init=False, repr=False)

    ps_file_path: Path | None = field(default=None, repr=False)
    max_wavenumber: float = field(default=5000.0, repr=False)
    powerspec_type: str = field(default="cdm", repr=False)

    def __attrs_post_init__(self) -> None:
        self.h = self.H0 / 100
        self.set_cosmology()

    def set_cosmology(self) -> None:
        cosmology.addCosmology(
            self.name,
            {
                "flat": True,
                "H0": self.H0,
                "Om0": self.Omega_m,
                "Ob0": self.Omega_b,
                "sigma8": self.sigma_8,
                "ns": self.nspec,
                "w0": self.w0,
                "wa": self.wa,
            },
        )
        self.__cosmo = cosmology.setCosmology(self.name)

    @classmethod
    def from_directory(
            cls, sim_dir: Path, 
            name: str = "sim_cosmo",
            from_music: bool = True, 
            param_path: Path | None = None
        ) -> Cosmology:

        filepath = get_cosmo_filepath(sim_dir, from_music, param_path=param_path)

        cosmo_cfg = (
            read_from_music(filepath) if from_music else read_from_gadget(filepath)
        )

        transfer = cosmo_cfg.pop("transfer", "simEfstathiou")

        # Create Cosmology object
        return cls(cfg = CosmoConfig.from_str(name, transfer), **cosmo_cfg)

    @classmethod
    def from_pickle(cls, directory: Path) -> Cosmology:
        filepath = Path(directory, "cosmology.pkl")
        with open(filepath, "rb") as f:
            cosmo = pickle.load(f)
        cosmo._set_cosmology()
        return cosmo

    @classmethod
    def from_hdf5(cls, cosmo_h5py_group: h5py.Group) -> Cosmology:
        return cls(
            H0=cosmo_h5py_group.attrs["H0"],
            Omega_m=cosmo_h5py_group.attrs["Omega_m"],
            Omega_b=cosmo_h5py_group.attrs["Omega_b"],
            Omega_L=cosmo_h5py_group.attrs["Omega_L"],
            sigma_8=cosmo_h5py_group.attrs["sigma_8"],
            nspec=cosmo_h5py_group.attrs["nspec"],
            w0=cosmo_h5py_group.attrs["w0"],
            wa=cosmo_h5py_group.attrs["wa"],
            cfg=CosmoConfig(
                cosmo=CosmoType.DUMMY.value,
                transfer_fn=cosmo_h5py_group.attrs["transfer"],
            )
            
        )

    @classmethod
    def from_loss_sim_cfgs(
            cls, cosmo_name: str, seed: int = sys.maxsize
        ) -> Cosmology:
        cosmo_params = get_cosmo_parameters(cosmo_name)
        if seed != sys.maxsize:
            cosmo_params["cfg"]["seed"] = seed
        params_dict = {k : v for k, v in cosmo_params.items() if k != "cfg"}
        cfg_dict = cosmo_params["cfg"]
        return cls(**params_dict, cfg=CosmoConfig(**cfg_dict))

    def save(self, directory: Path) -> None:
        filepath = Path(directory, "cosmology.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other: Cosmology) -> bool:
        return np.logical_and.reduce([
            np.isclose(self.H0, other.H0),
            np.isclose(self.Omega_m, other.Omega_m),
            np.isclose(self.Omega_b, other.Omega_b),
            np.isclose(self.Omega_L, other.Omega_L),
            np.isclose(self.sigma_8, other.sigma_8),
            np.isclose(self.nspec, other.nspec),
            np.isclose(self.w0, other.w0),
            np.isclose(self.wa, other.wa),
            self.transfer == other.transfer,
        ]).astype(bool) # type: ignore
    
    def __hash__(self) -> int:
        return hash(
            (
                self.H0,
                self.Omega_m,
                self.Omega_b,
                self.Omega_L,
                self.sigma_8,
                self.nspec,
                self.w0,
                self.wa,
                self.transfer,
            )
        )

    @property
    def name(self) -> str:
        return self.cfg.cosmo

    @property
    def transfer(self) -> str:
        return self.cfg.transfer_fn

    @property
    def seed(self) -> int:
        return self.cfg.seed

    @property
    def _ps_args(self) -> dict:
        return {
            "model" : self.transfer,
            "kmax" : self.max_wavenumber, 
            "ps_type" : self.powerspec_type,
            "path" : self.ps_file_path 
        }

    def update_ps_args(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self._ps_args:
                self._ps_args[key] = value
    
    @property
    def Omega_k(self) -> float:
        return 1 - self.Omega_m - self.Omega_L
    
    @property
    def Omega_cdm(self) -> float:
        return self.Omega_m - self.Omega_b
    
    @property
    def Omega_m_h2(self) -> float:
        return self.Omega_m * self.h**2
    @property
    def Omega_b_h2(self) -> float:
        return self.Omega_b * self.h**2
    
    @property
    def Omega_cdm_h2(self) -> float:
        return self.Omega_cdm * self.h**2

    @property
    def is_flat(self) -> bool: 
        return self.__cosmo.isFlat()
    
    def _get_redshift_for_colossus(
            self,
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> Number | Numbers:

        if not is_valid_time_input(t=t, a=a, z=z):
            raise ValueError(
                "Must provide exactly one of z, t, or a to calculate w"
            )
        if a is not None:
            z = (1.0 / a) - 1.0
        if t is not None:
            z = self.__cosmo.age((t * u.Gyr).value, inverse=True)

        return z

    def _get_scale_factor_for_integration(
            self,
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> Number | Numbers:

        if not is_valid_time_input(t=t, a=a, z=z):
            raise ValueError(
                "Must provide exactly one of z, t, or a to calculate w"
            )
        if z is not None:
            a = 1.0 / (1.0 + z)
        if t is not None:
            z = self.__cosmo.age((t * u.Gyr).value, inverse=True)
            a = 1.0 / (1.0 + z)
        
        return a
    
    def get_eos_w(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> float | Collection[float] | np.ndarray:

        return self.__cosmo.w(z=self._get_redshift_for_colossus(z=z, t=t, a=a))
    
    @property
    def colossus_cosmo(self) -> ColossusCosmology:
        return self.__cosmo
    @property
    def colossus_density_units(self) -> float: 
        H0 = self.H0 * u.km / u.s / u.Mpc
        colossus_units = (u.Msun / cu.littleh) / (u.kpc / cu.littleh)**3
        my_units = (u.Msun / cu.littleh) / (u.Mpc / cu.littleh)**3
        return (1.0 * colossus_units).to(my_units, cu.with_H0(H0)).value


    @property
    def present_day_critical_density(self) -> float | Collection[float] | np.ndarray:
        return self.critical_density(0)
    
    @property
    def present_day_matter_density(self) -> float | Collection[float] | np.ndarray:
        return self.matter_density(0)

    @property
    def cosmological_constant(self) -> float:
        Mpc_in_m = 3.085677581e22  # meters
        H0_si = self.H0 * 1e3 / Mpc_in_m  # convert to s^-1
        return 3.0 * self.Omega_L * H0_si**2  # units: s^-2

    @classmethod
    def load(cls, saved_data_dir: Path) -> Cosmology:
        with h5py.File(Path(saved_data_dir, "sim_data.h5"), "r") as sim_h5_file:
            return cls.from_hdf5(sim_h5_file["cosmological_model"])


    def critical_density(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> float | Collection[float] | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return self.__cosmo.rho_c(z) * self.colossus_density_units

    def matter_density(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> float | Collection[float] | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return self.__cosmo.rho_m(z) * self.colossus_density_units

    
    def virial_overdensity(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> float | Collection[float] | np.ndarray:
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        x = self.__cosmo.Om(z) - 1
        return (18 * np.pi**2 + 82 * x - 39 * x**2) / self.__cosmo.Om(z)
    
    def virial_density(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None
        ) -> float | Collection[float] | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return self.virial_overdensity(z) * self.matter_density(z)

    def get_density_threshold(
            self, 
            overdensity: int | str, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            base: str = "critical"
        ) -> float | Collection[float] | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        if base == "critical":
            reference_density = self.critical_density(z)
        elif base == "mean":
            reference_density = self.matter_density(z)
        else:
            raise ValueError("Base density must be either critical or mean")

        if overdensity == "virial":
            return reference_density * self.virial_overdensity(z)
        elif isinstance(overdensity, int):
            return reference_density * overdensity
        else:
            raise ValueError("Overdensity must be either virial or integer")
        
    def get_corrected_collapse_overdensity(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return 1.686 * self.__cosmo.Om(z)**0.0055
    
    def _normalized_linear_growth_factor(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        '''Returns the linear growth factor normalized to 1 at z=0'''
        if isinstance(z, float): return float(self.__cosmo.growthFactor(z))
        return self.__cosmo.growthFactor(z)

    def _unnormalized_linear_growth_factor(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        '''Returns the linear growth factor normalized to 1 at z=0'''
        if isinstance(z, float): return float(self.__cosmo.growthFactorUnnormalized(z))
        return self.__cosmo.growthFactorUnnormalized(z)
    
    def _number_linear_growth_factor(
            self, 
            z: Number,
            normalize: bool = True
        ) -> float:
        
        if -0.998 <= z < 500:
            return (
                self._normalized_linear_growth_factor(z=z)
                if normalize else 
                self._unnormalized_linear_growth_factor(z=z)
            ) # type: ignore
        Dplus = self._unnormalized_linear_growth_factor(z=z)
        present_day_Dplus = self._unnormalized_linear_growth_factor(z=0.0)
        return Dplus / present_day_Dplus if normalize else Dplus # type: ignore


    def _numbers_linear_growth_factor(
            self, 
            z: Numbers,
            normalize: bool = True
        ) -> np.ndarray:

          # type: ignore

        colossus_valid_idx = np.where((z >= -0.998) & (z < 500))[0]
        colossus_invalid_idx = np.where((z < -0.998) | (z >= 500))[0]

        D = np.zeros_like(z, dtype=float)
        D[colossus_valid_idx] = (
            self._normalized_linear_growth_factor(z=z[colossus_valid_idx])
            if normalize else 
            self._unnormalized_linear_growth_factor(z=z[colossus_valid_idx])
        )

        if normalize:
            D[colossus_invalid_idx]= (
                self._unnormalized_linear_growth_factor(z=z[colossus_invalid_idx])
                / self._unnormalized_linear_growth_factor(z=0.0)
            )
        else:
            D[colossus_invalid_idx] = self._unnormalized_linear_growth_factor(z=z[colossus_invalid_idx])

        return D
    
    def linear_growth_factor(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            normalize: bool = True
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)


        if isinstance(z, (int, float | np.integer | np.floating)):
            return self._number_linear_growth_factor(z=z, normalize=normalize)
        else:
            return self._numbers_linear_growth_factor(z=z, normalize=normalize)

    def linear_matter_ps(
            self, 
            k: np.ndarray, 
            redshift: float,
            derivative: bool = False
        ) -> np.ndarray:
        
        _ps_args = {
            "model" : self.transfer,
            "kmax" : self.max_wavenumber,
            "ps_type" : self.powerspec_type,
        }

        if (self.ps_file_path is not None) and (self.transfer != "camb"):
            _ps_args["path"] = self.ps_file_path

        lin_ps_present = self.__cosmo.matterPowerSpectrum(
            k, 
            z=0.0, 
            derivative=derivative,
            **_ps_args
        )

        D = self.linear_growth_factor(z=redshift, normalize=True)
        return D**2 * lin_ps_present
    

    def transfer_function(
            self, 
            k: np.ndarray, 
            redshift: float = 0.0
        ) -> np.ndarray:
        Pk = self.linear_matter_ps(k=k, redshift=redshift)
        return k**(-self.nspec / 2) * np.sqrt(Pk)
    
    @property
    def linear_matter_ps_normalization(self) -> float:
        sigma_8Mpc = self.rms_mass_variance(np.asarray(8.0), z=0.0) 
        return float(self.sigma_8 / sigma_8Mpc)**2

    def _get_integrated_age(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            unit: str = "Gyr",
            a_eps: float = 1e-8,
            eps_abs: float = 0,
            eps_rel: float = 1e-8
        ) -> float | Collection[float] | np.ndarray:

        a = self._get_scale_factor_for_integration(z=z, t=t, a=a)

        return age_of_universe(
            a=a,
            unit=unit,
            H0=self.H0,
            Omega_m=self.Omega_m,
            Omega_L=self.Omega_L,
            Omega_r=0.0,
            Omega_k=0.0,
            a_eps=a_eps,
            eps_abs=eps_abs,
            eps_rel=eps_rel
        )

    def _get_integrated_lookback_time(
            self, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            unit: str = "Gyr",
            a_eps: float = 1e-8,
            eps_abs: float = 0,
            eps_rel: float = 1e-8
        ) -> float | Collection[float] | np.ndarray:

        a = self._get_scale_factor_for_integration(z=z, t=t, a=a)

        age = self._get_integrated_age(
            z=z,
            t=t,
            a=a,
            unit=unit,
            a_eps=a_eps,
            eps_abs=eps_abs,
            eps_rel=eps_rel
        )
    
        present_day_age = self._get_integrated_age(
            a=1.0,
            unit=unit,
            a_eps=a_eps,
            eps_abs=eps_abs,
            eps_rel=eps_rel
        )

        if isinstance(age, (int, float)):
            return present_day_age - age
        else:
            # Ensure the output is an array of the same shape as age
            return np.array([present_day_age - a_i for a_i in age])

    def linear_matter_correlation(
            self, 
            r: np.ndarray, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            derivative: bool = False
        ) -> np.ndarray:
        xi_present = self.__cosmo.correlationFunction(
            r, 
            z=0.0,
            derivative=derivative,
            ps_args={
                "model": self.transfer,
                "path": self.ps_file_path
            }
        )
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        return (self.linear_growth_factor(z, normalize=True))**2 * xi_present

    def rms_mass_variance(
            self,
            r: np.ndarray, 
            z: Number | Numbers | None = None, 
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            filter_type: str = "tophat"
        ) -> np.ndarray:
        sigma = self.__cosmo.sigma(
            r, 
            z=0.0,
            filt=filter_type, 
            kmax=self._ps_args["kmax"],
            ps_args={
                "model": self.transfer,
                "path": self.ps_file_path
            }
        )
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        return self.linear_growth_factor(z, normalize=True) * sigma
    
    def _float_age(self, z: float, unit: str = "Gyr") -> float:
        if z >= -0.99:
            return (self.__cosmo.age(z) * u.Gyr).to(unit).value
        else:
            # For z > -0.99, we use the integrated age function
            return self._get_integrated_age(z=z, unit=unit)
        
    def _array_age(self, redshifts: np.ndarray, unit: str = "Gyr") -> np.ndarray:
        ages = []
        for z in redshifts:
            if z >= -0.99:
                ages.append((self.__cosmo.age(z) * u.Gyr).to(unit).value)
            else:
                # For z < -0.99, we use the integrated age function
                ages.append(self._get_integrated_age(z=z, unit=unit))
        
        return np.array(ages)

    def age(self, z: float | np.ndarray, unit: str = "Gyr") -> float | np.ndarray:
        if isinstance(z, float):
            return self._float_age(z=z, unit=unit)
        elif isinstance(z, np.ndarray):
            return self._array_age(redshifts=z, unit=unit)
        
    def _float_lookback_time(self, z: float, unit: str = "Gyr") -> float:
        if z >= -0.99:
            return (self.__cosmo.age(z) * u.Gyr).to(unit).value
        else:
            # For z < -0.99, we use the integrated age function
            return self._get_integrated_age(z=z, unit=unit)
        
    def _array_lookback_time(self, redshifts: np.ndarray, unit: str = "Gyr") -> np.ndarray:
        lookback_times = []
        for z in redshifts:
            if z >= -0.99:
                t_lb = (self.__cosmo.lookbackTime(z) * u.Gyr).to(unit).value
            else:
                # For z < -0.99, we use the integrated age function
                t_lb = self._get_integrated_lookback_time(z=z, unit=unit)

            lookback_times.append(t_lb)
        
        return np.array(lookback_times)

    def lookback_time(self, z: float | np.ndarray, unit: str = "Gyr") -> float | np.ndarray:
        if isinstance(z, float):
            return self._float_lookback_time(z=z, unit=unit)
        elif isinstance(z, np.ndarray):
            return self._array_lookback_time(redshifts=z, unit=unit)
        
    
    def get_dynamical_time(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            unit: str = "Gyr"
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        hubble_time = self.hubble_time(z=z, unit=unit)

        return hubble_time / (5 * np.sqrt(self.colossus_cosmo.Om(z)))

    def hubble_parameter(
            self,
            z: float | Collection[float] | None = None,
            t: float | Collection[float] | None = None,
            a: float | Collection[float] | None = None,
            unit: str = "Gyr",
            scaled: bool = False,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        if scaled:
            return self.__cosmo.Ez(z)
        else:
            return (self.__cosmo.Hz(z) * u.km / u.s / u.Mpc).to(f"1/{unit}").value
        
    def hubble_time(
            self,
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            unit: str = "Gyr",
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        
        return (self.__cosmo.hubbleTime(z) * u.Gyr).to(f"{unit}").value
    
    def hubble_epoch(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return self.age(z=z, unit="Gyr") / self.hubble_time(z=0.0)

    def hubble_wrt_present(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)

        return (self.age(z=z, unit="Gyr") / self.age(z=0.0, unit="Gyr")) - 1.0
    

    def deceleration_parameter(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        matter = self.__cosmo.Om(z)
        dark_energy = self.__cosmo.Ode(z) * (1 + 3.0 * self.__cosmo.w(z))

        return 0.5 * (matter + dark_energy)

    def deceleration_rate(
            self, 
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
            unit: str = "Gyr"
        ) -> float | np.ndarray:
        q = self.deceleration_parameter(z=z, t=t, a=a)
        Hz = self.hubble_parameter(z=z, t=t, a=a, unit=unit)
        return q * Hz**2
    
    def get_hubble_velocity(
            self, 
            distances: np.ndarray,
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray:

        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        hubble = (self.hubble_parameter(z=z, t=t, a=a, unit="s") / u.s)
        return (hubble * (distances * u.Mpc / self.h)).to("km/s").value
    

    def growth_factor_exact(
            self, 
            z: float | np.ndarray | None = None,
            a: float | np.ndarray | None = None,
            normalize: bool = True
        ) -> np.ndarray:
        """
        Linear growth factor D(a) (normalized to D(1)=1). Valid for arbitrarily large a (e.g., a=1e4).
        Provide exactly one of (z, a).
        """
        if (z is None) == (a is None):
            raise ValueError("Provide exactly one of z or a to growth_factor_exact")
        a_arr = np.atleast_1d((1.0 / (1.0 + z)) if z is not None else a).astype(float)

        interp = _growth_interpolator_cached(
            Om0=float(self.Omega_m), 
            Ode0=float(self.Omega_L), 
            w0=float(getattr(self, "w0", -1.0)), 
            wa=float(getattr(self, "wa", 0.0)), 
            a_min=max(1e-3, np.min(a_arr) * 0.5), 
            a_max=max(1.0, np.max(a_arr) * 1.1)
        )

        return interp(a_arr) if normalize else interp(a_arr) * 1.0

    def sigma_linear(
            self, 
            R: float | np.ndarray, 
            z: float, 
            ps_args: dict[str, str | Path] | None = None,
            additional_sigma_args: dict[str, float] | None = None
        ) -> float | np.ndarray:
        """
        Drop-in replacement for Colossus' sigma(R, z, ...).
        Tries Colossus first; if z is 'too future' for its tables, rescales σ(R,0) by D(a).
        """

        if ps_args is None:
            ps_args = {}
        
        if "model" not in ps_args:
            ps_args["model"] = self.transfer

        if additional_sigma_args is None:
            additional_sigma_args = {}
        try:
            return self.colossus_cosmo.sigma(
                R=R, z=z, 
                ps_args=ps_args, 
                **additional_sigma_args
            )
        except Exception as e:
            msg = str(e)
            # Colossus throws an 'outside range of interpolation table' error for very negative z.
            if "outside range of interpolation table" not in msg and "outside range of interpolation" not in msg:
                raise
            sigma0 = self.colossus_cosmo.sigma(
                R=R, z=0.0, 
                ps_args=ps_args,
                **additional_sigma_args 
            )
            D = self.linear_growth_factor(z, normalize=False) 
            return np.asarray(sigma0) * np.asarray(D)

    def peak_height_from_sigma(
            self, 
            sigma_Rz: float | np.ndarray,
            z: Number | Numbers | None,
        ) -> float | np.ndarray:
        """ν = δ_c / σ with δ_c≈1.686 (sufficient for late-time ΛCDM usage)."""
        delta_c = colossus.lss.peaks.collapseOverdensity(z=z, corrections=True)
        return delta_c / np.asarray(sigma_Rz)
    

    def get_lagrangian_mass(
            self, 
            radii: float | np.ndarray,
            z: Number | Numbers | None = None,
            t: Number | Numbers | None = None,
            a: Number | Numbers | None = None,
        ) -> float | np.ndarray: 
        z = self._get_redshift_for_colossus(z=z, t=t, a=a)
        a = 1.0 / (1.0 + z)
        R = radii / a
        return (4/3 * np.pi * self.present_day_matter_density * R**3)
    
    def get_lagrangian_radius(self, mass: float | np.ndarray) -> float | np.ndarray: 
        return (3/4 * mass / (np.pi * self.present_day_matter_density))**(1/3)

def get_cosmo_filepath(
        sim_dir: Path, 
        from_music: bool, 
        param_path: Path | None
    ) -> Path:
    
    sim_path = sim_dir.as_posix().lower()
    
    match (param_path, from_music, sim_path):
        # If param_path is provided and exists, return it.
        case (p, _, _) if ((p is not None) and p.exists()):
            return p

        # If using music and simulation directory indicates a uniform simulation.
        case (_, True, s) if ("uniform" in s):
            return Path(sim_dir.parent.parent, "music", "uniform_ics.conf")

        # If using music and simulation directory indicates a zoom-in simulation.
        case (_, True, s) if ("zoom" in s):
            return Path(sim_dir.parent.parent, "music", "zoom_in_ics.conf")

        # If param_path is None and not using music, return the default param file.
        case (None, False, _):
            return Path(sim_dir.parent, "param.txt")

        # For any other case, raise an error.
        case _:
            raise ValueError("Must be either uniform or zoom-in simulation")

def is_valid_time_input(
        t: Number | Numbers | None,
        a: Number | Numbers | None,
        z: Number | Numbers | None
    ) -> bool:
    none_count = sum(val is None for val in (t, a, z))
    # change to `none_count == 1` if you only want exactly two inputs provided
    return none_count in (1, 2)
    # return np.logical_or((z, t, a).count(None) == 2, (z, t, a).count(None) != 0)


def age_of_universe(
        a: Number | Numbers,
        unit: str = "Gyr",
        H0: float = 70.0,
        Omega_m: float = 0.3,
        Omega_L: float = 0.7,
        Omega_r: float = 0.0,
        Omega_k: float = 0.0,
        a_eps: float = 1e-8,
        eps_abs: float = 0,
        eps_rel: float = 1e-8
    ) -> float:
        
    """
    Compute the age of the Universe at a given redshift z OR scale factor a
    in a ΛCDM cosmology with parameters (H0, Omega_m, Omega_L, Omega_r, Omega_k).

    Exactly one of (z, a) must be provided.  If z is given, then a = 1/(1+z).

    Parameters
    ----------
    a : float or None
        Scale factor at which to compute the age.  Must satisfy 0 < a <= 1.
        If a is not None, `z` must be None.
    H0 : float
        Hubble constant today, in units km s⁻¹ Mpc⁻¹.  (Default: 70.)
    Omega_m : float
        Matter density parameter today (Ωₘ).  (Default: 0.3)
    Omega_L : float
        Dark energy density parameter today (ΩΛ).  (Default: 0.7)
    Omega_r : float
        Radiation density parameter today (Ωᵣ).  (Default: 0.0)
    Omega_k : float
        Curvature density parameter today (Ωₖ).  (Default: 0.0)

    Returns
    -------
    t_Gyr : float
        Age of the Universe at redshift `z` or scale factor `a`, in Gyr.

    Raises
    ------
    ValueError
        If both z and a are provided, or if neither is provided, or if inputs are out of range.
    """

    # 1) Convert H0 to SI (s⁻¹)
    H0_GYR = (H0 * u.km / u.s / u.Mpc).to("1/Gyr").value

    # 2) Define the Hubble parameter H(a') function
    def H_of_a(a_prime: float) -> float:
        """
        H(a') = H0 * sqrt(Ωr / a'^4 + Ωm / a'^3 + Ωk / a'^2 + ΩΛ).
        """
        return H0_GYR * np.sqrt(
            Omega_r / (a_prime**4) +
            Omega_m / (a_prime**3) +
            Omega_k / (a_prime**2) +
            Omega_L
        )

    # 3) Define the integrand: 1 / [a' * H(a')]
    def integrand(a_prime: float) -> float:
        return 1.0 / (a_prime * H_of_a(a_prime))

    # 4) Perform the integral from a_small up to a
    #    We cannot start exactly at 0 because of the 1/a' singularity, so pick a tiny epsilon.
    lower_limit = a_eps
    upper_limit = a

    # quad will return (value, error_estimate).  We only need the value.
    integral_value, _ = quad(
        integrand,
        lower_limit,
        upper_limit,
        epsabs=eps_abs,
        epsrel=eps_rel,
    )

    return (integral_value * u.Gyr).to(unit).value


def _f_de(a: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """ρ_de(a)/ρ_de(1) for CPL: a^{-3(1+w0+wa)} * exp[-3 wa (1-a)]"""
    return a**(-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))

def _E2(a: np.ndarray, Om0: float, Ode0: float, w0: float, wa: float) -> np.ndarray:
    """E(a)^2 / H0^2 for flat geometry, negligible radiation."""
    return Om0 * a**(-3.0) + Ode0 * _f_de(a, w0, wa)

def _omega_m_a(a: np.ndarray, Om0: float, Ode0: float, w0: float, wa: float) -> np.ndarray:
    return (Om0 * a**(-3.0)) / _E2(a, Om0, Ode0, w0, wa)

def _omega_de_a(a: np.ndarray, Om0: float, Ode0: float, w0: float, wa: float) -> np.ndarray:
    return (Ode0 * _f_de(a, w0, wa)) / _E2(a, Om0, Ode0, w0, wa)

def _w_of_a(a: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """CPL: w(a) = w0 + wa (1 - a)."""
    return w0 + wa * (1.0 - a)

# def _growth_rhs(ln_a: float, y: np.ndarray, Om0: float, Ode0: float, w0: float, wa: float) -> np.ndarray:
#     """
#     ODE in x = ln a for linear growth D:
#       y = [D, dD/dx], with
#       D'' + [1/2 - (3/2) w(a) Ω_de(a)] D' - (3/2) Ω_m(a) D = 0
#     """
#     a = np.exp(ln_a)
#     Om_a = _omega_m_a(a, Om0, Ode0, w0, wa)
#     Ode_a = _omega_de_a(a, Om0, Ode0, w0, wa)
#     w_a = _w_of_a(a, w0, wa)
#     D, Dp = y
#     coeff1 = 0.5 - 1.5 * w_a * Ode_a
#     coeff2 = -1.5 * Om_a
#     return np.array([Dp, -coeff1 * Dp + coeff2 * D])

def _growth_rhs(ln_a: float, y: np.ndarray, Om0: float, Ode0: float, w0: float, wa: float) -> np.ndarray:
    """
    ODE in x = ln a for linear growth D:
      G = D/a
      X(a) = Ω_m(a)/Ω_de(a)
      y = [G, dG/da], with
      coeff1 = [7/2 - (3/2) * w(a) / (1 + X(a))]
      coeff2 = - (3/2) * (1 - w(a)) / (1 + X(a))
      G'' + coeff1 * (G'/a) + coeff2 * (G/a^2) = 0
    """
    a = np.exp(ln_a)
    Om_a = _omega_m_a(a, Om0, Ode0, w0, wa)
    Ode_a = _omega_de_a(a, Om0, Ode0, w0, wa)
    w_a = _w_of_a(a, w0, wa)
    G, Gp = y
    X_a = Om_a / Ode_a
    coeff1 = -(3.5 - 1.5 * w_a / (1.0 + X_a)) / a
    coeff2 = -1.5 * (1.0 - w_a) / (1.0 + X_a) / a**2
    return np.array([Gp, coeff1 * Gp + coeff2 * G])
@lru_cache(maxsize=32)
def _growth_interpolator_cached(Om0: float, Ode0: float, w0: float, wa: float,
                                a_min: float, a_max: float) -> interp1d:
    """
    Integrate once on ln a ∈ [ln a_min, ln a_max]; normalize to D(1)=1; return D(a) interpolator.
    """
    x0 = float(np.log(a_min))
    # Matter-era initialization: D ∝ a ⇒ dD/dln a = D
    # y0 = np.array([1.0, 0.0], dtype=float)
    y0 = np.array([a_min, a_min], dtype=float)
    x1 = float(np.log(a_max))

    sol = solve_ivp(
        fun=lambda x, y: _growth_rhs(x, y, Om0, Ode0, w0, wa),
        t_span=(x0, x1),
        y0=y0,
        rtol=1e-6, atol=1e-6,
        dense_output=True, method="RK45"
    )
    if not sol.success:
        raise RuntimeError(f"Growth ODE failed: {sol.message}")

    x_grid = np.linspace(x0, x1, 4096)
    a_grid = np.exp(x_grid)
    D_grid = sol.sol(x_grid)[0] * a_grid  # convert G(a) to D(a)=a*G(a)
    D1 = np.interp(1.0, a_grid, D_grid)
    Dn = D_grid / D1  # normalize so D(1)=1

    return interp1d(a_grid, Dn, kind="cubic", bounds_error=False,
                    fill_value=(Dn[0], Dn[-1]))

# @lru_cache(maxsize=32)
# def _growth_interpolator_cached(Om0: float, Ode0: float, w0: float, wa: float,
#                                 a_min: float, a_max: float) -> interp1d:
#     """
#     Integrate once on ln a ∈ [ln a_min, ln a_max]; normalize to D(1)=1; return D(a) interpolator.
#     """
#     x0 = float(np.log(a_min))
#     # Matter-era initialization: D ∝ a ⇒ dD/dln a = D
#     y0 = np.array([a_min, a_min], dtype=float)
#     x1 = float(np.log(a_max))

#     sol = solve_ivp(
#         fun=lambda x, y: _growth_rhs(x, y, Om0, Ode0, w0, wa),
#         t_span=(x0, x1),
#         y0=y0,
#         rtol=1e-8, atol=1e-10,
#         dense_output=True, method="RK45"
#     )
#     if not sol.success:
#         raise RuntimeError(f"Growth ODE failed: {sol.message}")

#     x_grid = np.linspace(x0, x1, 4096)
#     a_grid = np.exp(x_grid)
#     D_grid = sol.sol(x_grid)[0]
#     D1 = np.interp(1.0, a_grid, D_grid)
#     Dn = D_grid / D1  # normalize so D(1)=1

#     return interp1d(a_grid, Dn, kind="cubic", bounds_error=False,
#                     fill_value=(Dn[0], Dn[-1]))