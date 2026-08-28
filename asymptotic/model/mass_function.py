from __future__ import annotations

import numpy as np, pdb 
import matplotlib.pyplot as plt

from copy import deepcopy
from attrs import define, field, fields
from scipy.optimize import curve_fit, Bounds
from scipy.special import exp1
from collections.abc import Collection
from collections import OrderedDict, defaultdict

from ..abundance.type_defs import AbundanceType
# from ..mass_function_old.type_defs import AbundanceType
from ..model.evo import EvolutionModel
from ..fields.peaks import DensityPeaks
# from ..mass_def.mapping import global_mass_def_mapping
from ..simulation.moments import MomentsInTime
from .viz import display_mf_model_parameter_evolution
from ..mass_def.base import (
    MassDefinitionDependentType, MassDefinitionDependentEvoType
)



# Need a way to toggle between nu parameterization and sigma parameterization

def halo_multiplicity(
        peak_heights: float | np.ndarray,
        A: float = 0.124399,
        a: float = 1.191457,
        b: float = 0.337871,
        c: float = 0.431710, 
    ) -> float | np.ndarray:
    return A * ((peak_heights / b) ** a + 1.0) * np.exp(-c * peak_heights**2)


def ryu_lee_splashback_multiplicity(
        sigma: float | np.ndarray,
        diffusion_coefficient: float,
        delta_sp: float = 1.52,
        kappa: float | np.ndarray = 0.475,
    ) -> float | np.ndarray:
    """Ryu & Lee (2021) splashback multiplicity function.

    This is their one-parameter modified Maggiore--Riotto expression
    (ApJ 917, 98, Eq. 12), also used by Ryu & Lee (2022, ApJ 933, 189).
    ``exp1(x)`` is the upper incomplete gamma function Gamma(0, x).

    Parameters
    ----------
    sigma
        Linear rms density fluctuation on the Lagrangian mass scale.
    diffusion_coefficient
        Non-negative boundary diffusion coefficient D_B.
    delta_sp
        Mean splashback barrier. Both papers adopt 1.52.
    kappa
        Non-Markovian coefficient. Ryu & Lee (2021) use 0.475; the 2022
        implementation evaluates a mass-dependent value from the power
        spectrum and may pass an array here.
    """
    sigma = np.asarray(sigma, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    q = 1.0 + float(diffusion_coefficient)
    if q <= 0:
        return np.full(np.broadcast_shapes(sigma.shape, kappa.shape), np.nan)
    x = delta_sp**2 / (2.0 * sigma**2 * q)
    prefactor = (delta_sp / sigma) * np.sqrt(2.0 / (np.pi * q))
    correction = (
        (1.0 - kappa / q) * np.exp(-x)
        + (kappa / (2.0 * q)) * exp1(x)
    )
    return prefactor * correction

# Maybe make a nu and sigma function, then halo_multiplicity is just a wrapper
# that calls the appropriate function. Additionally, we can set a property that
# toggles between the appropriate b and c values.


@define(slots=True)
class AsymptoticAbundanceFit: 
    present_day_mean_density: float

    # From Diemer 2020 Paper - Universal At Last?
    A: float = field(default=0.124399)
    a: float = field(default=1.191457)
    b: float = field(default=0.337871)
    c: float = field(default=0.431710)

    is_fitted: bool = field(default=False)
    parameter_covariance: np.ndarray | None = field(default=None)

    # Bounds for the curve fitting
    A_bounds: tuple[float, float] = (0, np.inf)
    a_bounds: tuple[float, float] = (-np.inf, np.inf)
    b_bounds: tuple[float, float] = (-np.inf, np.inf)
    c_bounds: tuple[float, float] = (-np.inf, np.inf)

    def __call__(
            self, 
            peak_heights: float | np.ndarray, 
            masses: float | np.ndarray | None = None,
            mf_type: str = "multiplicity",
            return_log: bool = False
        ) -> float | np.ndarray:

        match mf_type:
            case AbundanceType.MULTIPLICITY.value:
                f_nu = self.halo_multiplicity(peak_heights)
                return np.log10(f_nu) if return_log else f_nu
            
            case AbundanceType.NUMBER_DENSITY.value if masses is not None:
                dndM = self.number_density(peak_heights, masses)
                return np.log10(dndM) if return_log else dndM
            
            case AbundanceType.DIFFERENTIAL.value if masses is not None:
                dndlnM = self.differential(peak_heights, masses)
                return np.log10(dndlnM) if return_log else dndlnM
            
            case AbundanceType.CUMULATIVE.value if masses is not None:
                n_gtm = self.cumulative(peak_heights, masses)
                return np.log10(n_gtm) if return_log else n_gtm
            
            case AbundanceType.NORMALIZED.value if masses is not None:
                m_sq_dndM = self.normalized(peak_heights, masses)
                return np.log10(m_sq_dndM) if return_log else m_sq_dndM

            case _ if masses is None:
                raise ValueError(
                    f"Masses must be provided for {mf_type} mass function"
                )
            
            case _:
                raise ValueError(f"Mass function type {mf_type} not recognized")
        
    def __repr__(self) -> str:
        return (
            f"AsymptoticAbundanceFit(A={self.A:.6f}, "
            f"a={self.a:.6f}, b={self.b:.6f}, c={self.c:.6f})"
        )

        
    def halo_multiplicity(self, peak_heights: float | np.ndarray) -> float | np.ndarray:
        return halo_multiplicity(
            peak_heights, 
            A=self.A, 
            a=self.a, 
            b=self.b, 
            c=self.c
        )
    
    def number_density(
            self, peak_heights: float | np.ndarray, masses: float | np.ndarray
        ) -> float | np.ndarray:
        return self.differential(peak_heights, masses) / masses
    
    def differential(
            self, peak_heights: float | np.ndarray, masses: float | np.ndarray
        ) -> float | np.ndarray:
        normalized = self.normalized(peak_heights, masses)
        return (self.present_day_mean_density / masses) * normalized
    
    def cumulative(
            self, peak_heights: float | np.ndarray, masses: float | np.ndarray
        ) -> float | np.ndarray:
        dndlnM = self.differential(peak_heights, masses)
        return np.cumsum((dndlnM * np.gradient(np.log(masses)))[::-1])[::-1]
    
    def normalized(
            self, peak_heights: float | np.ndarray, masses: float | np.ndarray
        ) -> float | np.ndarray:
        dln_nu_dln_M = np.abs(np.gradient(np.log(peak_heights), np.log(masses)))
        return self.halo_multiplicity(peak_heights) * dln_nu_dln_M
        
    @property
    def visual_parameters(self) -> dict[str, float]:
        return {r"$A$": self.A, r"$a$": self.a, r"$b$": self.b, r"$c$": self.c}

    @property   
    def lower_bounds(self) -> np.ndarray:
        return np.asarray([
            self.A_bounds[0], 
            self.a_bounds[0], 
            self.b_bounds[0], 
            self.c_bounds[0]
        ])

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.asarray([
            self.A_bounds[1], 
            self.a_bounds[1], 
            self.b_bounds[1], 
            self.c_bounds[1]
        ])

    @property
    def bounds(self) -> Bounds:
        return Bounds(lb=self.lower_bounds, ub=self.upper_bounds)
    
    def fit(
            self, 
            peak_heights: np.ndarray, 
            multiplicity: np.ndarray,
            multiplicity_errors: np.ndarray | None = None,
            verbose: bool = False
        ) -> None:

        if self.is_fitted:
            print("This object has already been fitted")
            return

        initial_guess = np.asarray([self.A, self.a, self.b, self.c])

        model = deepcopy(self)
        def fit_model(
                peak_heights: float | np.ndarray, 
                A: float, 
                a: float,    
                b: float, 
                c: float
            ) -> float | np.ndarray:
            model.A, model.a, model.b, model.c = A, a, b, c
            f_nu = halo_multiplicity(
                peak_heights, A=model.A, a=model.a, b=model.b, c=model.c
            )
            return np.hstack((f_nu, -np.log10(f_nu)))

        try:
            if multiplicity_errors is None:
                errors = None
            else:
                delta = (multiplicity_errors / multiplicity) / np.log(10.0)
                errors = np.hstack((delta, delta))

            # errors = None

            optimal_parameters, parameter_covariance = curve_fit(
                f=fit_model, 
                xdata=peak_heights, 
                ydata=np.hstack((multiplicity, -np.log10(multiplicity))), 
                p0=initial_guess, 
                bounds=self.bounds,
                # bounds=(self.lower_bounds, self.upper_bounds),
                sigma=errors,
                absolute_sigma=True
            )
            self.A, self.a, self.b, self.c = optimal_parameters
            self.parameter_covariance = parameter_covariance
            self.is_fitted = True
        except Exception as e: 
            if verbose:
                print(f"{e}\n")
                print("Original parameters kept for this fit\n")


@define(slots=True)
class AsymptoticAbundanceModel(MassDefinitionDependentType):
    fof: AsymptoticAbundanceFit | None = field(default=None)
    subfind: AsymptoticAbundanceFit | None = field(default=None)
    crit200: AsymptoticAbundanceFit | None = field(default=None)
    crit500: AsymptoticAbundanceFit | None = field(default=None)
    mean200: AsymptoticAbundanceFit | None = field(default=None)
    virial: AsymptoticAbundanceFit | None = field(default=None)
    splashback: AsymptoticAbundanceFit | None = field(default=None)
    
    # This can be computed then stored in the object
    asymptotic: AsymptoticAbundanceFit | None = field(default=None) 

    def __repr__(self) -> str:
        return super().__repr__()

    # This will control the overdensity parameterization of the model

    @property
    def parameters_A(self) -> dict[int, float]:
        return {
            self.contained_keys[i]: getattr(self, mdef).A
            for i, mdef in enumerate(self.contained_definitions)
        }

    @property
    def parameters_a(self) -> dict[int, float]:
        return {

            self.contained_keys[i]: getattr(self, mdef).a
            for i, mdef in enumerate(self.contained_definitions)
        }
    
    @property
    def parameters_b(self) -> dict[int, float]:
        return {
            self.contained_keys[i]: getattr(self, mdef).b
            for i, mdef in enumerate(self.contained_definitions)
        }

    @property
    def parameters_c(self) -> dict[int, float]:
        return {
            self.contained_keys[i]: getattr(self, mdef).A
            for i, mdef in enumerate(self.contained_definitions)
        }
    

    @classmethod
    def default_init(
            cls, 
            mass_def_keys_to_init: Collection[str],
            present_day_mean_density: float
        ) -> AsymptoticAbundanceModel:

        mass_defs_to_init = get_mass_defs_to_init(mass_def_keys_to_init)

        return AsymptoticAbundanceModel(
            **{
                mdef: AsymptoticAbundanceFit(present_day_mean_density)
                for mdef in mass_defs_to_init
            }
        )


# Maybe just make this a AsymptoticMassFunctionEvo and let the 
# AsymptoticMassFunctionModel be the object that contains the AsymptoticMassFunctionEvo
# and the AsymptoticMassFunctionFit objects for each mass definition. That 
# way we can have a single object that contains all the information we need. 
@define(slots=True)
class AsymptoticAbundanceFitEvo(EvolutionModel):
    # present_day_mean_density: float

    # fit_scale_factors: np.ndarray
    fits: dict[int, AsymptoticAbundanceFit] 
    # params: AsymptoticMassFunctionModelParams = field(init=False)

    # This will control the scale factor evolution of the model

    def __repr__(self) -> str:
        return super().__repr__()
    
    # def __getitem__(self, snapshot_id: int) -> AsymptoticAbundanceFit:
    #     return super().__getitem__(snapshot_id)
    
    @property
    def present_day_mean_density(self) -> float:
        return self.fits[max(self.fits)].present_day_mean_density
    
    @property
    def parameter_evolution(self) -> dict[str, np.ndarray]:
        parameter_evolution = defaultdict(list)

        for snap_id, fit in self.fits.items():
            if not fit.is_fitted: continue
            scale_factor = self.moments.map_by_attribute(
                key_attr="snapshot_id",
                attr_value=snap_id,
                return_attr="scale_factor"
            )
            for key, value in fit.visual_parameters.items():
                parameter_evolution[key].append((scale_factor, value))
        
        for parameter, evo_data in parameter_evolution.items(): 
            evo_data_array = np.asarray(evo_data)
            sorted_by_times = np.argsort(evo_data_array[:,0])
            parameter_evolution[parameter] = evo_data_array[sorted_by_times]
            
        return parameter_evolution


    @property
    def parameter_evo_A(self) -> np.ndarray:
        return self.parameter_evolution[r"$A$"] 
    
    @property
    def parameter_evo_a(self) -> np.ndarray:
        return self.parameter_evolution[r"$a$"]
    
    @property
    def parameter_evo_b(self) -> np.ndarray:
        return self.parameter_evolution[r"$b$"]

    @property
    def parameter_evo_c(self) -> np.ndarray:
        return self.parameter_evolution[r"$c$"]
    
    @classmethod
    def default_init(
            cls,
            present_day_mean_density: float,
            moments: MomentsInTime
        ) -> AsymptoticAbundanceFitEvo:
        return AsymptoticAbundanceFitEvo(
            moments=moments,
            fits={
                snapshot_id: AsymptoticAbundanceFit(
                    present_day_mean_density=present_day_mean_density
                )
                for snapshot_id in moments.snapshot_ids
            }
        )
    # def interpolated_parameters(self, scale_factors: np.ndarray) -> dict[str, np.ndarray]:
    #     ...
    

    # def fit(
    #         self, peak_height_evo: dict[float, np.ndarray], 
    #         multiplicity_evo: dict[float, np.ndarray], 
    #     ) -> None:
    #     ...

    # def number_density_evo(self, mass: float, scale_factors: np.ndarray) -> np.ndarray:
    #    ...

    def display_parameter_evolution(
            self, text_size: int = 12,
            plot_loglog: bool = False, 
            cutoff_scale_factor: float = 0.4,
        ) -> None: 

        _, ax = plt.subplots(
            4, 1, figsize=(4, 6), sharex=True, gridspec_kw={"hspace": 0.0}
        )

        masked = lambda params: params[params[:, 0] > cutoff_scale_factor]

        evo_A = masked(self.parameter_evo_A)
        evo_a = masked(self.parameter_evo_a)
        evo_b = masked(self.parameter_evo_b)
        evo_c = masked(self.parameter_evo_c)

        ax[0].semilogx(evo_A[:, 0], evo_A[:, 1], 'o')
        ax[1].semilogx(evo_a[:, 0], evo_a[:, 1], 'o')
        ax[2].semilogx(evo_b[:, 0], evo_b[:, 1], 'o')
        ax[3].semilogx(evo_c[:, 0], evo_c[:, 1], 'o')

        for i in range(4):
            ax[i].yaxis.set_tick_params(labelsize=text_size)

            if plot_loglog:
                ax[i].set_yscale("log")


        ax[0].set_ylabel(r"$A$")
        ax[1].set_ylabel(r"$a$")
        ax[2].set_ylabel(r"$b$")
        ax[3].set_ylabel(r"$c$")
        ax[3].set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$", fontsize=text_size)

        # ax[3].set_xlim(left=cutoff_scale_factor)

        plt.show()


@define(slots=True)
class AsymptoticAbundanceEvoModel(MassDefinitionDependentEvoType):
    fof: AsymptoticAbundanceFitEvo | None = field(default=None)
    subfind: AsymptoticAbundanceFitEvo | None = field(default=None)
    crit200: AsymptoticAbundanceFitEvo | None = field(default=None)
    crit500: AsymptoticAbundanceFitEvo | None = field(default=None)
    mean200: AsymptoticAbundanceFitEvo | None = field(default=None)
    virial: AsymptoticAbundanceFitEvo | None = field(default=None)
    splashback: AsymptoticAbundanceFitEvo | None = field(default=None)
    asymptotic: AsymptoticAbundanceFitEvo | None = field(default=None) 

    present_day_peaks: DensityPeaks | None = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()

    def get_snapshot_instance(self, snapshot_id: int) -> AsymptoticAbundanceModel:
        return AsymptoticAbundanceModel(
            **self.get_mass_def_attrs_instance("fits", snapshot_id)
        )

    @property
    def parameter_evos_A(self) -> dict[str, np.ndarray]:
        evo_values = {}
        for i, mdef in enumerate(self.contained_definitions):
            key = self.contained_keys[i]
            evo_values[key] = getattr(self, mdef).parameter_evo_A

        return evo_values

    @property
    def parameter_evos_a(self) -> dict[str, np.ndarray]:
        evo_values = {}
        for i, mdef in enumerate(self.contained_definitions):
            key = self.contained_keys[i]
            evo_values[key] = getattr(self, mdef).parameter_evo_a

        return evo_values
    

    @property
    def parameter_evos_b(self) -> dict[str, np.ndarray]:
        evo_values = {}
        for i, mdef in enumerate(self.contained_definitions):
            key = self.contained_keys[i]
            evo_values[key] = getattr(self, mdef).parameter_evo_b
            
        return evo_values

    @property
    def parameter_evos_c(self) -> dict[str, np.ndarray]:
        evo_values = {}
        for i, mdef in enumerate(self.contained_definitions):
            key = self.contained_keys[i]
            evo_values[key] = getattr(self, mdef).parameter_evo_c
            
        return evo_values

    @classmethod
    def default_init(
            cls,
            present_day_mean_density: float,
            moments_in_time: MomentsInTime,
            mass_def_keys_to_init: Collection[str] | None = None,
        )  -> AsymptoticAbundanceEvoModel:

        mass_defs_to_init = get_mass_defs_to_init(mass_def_keys_to_init)

        
        return AsymptoticAbundanceEvoModel(
            **{
                mdef: AsymptoticAbundanceFitEvo.default_init(
                    present_day_mean_density=present_day_mean_density,
                    moments=moments_in_time
                ) 
                for mdef in mass_defs_to_init
            }
        )
    
    def get_accumulation_history(self, *args, **kwargs) -> np.ndarray:
        ... 
    

def get_mass_defs_to_init(
        mass_def_keys_to_init: Collection[str] | None
    ) -> Collection[str]:

    all_mass_defs = (
            "fof", "subfind", "crit200", "crit500",
            "mean200", "virial", "splashback", "asymptotic"
        )

    if mass_def_keys_to_init is None:
        return all_mass_defs
    
    all_mass_def_keys = (
        "fof", "subfind", "200c", "500c",
        "200m", "virial", "splashback", "asymptotic"
    )

    mdefs = {
        mdef
        for i, mdef in enumerate(all_mass_defs)
        if all_mass_def_keys[i] in mass_def_keys_to_init
    }

    return set(mdefs).intersection(all_mass_defs)
