from __future__ import annotations

import inspect
import numpy as np, pdb

from enum import StrEnum
from typing import Callable
from functools import partial
from attrs import define, field
from collections import OrderedDict
from abc import ABC, abstractmethod 
from scipy import integrate, optimize
from collections.abc import Collection
from colossus.utils.utilities import safeExp, safeLog

from ..model.curve.fit import FitConfig
from ..particles.collection import get_circular_velocity 
from ..model.curve.distribution import MassDistributionFitter

class MassDistributionQuantityType(StrEnum):
    SLOPE = "slope"
    ENCLOSED_MASS = "enclosed_mass"
    DENSITY = "density"
    ENCLOSED_DENSITY = "enclosed_density"
    KAPPA = "kappa"

class ProfileType(StrEnum):
    ORBITAL = "orbital"
    INFALL = "infall"
    ASYMPTOTIC = "asymptotic"


@define(slots=True)
class ProfileModelParameters(ABC):

    covariances: np.ndarray | None
    is_fitted: bool 

    @abstractmethod
    def update(
            self, 
            model_params: dict[str, float],
            covariances: np.ndarray | None = None,
            is_fitted: bool | None = None
        ) -> None:
        ... 

    @property
    @abstractmethod
    def param_order(self) -> dict[str, int]:
        ...

    @property
    @abstractmethod
    def param_labels(self) -> dict[str, str]:
        ...

    @property
    @abstractmethod
    def as_array(self) -> np.ndarray:
        ...

    @property
    def log_parameters(self) -> np.ndarray:
        return np.log10(self.as_array)

    @property
    @abstractmethod
    def free_params_as_array(self) -> np.ndarray:
        ...

    @property
    def log_free_params(self) -> np.ndarray:
        return np.log10(self.free_params_as_array)

    @property
    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        ...

    @property
    @abstractmethod
    def free_params_as_dict(self) -> dict[str, float]:
        ...

    @property
    def param_names(self) -> list[str]:
        return list(self.as_dict.keys())

    @property
    @abstractmethod
    def upper_bounds(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def lower_bounds(self) -> np.ndarray:
        ...
    
    @property
    def bounds(self) -> optimize.Bounds:
        return optimize.Bounds(lb=self.lower_bounds, ub=self.upper_bounds)
    

    @property
    def log_lower_bounds(self) -> np.ndarray:
        return np.log10(self.lower_bounds)
    
    @property
    def log_upper_bounds(self) -> np.ndarray:
        return np.log10(self.upper_bounds)
    
    @property
    def log_bounds(self) -> optimize.Bounds:
        return optimize.Bounds(
            lb=np.log10(self.log_lower_bounds),
            ub=np.log10(self.log_upper_bounds)
        )
    
    @property
    def param_errors(self) -> np.ndarray:
        if self.covariances is None: return np.nan * np.ones_like(self.as_array)
        return np.sqrt(np.diag(self.covariances))
    
    @property
    def condition_number(self) -> float:
        return np.nan if self.covariances is None else np.linalg.cond(self.covariances)
    

    def _get_masked_bounds(
            self, params_to_freeze: Collection[str], use_log_params: bool = False
        ) -> optimize.Bounds:

        param_mask = np.asarray([
            value for key, value in self.param_order.items()
            if key not in params_to_freeze
        ])

        lower_bounds = (
            self.log_lower_bounds[param_mask]
            if use_log_params else
            self.lower_bounds[param_mask]
        )

        upper_bounds = (
            self.log_upper_bounds[param_mask]
            if use_log_params else
            self.upper_bounds[param_mask]
        )

        return optimize.Bounds(lb=lower_bounds, ub=upper_bounds)
    
    def _split_initial_free_and_fixed_params(
            self, params_to_freeze: Collection[str], use_log_params: bool = False
        ) -> tuple[dict[str, float], dict[str, float]]:

        initial_guess, fixed_params = {}, {}
        for key, value in self.as_dict.items():
            if key in params_to_freeze:
                fixed_params[key] = np.log10(value) if use_log_params else value
            else:
                initial_guess[key] = np.log10(value) if use_log_params else value

        return initial_guess, fixed_params


    def _get_initial_guess_info(
            self, 
            initial_guess_dict: dict[str, float],
            params_to_freeze: Collection[str]
        ) -> tuple[np.ndarray, dict[str, float]]: 
        
        initial_guess_array, initial_guess_order = [], {}
        idx_counter = 0
        for key, _ in sorted(self.param_order.items(), key=lambda x: x[1]):
            if ((key not in params_to_freeze) and (key in initial_guess_dict)):
                initial_guess_array.append(initial_guess_dict[key])
                initial_guess_order[key] = idx_counter
                idx_counter += 1

        initial_guess_array = np.asarray(initial_guess_array)

        # initial_guess_array = np.asarray([
        #     initial_guess[key]
        #     for key, _ in sorted(self.param_order.items(), key=lambda x: x[1])
        #     if key not in params_to_freeze
        # ])

        return initial_guess_array, initial_guess_order
    @abstractmethod
    def get_initial_fit_setup(
            self, *args, 
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None, 
            **kwargs
        ) -> dict[str, dict[str, float] | optimize.Bounds]:
        ...


@define(slots=True)
class OrbitalParameters(ProfileModelParameters):
    scale_density: float
    transition_radius: float
    scale_radius: float = field(default=0.2)
    alpha: float = field(default=0.18)
    beta: float = field(default=3.0)
    eta: float = field(default=0.1)

    covariances: np.ndarray | None = field(default=None)
    is_fitted: bool = field(default=False)

    def update(
            self, 
            model_params: dict[str, float], 
            covariances: np.ndarray | None = None,
            is_fitted: bool | None = None
        ) -> None:

        # pdb.set_trace()

        self.scale_density = (
            self.scale_density
            if "scale_density" not in model_params
            else model_params["scale_density"]
        )

        self.scale_radius = (
            self.scale_radius
            if "scale_radius" not in model_params
            else model_params["scale_radius"]
        )

        self.transition_radius = (
            self.transition_radius
            if "transition_radius" not in model_params
            else model_params["transition_radius"]
        )

        self.alpha = (
            self.alpha
            if "alpha" not in model_params
            else model_params["alpha"]
        )

        self.beta = (
            self.beta
            if "beta" not in model_params
            else model_params["beta"]
        )

        self.eta = (
            self.eta
            if "eta" not in model_params
            else model_params["eta"]
        )

        # pdb.set_trace()

        self.covariances = covariances if covariances is not None else self.covariances
        self.is_fitted = is_fitted if is_fitted is not None else self.is_fitted
    
    @property
    def param_order(self) -> dict[str, int]:
        return {
            "scale_density": 0,
            "scale_radius": 1,
            "transition_radius": 2,
            "alpha": 3,
            "beta": 4,
            "eta": 5,
        }
    
    @property
    def param_labels(self) -> dict[str, str]:
        return {
            "scale_density": r"$\rho_{\rm s}$",
            "scale_radius": r"$r_{\rm s}$",
            "transition_radius": r"$r_{\rm t}$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "eta": r"$\eta$",
        }
    
    # These are in units of rho_m and R200m
    @property
    def upper_bounds(self) -> np.ndarray: 
        return np.array([10.0**7, 0.45, 3.0, 0.4, 10.0, 0.10000001])
    
    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([10.0, 0.01, 0.5, 0.03, 0.1, 0.099999999])

    @property  
    def as_array(self) -> np.ndarray:
        return np.array([
            self.scale_density,
            self.scale_radius,
            self.transition_radius,
            self.alpha,
            self.beta,
            self.eta,
        ])

    @property
    def log_parameters(self) -> np.ndarray:
        return np.log10(self.as_array)

    @property
    def free_params_as_array(self) -> np.ndarray:
        return self.as_array
    
    @property
    def as_dict(self) -> dict[str, float]:
        return OrderedDict({    
            "scale_density": self.scale_density,
            "scale_radius": self.scale_radius,
            "transition_radius": self.transition_radius,
            "alpha": self.alpha,
            "beta": self.beta,
            "eta": self.eta,
        })
    
    @property
    def free_params_as_dict(self) -> dict[str, float]:
        return self.as_dict
    
    def get_initial_fit_setup(
            self, 
            R200m: float, 
            rho_mean: float, 
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None
        ) -> dict[str, dict[str, float] | np.ndarray | optimize.Bounds]:

        if params_to_freeze is None:
            params_to_freeze = []

        bounds = self._get_masked_bounds(params_to_freeze, use_log_params)
        splits = self._split_initial_free_and_fixed_params(params_to_freeze, use_log_params)
        initial_guess, fixed_params = splits
        
        for key in initial_guess:
            match key:
                case "scale_density":
                    initial_guess[key] = (
                        initial_guess[key] - np.log10(rho_mean)
                        if use_log_params else
                        initial_guess[key] / rho_mean
                    )
                case "scale_radius" | "transition_radius":
                    initial_guess[key] = (
                        initial_guess[key] - np.log10(R200m) 
                        if use_log_params else 
                        initial_guess[key] / R200m
                    )
                case _:
                    initial_guess[key] = (
                        np.log10(initial_guess[key]) 
                        if use_log_params else 
                        initial_guess[key]
                    )

        initial_info = self._get_initial_guess_info(initial_guess, params_to_freeze)
        initial_guess_array, initial_guess_order = initial_info

        assert are_params_in_bounds(initial_guess_array, bounds)

        return {
            "fixed_params": fixed_params, 
            "initial_guess": initial_guess, 
            "initial_guess_order": initial_guess_order,
            "initial_guess_array": initial_guess_array,
            "bounds": bounds
        }
    

@define(slots=True)
class InfallParameters(ProfileModelParameters):
    r_pivot: float
    delta1: float = field(default=10.0)        
    delta_max: float = field(default=30.0)      
    slope: float = field(default=1.0)  
    zeta: float = field(default=0.5)  
    shift: float = field(default=0.0)

    covariances: np.ndarray | None = field(default=None)
    is_fitted: bool = field(default=False)

    def update(
            self, 
            model_params: dict[str, float],
            covariances: np.ndarray | None = None,
            is_fitted: bool | None = None
        ) -> None:

        # pdb.set_trace()

        self.r_pivot = (
            self.r_pivot
            if "r_pivot" not in model_params
            else model_params["r_pivot"]
        )

        self.delta1 = (
            self.delta1
            if "delta1" not in model_params
            else model_params["delta1"]
        )

        self.delta_max = (
            self.delta_max
            if "delta_max" not in model_params
            else model_params["delta_max"]
        )

        self.slope = (
            self.slope
            if "slope" not in model_params
            else model_params["slope"]
        )

        self.zeta = (
            self.zeta
            if "zeta" not in model_params
            else model_params["zeta"]
        )

        self.shift = (
            self.shift
            if "shift" not in model_params
            else model_params["shift"]
        )

        # pdb.set_trace()

        self.covariances = covariances if covariances is not None else self.covariances
        self.is_fitted = is_fitted if is_fitted is not None else self.is_fitted
    
    @property
    def param_order(self) -> dict[str, int]:
        return {
            "r_pivot": 0,
            "delta1": 1,
            "delta_max": 2,
            "s": 3,
            "zeta": 4,
            "shift": 5,
        }
    
    @property
    def param_labels(self) -> dict[str, str]:
        return {
            "r_pivot": r"$r_{\rm pivot}$",
            "delta1": r"$\delta_1$",
            "delta_max": r"$\delta_{\rm max}$",
            "s": r"$s$",
            "zeta": r"$\zeta$",
            "shift": r"$\epsilon$",
        }
    
    @property
    def as_array(self) -> np.ndarray:
        return np.array([
            self.r_pivot,
            self.delta1,
            self.delta_max,
            self.slope,
            self.zeta,
            self.shift,
        ])
    
    @property
    def free_params_as_array(self) -> np.ndarray:
        return np.array([
            self.delta1,
            self.delta_max,
            self.slope,
            self.zeta,
            self.shift,
        ])
    
    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "r_pivot": self.r_pivot,
            "delta1": self.delta1,
            "delta_max": self.delta_max,
            "s": self.slope,
            "zeta": self.zeta,
            "shift": self.shift,
        }
    
    @property
    def free_params_as_dict(self) -> dict[str, float]:
        return {
            "delta1": self.delta1,
            "delta_max": self.delta_max,
            "s": self.slope,
            "zeta": self.zeta,
            "shift": self.shift,
        }
    
    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array([0.0, 1.0, 10.0, 0.01, 0.4999999, 0.0])
    
    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array([np.inf, 100.0, 2000.0, 4.0, 0.5000001, 1.0])
    
    
    
    def get_initial_fit_setup(
            self, 
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None
        ) -> dict[str, dict[str, float] | np.ndarray | optimize.Bounds]:

        if params_to_freeze is None:
            params_to_freeze = []

        # for key in ("r_pivot", "zeta"):
        #     if key not in params_to_freeze: params_to_freeze.append(key)

        bounds = self._get_masked_bounds(params_to_freeze, use_log_params)
        splits = self._split_initial_free_and_fixed_params(params_to_freeze, use_log_params)
        initial_guess, fixed_params = splits

        initial_info = self._get_initial_guess_info(initial_guess, params_to_freeze)
        initial_guess_array, initial_guess_order = initial_info

        assert are_params_in_bounds(initial_guess_array, bounds)

        return {
            "fixed_params": fixed_params, 
            "initial_guess": initial_guess, 
            "initial_guess_order": initial_guess_order,
            "initial_guess_array": initial_guess_array,
            "bounds": bounds
        }


@define(slots=True)
class AsymptoticProfileParameters(ProfileModelParameters):
    M200m: float
    R200m: float
    mean_density: float
    orbital: OrbitalParameters
    infall: InfallParameters

    # Need to get the dimensions of this correct for 
    # splitting this up into the orbital and infall parameters
    covariances: np.ndarray | None = field(default=None)
    is_fitted: bool = field(default=False)

    def update(
            self, 
            model_params: dict[str, float],
            covariances: np.ndarray | None = None,
            is_fitted: bool | None = None
        ) -> None:

        self.M200m = (
            self.M200m
            if "M200m" not in model_params
            else model_params["M200m"]
        )

        self.R200m = (
            self.R200m
            if "R200m" not in model_params
            else model_params["R200m"]
        )

        self.mean_density = (
            self.mean_density
            if "mean_density" not in model_params
            else model_params["mean_density"]
        )

        self.covariances = covariances if covariances is not None else self.covariances
        self.is_fitted = is_fitted if is_fitted is not None else self.is_fitted

        self.orbital.update(
            model_params=model_params,
            covariances=covariances,
            is_fitted=is_fitted
        )
        self.infall.update(
            model_params=model_params,
            covariances=covariances,
            is_fitted=is_fitted
        )
    
    @property
    def param_order(self) -> dict[str, int]:
        orbital_order = self.orbital.param_order
        infall_shift = len(orbital_order)
        infall_order = {
            key: value + infall_shift 
            for key, value in self.infall.param_order.items()
        }
        shift = len(orbital_order) + len(infall_order)
        return {
            **orbital_order,
            **infall_order,
            "mean_density": shift + 1,
            "R200m": shift + 2,
            "M200m": shift + 3,
        }
    
    @property
    def param_labels(self) -> dict[str, str]:
        return {
            **self.orbital.param_labels,
            **self.infall.param_labels,
            "mean_density": r"$\rho_{\rm m}(a)$",
            "R200m": r"$R_{\rm 200m}$",
            "M200m": r"$M_{\rm 200m}$",
        }

    @property
    def mean_density_200(self) -> float:
        return 200.0 * self.mean_density
    
    @property
    def as_array(self) -> np.ndarray:
        return np.concatenate([
            self.orbital.as_array, 
            self.infall.as_array, 
            np.asarray([self.mean_density]),
            np.asarray([self.R200m])
        ])

    @property
    def free_params_as_array(self) -> np.ndarray:
        return np.concatenate([
            self.orbital.free_params_as_array, 
            self.infall.free_params_as_array
        ])
    
    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "M200m": self.M200m,
            "R200m": self.R200m,
            "mean_density": self.mean_density, 
            **self.orbital.as_dict, 
            **self.infall.as_dict
        }
    
    @property
    def free_params_as_dict(self) -> dict[str, float]:
        return {
            **self.orbital.free_params_as_dict, 
            **self.infall.free_params_as_dict
        }

    
    @property
    def upper_bounds(self) -> np.ndarray:
        return np.concatenate([
            self.orbital.upper_bounds, self.infall.upper_bounds,
        ])
    
    @property
    def lower_bounds(self) -> np.ndarray:
        return np.concatenate([
            self.orbital.lower_bounds, self.infall.lower_bounds,
        ])
    

    def get_initial_fit_setup(
            self, 
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None
        ) -> dict[str, dict[str, float] | np.ndarray | optimize.Bounds]:

        orbital_setup = self.orbital.get_initial_fit_setup(
            R200m=self.R200m,
            rho_mean=self.mean_density,
            use_log_params=use_log_params,
            params_to_freeze=params_to_freeze
        )

        infall_setup = self.infall.get_initial_fit_setup(
            use_log_params=use_log_params,
            params_to_freeze=params_to_freeze
        )

        bounds = optimize.Bounds(
            lb=np.concatenate([
                orbital_setup["bounds"].lb,
                infall_setup["bounds"].lb,
            ]),
            ub=np.concatenate([
                orbital_setup["bounds"].ub,
                infall_setup["bounds"].ub,
            ])
        )

        initial_guess = {
            **orbital_setup["initial_guess"],
            **infall_setup["initial_guess"],
        }

        fixed_params = {
            **orbital_setup["fixed_params"],
            **infall_setup["fixed_params"],
            "R200m": self.R200m,
            "mean_density": self.mean_density,
            "M200m": self.M200m
        }

        initial_guess_array = np.concatenate([
            orbital_setup["initial_guess_array"],
            infall_setup["initial_guess_array"],
        ])

        num_free_orbital_params = len(orbital_setup["initial_guess_array"])
        initial_guess_order = orbital_setup["initial_guess_order"]
        for key, value in infall_setup["initial_guess_order"].items():
            initial_guess_order[key] = value + num_free_orbital_params 

        assert are_params_in_bounds(initial_guess_array, bounds)

        return {
            "fixed_params": fixed_params,
            "initial_guess": initial_guess, 
            "initial_guess_order": initial_guess_order,
            "orbital_initial_guess": orbital_setup["initial_guess"],
            "orbital_initial_guess_order" : orbital_setup["initial_guess_order"],
            "orbital_initial_guess_array": orbital_setup["initial_guess_array"],
            "infall_initial_guess": infall_setup["initial_guess"],
            "infall_initial_guess_order": infall_setup["initial_guess_order"],
            "infall_initial_guess_array": infall_setup["initial_guess_array"],
            "initial_guess_array": initial_guess_array,
            "bounds": bounds
        }


@define(slots=True)
class MassDistributionProfile(ABC):
    params: ProfileModelParameters
    optimizer: MassDistributionFitter

    @abstractmethod
    def __call__(self, radii: float | np.ndarray) -> float | np.ndarray:
        ...

    @abstractmethod
    def slope(self, radii: float | np.ndarray) -> float | np.ndarray:
        ...

    @abstractmethod
    def update_params(self, params: dict[str, float]) -> None:
        ...

    @property
    def is_fitted(self) -> bool:
        return self.params.is_fitted

    def _integrand(self, radii: float | np.ndarray) -> float | np.ndarray:
        return 4 * np.pi * radii**2 * self(radii)
    
    def enclosed_mass(self, radii: float | np.ndarray) -> float | np.ndarray:
        if isinstance(radii, float):
            radii = np.array([radii])
        return np.array([integrate.quad(self._integrand, 0, r)[0] for r in radii])
    
    def enclosed_density(self, radii: float | np.ndarray) -> float | np.ndarray:
        return (3.0 * self.enclosed_mass(radii)) / (4 * np.pi * radii**3)
    
    def mu(self, radii: float | np.ndarray) -> float | np.ndarray:
        return self(radii) / (3.0 * self.enclosed_density(radii))

    def enclosed_density_slope(self, radii: float | np.ndarray) -> float | np.ndarray:
        return self.mu(radii) + 3.0
    
    def kappa(self, radii: float | np.ndarray) -> float | np.ndarray:
        return 1.0 - (3.0 / self.mu(radii)) 

    def linear_derivative(self, radii: float | np.ndarray) -> float | np.ndarray:
        return (self(radii) / radii) * self.slope(radii)
    
    def rotation_curve(self, radii: float | np.ndarray) -> float | np.ndarray:
        return get_circular_velocity(self.enclosed_mass(radii), radii)
    

    @abstractmethod
    def get_density_parameter_gradients(
            self, 
            radii: float | np.ndarray, 
            params_to_freeze: Collection[str] | None = None
        ) -> np.ndarray:
        ...
    
    def fit(
            self,
            radii: np.ndarray,
            quantities: np.ndarray,
            quantity_type: str,
            profile_type: str,  
            min_float_log: float = np.exp(-100.0),
            max_float_log: float = np.exp(100.0),
            quantity_errors: np.ndarray | None = None,
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None,
            fit_cfg_kwargs: dict[str, str | float | int | bool] | None = None
        ) -> None:
    
        fit_results = self.optimizer(
            radii=radii,
            quantities=quantities,
            fitting_term=partial(
                fitting_function,
                quantities=quantities,
                profile_type=profile_type,
                quantity_type=quantity_type,
                min_float_log=min_float_log,
                max_float_log=max_float_log,
                quantity_errors=quantity_errors
            ),
            init_param_setup=self.params.get_initial_fit_setup(
                use_log_params=False,
                params_to_freeze=params_to_freeze
            ),
            use_log_params=use_log_params,
            quantity_errors=quantity_errors,
            # param_gradients="3-point",
            # param_gradients = (
            #     partial(
            #         fitting_gradients,
            #         profile_type=profile_type,
            #         params_to_freeze=params_to_freeze,
            #         quantity_errors=quantity_errors,
            #     ) 
            #     if quantity_type == MassDistributionQuantityType.DENSITY
            #     else "2-point"
            # ),
            fit_cfg_kwargs=fit_cfg_kwargs
        )

        self.params.update(
            model_params=fit_results["best_fit_params"],
            covariances=fit_results["covariances"],
            is_fitted=fit_results["is_success"]
        )




@define(slots=True)
class OrbitalProfile(MassDistributionProfile):
    params: OrbitalParameters

    optimizer: MassDistributionFitter = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.optimizer = MassDistributionFitter()

    def __call__(self, radii: float | np.ndarray) -> float | np.ndarray:
        return orbital_density_profile(
            radii=radii,
            scale_density=self.params.scale_density,
            scale_radius=self.params.scale_radius,
            transition_radius=self.params.transition_radius,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
        )
    
    def update_params(self, params: dict[str, float]) -> None:
        self.params.update(params)
    
    def slope(self, radii: float | np.ndarray) -> float | np.ndarray:
        return orbital_density_slope(
            radii=radii,
            scale_radius=self.params.scale_radius,
            transition_radius=self.params.transition_radius,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
        )

    def get_concentration(self, ref_radius: float) -> float:
        return ref_radius / self.params.scale_radius
    
    def get_density_parameter_gradients(
            self, 
            radii: float | np.ndarray,
            params_to_freeze: Collection[str] | None = None
        ) -> np.ndarray:
        return orbital_parameter_gradients(
            radii=radii,
            scale_radius=self.params.scale_radius,
            transition_radius=self.params.transition_radius,
            alpha=self.params.alpha,
            beta=self.params.beta,
            eta=self.params.eta,
            params_to_freeze=params_to_freeze
        )
    
    def fit(
            self,
            radii: np.ndarray,
            quantities: np.ndarray,
            quantity_type: str, 
            min_float_log: float = np.exp(-100.0),
            max_float_log: float = np.exp(100.0),
            quantity_errors: np.ndarray | None = None,
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None,
            fit_cfg_kwargs: dict[str, str | float | int | bool] | None = None
        ) -> None:
    
        super().fit(
            radii=radii,
            quantities=quantities,
            quantity_type=quantity_type,
            profile_type="orbital",
            min_float_log=min_float_log,
            max_float_log=max_float_log,
            quantity_errors=quantity_errors,
            use_log_params=use_log_params,
            params_to_freeze=params_to_freeze,
            fit_cfg_kwargs=fit_cfg_kwargs
        )

@define(slots=True)
class InfallProfile(MassDistributionProfile):
    mean_density: float
    params: InfallParameters

    optimizer: MassDistributionFitter = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.optimizer = MassDistributionFitter()

    def __call__(self, radii: float | np.ndarray) -> float | np.ndarray:
        return infall_density_profile(
            radii=radii,
            mean_density=self.mean_density,
            delta1=self.params.delta1,
            delta_max=self.params.delta_max,
            r_pivot=self.params.r_pivot,
            slope=self.params.slope,
            zeta=self.params.zeta,
            shift=self.params.shift
        )
    
    def update_params(self, params: dict[str, float]) -> None:
        self.mean_density = (
            self.mean_density
            if "mean_density" not in params
            else params["mean_density"]
        )
        self.params.update(params)
    
    def slope(self, radii: float | np.ndarray) -> float | np.ndarray:
        return infall_density_slope(
            radii=radii,
            delta1=self.params.delta1,
            delta_max=self.params.delta_max,
            r_pivot=self.params.r_pivot,
            slope=self.params.slope,
            zeta=self.params.zeta,
            shift=self.params.shift
        )
    
    def get_density_parameter_gradients(
            self, 
            radii: float | np.ndarray, 
            params_to_freeze: Collection[str] | None = None
        ) -> np.ndarray:
        return infall_parameter_gradients(
            radii=radii,
            delta1=self.params.delta1,
            delta_max=self.params.delta_max,
            r_pivot=self.params.r_pivot,
            s=self.params.slope,
            zeta=self.params.zeta,
            params_to_freeze=params_to_freeze
        )
        
    def fit(
            self,
            radii: np.ndarray,
            quantities: np.ndarray,
            quantity_type: str, 
            min_float_log: float = np.exp(-100.0),
            max_float_log: float = np.exp(100.0),
            quantity_errors: np.ndarray | None = None,
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None,
            fit_cfg_kwargs: dict[str, str | float | int | bool] | None = None
        ) -> None:
    
        super().fit(
            radii=radii,
            quantities=quantities,
            quantity_type=quantity_type,
            profile_type="infall",
            min_float_log=min_float_log,
            max_float_log=max_float_log,
            quantity_errors=quantity_errors,
            use_log_params=use_log_params,
            params_to_freeze=params_to_freeze,
            fit_cfg_kwargs=fit_cfg_kwargs
        )




@define(slots=True)
class AsymptoticProfile(MassDistributionProfile):
    orbital: OrbitalProfile
    infall: InfallProfile

    nu200m: float = field(default=0.0)
    R200m: float = field(default=0.0)
    M200m: float = field(default=0.0)

    optimizer: MassDistributionFitter = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.optimizer = MassDistributionFitter()
    def __call__(self, radii: float | np.ndarray) -> float | np.ndarray:
        return self.orbital(radii) + self.infall(radii)
    
    def __repr__(self) -> str:
        return (
            rf"AsymptoticProfile(nu200m={self.nu200m:.3f}, "
            rf" R200m={self.R200m:.3f} Mpc/h, "
            rf"M200m={self.M200m:.3e} Msun/h)"
        )
    
    @property
    def parameters(self) -> AsymptoticProfileParameters:
        return AsymptoticProfileParameters(
            M200m=self.M200m,
            R200m=self.R200m,
            mean_density=self.infall.mean_density,
            orbital=self.orbital.params,
            infall=self.infall.params,
        )
    
    def update_params(self, params: dict[str, float]) -> None:
        self.M200m = self.M200m if "M200m" not in params else params["M200m"]
        self.R200m = self.R200m if "R200m" not in params else params["R200m"]
        self.nu200m = self.nu200m if "nu200m" not in params else params["nu200m"]
        self.infall.update_params(params)
        self.orbital.update_params(params)
    
    def update_parameters(self, new_params: AsymptoticProfileParameters) -> None:
        self.M200m = new_params.M200m
        self.R200m = new_params.R200m
        self.infall.mean_density = new_params.mean_density
        self.orbital.params = new_params.orbital
        self.infall.params = new_params.infall
    
    def slope(self, radii: float | np.ndarray) -> float | np.ndarray:
        
        rho = self(radii)
        drho_orb = self.orbital.linear_derivative(radii)
        drho_inf = self.infall.linear_derivative(radii)
        drho = drho_orb + drho_inf

        return (radii / rho) * drho
    
    def get_concentration(self, ref_radius: float) -> float:
        return self.orbital.get_concentration(ref_radius)

    @property
    def r_splashback(self) -> float | np.ndarray:
        minima = find_funciton_minimum(
            func=self.slope,
            bounds=optimize.Bounds(lb=0.5 * self.R200m, ub=4.0 * self.R200m)
        )
        return np.nan if minima["r_minimum"] is None else minima["r_minimum"]

    @property
    def r_bound(self) -> float | np.ndarray:
        minima = find_funciton_minimum(
            func=self.kappa,
            bounds=optimize.Bounds(lb=0.5 * self.R200m, ub=4.0 * self.R200m)
        )
        return np.nan if minima["r_minimum"] is None else minima["r_minimum"]
    
    @property
    def scale(self) -> float | np.ndarray: # milestones object
        return self.get_distribution_milestone(self.orbital.params.scale_radius)

    @property
    def four_scale(self) -> float | np.ndarray:
        return self.get_distribution_milestone(4 * self.orbital.params.scale_radius)

    @property
    def bound(self) -> float | np.ndarray:
        return self.get_distribution_milestone(self.r_bound)

    @property
    def splashback(self) -> float | np.ndarray:
        return self.get_distribution_milestone(self.r_splashback)

    @classmethod
    def from_halo_properties(
        cls, 
        peak_height_200m: float, 
        mass_200m: float,
        radius_200m: float,
        scale_radius: float,
        scale_density: float,
        mean_density: float,
        init_model_params: dict[str, float] | None = None,
    ) -> AsymptoticProfile:
        
        if init_model_params is None:
            init_model_params = {}

        # From Diemer et. al. 2022
        transition_radius = radius_200m * (1.4 - 0.21 * peak_height_200m)

        orbital_params = OrbitalParameters(
            scale_density=scale_density,
            scale_radius=scale_radius,
            transition_radius=transition_radius,
        )

        infall_params = InfallParameters(r_pivot=radius_200m)

        if init_model_params: 
            orbital_params.update(init_model_params)
            infall_params.update(init_model_params)

        orbital = OrbitalProfile(params=orbital_params)
        infall = InfallProfile(mean_density=mean_density, params=infall_params)

        return cls(
            orbital=orbital,
            infall=infall,
            nu200m=peak_height_200m,
            R200m=radius_200m,
            M200m=mass_200m,
            params=AsymptoticProfileParameters(
                M200m=mass_200m,
                R200m=radius_200m,
                mean_density=infall.mean_density,
                orbital=orbital.params,
                infall=infall.params,
            )
        )
    

    def get_density_parameter_gradients(
            self, 
            radii: float | np.ndarray,
            params_to_freeze: Collection[str] | None = None
        ) -> np.ndarray:
        return np.concatenate([
            self.orbital.get_density_parameter_gradients(
                radii=radii,
                params_to_freeze=params_to_freeze
            ),
            self.infall.get_density_parameter_gradients(
                radii=radii,
                params_to_freeze=params_to_freeze
            )
        ])

    def get_distribution_milestone(self, radii: float | np.ndarray) -> float | np.ndarray:
        ... 

    def fit(
            self,
            radii: np.ndarray,
            quantities: np.ndarray,
            quantity_type: str, 
            min_float_log: float = np.exp(-100.0),
            max_float_log: float = np.exp(100.0),
            quantity_errors: np.ndarray | None = None,
            use_log_params: bool = False,
            params_to_freeze: Collection[str] | None = None,
            fit_cfg_kwargs: dict[str, str | float | int | bool] | None = None
        ) -> None:
    
        super().fit(
            radii=radii,
            quantities=quantities,
            quantity_type=quantity_type,
            profile_type="asymptotic",
            min_float_log=min_float_log,
            max_float_log=max_float_log,
            quantity_errors=quantity_errors,
            use_log_params=use_log_params,
            params_to_freeze=params_to_freeze,
            fit_cfg_kwargs=fit_cfg_kwargs
        )


def orbital_density_profile(
        radii: float | np.ndarray,
        scale_density: float,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
    ) -> float | np.ndarray:
    
    r_rs = (radii / scale_radius)
    r_rt = (radii / transition_radius)
    rs_rt = (scale_radius / transition_radius)

    term1 = -(2.0 / alpha) * (r_rs ** alpha - 1.0)
    term2 = -(1.0 / beta) * (r_rt ** beta - rs_rt ** beta)
    term3 = (1.0 / eta) * (rs_rt)**beta * (r_rs ** eta - 1.0)
    S = term1 + term2 + term3

    return scale_density * safeExp(S)

def infall_density_profile(
        radii: float | np.ndarray,
        mean_density: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        slope: float,
        zeta: float,
        shift: float = 0.0,
    ) -> float | np.ndarray:
    

    Q = (delta1 / delta_max)**(1.0/zeta) + (radii / r_pivot)**(slope/zeta)
    return mean_density * ((delta1 * Q**(-zeta)) + shift) 

def density_profile(
        radii: float | np.ndarray,
        scale_density: float,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
        mean_density: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        slope: float,
        zeta: float,
        shift: float = 0.0,
    ) -> float | np.ndarray:

    orbital = orbital_density_profile(
        radii=radii,
        scale_density=scale_density,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
    )

    infall = infall_density_profile(
        radii=radii,
        mean_density=mean_density,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        slope=slope,
        zeta=zeta,
        shift=shift,
    )

    return orbital + infall

def orbital_density_slope(
        radii: float | np.ndarray,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
    ) -> float | np.ndarray:

    r_rs = (radii / scale_radius)
    r_rt = (radii / transition_radius)
    rs_rt = (scale_radius / transition_radius)

    return -2.0 * r_rs ** alpha - r_rt ** beta + (rs_rt ** beta) * r_rs ** eta

def infall_density_slope(
        radii: float | np.ndarray,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        slope: float,
        zeta: float,
        shift: float = 0.0,
    ) -> float | np.ndarray:

    radial_term = (radii / r_pivot) ** (slope / zeta)
    Q = (delta1 / delta_max) ** (1.0 / zeta) + radial_term
    norm_rho = (delta1 * Q**(-zeta) + shift)

    return - (1.0 - (1.0 / norm_rho)) * (slope/Q) * radial_term


def orbital_linear_derivative(
        radii: float | np.ndarray,
        scale_density: float,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
    ) -> float | np.ndarray:

    slope = orbital_density_slope(
        radii=radii,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
    )

    density = orbital_density_profile(
        radii=radii,
        scale_density=scale_density,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
    )

    return (density / radii) * slope


def infall_linear_derivative(
        radii: float | np.ndarray,
        mean_density: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        slope: float,
        zeta: float,
        shift: float = 0.0,
    ) -> float | np.ndarray:

    slope = infall_density_slope(
        radii=radii,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        slope=slope,
        zeta=zeta,
        shift=shift,
    ) # type: ignore

    density = infall_density_profile(
        radii=radii,
        mean_density=mean_density,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        slope=slope,
        zeta=zeta,
        shift=shift
    )

    return (density / radii) * slope

def density_linear_derivative(
        radii: float | np.ndarray,
        scale_density: float,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
        mean_density: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        slope: float,
        zeta: float,
        shift: float = 0.0,
    ) -> float | np.ndarray:

    orbital = orbital_linear_derivative(
        radii=radii,
        scale_density=scale_density,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
    )

    infall = infall_linear_derivative(
        radii=radii,
        mean_density=mean_density,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        slope=slope,
        zeta=zeta,
        shift=shift,
    )

    return orbital + infall


def density_slope(
        radii: float | np.ndarray,
        scale_density: float,
        scale_radius: float,
        mean_density: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        s: float,
        zeta: float,
    ) -> float | np.ndarray:

    drho_dr = density_linear_derivative(
        radii=radii,
        scale_density=scale_density,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
        mean_density=mean_density,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        s=s,
        zeta=zeta,
    )

    rho = density_profile(
        radii=radii,
        scale_density=scale_density,
        scale_radius=scale_radius,
        transition_radius=transition_radius,
        alpha=alpha,
        beta=beta,
        eta=eta,
        mean_density=mean_density,
        delta1=delta1,
        delta_max=delta_max,
        r_pivot=r_pivot,
        s=s,
        zeta=zeta,
    )

    return (rho / radii) * drho_dr

def make_sure_array_like(x: float | np.ndarray) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.array([x])

def orbital_parameter_gradients(
        radii: float | np.ndarray,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
        params_to_freeze: Collection[str] | None = None,
    ) -> np.ndarray:

    if params_to_freeze is None:
        params_to_freeze = []

    r_rs = (radii / scale_radius)
    r_rt = (radii / transition_radius)
    rs_rt = (scale_radius / transition_radius)

    dln_rs = 2.0 * r_rs ** alpha + ((beta / eta) - 1.0) * (rs_rt ** beta) * (r_rt ** eta - 1)
    dln_alpha = (2.0/alpha) * (r_rs ** alpha * (1.0 - alpha * np.log(r_rs)) - 1.0)
    dln_rt = r_rt ** beta - (rs_rt ** beta) * ((beta / eta) * (r_rs ** eta - 1) + 1)
    dln_eta = (1.0/eta) * (rs_rt ** beta) * (r_rt ** eta * (eta * np.log(r_rs) - 1.0) + 1.0)
    dln_beta = r_rt**beta * ((1.0/beta) - np.log(r_rt))
    dln_beta -= (rs_rt**beta) * ((1.0/beta) - np.log(rs_rt) * ((beta/eta) * (r_rs**eta - 1)) + 1.0)

    gradient_terms = {
        'scale_density': np.ones_like(radii),
        'scale_radius': make_sure_array_like(dln_rs),
        'alpha': make_sure_array_like(dln_alpha),
        'transition_radius': make_sure_array_like(dln_rt),
        'eta': make_sure_array_like(dln_eta),
        'beta': make_sure_array_like(dln_beta),
    }

    return np.column_stack([
        gradient for param, gradient in gradient_terms.items()
        if param not in params_to_freeze
    ])

def infall_parameter_gradients(
        radii: float | np.ndarray,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        s: float,
        zeta: float,
        params_to_freeze: Collection[str] | None = None,
    ) -> np.ndarray:

    if params_to_freeze is None:
        params_to_freeze = []

    if not isinstance(params_to_freeze, list):
        params_to_freeze = list(params_to_freeze)

    # if "r_pivot" not in params_to_freeze:
    #     params_to_freeze.append("r_pivot")

    r_rp, d1_dmax = (radii / r_pivot), (delta1 / delta_max)
    radial_term = (r_rp) ** (s / zeta)
    delta_term = d1_dmax ** (1.0 / zeta)
    Q = delta_term + radial_term
    fm = (1.0 - (1.0 / (delta1 * Q**(-zeta) + 1)))

    dln_delta1 = fm * (1.0 - (delta_term / Q))
    dln_s = - fm * (s/Q) * radial_term * np.log(r_rp)
    dln_deltaMax = (fm / Q) * delta_term
    dln_zeta = fm * (np.log(Q) / zeta) * (
        delta_term * np.log(d1_dmax) + s * radial_term * np.log(r_rp)
    )

    gradient_terms = {
        'delta1': make_sure_array_like(dln_delta1),
        's': make_sure_array_like(dln_s),
        'delta_max': make_sure_array_like(dln_deltaMax),
        'zeta': make_sure_array_like(dln_zeta),
    }

    return np.column_stack([
        gradient for param, gradient in gradient_terms.items()
        if param not in params_to_freeze
    ])

def parameter_gradients(
        radii: float | np.ndarray,
        scale_radius: float,
        transition_radius: float,
        alpha: float,
        beta: float,
        eta: float,
        delta1: float,
        delta_max: float,
        r_pivot: float,
        s: float,
        zeta: float,
        params_to_freeze: Collection[str] | None = None,
    ) -> np.ndarray:

    return np.column_stack([
        orbital_parameter_gradients(
            radii=radii,
            scale_radius=scale_radius,
            transition_radius=transition_radius,
            alpha=alpha,
            beta=beta,
            eta=eta,
            params_to_freeze=params_to_freeze,
        ),
        infall_parameter_gradients(
            radii=radii,
            delta1=delta1,
            delta_max=delta_max,
            r_pivot=r_pivot,
            s=s,
            zeta=zeta,
            params_to_freeze=params_to_freeze,
        )
    ])
    


def are_params_in_bounds(
        param_array: np.ndarray, 
        bounds: optimize.Bounds, 
    ) -> bool:

    if not np.all(param_array >= bounds.lb):
        return False

    return bool(np.all(param_array <= bounds.ub))


def find_funciton_minimum(func: Callable, bounds: optimize.Bounds) -> dict[str, float | None]:

    try: 
        minimizer = optimize.fminbound(
            func=func,
            x1=bounds.lb,
            x2=bounds.ub,
        )
        return {"r_minimum" : minimizer, "f_minimum" : func(minimizer)}
    except ValueError:
        return {"r_minimum" : None, "f_minimum" : None}
    
def get_enclosed_mass(
        density_profile_fit: Callable,
        profile_params: dict[str, float],
        radii: float | np.ndarray
    ) -> float | np.ndarray:

    if isinstance(radii, float):
        radii = np.asarray([radii])

    def integrand(r: float) -> float:
        return 4 * np.pi * r**2 * density_profile_fit(r, **profile_params)

    return np.array([integrate.quad(integrand, 0, r)[0] for r in radii])

def get_enclosed_density(
        density_profile_fit: Callable,
        profile_params: dict[str, float],
        radii: float | np.ndarray
    ) -> float | np.ndarray:
    enclosed_mass = get_enclosed_mass(density_profile_fit, profile_params, radii)
    return (3.0 * enclosed_mass) / (4 * np.pi * radii**3)

def get_kappa_slope(
        density_profile_fit: Callable,
        profile_params: dict[str, float],
        radii: float | np.ndarray
    ) -> float | np.ndarray:

    enclosed_density = get_enclosed_density(density_profile_fit, profile_params, radii)
    mu = enclosed_density / (3.0 * enclosed_density)
    return mu * (mu + 3.0)

def get_asymptotic_fitting_form(quantity_type: str) -> Callable: 

    match quantity_type:
        case MassDistributionQuantityType.SLOPE:
            return density_slope
        case MassDistributionQuantityType.ENCLOSED_MASS:
            return partial(get_enclosed_mass, density_profile_fit=density_profile)
        case MassDistributionQuantityType.DENSITY:
            return density_profile
        case MassDistributionQuantityType.ENCLOSED_DENSITY:
            return partial(get_enclosed_density, density_profile_fit=density_profile)
        case MassDistributionQuantityType.KAPPA:
            return partial(get_kappa_slope, density_profile_fit=density_profile)
        case _:
            raise NotImplementedError(
                f"{quantity_type} is not a valid quantity type for asymptotic fitting"
            )
        
def get_orbital_fitting_form(quantity_type: str) -> Callable: 

    match quantity_type:
        case MassDistributionQuantityType.SLOPE:
            return orbital_density_slope
        case MassDistributionQuantityType.ENCLOSED_MASS:
            return partial(get_enclosed_mass, density_profile_fit=orbital_density_slope)
        case MassDistributionQuantityType.DENSITY:
            return orbital_density_profile
        case MassDistributionQuantityType.ENCLOSED_DENSITY:
            return partial(get_enclosed_density, density_profile_fit=orbital_density_profile)
        case MassDistributionQuantityType.KAPPA:
            return partial(get_kappa_slope, density_profile_fit=orbital_density_profile)
        case _:
            raise NotImplementedError(
                f"{quantity_type} is not a valid quantity type for orbital fitting"
            )
        
def get_infall_fitting_form(quantity_type: str) -> Callable:

    match quantity_type:
        case MassDistributionQuantityType.SLOPE:
            return infall_density_slope
        case MassDistributionQuantityType.ENCLOSED_MASS:
            return partial(get_enclosed_mass, density_profile_fit=infall_density_slope)
        case MassDistributionQuantityType.DENSITY:
            return infall_density_profile
        case MassDistributionQuantityType.ENCLOSED_DENSITY:
            return partial(get_enclosed_density, density_profile_fit=infall_density_profile)
        case MassDistributionQuantityType.KAPPA:
            return partial(get_kappa_slope, density_profile_fit=infall_density_profile)
        case _:
            raise NotImplementedError(
                f"{quantity_type} is not a valid quantity type for infall fitting"
            )

def get_fitting_form(quantity_type: str, profile_type: str) -> Callable: 

    match profile_type:
        case ProfileType.ASYMPTOTIC:
            return get_asymptotic_fitting_form(quantity_type)
        case ProfileType.ORBITAL:
            return get_orbital_fitting_form(quantity_type)
        case ProfileType.INFALL:
            return get_infall_fitting_form(quantity_type)
        case _:
            raise NotImplementedError(f"{profile_type} is not a valid profile type")
        
def get_parameter_gradient_forms(profile_type: str) -> Callable: 

    match profile_type:
        case ProfileType.ASYMPTOTIC:
            return parameter_gradients
        case ProfileType.ORBITAL:
            return orbital_parameter_gradients
        case ProfileType.INFALL:
            return infall_parameter_gradients
        case _:
            raise NotImplementedError(f"{profile_type} is not a valid profile type")

def safe_log_array(
        values: np.ndarray, min_float_log: float = np.exp(-100.0),
    ) -> np.ndarray:

    if np.nanmin(values) < min_float_log:
        mask = values < min_float_log
        log_values = np.zeros_like(values)
        log_values[mask] = -min_float_log
        log_values[~mask] = np.log(values[~mask])
        values[values < min_float_log] = min_float_log
    else:
        log_values = np.log(values)
    return log_values

def safe_log_difference(
        quantities: np.ndarray,
        fitted_values: np.ndarray, 
        min_float_log: float = np.exp(-100.0),
        max_float_log: float = np.exp(100.0),
        quantity_errors: np.ndarray | None = None,
    ) -> np.ndarray:

    # log_fit = safe_log_array(fitted_values, min_float_log)
    log_fit = safeLog(fitted_values)
    log_difference = log_fit - np.log(quantities)

    if max_float_log is not None:
        log_difference[log_difference > max_float_log] = max_float_log
        log_difference[log_difference < -max_float_log] = -max_float_log

    if quantity_errors is not None:
        log_difference /= np.log(1.0 + quantity_errors / quantities)

    return log_difference

def fitting_function(
        radii: float | np.ndarray, 
        quantities: np.ndarray,
        quantity_type: str,
        profile_type: str,
        profile_params: dict[str, float],
        min_float_log: float = np.exp(-100.0),
        max_float_log: float = np.exp(100.0),
        quantity_errors: np.ndarray | None = None,
    ) -> float | np.ndarray:

    # Compute the fitted quantities
    fit_func = get_fitting_form(quantity_type, profile_type)
    valid_keys = inspect.signature(fit_func).parameters.keys()
    filtered_params = {k: v for k, v in profile_params.items() if k in valid_keys}
    quantities_fit = fit_func(radii, **filtered_params) 

    # This way if I need to use different quantity weightings I 
    # can add them and call them with a match-case to the fitting type 
    # (will create enum type for quantity weights)
    return safe_log_difference(
        quantities=quantities,
        fitted_values=quantities_fit,
        min_float_log=min_float_log,
        max_float_log=max_float_log,
        quantity_errors=quantity_errors
    )

def fitting_gradients(
        radii: float | np.ndarray,
        profile_type: str,
        profile_params: dict[str, float],
        quantities: np.ndarray | None = None,
        quantity_errors: np.ndarray | None = None,
        params_to_freeze: Collection[str] | None = None
    ) -> np.ndarray:

    if params_to_freeze is None:
        params_to_freeze = []
    
    grad_form = get_parameter_gradient_forms(profile_type)
    valid_keys = inspect.signature(grad_form).parameters.keys()
    filtered_params = {k: v for k, v in profile_params.items() if k in valid_keys}
    gradients = grad_form(
        radii=radii, **filtered_params, params_to_freeze=params_to_freeze
    )

    if (quantities is not None) and (quantity_errors is not None):
        gradients[:, ] /= np.log(1.0 + quantity_errors / quantities)
    
    return gradients