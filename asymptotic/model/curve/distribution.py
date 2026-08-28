from __future__ import annotations

import numpy as np, pdb

from typing import Callable
from attrs import define, field
from scipy.optimize import Bounds
from collections.abc import Collection

from .fit import CurveFitter, FitConfig

@define(slots=True)
class MassDistributionFitter(CurveFitter):

    cfg: FitConfig = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.cfg = FitConfig()

    def __call__(
            self,
            radii: np.ndarray,
            quantities: np.ndarray,
            fitting_term: Callable,
            init_param_setup: dict[str, dict[str, float] | Bounds],
            quantity_errors: np.ndarray | None = None,
            use_log_params: bool = False,
            param_gradients: Callable | str = '2-point',
            fit_cfg_kwargs: dict[str, str | float | int | bool] | None = None
        ) -> dict:

        if fit_cfg_kwargs is not None:
            self.cfg.update(**fit_cfg_kwargs)

        free_params_keys = list(init_param_setup["initial_guess"].keys())
        initial_guess_order = init_param_setup["initial_guess_order"]
        free_initial_guess = (
            np.log(init_param_setup["initial_guess_array"])
            if use_log_params else 
            init_param_setup["initial_guess_array"]
        )

        if use_log_params:
            init_param_setup["bounds"].lb = np.log(init_param_setup["bounds"].lb)
            init_param_setup["bounds"].ub = np.log(init_param_setup["bounds"].ub)

            

        fit_function_wrapper, fit_gradients_wrapper = get_function_wrappers(
            fitting_function=fitting_term,
            gradient_function=param_gradients,
            free_params_keys=free_params_keys,
            fixed_params=init_param_setup["fixed_params"],
            initial_guess_order=initial_guess_order,
            use_log_params=use_log_params
        )

        results = self.optimize(
            fitting_function=fit_function_wrapper,
            x=radii,
            y=quantities,
            init_guess=free_initial_guess,
            bounds=init_param_setup["bounds"],
            yerr=quantity_errors,
            param_gradients=fit_gradients_wrapper
        )

        return get_finalized_results(
            results=results,
            R200m=init_param_setup["fixed_params"]["R200m"],
            mean_density=init_param_setup["fixed_params"]["mean_density"],
            initial_guess_order=initial_guess_order,
            use_log_params=use_log_params
        )
    
def get_updated_full_params(
        free_params: list[float] | tuple[float, ...] | np.ndarray,
        free_params_keys: list[str],
        fixed_params: dict[str, float],
        initial_guess_order: dict[str, int],
        use_log_params: bool = False,
    ) -> dict[str, float]:

    transformed_free_params = (
        np.exp(free_params) if use_log_params else free_params
    )

    free_params_dict = {
        key: transformed_free_params[idx]
        for key, idx in initial_guess_order.items()
        if key in free_params_keys
    }
    return free_params_dict | fixed_params


def get_function_wrappers(
        fitting_function: Callable,
        gradient_function: Callable | str,
        free_params_keys: list[str],
        fixed_params: dict[str, float],
        initial_guess_order: dict[str, int],
        use_log_params: bool = False,
    ) -> tuple[Callable, Callable | str]:

    def fit_function_wrapper(radii: np.ndarray, *free_params) -> float |np.ndarray:
        full_params = get_updated_full_params(
            free_params=free_params,
            free_params_keys=free_params_keys,
            fixed_params=fixed_params,
            initial_guess_order=initial_guess_order,
            use_log_params=use_log_params
        )
        return fitting_function(radii=radii, profile_params=full_params)
    
    if isinstance(gradient_function, str):
        return fit_function_wrapper, gradient_function
    elif callable(gradient_function):
        def fit_gradients_wrapper(radii: np.ndarray, *free_params) -> np.ndarray:
            full_params = get_updated_full_params(
                free_params=free_params,
                free_params_keys=free_params_keys,
                fixed_params=fixed_params,
                initial_guess_order=initial_guess_order,
                use_log_params=use_log_params
            )
            return gradient_function(
                radii=radii, 
                profile_params=full_params, 
                params_to_freeze=fixed_params.keys()
            )
        
    else:
        raise ValueError("Invalid gradient function provided")

    return fit_function_wrapper, fit_gradients_wrapper

def rescale_regular_params(
        free_param_dict: dict[str, float],
        R200m: float,
        mean_density: float
    ) -> None:

    if "scale_density" in free_param_dict:
        free_param_dict["scale_density"] *= mean_density

    if "scale_radius" in free_param_dict:
        free_param_dict["scale_radius"] *= R200m

    if "transition_radius" in free_param_dict:
        free_param_dict["transition_radius"] *= R200m

def rescale_log_params(
        free_param_dict: dict[str, float],
        R200m: float,
        mean_density: float
    ) -> None:

    if "scale_density" in free_param_dict:
        new_rho_s = np.exp(free_param_dict["scale_density"]) * mean_density
        free_param_dict["scale_density"] = new_rho_s

    if "scale_radius" in free_param_dict:
        new_rs = np.exp(free_param_dict["scale_radius"]) * R200m
        free_param_dict["scale_radius"] = new_rs

    if "transition_radius" in free_param_dict:
        new_rt = np.exp(free_param_dict["transition_radius"]) * R200m
        free_param_dict["transition_radius"] = new_rt

def get_rescaled_parameters(
        best_fit_params: np.ndarray,
        R200m: float,
        mean_density: float,
        initial_guess_order: dict[str, int],
        use_log_params: bool = False
    ) -> dict[str, float]:

    optimized_free_dict = {
        key: best_fit_params[idx] 
        for key, idx in initial_guess_order.items()
    }

    if use_log_params:
        rescale_log_params(
            free_param_dict=optimized_free_dict,
            R200m=R200m,
            mean_density=mean_density
        )
    else:
        rescale_regular_params(
            free_param_dict=optimized_free_dict,
            R200m=R200m,
            mean_density=mean_density
        )

    return optimized_free_dict

def get_finalized_results(
        results: dict,
        R200m: float,
        mean_density: float,
        initial_guess_order: dict[str, int],
        use_log_params: bool = False
    ) -> dict:

    best_fit_params = results["best_fit_params"] 

    if len(best_fit_params) == 0:
        return {**results, "best_fit_params": {}}

    return {
        **results,
        "best_fit_params": get_rescaled_parameters(
            best_fit_params=best_fit_params,
            R200m=R200m,
            mean_density=mean_density,
            initial_guess_order=initial_guess_order,
            use_log_params=use_log_params
        )
    }
