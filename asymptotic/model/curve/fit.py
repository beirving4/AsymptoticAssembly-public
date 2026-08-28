from __future__ import annotations

import numpy as np, pdb

from typing import Callable
from enum import Enum, StrEnum
from copy import deepcopy
from abc import ABC, abstractmethod
from attrs import define, field, fields
from scipy.optimize import curve_fit, Bounds, OptimizeWarning
from collections.abc import Collection
from collections import OrderedDict, defaultdict

class FittingType(StrEnum):
    STANDARD = "standard"
    STANDARD_W_ERRORS = "standard_w_errors"
    STACKED = "stacked"
    STACKED_W_ERRORS = "stacked_w_errors"

class OptimizationResult(Enum):
    IMPROPER_INPUT_PARAMETERS = 0
    FTOL_CONVERGENCE = 1  # Both actual and predicted relative reductions are at most FTOL.
    XTOL_CONVERGENCE = 2  # Relative error between two consecutive iterates is at most XTOL.
    FTOL_AND_XTOL_CONVERGENCE = 3  # Both FTOL and XTOL criteria are met.
    GTOL_CONVERGENCE = 4  # Cosine of the angle between FVEC and any Jacobian column is at most GTOL.
    MAXFEV_REACHED = 5  # Maximum number of function evaluations (MAXFEV) reached.
    FTOL_TOO_SMALL = 6  # FTOL is too small; no further reduction in the sum of squares is possible.
    XTOL_TOO_SMALL = 7  # XTOL is too small; no further improvement in the solution X is possible.
    GTOL_TOO_SMALL = 8  # GTOL is too small; FVEC is orthogonal to the Jacobian columns to machine precision.


def did_optimization_converge(result: int) -> bool:
    return result in {1, 2, 3, 4}

def get_optimization_result(result: int) -> OptimizationResult:
    try:
        return OptimizationResult(result)
    except ValueError as e:
        raise ValueError(f"Invalid optimization result flag: {result}") from e

"""

The point of this is to allow for different implementations of curve fitting
given different requirements needed to handle mass definition dependent relations.
We will implement different approaches of fitting, and which ever form produces 
the best fit will be used. The best fit will be determined by the smallest 
chi-squared statistic from the different fits.

Standard = Least squares with standard Gaussian errors.

Standard_w_errors = Least squares with jackknife computed errors from sim.

Stacked = Least squares with standard Gaussian errors on target data and its
          complement (i.e., inverse, reversed, etc.) at the same time to help
          capture asymptotic behavior from both endpoints.

Stacked_w_errors = Stacked with jackknife computed errors from sim.

"""

@define(slots=True)
class FitConfig:
    method: str = field(default="lm")
    x_scale: str = field(default="jac")
    xtol: float | None = field(default=None)
    ftol: float | None = field(default=None)
    gtol: float | None = field(default=None)
    epsfcn: float | None = field(default=None)
    loss: str = field(default="linear")
    nan_policy: str = field(default="omit")
    verbose: bool = field(default=False)
    maxfev: int | None = field(default=None)

    @property
    def as_dict(self) -> dict[str, str | bool | int | float | None]:
        return {
            "method": self.method,
            "x_scale": self.x_scale,
            "loss": self.loss,
            "nan_policy": self.nan_policy,
            "verbose": self.verbose,
            # "maxfev": self.maxfev,
            # "xtol": self.xtol,
            # "ftol": self.ftol,
            # "gtol": self.gtol,
            # "epsfcn": self.epsfcn,
        }
    
    # @property
    # def lst_sq_kwargs(self) -> dict[str, str | bool | int | float | None]:
    #     return {
    #         "x_scale": self.x_scale,
    #         "loss": self.loss,
    #         "nan_policy": self.nan_policy,
    #         "verbose": self.verbose,
    #         "maxfev": self.maxfev,
    #         "xtol": self.xtol,
    #         "ftol": self.ftol,
    #         "gtol": self.gtol,
    #         "epsfcn": self.epsfcn,
    #     }
    
    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@define(slots=True)
class CurveFitter(ABC):

    cfg: FitConfig

    @abstractmethod
    def __call__(self, verbose: bool, *args, **kwargs) -> np.ndarray:
        ...

    def optimize(
            self,
            fitting_function: Callable, 
            x: np.ndarray, 
            y: np.ndarray, 
            init_guess: np.ndarray,
            bounds: Bounds,
            *args, 
            yerr: np.ndarray | None = None,
            param_gradients: Callable | str = '2-point',
            **kwargs
        ) -> dict:

        try: 
            optimal_params, param_cov_matrix, info_dict, messeges, int_err = curve_fit(
                f=fitting_function,
                xdata=x, 
                ydata=y, 
                p0=init_guess,
                jac=param_gradients,
                bounds=bounds,
                full_output=True,
                sigma=yerr, 
                **self.cfg.as_dict, 
                **kwargs
            )
    
            return {
                "best_fit_params": optimal_params,
                "covariances" : param_cov_matrix,
                "message": messeges,
                "optimization_result" : get_optimization_result(int_err),
                "is_success": did_optimization_converge(int_err),
                **info_dict
            }
        
        except (ValueError, TypeError, OptimizeWarning) as error:

            return {
                "best_fit_params": [],
                "covariances" : np.array([]),
                "message": f"{error}",
                "optimization_result" : 6,
                "is_success": False,
            }


    # @abstractmethod
    # def standard(self, verbose: bool, *args, **kwargs) -> np.ndarray:
    #     ...

    # def standard_w_errors(self, verbose: bool, *args, **kwargs) -> np.ndarray:
    #     ...


    # @abstractmethod
    # def stacked(self, verbose: bool, *args, **kwargs) -> np.ndarray:
    #     ...

    # @abstractmethod
    # def stacked_w_errors(self, verbose: bool, *args, **kwargs) -> np.ndarray:
    #     ...