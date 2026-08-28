"""The finite-reference splashback mass-accretion history (Chapter 5).

The fitted quantity is `Psi_hat(a) = M(a) / M(a_ref)` — normalized at a
**finite** reference epoch supplied by the data, never at infinity and never
hard-coded:

    ln Psi_hat(a) = -K [B(a) - B(a_ref)] + T(a) - T(a_ref)

    B(a) = (1 - w) (a / a_f)^(-gamma_1) + w (a / a_f)^(-gamma_2)
    T(a) = 2 h / [(a / a_pk)^(-2) + (a / a_pk)^v]
    K    = [ln 2 + T(a_f) - T(a_ref)] / [1 - B(a_ref)]

`B` is the milestone-anchored double-power core, `T` the transient, and `K` the
normalization that makes the half-mass anchor exact. Because `B(a_f) = 1` by
construction, the two anchors hold for every valid parameter set:

    Psi_hat(a_f)   = 1/2
    Psi_hat(a_ref) = 1
    Psi_hat(a)    -> 0  as a -> 0+

Nested limits are parameter limits, not separate implementations: `h = 0` and
`w = 0` reduce the core to the finite-reference Busha (2007) form, and
additionally `gamma_1 = 1` to Wechsler (2002).

The record holds state; the three module-level functions perform the actions.
This module imports numpy and attrs only — no optimizer, no plotting, no I/O.
"""
from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from attrs import define, field
from numpy.typing import ArrayLike, NDArray

from .core import AsymptoticCalibration, _as_float

__all__ = [
    "AsymptoticSplashbackMassHistoryFit",
    "splashback_mass_accretion_rate",
    "splashback_mass_history_fraction",
    "splashback_mass_history_log_fraction",
]

#: a positive finite scale factor, scalar or array-like
ScaleFactor: TypeAlias = float | ArrayLike
#: the matching result: a floating scalar for scalar input, else an array
FloatResult: TypeAlias = np.float64 | NDArray[np.float64]


#: relative tolerance when comparing an epoch against its calibration, wide
#: enough for 100.0 vs 100.0000000000001 and far too tight to hide 50 vs 100
_EPOCH_TOLERANCE = 1e-9


def _real(instance: object, attribute: object, value: float) -> None:
    """A finite parameter. `as_float` has already enforced realness."""
    name = getattr(attribute, "name", "field")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _positive(instance: object, attribute: object, value: float) -> None:
    _real(instance, attribute, value)
    name = getattr(attribute, "name", "field")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _non_negative(instance: object, attribute: object, value: float) -> None:
    _real(instance, attribute, value)
    name = getattr(attribute, "name", "field")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@define(frozen=True, slots=True)
class AsymptoticSplashbackMassHistoryFit:
    """One immutable parameter set for the finite-reference splashback MAH.

    Names follow the thesis. The optimizer's transformed coordinates
    (`phi`, `delta`, `ln a_f`, `ln a_pk`) belong to the fitting boundary and
    deliberately do not appear here.
    """

    a_f: float = field(converter=_as_float, validator=_positive)
    w: float = field(converter=_as_float, validator=_real)
    gamma_1: float = field(converter=_as_float, validator=_positive)
    gamma_2: float = field(converter=_as_float, validator=_positive)
    h: float = field(converter=_as_float, validator=_non_negative)
    a_pk: float = field(converter=_as_float, validator=_positive)
    v: float = field(converter=_as_float, validator=_positive)
    a_ref: float = field(converter=_as_float, validator=_positive)
    calibration: AsymptoticCalibration = field()

    @calibration.validator
    def _is_calibration(self, attribute: object,
                        value: AsymptoticCalibration) -> None:
        if not isinstance(value, AsymptoticCalibration):
            raise TypeError(
                "calibration must be an AsymptoticCalibration, got "
                f"{type(value).__name__}")

    def __attrs_post_init__(self) -> None:
        if not 0.0 <= self.w <= 1.0:
            raise ValueError(f"w must lie in [0, 1], got {self.w!r}")
        if not self.gamma_1 > self.gamma_2:
            raise ValueError(
                f"gamma_1 must exceed gamma_2, got {self.gamma_1!r} "
                f"and {self.gamma_2!r}")
        if not self.a_f < self.a_ref:
            raise ValueError(
                f"a_f must precede a_ref, got {self.a_f!r} and {self.a_ref!r}")
        denominator = 1.0 - _core(np.float64(self.a_ref), self)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "the normalization denominator 1 - B(a_ref) must be finite and "
                f"positive, got {float(denominator)!r}")
        epoch = self.calibration.reference_epoch
        if epoch is not None and not math.isclose(
                self.a_ref, epoch, rel_tol=_EPOCH_TOLERANCE, abs_tol=0.0):
            raise ValueError(
                f"a_ref={self.a_ref!r} disagrees with the calibration's "
                f"reference_epoch={epoch!r}")


def _scale_factors(scale_factor: ScaleFactor) -> NDArray[np.float64]:
    """Validate and promote the independent variable to float64."""
    values = np.asarray(scale_factor, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("scale_factor must be finite")
    if np.any(values <= 0.0):
        raise ValueError("scale_factor must be positive")
    return values


def _core(scale_factor: NDArray[np.float64],
          fit: AsymptoticSplashbackMassHistoryFit) -> NDArray[np.float64]:
    """B(a): the milestone-anchored double-power core, B(a_f) = 1."""
    x = scale_factor / fit.a_f
    return (1.0 - fit.w) * x ** (-fit.gamma_1) + fit.w * x ** (-fit.gamma_2)


def _transient(scale_factor: NDArray[np.float64],
               fit: AsymptoticSplashbackMassHistoryFit) -> NDArray[np.float64]:
    """T(a): the fixed-u=2 transient bump."""
    y = scale_factor / fit.a_pk
    return 2.0 * fit.h / (y ** (-2.0) + y ** fit.v)


def _normalization(fit: AsymptoticSplashbackMassHistoryFit) -> np.float64:
    """K, the constant that makes Psi_hat(a_f) = 1/2 exact."""
    reference, half_mass = np.float64(fit.a_ref), np.float64(fit.a_f)
    return np.float64(
        (math.log(2.0) + _transient(half_mass, fit) - _transient(reference, fit))
        / (1.0 - _core(reference, fit)))


def _shaped(values: NDArray[np.float64],
            scale_factor: NDArray[np.float64]) -> FloatResult:
    """Scalar in, floating scalar out; array in, array out."""
    return values[()] if scale_factor.ndim == 0 else values


def _log_fraction(a: NDArray[np.float64],
                  fit: AsymptoticSplashbackMassHistoryFit) -> NDArray[np.float64]:
    """The equation, on an already-validated float64 array. Written once."""
    reference = np.float64(fit.a_ref)
    return (-_normalization(fit) * (_core(a, fit) - _core(reference, fit))
            + _transient(a, fit) - _transient(reference, fit))


def splashback_mass_history_log_fraction(
    scale_factor: ScaleFactor, fit: AsymptoticSplashbackMassHistoryFit
) -> FloatResult:
    """ln Psi_hat(a) = -K [B(a) - B(a_ref)] + T(a) - T(a_ref)."""
    a = _scale_factors(scale_factor)
    return _shaped(_log_fraction(a, fit), a)


def splashback_mass_history_fraction(
    scale_factor: ScaleFactor, fit: AsymptoticSplashbackMassHistoryFit
) -> FloatResult:
    """Psi_hat(a) = M(a) / M(a_ref).

    Validates once and shares the private kernel: the input is never scanned
    twice on the hot path.
    """
    a = _scale_factors(scale_factor)
    return _shaped(np.exp(_log_fraction(a, fit)), a)


def splashback_mass_accretion_rate(
    scale_factor: ScaleFactor, fit: AsymptoticSplashbackMassHistoryFit
) -> FloatResult:
    """d ln Psi_hat / d ln a, differentiated analytically.

    dB/dln a = -gamma_1 (1 - w) x^(-gamma_1) - gamma_2 w x^(-gamma_2)
    dT/dln a = -2 h (v y^v - 2 y^-2) / [y^-2 + y^v]^2
    """
    a = _scale_factors(scale_factor)
    x = a / fit.a_f
    core_rate = (-fit.gamma_1 * (1.0 - fit.w) * x ** (-fit.gamma_1)
                 - fit.gamma_2 * fit.w * x ** (-fit.gamma_2))
    y = a / fit.a_pk
    denominator = y ** (-2.0) + y ** fit.v
    transient_rate = (-2.0 * fit.h * (fit.v * y ** fit.v - 2.0 * y ** (-2.0))
                      / denominator**2)
    return _shaped(-_normalization(fit) * core_rate + transient_rate, a)
