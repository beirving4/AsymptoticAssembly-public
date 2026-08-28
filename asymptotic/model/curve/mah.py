"""Bounded multistart fit of the finite-reference splashback mass history.

This module is the optimizer boundary for `AsymptoticSplashbackMassHistoryFit`.
It adds no equation: every residual evaluation maps the optimizer vector onto
that accepted record and reuses the canonical kernels in `model.assembly`.

The convention is the Chapter 5 one. The optimizer works in transformed
coordinates so the ordering constraint is structural rather than a penalty:

    theta = (ln a_f, phi, delta, gamma_2, h, ln a_pk, v)
    gamma_1 = gamma_2 + delta,  w = phi

Nine deterministic starts are tried in a fixed order and the best converged
result wins. There is no randomness and no internal parallelism: callers own
the parallelism across independent histories, so nesting threads here would
only risk BLAS oversubscription.

`fit.py` in this package was audited and deliberately not reused: its
`FitConfig` is mutable and shaped around `curve_fit`, and `CurveFitter.optimize`
returns an untyped dict and swallows failures. This package needs bounded
`least_squares`, an immutable typed result, and explicit failure.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
from attrs import define, field
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from ..assembly import AsymptoticSplashbackMassHistoryFit, _log_fraction, _transient
from ..core import AsymptoticCalibration

__all__ = [
    "AsymptoticSplashbackMassHistoryFitResult",
    "fit_asymptotic_splashback_mass_history",
]

#: (ln a_f, phi, delta, gamma_2, h, ln a_pk, v) — the source bounds, frozen
LOWER_BOUNDS: tuple[float, ...] = (
    math.log(0.05), 0.0, 0.05, 0.35, 0.0, math.log(2.0), 0.15)
UPPER_BOUNDS: tuple[float, ...] = (
    math.log(5.0), 0.98, 8.0, 4.0, 0.12, math.log(60.0), 4.0)
#: the deterministic start grid, outer loop first
PHI_STARTS: tuple[float, ...] = (0.1, 0.4, 0.7)
CORE_STARTS: tuple[tuple[float, float], ...] = ((0.6, 1.2), (1.0, 0.8), (0.5, 2.0))
#: the source bump seed, (h, ln a_pk, v)
BUMP_START: tuple[float, float, float] = (0.03, math.log(6.0), 0.9)

#: a scale factor may equal `a_ref` to this relative tolerance, and no more
_REFERENCE_RTOL = 1.0e-9
#: the reference row's mass fraction must be one to this relative tolerance
_UNIT_RTOL = 1.0e-6

Theta: TypeAlias = NDArray[np.float64]


@define(frozen=True, slots=True)
class AsymptoticSplashbackMassHistoryFitResult:
    """One fit plus the diagnostics needed to judge it.

    No residual, Jacobian or covariance array is stored: the thesis
    uncertainty contract is jackknife-based, so a local Jacobian covariance
    would be scientifically misleading here.
    """

    fit: AsymptoticSplashbackMassHistoryFit = field()
    success: bool = field()
    status: int = field()
    message: str = field()
    n_points: int = field()
    starts_attempted: int = field()
    starts_succeeded: int = field()
    nfev: int = field()
    optimality: float = field()
    objective_rss: float = field()
    data_weighted_rss: float = field()
    fractional_rms: float = field()
    fractional_max_abs: float = field()
    regularizer_residual: float = field()


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not a boolean")
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {result!r}")
    return result


def _non_negative_real(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not a boolean")
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {result!r}")
    return result


def _observations(scale_factor: ArrayLike,
                  mass_fraction: ArrayLike) -> tuple[NDArray[np.float64],
                                                     NDArray[np.float64]]:
    """Copy, shape-check and domain-check the caller's history."""
    a = np.array(scale_factor, dtype=np.float64, copy=True)
    psi = np.array(mass_fraction, dtype=np.float64, copy=True)
    if a.ndim != 1 or psi.ndim != 1:
        raise ValueError("scale_factor and mass_fraction must be one-dimensional")
    if a.size != psi.size:
        raise ValueError(
            f"scale_factor and mass_fraction must have equal length, got "
            f"{a.size} and {psi.size}")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("scale_factor must be finite and positive")
    if not np.all(np.diff(a) > 0.0):
        raise ValueError("scale_factor must be strictly increasing")
    if not np.all(np.isfinite(psi)) or np.any(psi <= 0.0):
        raise ValueError("mass_fraction must be finite and positive")
    return a, psi


def half_mass_epoch(scale_factor: NDArray[np.float64],
                    mass_fraction: NDArray[np.float64]) -> float:
    """First epoch the cumulative-max envelope of Psi_hat reaches one half.

    The same log-log crossing the source uses, so the initialization matches
    the published fits. Raises when the history never reaches one half.
    """
    envelope = np.maximum.accumulate(mass_fraction)
    if envelope[0] >= 0.5:
        return float(scale_factor[0])
    if envelope[-1] < 0.5:
        raise ValueError(
            "the history never reaches half of the reference mass; supply "
            "initial_a_f explicitly")
    index = int(np.searchsorted(envelope, 0.5, side="left"))
    a_low, a_high = scale_factor[index - 1], scale_factor[index]
    low, high = envelope[index - 1], envelope[index]
    if high <= low:
        return float(a_high)
    weight = ((math.log10(0.5) - math.log10(low))
              / (math.log10(high) - math.log10(low)))
    return float(10.0 ** (math.log10(a_low)
                          + weight * (math.log10(a_high) - math.log10(a_low))))


def _weights(fractional_error: ArrayLike | None, size: int, error_floor: float,
             systematic_floor: float) -> NDArray[np.float64]:
    """1 / sqrt(max(sigma_rel, floor)^2 + systematic^2), or equal weights."""
    if fractional_error is None:
        return np.ones(size, dtype=np.float64)
    errors = np.asarray(fractional_error, dtype=np.float64)
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("fractional_error must be finite and non-negative")
    try:
        errors = np.broadcast_to(errors, (size,))
    except ValueError as error:
        raise ValueError(
            f"fractional_error is not broadcastable to {size} points") from error
    return 1.0 / np.sqrt(
        np.maximum(errors, error_floor) ** 2 + systematic_floor**2)


def _theta_to_fit(theta: Theta, a_ref: float,
                  calibration: AsymptoticCalibration
                  ) -> AsymptoticSplashbackMassHistoryFit:
    """Map optimizer coordinates onto the accepted record, revalidating it."""
    log_a_f, phi, delta, gamma_2, h, log_a_pk, v = (float(x) for x in theta)
    return AsymptoticSplashbackMassHistoryFit(
        a_f=math.exp(log_a_f), w=phi, gamma_1=gamma_2 + delta, gamma_2=gamma_2,
        h=h, a_pk=math.exp(log_a_pk), v=v, a_ref=a_ref,
        calibration=calibration)


def _late_tail_residual(fit: AsymptoticSplashbackMassHistoryFit,
                        late_tail_cap: float, late_tail_sigma: float) -> float:
    """max(0, T(a_ref) - cap) / sigma — the transient must have decayed."""
    tail = float(_transient(np.float64(fit.a_ref), fit))
    return max(0.0, tail - late_tail_cap) / late_tail_sigma


def _starts(initial_a_f: float) -> Iterator[Theta]:
    """The nine frozen starts, in their stable order."""
    for phi in PHI_STARTS:
        for gamma_2, delta in CORE_STARTS:
            yield np.clip(
                np.array([math.log(initial_a_f), phi, delta, gamma_2,
                          *BUMP_START], dtype=np.float64),
                LOWER_BOUNDS, UPPER_BOUNDS)


def fit_asymptotic_splashback_mass_history(
    scale_factor: ArrayLike,
    mass_fraction: ArrayLike,
    *,
    a_ref: float,
    calibration: AsymptoticCalibration,
    fractional_error: ArrayLike | None = None,
    initial_a_f: float | None = None,
    error_floor: float = 0.002,
    systematic_floor: float = 0.005,
    late_tail_cap: float = 0.005,
    late_tail_sigma: float = 0.002,
    max_nfev: int = 15_000,
) -> AsymptoticSplashbackMassHistoryFitResult:
    """Fit one finite-reference splashback mass history.

    `mass_fraction` is Psi_hat = M(a)/M(a_ref); values above one are legitimate
    because the splashback transient overshoots. `fractional_error` is a
    *fractional* uncertainty on that quantity. A row at `a_ref` is verified to
    be unity and then excluded, because the model anchors it independently of
    the parameters.

    Raises `ValueError` for any invalid input and `RuntimeError` when no start
    produces a usable fit. It never returns a fabricated default.
    """
    a_ref = _positive_real(a_ref, "a_ref")
    if not isinstance(calibration, AsymptoticCalibration):
        raise TypeError(
            f"calibration must be an AsymptoticCalibration, got "
            f"{type(calibration).__name__}")
    epoch = calibration.reference_epoch
    if epoch is not None and not math.isclose(
            a_ref, epoch, rel_tol=_REFERENCE_RTOL, abs_tol=0.0):
        raise ValueError(
            f"a_ref={a_ref!r} disagrees with the calibration's "
            f"reference_epoch={epoch!r}")
    error_floor = _non_negative_real(error_floor, "error_floor")
    systematic_floor = _non_negative_real(systematic_floor, "systematic_floor")
    late_tail_cap = _non_negative_real(late_tail_cap, "late_tail_cap")
    late_tail_sigma = _positive_real(late_tail_sigma, "late_tail_sigma")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)):
        raise TypeError("max_nfev must be an integer")
    max_nfev = int(max_nfev)
    if max_nfev <= 0:
        raise ValueError(f"max_nfev must be positive, got {max_nfev!r}")

    a, psi = _observations(scale_factor, mass_fraction)
    beyond = a > a_ref * (1.0 + _REFERENCE_RTOL)
    if np.any(beyond):
        raise ValueError("no scale factor may exceed a_ref")
    at_reference = np.isclose(a, a_ref, rtol=_REFERENCE_RTOL, atol=0.0)
    if int(np.count_nonzero(at_reference)) > 1:
        raise ValueError("at most one observation may represent a_ref")
    if np.any(at_reference) and not math.isclose(
            float(psi[at_reference][0]), 1.0, rel_tol=_UNIT_RTOL, abs_tol=0.0):
        raise ValueError(
            "the observation at a_ref must have a mass fraction of one, got "
            f"{float(psi[at_reference][0])!r}")

    start_a_f = (half_mass_epoch(a, psi) if initial_a_f is None
                 else _positive_real(initial_a_f, "initial_a_f"))
    if not math.exp(LOWER_BOUNDS[0]) <= start_a_f <= math.exp(UPPER_BOUNDS[0]):
        raise ValueError(
            f"initial_a_f must lie within the fitter's a_f bounds, got "
            f"{start_a_f!r}")

    fitted = ~at_reference
    a_fit, psi_fit = a[fitted], psi[fitted]
    if a_fit.size < 12:
        raise ValueError(
            f"at least 12 fitted points are required, got {a_fit.size}")
    weights = _weights(fractional_error, a.size, error_floor,
                       systematic_floor)[fitted]
    log_psi = np.log(psi_fit)

    def residual(theta: Theta) -> NDArray[np.float64]:
        fit = _theta_to_fit(theta, a_ref, calibration)
        data = (log_psi - _log_fraction(a_fit, fit)) * weights
        return np.append(
            data, _late_tail_residual(fit, late_tail_cap, late_tail_sigma))

    attempted, succeeded, best = 0, 0, None
    for theta_0 in _starts(start_a_f):
        attempted += 1
        try:
            solution = least_squares(
                residual, theta_0, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                method="trf", max_nfev=max_nfev)
        except (ValueError, FloatingPointError):
            continue
        objective = float(np.sum(np.asarray(solution.fun) ** 2))
        if not (np.all(np.isfinite(solution.x)) and math.isfinite(objective)):
            continue
        succeeded += int(bool(solution.success))
        # a converged result always beats an unconverged one
        key = (not bool(solution.success), objective)
        if best is None or key < best[0]:
            best = (key, solution, objective)

    if best is None:
        raise RuntimeError(
            f"no usable fit: {attempted} starts attempted, {succeeded} converged")
    _, solution, objective = best

    fit = _theta_to_fit(solution.x, a_ref, calibration)
    penalty = _late_tail_residual(fit, late_tail_cap, late_tail_sigma)
    data_residual = (log_psi - _log_fraction(a_fit, fit)) * weights
    fractional = np.exp(_log_fraction(a_fit, fit)) / psi_fit - 1.0
    return AsymptoticSplashbackMassHistoryFitResult(
        fit=fit,
        success=bool(solution.success),
        status=int(solution.status),
        message=str(solution.message),
        n_points=int(a_fit.size),
        starts_attempted=attempted,
        starts_succeeded=succeeded,
        nfev=int(solution.nfev),
        optimality=float(solution.optimality),
        objective_rss=objective,
        data_weighted_rss=float(np.sum(data_residual**2)),
        fractional_rms=float(np.sqrt(np.mean(fractional**2))),
        fractional_max_abs=float(np.max(np.abs(fractional))),
        regularizer_residual=penalty,
    )
