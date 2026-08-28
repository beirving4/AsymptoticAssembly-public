"""Canonical Klypin/Chen one-point density PDF: state, equation, normalization.

Conventions (must hold everywhere; `x = rho/rho_mean = 1 + delta`):

    P_V(x): volume-weighted PDF,  int P_V dx = 1  AND  int x P_V dx = 1.
    P_M(x) = x P_V(x): mass-weighted PDF (derived, never fit independently).
    F_M(>x_t) = int_{x_t}^inf x P_V dx: upper cumulative MASS fraction.

Fiducial model (Klypin et al. 2018, eq. 18 form; tail exponents fixed):

    P_V(x) = A x^-alpha exp[-(x0/x)^1.1] exp[-(x/x1)^0.55]

Sensitivity model (Chen, Gnedin & Mansfield 2022, eq. 11 form; free exponents):

    P_V(x) = A x^-alpha exp[-(x0/x)^a0] exp[-(x/x1)^a1]

Both normalization constraints are imposed EXACTLY (to quadrature accuracy)
through a scaling closure: the family is closed under x -> x/s with
(x0, x1) -> (x0/s, x1/s), which maps a unit-area member with mean m onto a
unit-area member with mean m/s. The reported (A, alpha, x0, x1) always satisfy
both integrals within the caller's `norm_tol`.

This module owns the equation and its normalization. Fitting, trust masking and
diagnostics live in `asymptotic.model.curve.density_pdf`. NumPy and attrs only:
no SciPy, no I/O, no plotting.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from numbers import Integral
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
from attrs import define, field
from numpy.typing import ArrayLike, NDArray

from .core import AsymptoticCalibration, _as_float

__all__ = [
    "KLYPIN_EXPONENTS",
    "AsymptoticDensityPdfFit",
    "AsymptoticDensityPdfModel",
    "density_pdf_log_value",
    "density_pdf_value",
    "density_pdf_mass_fraction_above",
    "model_ln_pv",
    "normalized_params",
    "evaluate_pv",
    "model_cdf_above",
    "direct_cdf_above",
]

# numpy 2 renamed trapz to trapezoid; support both lines
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

KLYPIN_EXPONENTS = (1.1, 0.55)          # fixed (a0, a1) of the fiducial model
_LN_X_MIN, _LN_X_MAX = np.log(1e-10), np.log(1e14)
_N_GRID = 6001                           # quadrature grid (trapz in ln x)
_EXP_CLIP = 700.0                        # exp() overflow guard

#: the fitted parameter mapping produced by `normalized_params`
Params: TypeAlias = dict[str, float]
#: which member of the family a fit belongs to
ModelName: TypeAlias = Literal["klypin", "chen"]
#: floating scalar for scalar input, float64 array otherwise
FloatResult: TypeAlias = np.float64 | NDArray[np.float64]
#: the legacy CDF helper returns a builtin float for a scalar threshold
CdfResult: TypeAlias = float | NDArray[np.float64]
#: (int P dx, int x P dx) on the fixed quadrature grid
Integrals: TypeAlias = tuple[float, float]
#: the Klypin exponents are fixed, so a klypin fit must carry them exactly
_EXPONENT_TOLERANCE = 1e-12


# ----------------------------------------------------------------- model core
def model_ln_pv(x: ArrayLike, alpha: float, x0: float, x1: float,
                a0: float, a1: float, ln_amp: float = 0.0) -> FloatResult:
    """ln P_V(x) for amplitude exp(ln_amp). Vectorized, under/overflow safe."""
    lx = np.log(np.asarray(x, dtype=np.float64))
    t0 = np.exp(np.clip(a0 * (np.log(x0) - lx), None, _EXP_CLIP))
    t1 = np.exp(np.clip(a1 * (lx - np.log(x1)), None, _EXP_CLIP))
    return ln_amp - alpha * lx - t0 - t1


def _integrals(alpha: float, x0: float, x1: float, a0: float, a1: float,
               ln_amp: float = 0.0) -> Integrals:
    """(I0, I1) = (int P dx, int x P dx) by trapz on a fixed ln-x grid."""
    u = np.linspace(_LN_X_MIN, _LN_X_MAX, _N_GRID)
    x = np.exp(u)
    with np.errstate(under="ignore"):
        p = np.exp(model_ln_pv(x, alpha, x0, x1, a0, a1, ln_amp))
    i0 = _trapezoid(p * x, u)              # int P dx   = int P x d(ln x)
    i1 = _trapezoid(p * x * x, u)          # int x P dx = int P x^2 d(ln x)
    return float(i0), float(i1)


def normalized_params(alpha: float, x0_int: float, a0: float,
                      a1: float) -> Params | None:
    """Map gauge parameters (x1_int = 1) onto the physical, doubly-normalized
    parameter set. Returns dict with A, alpha, x0, x1, a0, a1 and the achieved
    integrals (I0, I1), which equal 1 to quadrature accuracy by construction."""
    i0, i1 = _integrals(alpha, x0_int, 1.0, a0, a1)
    if not (np.isfinite(i0) and np.isfinite(i1)) or i0 <= 0 or i1 <= 0:
        return None
    m = i1 / i0                          # mean of the unit-area gauge member
    x0, x1 = x0_int / m, 1.0 / m         # scaling closure -> mean one
    j0, j1 = _integrals(alpha, x0, x1, a0, a1)
    if not np.isfinite(j0) or j0 <= 0:
        return None
    amp = 1.0 / j0                       # numerical amplitude -> unit area
    return {
        "A": amp, "alpha": float(alpha), "x0": float(x0), "x1": float(x1),
        "a0": float(a0), "a1": float(a1),
        "I0": j0 * amp, "I1": (j1 / j0),
    }


def evaluate_pv(x: ArrayLike, params: Params) -> FloatResult:
    """P_V(x) for a `normalized_params` dict."""
    with np.errstate(under="ignore"):
        return np.exp(model_ln_pv(
            x, params["alpha"], params["x0"], params["x1"],
            params["a0"], params["a1"], np.log(params["A"])))


def model_cdf_above(params: Params, x_t: ArrayLike) -> CdfResult:
    """Fitted F_M(>x_t) = int_{x_t}^inf x P_V dx.

    The bounded loop runs over flattened thresholds and the answer is reshaped
    back, so any threshold shape is preserved. The quadrature, its limits and
    the scalar builtin-float return are unchanged.
    """
    scalar = np.isscalar(x_t)
    thresholds = np.atleast_1d(np.asarray(x_t, dtype=np.float64))
    out: list[float] = []
    for xt in thresholds.reshape(-1):
        lo = np.log(max(float(xt), np.exp(_LN_X_MIN)))
        if lo >= _LN_X_MAX:
            out.append(0.0)
            continue
        u = np.linspace(lo, _LN_X_MAX, _N_GRID)
        x = np.exp(u)
        out.append(float(_trapezoid(evaluate_pv(x, params) * x * x, u)))
    values = np.asarray(out, dtype=np.float64).reshape(thresholds.shape)
    return float(values.reshape(-1)[0]) if scalar else values


def direct_cdf_above(x_edges: ArrayLike, pdf_mass: ArrayLike,
                     x_t: ArrayLike) -> FloatResult:
    """Direct F_M(>x_t) from the cached mass-weighted PDF on its own bins.
    Linear interpolation of the cumulative between bin edges is the EXACT
    partial-bin integral for a histogram (piecewise-constant) PDF."""
    xe = np.asarray(x_edges, dtype=np.float64)
    pm = np.asarray(pdf_mass, dtype=np.float64)
    w = np.diff(xe)
    below = np.concatenate([[0.0], np.cumsum(pm * w)])
    below = below / below[-1]
    return 1.0 - np.interp(np.asarray(x_t, dtype=np.float64), xe, below)


# ------------------------------------------------------------ public boundary
def _finite(instance: object, attribute: object, value: float) -> None:
    name = getattr(attribute, "name", "field")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _finite_positive(instance: object, attribute: object, value: float) -> None:
    _finite(instance, attribute, value)
    name = getattr(attribute, "name", "field")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@define(frozen=True, slots=True)
class AsymptoticDensityPdfFit:
    """One immutable, doubly-normalized density-PDF parameter set.

    State only: status, gauge, residual summaries and mask rules belong to
    `AsymptoticDensityPdfFitResult` in the curve module. Double normalization
    is established by the fitting action and proved numerically in tests, not
    re-integrated on every construction — the record stays cheap.
    """

    A: float = field(converter=_as_float, validator=_finite_positive)
    alpha: float = field(converter=_as_float, validator=_finite)
    x0: float = field(converter=_as_float, validator=_finite_positive)
    x1: float = field(converter=_as_float, validator=_finite_positive)
    a0: float = field(converter=_as_float, validator=_finite_positive)
    a1: float = field(converter=_as_float, validator=_finite_positive)
    model: ModelName = field()
    calibration: AsymptoticCalibration = field()

    @model.validator
    def _known_model(self, attribute: object, value: str) -> None:
        if value not in ("klypin", "chen"):
            raise ValueError(f"unknown model {value!r}")

    @calibration.validator
    def _is_calibration(self, attribute: object,
                        value: AsymptoticCalibration) -> None:
        if not isinstance(value, AsymptoticCalibration):
            raise TypeError(
                "calibration must be an AsymptoticCalibration, got "
                f"{type(value).__name__}")

    def __attrs_post_init__(self) -> None:
        if self.model != "klypin":
            return
        fixed_a0, fixed_a1 = KLYPIN_EXPONENTS
        if not (math.isclose(self.a0, fixed_a0, rel_tol=_EXPONENT_TOLERANCE)
                and math.isclose(self.a1, fixed_a1,
                                 rel_tol=_EXPONENT_TOLERANCE)):
            raise ValueError(
                f"a klypin fit must carry the fixed exponents {KLYPIN_EXPONENTS}, "
                f"got ({self.a0!r}, {self.a1!r})")


def _params(fit: AsymptoticDensityPdfFit) -> Params:
    """The mapping the canonical kernels consume. Not a second equation."""
    return {"A": fit.A, "alpha": fit.alpha, "x0": fit.x0, "x1": fit.x1,
            "a0": fit.a0, "a1": fit.a1}


def _abscissa(x: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must be positive")
    return values


def _shaped(values: NDArray[np.float64],
            reference: NDArray[np.float64]) -> FloatResult:
    return values[()] if reference.ndim == 0 else values


def density_pdf_log_value(
    x: ArrayLike, fit: AsymptoticDensityPdfFit
) -> FloatResult:
    """ln P_V(x) at the fitted, doubly-normalized parameters."""
    values = _abscissa(x, "x")
    with np.errstate(under="ignore"):
        return _shaped(
            np.asarray(model_ln_pv(values, fit.alpha, fit.x0, fit.x1, fit.a0,
                                   fit.a1, math.log(fit.A)),
                       dtype=np.float64), values)


def density_pdf_value(x: ArrayLike, fit: AsymptoticDensityPdfFit) -> FloatResult:
    """P_V(x), the volume-weighted density PDF."""
    values = _abscissa(x, "x")
    with np.errstate(under="ignore"):
        return _shaped(
            np.asarray(evaluate_pv(values, _params(fit)), dtype=np.float64),
            values)


def density_pdf_mass_fraction_above(
    x_threshold: ArrayLike, fit: AsymptoticDensityPdfFit
) -> FloatResult:
    """F_M(>x_t) = int_{x_t}^inf x P_V dx, the upper cumulative mass fraction."""
    thresholds = _abscissa(x_threshold, "x_threshold")
    return _shaped(
        np.asarray(model_cdf_above(_params(fit), thresholds),
                   dtype=np.float64).reshape(thresholds.shape), thresholds)

# ------------------------------------------------- snapshot-indexed collection
#: one accepted fit per snapshot id; the only axis this model has
DensityPdfFits: TypeAlias = Mapping[int, AsymptoticDensityPdfFit]

def _snapshot_id(value: Integral) -> int:
    """A genuine integer snapshot id as a builtin `int`.

    NumPy integers are accepted; `bool` is not a snapshot id, and neither is
    `1.0` — a float key would silently alias an integer one.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a snapshot id")
    if not isinstance(value, Integral):
        raise TypeError(
            f"snapshot id must be an integer, got {type(value).__name__}")
    return int(value)

def _snapshot_fits(fits: DensityPdfFits) -> DensityPdfFits:
    """Copy, normalize the keys and sort ascending; never retain the caller's
    mapping. Each fit object is stored exactly as given."""
    if not isinstance(fits, Mapping):
        raise TypeError(f"fits must be a mapping, got {type(fits).__name__}")
    if not fits:
        raise ValueError("fits must not be empty")
    collected: dict[int, AsymptoticDensityPdfFit] = {}
    for key, fit in fits.items():
        snapshot_id = _snapshot_id(key)
        if snapshot_id in collected:
            raise ValueError(f"duplicate snapshot id {snapshot_id}")
        if not isinstance(fit, AsymptoticDensityPdfFit):
            raise TypeError(
                f"snapshot {snapshot_id} must hold an AsymptoticDensityPdfFit, "
                f"got {type(fit).__name__}")
        collected[snapshot_id] = fit
    return MappingProxyType(dict(sorted(collected.items())))

def _model_name(instance: object, attribute: object, value: str) -> None:
    if value not in ("klypin", "chen"):
        raise ValueError(f"unknown model {value!r}")

@define(frozen=True, slots=True)
class AsymptoticDensityPdfModel:
    """One density-PDF family, one accepted fit per snapshot.

    State only: no fitting, interpolation, quadrature or I/O. Callers choose
    which fits enter the model — nothing here filters, promotes or repairs an
    `AsymptoticDensityPdfFitResult`. Epoch and smoothing scope stay in each
    fit's `AsymptoticCalibration`; snapshot id is the model's only axis.

    Evaluate a snapshot with the module-level actions::

        density_pdf_value(x, model[snapshot_id])
    """

    model: ModelName = field(validator=_model_name)
    fits: DensityPdfFits = field(converter=_snapshot_fits)

    def __attrs_post_init__(self) -> None:
        wrong = [snapshot_id for snapshot_id, fit in self.fits.items()
                 if fit.model != self.model]
        if wrong:
            raise ValueError(
                f"snapshots {wrong} do not carry the {self.model!r} model")

    def __len__(self) -> int:
        return len(self.fits)

    def __iter__(self) -> Iterator[int]:
        return iter(self.fits)

    def __getitem__(self, snapshot_id: Integral) -> AsymptoticDensityPdfFit:
        return self.fits[_snapshot_id(snapshot_id)]

    @property
    def snapshot_ids(self) -> tuple[int, ...]:
        """Ascending, immutable."""
        return tuple(self.fits)
