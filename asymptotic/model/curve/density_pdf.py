"""Trust masking, bounded multistart fitting and diagnostics for the density PDF.

The equation, its normalization and the CDF helpers live in
`asymptotic.model.density_pdf`; this module never restates them. It owns the
per-bin trust mask, the deterministic nine-start `soft_l1` fit in log10 P_V,
and the quality gates that decide whether a fit is publishable.

Two result shapes exist on purpose. `DensityPdfFitResult` is the legacy
mutable dataclass, preserved unchanged so existing callers keep working;
`AsymptoticDensityPdfFitResult` is the immutable record new code receives.
There is exactly one optimizer implementation behind both.

`model/curve/fit.py` was audited again and deliberately not reused: its
`FitConfig` is mutable and shaped around `curve_fit`, and `CurveFitter.optimize`
returns an untyped dict and swallows failures.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
from attrs import define
from attrs import field as attrs_field
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from ..core import AsymptoticCalibration
from ..density_pdf import (
    _LN_X_MAX,
    KLYPIN_EXPONENTS,
    AsymptoticDensityPdfFit,
    ModelName,
    Params,
    model_cdf_above,
    model_ln_pv,
    normalized_params,
)

__all__ = [
    "DensityPdfFitResult",
    "AsymptoticDensityPdfFitResult",
    "build_trust_mask",
    "fit_density_pdf",
    "fit_asymptotic_density_pdf",
]

#: per-bin boolean trust mask
Mask: TypeAlias = NDArray[np.bool_]
#: the internal gauge parameters recorded alongside a fit
Gauge: TypeAlias = Mapping[str, float]
#: the per-bin log10 sigma and whether real errors were supplied
SigmaResult: TypeAlias = tuple[NDArray[np.float64], bool]
#: the free optimizer vector (alpha, ln x0_int[, a0, a1])
Theta: TypeAlias = NDArray[np.float64]
#: (alpha, x0_int, a0, a1) unpacked from a theta vector
Unpacked: TypeAlias = tuple[float, float, float, float]


# ------------------------------------------------------------------ fit setup
def build_trust_mask(x: ArrayLike, pv: ArrayLike,
                     support: ArrayLike | None = None,
                     rel_err: ArrayLike | None = None,
                     n_min: float = 100.0, rel_err_max: float = 5.0,
                     cells: ArrayLike | None = None,
                     n_cells_min: float = 5.0) -> Mask:
    """Per-bin trust mask for fitting P_V.

    Rules (each recorded in DensityPdfFitResult.mask_rules):
      finite positive x and P_V;
      mass-weighted support (particle weight per bin) >= n_min, which trims
        both the truncated low-density tail and the unresolved high-density
        tail;
      grid-cell support >= n_cells_min (only when a cell-count array is
        provided) — on coarse grids every cell carries thousands of particles,
        so the particle threshold alone would admit single-cell tail spikes;
      jackknife relative error finite, positive, and <= rel_err_max (only when
        an error array is provided);
      final mask = the largest contiguous True run (no ragged holes).
    """
    x = np.asarray(x, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)
    m = np.isfinite(x) & (x > 0) & np.isfinite(pv) & (pv > 0)
    if support is not None:
        s = np.asarray(support, dtype=np.float64)
        m &= np.isfinite(s) & (s >= n_min)
    if cells is not None:
        c = np.asarray(cells, dtype=np.float64)
        m &= np.isfinite(c) & (c >= n_cells_min)
    if rel_err is not None:
        r = np.asarray(rel_err, dtype=np.float64)
        m &= np.isfinite(r) & (r > 0) & (r <= rel_err_max)
    if not m.any():
        return m
    # largest contiguous run
    idx = np.flatnonzero(m)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, breaks + 1)
    best = max(runs, key=len)
    out = np.zeros_like(m)
    out[best] = True
    return out


@dataclass
class DensityPdfFitResult:
    model: str
    status: str                          # "ok" | "failed"
    reason: str                          # "" when ok
    params: dict | None                  # normalized_params dict (physical)
    gauge: dict | None                   # internal gauge parameters
    n_bins: int
    x_lo: float
    x_hi: float
    rms_log10_resid: float
    max_abs_log10_resid: float
    on_bound: list = field(default_factory=list)
    tail_mass_below: float = np.nan      # fitted int_0^{x_lo} x P dx
    tail_mass_above: float = np.nan      # fitted int_{x_hi}^inf x P dx
    norm_tol: float = 1e-3
    mask_rules: str = ""
    weighted: bool = False


def _prepare_sigma_log10(pv: ArrayLike, rel_err: ArrayLike | None,
                         mask: Mask, err_floor: float) -> SigmaResult:
    if rel_err is None:
        return np.full(int(mask.sum()), 1.0), False
    r = np.asarray(rel_err, dtype=np.float64)[mask]
    r = np.maximum(np.where(np.isfinite(r) & (r > 0), r, err_floor), err_floor)
    return r / np.log(10.0), True


def fit_density_pdf(x: ArrayLike, pv: ArrayLike,
                    support: ArrayLike | None = None,
                    rel_err: ArrayLike | None = None,
                    model: ModelName = "klypin", n_min: float = 100.0,
                    err_floor: float = 0.05, norm_tol: float = 1e-3,
                    exponent_bounds: tuple[float, float] = (0.15, 3.0),
                    cells: ArrayLike | None = None,
                    n_cells_min: float = 5.0) -> DensityPdfFitResult:
    """Fit the doubly-normalized model to a cached volume-weighted PDF curve.

    Parameters
    ----------
    x, pv : the enforced mean-one bin centers and P_V values (load_pdf_group
        outputs `_x_centers` and `vol_pdf`).
    support : per-bin mass-weighted counts (the raw `histogram` array; sums to
        the particle count) used for the trust mask.
    rel_err : per-bin relative jackknife error of the cached PDF
        (`pdf_error/pdf`); mapped 1:1 onto P_V (see module docstring).
    cells : optional per-bin grid-cell counts (support scaled by
        N_cells/N_particles); trims single-cell tail spikes on coarse grids.
    model : "klypin" (fixed exponents 1.1/0.55) or "chen" (free exponents).
    """
    if model not in ("klypin", "chen"):
        raise ValueError(f"unknown model {model!r}")
    x = np.asarray(x, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)

    mask = build_trust_mask(x, pv, support=support, rel_err=rel_err, n_min=n_min,
                            cells=cells, n_cells_min=n_cells_min)
    rules = (f"finite&positive; support>={n_min:g}; "
             f"cells>={n_cells_min:g} ({'on' if cells is not None else 'off'}); "
             f"rel_err finite&<=5 ({'on' if rel_err is not None else 'off'}); "
             "largest contiguous run")
    n_free = 2 if model == "klypin" else 4
    n_needed = n_free + 4
    if mask.sum() < n_needed:
        return DensityPdfFitResult(
            model=model, status="failed", reason="too_few_trusted_bins",
            params=None, gauge=None, n_bins=int(mask.sum()),
            x_lo=np.nan, x_hi=np.nan, rms_log10_resid=np.nan,
            max_abs_log10_resid=np.nan, mask_rules=rules)

    xm, pm = x[mask], pv[mask]
    log10_p = np.log10(pm)
    sig, weighted = _prepare_sigma_log10(pv, rel_err, mask, err_floor)

    a0_fix, a1_fix = KLYPIN_EXPONENTS
    lo_e, hi_e = exponent_bounds

    def unpack(theta: Theta) -> Unpacked:
        if model == "klypin":
            alpha, ln_x0 = theta
            return alpha, np.exp(ln_x0), a0_fix, a1_fix
        alpha, ln_x0, a0, a1 = theta
        return alpha, np.exp(ln_x0), a0, a1

    def resid(theta: Theta) -> NDArray[np.float64]:
        alpha, x0i, a0, a1 = unpack(theta)
        pars = normalized_params(alpha, x0i, a0, a1)
        if pars is None:
            return np.full(xm.size, 1e3)
        with np.errstate(under="ignore", divide="ignore"):
            ln10_model = model_ln_pv(
                xm, pars["alpha"], pars["x0"], pars["x1"],
                pars["a0"], pars["a1"], np.log(pars["A"])) / np.log(10.0)
        r = (ln10_model - log10_p) / sig
        return np.where(np.isfinite(r), r, 1e3)

    if model == "klypin":
        lo = np.array([-3.0, np.log(1e-8)])
        hi = np.array([5.0, np.log(1e4)])
        starts = [np.array([a, np.log(x0)])
                  for a in (0.2, 0.8, 1.6) for x0 in (0.01, 0.1, 1.0)]
    else:
        lo = np.array([-3.0, np.log(1e-8), lo_e, lo_e])
        hi = np.array([5.0, np.log(1e4), hi_e, hi_e])
        starts = [np.array([a, np.log(x0), a0_fix, a1_fix])
                  for a in (0.2, 0.8, 1.6) for x0 in (0.01, 0.1, 1.0)]

    best = None
    for s0 in starts:
        try:
            sol = least_squares(resid, s0, bounds=(lo, hi), loss="soft_l1",
                                f_scale=1.0, max_nfev=4000)
        except (ValueError, FloatingPointError):
            # the one authorized hardening: an unexpected exception propagates
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None or not np.all(np.isfinite(best.x)):
        return DensityPdfFitResult(
            model=model, status="failed", reason="optimizer_failed",
            params=None, gauge=None, n_bins=int(mask.sum()),
            x_lo=float(xm[0]), x_hi=float(xm[-1]), rms_log10_resid=np.nan,
            max_abs_log10_resid=np.nan, mask_rules=rules, weighted=weighted)

    alpha, x0i, a0, a1 = unpack(best.x)
    pars = normalized_params(alpha, x0i, a0, a1)
    on_bound = [name for name, value, lower, upper in zip(
        (["alpha", "ln_x0_int"] if model == "klypin"
         else ["alpha", "ln_x0_int", "a0", "a1"]),
        best.x, lo, hi)
        if min(value - lower, upper - value) < 1e-3 * (upper - lower)]

    with np.errstate(under="ignore", divide="ignore"):
        r_unw = (model_ln_pv(xm, pars["alpha"], pars["x0"], pars["x1"],
                             pars["a0"], pars["a1"], np.log(pars["A"]))
                 / np.log(10.0)) - log10_p
    rms = float(np.sqrt(np.mean(r_unw ** 2)))
    mx = float(np.max(np.abs(r_unw)))

    status, reason = "ok", ""
    if abs(pars["I0"] - 1) > norm_tol or abs(pars["I1"] - 1) > norm_tol:
        status, reason = "failed", "normalization_tolerance_exceeded"
    if on_bound:
        status, reason = "failed", "parameter_on_bound:" + ",".join(on_bound)

    tail_lo = float(np.clip(1.0 - model_cdf_above(pars, float(xm[0])), 0.0, 1.0))
    tail_hi = float(model_cdf_above(pars, float(xm[-1])))
    # upper-tail stability: the quadrature window must contain the mass integral
    if model_cdf_above(pars, np.exp(_LN_X_MAX - 2.0)) > 1e-8:
        status, reason = "failed", "upper_tail_unstable"

    return DensityPdfFitResult(
        model=model, status=status, reason=reason, params=pars,
        gauge={"alpha": float(alpha), "x0_int": float(x0i),
               "a0": float(a0), "a1": float(a1)},
        n_bins=int(mask.sum()), x_lo=float(xm[0]), x_hi=float(xm[-1]),
        rms_log10_resid=rms, max_abs_log10_resid=mx, on_bound=on_bound,
        tail_mass_below=tail_lo, tail_mass_above=tail_hi,
        norm_tol=norm_tol, mask_rules=rules, weighted=weighted)


# ------------------------------------------------------------ public boundary
def _read_only_gauge(gauge: Mapping[str, float] | None) -> Gauge | None:
    """Copy and freeze; the caller's mapping is never retained."""
    return None if gauge is None else MappingProxyType(dict(gauge))


def _frozen_names(names: Iterable[str]) -> tuple[str, ...]:
    """Copy any iterable of parameter names into a tuple."""
    return tuple(names)


@define(frozen=True, slots=True)
class AsymptoticDensityPdfFitResult:
    """One density-PDF fit attempt and everything needed to judge it.

    `fit` is `None` only for a hard failure that produced no parameters. A
    diagnostic gate (normalization tolerance, an active bound, upper-tail
    instability) sets `status="failed"` while retaining the real fit, the same
    honest fit-versus-success separation the mass-history fitter uses.
    """

    fit: AsymptoticDensityPdfFit | None = attrs_field()
    model: ModelName = attrs_field()
    status: Literal["ok", "failed"] = attrs_field()
    reason: str = attrs_field()
    gauge: Gauge | None = attrs_field(converter=_read_only_gauge)
    n_bins: int = attrs_field()
    x_lo: float = attrs_field()
    x_hi: float = attrs_field()
    rms_log10_resid: float = attrs_field()
    max_abs_log10_resid: float = attrs_field()
    on_bound: tuple[str, ...] = attrs_field(converter=_frozen_names)
    tail_mass_below: float = attrs_field()
    tail_mass_above: float = attrs_field()
    norm_tol: float = attrs_field()
    mask_rules: str = attrs_field()
    weighted: bool = attrs_field()


def _fit_from_params(params: Params | None, model: ModelName,
                     calibration: AsymptoticCalibration
                     ) -> AsymptoticDensityPdfFit | None:
    """Adapt the legacy parameter mapping; never fabricate one."""
    if params is None:
        return None
    return AsymptoticDensityPdfFit(
        A=params["A"], alpha=params["alpha"], x0=params["x0"],
        x1=params["x1"], a0=params["a0"], a1=params["a1"], model=model,
        calibration=calibration)


def fit_asymptotic_density_pdf(
    x: ArrayLike,
    pv: ArrayLike,
    *,
    calibration: AsymptoticCalibration,
    support: ArrayLike | None = None,
    rel_err: ArrayLike | None = None,
    model: ModelName = "klypin",
    n_min: float = 100.0,
    err_floor: float = 0.05,
    norm_tol: float = 1e-3,
    exponent_bounds: tuple[float, float] = (0.15, 3.0),
    cells: ArrayLike | None = None,
    n_cells_min: float = 5.0,
) -> AsymptoticDensityPdfFitResult:
    """Fit the density PDF and return the immutable thesis-model result.

    Wraps the one scientific engine: no equation, mask, optimizer or diagnostic
    is recomputed here. The caller supplies the calibration, which the returned
    fit retains by identity.
    """
    if not isinstance(calibration, AsymptoticCalibration):
        raise TypeError(
            "calibration must be an AsymptoticCalibration, got "
            f"{type(calibration).__name__}")

    legacy = fit_density_pdf(
        x, pv, support=support, rel_err=rel_err, model=model, n_min=n_min,
        err_floor=err_floor, norm_tol=norm_tol,
        exponent_bounds=exponent_bounds, cells=cells,
        n_cells_min=n_cells_min)

    return AsymptoticDensityPdfFitResult(
        fit=_fit_from_params(legacy.params, legacy.model, calibration),
        model=legacy.model,
        status=legacy.status,
        reason=legacy.reason,
        gauge=legacy.gauge,
        n_bins=legacy.n_bins,
        x_lo=legacy.x_lo,
        x_hi=legacy.x_hi,
        rms_log10_resid=legacy.rms_log10_resid,
        max_abs_log10_resid=legacy.max_abs_log10_resid,
        on_bound=legacy.on_bound,
        tail_mass_below=legacy.tail_mass_below,
        tail_mass_above=legacy.tail_mass_above,
        norm_tol=legacy.norm_tol,
        mask_rules=legacy.mask_rules,
        weighted=legacy.weighted,
    )
