"""Temporally-ordered assembly-milestone estimators (Chapter 5 §1.1).

Canonical location for the milestone estimators first prototyped under
``notebooks/chapter5_assembly/``. Pure NumPy, no I/O or path assumptions, so any
postprocessing driver or study can import them:

    from AsymptoticAssembly.asymptotic.studies.mah.milestone_estimators import (
        first_attainment, persistent_completion, mah_freeze,
    )

Replaces the inverse-interpolation milestone helpers (``get_mass_accumulation_time``,
``get_freeze_out_time``), which call ``interp1d`` on a non-monotonic independent
variable and therefore return a sorted blend that is *neither* a first, last, nor
sustained crossing (verified: stored ``a_mass_accum`` matches none of them).

Every estimator here is explicitly time-ordered and distinguishes:

  * first_attainment      — first epoch the cumulative-max envelope reaches x*M_ref
                            (a formation clock; for x=0.5 == the pipeline's a_half)
  * persistent_completion — earliest epoch after which M/M_ref stays >= (1-eps)
                            (a completion clock; right-censored if never satisfied)
  * mah_freeze            — earliest epoch after which |Gamma_tdyn| stays < tol
                            (a freeze clock on the SAME rate shown in figures;
                            right-censored if not frozen by a_ref)
  * isolation             — last major merger (handled at the call site from the
                            tree; no-event halos are CENSORED, not dropped)

Conventions: histories are on the global a-grid, normalized Psi(a)=M(a)/M(a_ref)
with the FINITE reference a_ref=100 (NOT infinity). Right-censoring returns
``np.inf`` + a censored flag so the caller can treat "not reached by a_ref" as a
survival observation rather than missing-at-random. Do NOT normalize by M_peak
(that changes the target from the finite reference object to the historical max).
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

A_REF = 100.0

#: the finite-filtered, ascending history arrays the estimators work on
_FloatArray: TypeAlias = NDArray[np.float64]
#: one event epoch with its right-censoring flag. Private: the observed/left/
#: right-censoring input boundary is designed in a later package, and this
#: two-tuple must not pre-empt it
_CensoredEpoch: TypeAlias = tuple[float, bool]


# --------------------------- low-level helpers ---------------------------- #
def _finite_sorted(a: ArrayLike,
                   y: ArrayLike) -> tuple[_FloatArray, _FloatArray]:
    a = np.asarray(a, float)
    y = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(y)
    aa, yy = a[m], y[m]
    o = np.argsort(aa)
    return aa[o], yy[o]


def _smooth(y: _FloatArray, window: int = 1) -> _FloatArray:
    """Edge-aware centered moving average over ``window`` points.

    Divides by the local in-window count (not the full window), so endpoints are
    averaged only with the neighbours that exist rather than depressed by implicit
    zero-padding. This matters because completion/freeze are judged at a_ref, the
    last point -- zero-padding there would fabricate spurious right-censoring.
    """
    if window <= 1:
        return y
    k = np.ones(window)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


# ------------------------------ estimators -------------------------------- #
def first_attainment(a: ArrayLike, psi: ArrayLike, x: float) -> float:
    """a_{x,first}: first epoch the cumulative-max envelope of Psi reaches ``x``.

    Returns NaN if <2 finite points or x never reached; the envelope makes the
    crossing single-valued even when Psi dips after a peak. This is the
    well-behaved FORMATION clock (and == stored a_half for x=0.5). It is NOT a
    completion clock for x near 1 (overshoot makes first attainment early).
    """
    aa, pp = _finite_sorted(a, psi)
    if aa.size < 2 or not (np.isfinite(x) and x > 0):
        return np.nan
    env = np.maximum.accumulate(pp)
    if env[0] >= x:
        return float(aa[0])
    if env[-1] < x:
        return np.nan
    j = int(np.searchsorted(env, x, side="left"))
    a0, a1, e0, e1 = aa[j - 1], aa[j], env[j - 1], env[j]
    if e1 <= e0:
        return float(a1)
    t = (np.log10(x) - np.log10(e0)) / (np.log10(e1) - np.log10(e0))
    return float(10.0 ** (np.log10(a0) + t * (np.log10(a1) - np.log10(a0))))


def persistent_completion(a: ArrayLike, psi: ArrayLike, eps: float = 0.05,
                          smooth_window: int = 1,
                          a_ref: float = A_REF) -> _CensoredEpoch:
    """a_comp(eps): earliest epoch after which Psi stays within the completion band.

    Completion = the (optionally smoothed) history has risen to within ``eps`` of
    the finite reference and never drops below ``1-eps`` again before a_ref. This
    is the SUSTAINED counterpart to first attainment; for an overshoot-then-settle
    history the two differ by a decade in a.

    Returns ``(a_comp, censored)``. ``censored=True`` (and a_comp=inf) when the
    last in-range point is still below ``1-eps`` (not completed by a_ref) -> treat
    as a right-censored survival observation, not a number.
    """
    aa, pp = _finite_sorted(a, psi)
    aa, pp = aa[aa <= a_ref], pp[aa <= a_ref]
    if aa.size < 2:
        return np.nan, False
    pp = _smooth(pp, smooth_window)
    below = pp < (1.0 - eps)
    if not below.any():
        return float(aa[0]), False           # always complete in range
    last_below = int(np.max(np.flatnonzero(below)))
    if last_below == aa.size - 1:
        return np.inf, True                  # still below at a_ref -> censored
    return float(aa[last_below + 1]), False


def mah_freeze(a: ArrayLike, gamma: ArrayLike, gamma_tol: float = 0.1,
               smooth_window: int = 3,
               a_ref: float = A_REF) -> _CensoredEpoch:
    """a_freeze(gamma_tol): earliest epoch after which |Gamma| stays below tol.

    ``gamma`` should be the SAME rate shown in the accretion figures
    (Gamma_tdyn = dlnM/dlna over one dynamical time), for consistency. Freeze =
    the smoothed |Gamma| drops below ``gamma_tol`` and never rises above it again
    before a_ref.

    Returns ``(a_freeze, censored)``; ``censored=True`` (a_freeze=inf) if still
    accreting (|Gamma|>=tol) at a_ref. Scan ``gamma_tol`` rather than fixing one
    tiny value (the legacy 1e-4 is below the per-halo noise floor and was applied
    through a broken inverse interpolation).
    """
    aa, gg = _finite_sorted(a, np.abs(np.asarray(gamma, float)))
    aa, gg = aa[aa <= a_ref], gg[aa <= a_ref]
    if aa.size < 2:
        return np.nan, False
    gg = _smooth(gg, smooth_window)
    above = gg >= gamma_tol
    if not above.any():
        return float(aa[0]), False
    last_above = int(np.max(np.flatnonzero(above)))
    if last_above == aa.size - 1:
        return np.inf, True
    return float(aa[last_above + 1]), False


def overshoot_max(a: ArrayLike, psi: ArrayLike,
                  a_ref: float = A_REF) -> float:
    """Diagnostic: max Psi over the history (the overshoot indicator, M_peak/M_ref)."""
    aa, pp = _finite_sorted(a, psi)
    pp = pp[aa <= a_ref]
    return float(np.max(pp)) if pp.size else np.nan
