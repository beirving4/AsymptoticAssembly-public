"""Apply the median-derived splashback prefix cutoff as per-halo left-censoring.

Canonical, path-free location for the prefix-mask / left-censoring machinery first
prototyped under ``notebooks/chapter5_assembly/splashback_prefix_mask.py``. Import
from any postprocessing driver or study:

    from AsymptoticAssembly.asymptotic.studies.mah.splashback_prefix_mask import (
        load_cutoffs, cutoff_for, apply_prefix_mask, first_attainment_censored,
    )

The early unresolved M_sp "hook" is RESOLUTION CENSORING, not physical mass
evolution. This module turns the population-level cutoff table (produced by the
``compute_bin_cutoffs`` estimator, e.g. ``save_splashback_mah_cutoffs.py``) into
per-halo operations:

  * ``cutoff_for``        — look up a_sp,start by box + final-M_sp bin;
  * ``apply_prefix_mask`` — set M_sp(a < a_cut) -> NaN (NO interpolation, cumulative
    maximum, or synthetic rising branch; genuine *later* mass loss is preserved);
  * ``first_attainment_censored`` — classify a first-attainment milestone as
    OBSERVED (crossing in the resolved domain) or LEFT_CENSORED (threshold already
    met at the first resolved epoch, so the crossing precedes a_first,resolved --
    the censoring bound, which is >= a_cut, NOT a_cut itself). It returns
    (NaN, LEFT_CENSORED); the milestone is never assigned to the cutoff epoch.

Convention: histories on the global a-grid, Psi=M_sp(a)/M_sp(a_ref=100), finite
a_ref. Splashback is the fiducial boundary; masking never fabricates mass.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .milestone_estimators import first_attainment
from .splashback_cutoff import BIN_WIDTH

# explicit milestone-observation statuses (no overloaded (NaN, False))
OBSERVED = "observed"              # crossing measured in the resolved domain
LEFT_CENSORED = "left_censored"    # crossing precedes a_first,resolved (bound >= a_cut)
RIGHT_CENSORED = "right_censored"  # resolved but never reaches x by a_ref
INVALID = "invalid"               # outside the accepted cutoff domain / no resolved data


def load_cutoffs(csv_path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """{box: {centers, a50, a80}} sorted by final-M_sp bin center (log10 Msun/h).

    ``csv_path`` is the cutoff table written by the ``compute_bin_cutoffs`` driver
    (``splashback_mah_hook_cutoffs.csv``); it is required (this module makes no
    assumption about where that file lives).
    """
    rows: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with open(csv_path, newline="") as stream:
        for r in csv.DictReader(stream):
            rows[int(float(r["box_mpc_h"]))].append((
                float(r["log10_final_msp_msun_h"]),
                float(r["a_adopted_50pct"]), float(r["a_adopted_80pct"]),
            ))
    out: dict[int, dict[str, np.ndarray]] = {}
    for box, lst in rows.items():
        lst.sort()
        c, a50, a80 = (np.array(x) for x in zip(*lst))
        out[box] = {"centers": c, "a50": a50, "a80": a80}
    return out


def cutoff_for(
        box: int, log_msp_final: float, cutoffs: dict, level: int = 50,
        floor: float | None = None,
    ) -> float:
    """a_sp,start for the 0.1-dex bin containing ``log_msp_final``.

    Returns NaN (the halo is OUTSIDE THE ACCEPTED CUTOFF DOMAIN, hence INVALID for
    splashback MAH) when any of:

    - the box is absent;
    - ``floor`` is given and ``log_msp_final < floor`` — enforces the per-halo
      1000-particle resolution floor. This is REQUIRED: without it, the ±half-bin
      lookup tolerance lets a halo just below the floor borrow the lowest accepted
      bin's cutoff (the bin straddles the floor), silently admitting below-floor halos;
    - the mass is in no accepted bin under HALF-OPEN membership ``[c-w/2, c+w/2)``.
    """
    if box not in cutoffs:
        return np.nan
    if floor is not None and log_msp_final < floor:
        return np.nan
    c = cutoffs[box]["centers"]
    col = cutoffs[box]["a50" if level == 50 else "a80"]
    j = int(np.argmin(np.abs(c - log_msp_final)))
    # half-open [center - w/2, center + w/2): a point on the upper edge belongs to the
    # next bin, so no halo is double-assigned.
    if not (c[j] - BIN_WIDTH / 2 - 1e-9 <= log_msp_final < c[j] + BIN_WIDTH / 2):
        return np.nan
    return float(col[j])


def apply_prefix_mask(a: np.ndarray, m_sp: np.ndarray, a_cut: float) -> np.ndarray:
    """Copy of ``m_sp`` with the unresolved prefix removed.

    For a finite ``a_cut``, sets M_sp(a < a_cut) -> NaN (no interpolation /
    cumulative maximum / synthetic branch; the resolved suffix incl. genuine later
    mass loss is untouched). For a **NaN** ``a_cut`` the halo is below the
    resolution floor / unusable, so the ENTIRE history is masked to NaN — this
    prevents below-floor halos from leaking back into downstream work.
    """
    out = np.asarray(m_sp, dtype=float).copy()
    if not np.isfinite(a_cut):
        out[:] = np.nan
        return out
    out[np.asarray(a, dtype=float) < a_cut] = np.nan
    return out


def first_attainment_censored(a, psi_raw, x, a_cut) -> tuple[float, str]:
    """(value, status) for the first epoch Psi reaches ``x`` in the resolved domain.

    ``status`` is one of ``OBSERVED`` / ``LEFT_CENSORED`` / ``RIGHT_CENSORED`` /
    ``INVALID`` (module constants):

    - ``INVALID`` — ``a_cut`` is NaN (outside the accepted cutoff domain: box absent,
      below the 1000-particle floor, or in no accepted bin) or no resolved data; value NaN.
    - ``LEFT_CENSORED`` — the threshold is already met at the first resolved epoch,
      so the crossing precedes a_first,resolved (the censoring bound, which is
      >= a_cut, not a_cut itself) and is unobservable; value NaN (NOT moved to a_cut).
    - ``RIGHT_CENSORED`` — resolved but ``x`` is never reached by a_ref; value NaN.
    - ``OBSERVED`` — crossing measured in the resolved domain; value = a_x.
    """
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a_cut):
        return np.nan, INVALID
    psi_m = apply_prefix_mask(a, psi_raw, a_cut)
    resolved = np.isfinite(psi_m)
    if resolved.sum() < 1:
        return np.nan, INVALID
    # The cumulative-max envelope starts at the first resolved epoch; if it is
    # already >= x there, the true crossing precedes a_cut. Test this explicitly
    # (before first_attainment, which needs >=2 points) so a lone resolved point
    # already above threshold is LEFT-, not RIGHT-, censored.
    a_res = a[resolved]
    psi_first = float(psi_m[resolved][np.argmin(a_res)])
    if psi_first >= x:
        return np.nan, LEFT_CENSORED              # crossing precedes a_first,resolved (>= a_cut)
    if resolved.sum() < 2:
        return np.nan, RIGHT_CENSORED             # one point, below x -> crossing not observed
    a_x = first_attainment(a, psi_m, x)
    if not np.isfinite(a_x):
        return np.nan, RIGHT_CENSORED             # never reaches x by a_ref
    return float(a_x), OBSERVED


def milestone_with_bound(a, psi_i, x, a_cut) -> tuple[float, str, float]:
    """(a_x_or_nan, status, bound) for a censored fractional-formation milestone.

    ``bound`` is the survival bound for later censoring-aware estimators: a_x if
    observed; a_first,resolved if left-censored (the true crossing is < bound);
    a_last,resolved if right-censored (crossing > bound / never by a_ref); NaN if
    invalid. Wraps :func:`first_attainment_censored`.
    """
    a_x, status = first_attainment_censored(a, psi_i, x, a_cut)
    if status == OBSERVED:
        return a_x, status, a_x
    if status == INVALID:
        return np.nan, status, np.nan
    psi_m = apply_prefix_mask(a, psi_i, a_cut)
    a_res = np.asarray(a, dtype=float)[np.isfinite(psi_m)]
    if status == LEFT_CENSORED:
        return np.nan, status, float(a_res.min())
    return np.nan, status, float(a_res.max())     # RIGHT_CENSORED
