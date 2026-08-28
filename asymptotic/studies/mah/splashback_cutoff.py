"""Population-level cutoff for the unresolved leading hook in M_sp MAHs.

Canonical, path-free location for the resolution-cutoff estimator first prototyped
under ``notebooks/chapter5_assembly/splashback_mah_hook_cutoff.py``. Pure NumPy;
import from any postprocessing driver or study:

    from AsymptoticAssembly.asymptotic.studies.mah.splashback_cutoff import (
        compute_bin_cutoffs, MIN_PARTICLES,
    )

The cutoff is intentionally simple and is *not* a per-halo shape classifier.
For each final-M_sp bin, find the first epoch after which

* at least a chosen fraction of the bin has finite positive M_sp at every later
  epoch, and
* at least ``MIN_FINITE`` halos remain at every later epoch.

The adopted cutoff is the later of that persistent-completeness epoch and the
exit from the last large downward step in the median before completeness. It can
therefore be looked up by final mass and applied as a uniform prefix mask to every
halo in that box/mass bin (see ``splashback_prefix_mask``).
"""
from __future__ import annotations

import numpy as np

BIN_WIDTH = 0.10
MIN_FINITE = 30
COMPLETENESS_LEVELS = (0.50, 0.80)
LARGE_DROP = 0.05          # absolute change in Psi between adjacent saved epochs
HOOK_SEARCH_MAX = 0.90     # search the hook while completeness < this (else it's real evolution)
MIN_PARTICLES = 1000       # splashback resolution floor (particles at the final M_sp)


def first_persistent(mask: np.ndarray) -> int | None:
    """First index from which a Boolean condition remains true."""
    suffix = np.logical_and.accumulate(mask[::-1])[::-1]
    where = np.flatnonzero(suffix)
    return int(where[0]) if where.size else None


def compute_bin_cutoffs(
        resolved_bin: np.ndarray,
        psi_bin: np.ndarray,
        a: np.ndarray,
        *,
        levels: tuple[float, ...] = COMPLETENESS_LEVELS,
        min_finite: int = MIN_FINITE,
        large_drop: float = LARGE_DROP,
        hook_search_max: float = HOOK_SEARCH_MAX,
    ) -> dict[str, object]:
    """Adopted prefix-mask cutoffs for a single final-M_sp bin.

    Single source of truth for the estimator, shared by the point-estimate driver
    and the bootstrap / bin-width sensitivity study. Inputs are the per-halo arrays
    for ONE bin, shape ``[n_halos, n_snaps]``:

    * ``resolved_bin`` — finite positive M_sp mask;
    * ``psi_bin``      — Psi_sp = M_sp(a) / M_sp(final) (only read where resolved).

    Returns the persistent-completeness epoch, the last-large-drop (hook) exit, and
    the adopted cutoff ``a_adopted_{level}`` = max(persistent-completeness, hook_exit)
    for each completeness level.
    """
    n_total = resolved_bin.shape[0]
    finite_count = resolved_bin.sum(axis=0)
    completeness = finite_count / n_total
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(resolved_bin, psi_bin, np.nan), axis=0)

    result: dict[str, object] = {}
    adopted_idx = []
    for level in levels:
        idx = first_persistent((completeness >= level) & (finite_count >= min_finite))
        result[f"a_persistent_{int(level * 100)}pct"] = a[idx] if idx is not None else np.nan
        adopted_idx.append(idx)

    idx_settle = first_persistent(completeness >= hook_search_max)
    hi = idx_settle if idx_settle is not None else len(median) - 1
    hook_exit = None
    if hi > 0:
        diffs = np.diff(median[: hi + 1])
        drops = np.flatnonzero(np.isfinite(diffs) & (diffs < -large_drop))
        if drops.size:
            hook_exit = int(drops[-1] + 1)
    result["a_last_large_drop_exit"] = a[hook_exit] if hook_exit is not None else np.nan

    for level, idx in zip(levels, adopted_idx):
        cut = idx
        shape_moved = False
        if cut is not None and hook_exit is not None and hook_exit > cut:
            cut, shape_moved = hook_exit, True   # residual hook past completeness
        result[f"a_adopted_{int(level * 100)}pct"] = a[cut] if cut is not None else np.nan
        result[f"shape_moved_{int(level * 100)}pct"] = int(shape_moved)
        result[f"finite_count_at_adopted_{int(level * 100)}pct"] = (
            int(finite_count[cut]) if cut is not None else 0
        )
    return result
