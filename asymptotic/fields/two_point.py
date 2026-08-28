from __future__ import annotations

import os
import treecorr 
import numpy as np, pdb

from tqdm.auto import tqdm
from typing import Iterator, Callable, Any
from Corrfunc.theory.DD import DD
from Corrfunc.theory.xi import xi as corrfunc_xi
from Corrfunc.utils import convert_3d_counts_to_cf
 
# Optional Halotools benchmark
try:
    from halotools.mock_observables import tpcf_jackknife as ht_tpcf_jackknife
    _HAS_HALOTOOLS = True
except Exception:
    _HAS_HALOTOOLS = False

from ..utils.threads import (
    _omp_nthreads,
    _ensure_omp_binding,
    assert_corrfunc_threads_active,
    _assert_threads_match_env,
)
from ..utils.jackknife import (
    jackknife_error,
    jackknife_resampling,
    jackknife_estimate,
    jackknife_correlation_matrix,
    get_jackknife_subbox_samples,
)

DEFAULT_NUM_TARGET_PAIRS = 100_000
DEFAULT_TOLERANCE = 1e-12
DEFAULT_RANDOM_MULTIPLIER = 10
DEFAULT_CIC_SUBSAMPLE_SIZE = 256 * 256 * 256

def _safe_corrfunc_run(
        nthreads: int,
        callsite: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
    """Run a Corrfunc call with optional thread-usage probe.

    If env var `AA_PROBE_OMP` is truthy, this will:
      1) `_ensure_omp_binding(nthreads)` (only sets defaults if missing)
      2) run `fn(*args, **kwargs)` inside `assert_corrfunc_threads_active(...)`,
         which verifies OMP workers actually spawn (Linux: samples /proc).
      3) If CPU affinity causes nthreads to be adjusted, updates args/kwargs accordingly

    Otherwise it just calls `fn` directly with zero overhead.
    """
    if os.environ.get("AA_PROBE_OMP", "0") in ("", "0"):
        return fn(*args, **kwargs)

    _ensure_omp_binding(nthreads)

    outbox: dict[str, Any] = {}
    adjusted_nthreads = {"value": nthreads}
    
    def _work() -> None:
        # Re-read nthreads in case assert_corrfunc_threads_active adjusted it
        actual_nthreads = _omp_nthreads()
        if actual_nthreads != nthreads:
            adjusted_nthreads["value"] = actual_nthreads
            # Update args if nthreads is in args (usually args[1])
            new_args = list(args)
            if len(new_args) > 1 and isinstance(new_args[1], int):
                new_args[1] = actual_nthreads
            outbox["res"] = fn(*new_args, **kwargs)
        else:
            outbox["res"] = fn(*args, **kwargs)

    assert_corrfunc_threads_active(nthreads, _work, callsite=callsite)
    return outbox.get("res")

def _validate_tpcf_inputs(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
    ) -> None:
    """Validate inputs for auto- and cross-correlation TPCF computations.

    Ensures:
    - coords are (N,3) (and (M,3) if provided)
    - weights match their catalog lengths when provided
    - basic numeric parameter sanity
    """
    # coords1
    if comoving_coordinates.ndim != 2 or comoving_coordinates.shape[1] != 3:
        raise ValueError("comoving_coordinates must be an (N, 3) array.")
    if not np.all(np.isfinite(comoving_coordinates)):
        raise ValueError("comoving_coordinates contains non-finite values.")

    # coords2 (optional)
    if other_coordinates is not None:
        if other_coordinates.ndim != 2 or other_coordinates.shape[1] != 3:
            raise ValueError("other_coordinates must be an (M, 3) array if provided.")
        if not np.all(np.isfinite(other_coordinates)):
            raise ValueError("other_coordinates contains non-finite values.")

    # scalar params
    if rmin <= 0 or rmax <= rmin:
        raise ValueError("Require 0 < rmin < rmax.")
    if nbins < 1:
        raise ValueError("nbins must be ≥ 1.")
    if not np.isfinite(boxsize) or boxsize <= 0:
        raise ValueError("boxsize must be a positive finite float.")

    # weights for catalog 1
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != comoving_coordinates.shape[0]:
            raise ValueError("weights must have shape (N,) matching comoving_coordinates.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values.")

    # weights for catalog 2 (only valid when other_coordinates provided)
    if weights2 is not None and other_coordinates is None:
        raise ValueError("weights2 was provided but other_coordinates is None.")
    if weights2 is not None:
        if weights2.ndim != 1 or weights2.shape[0] != other_coordinates.shape[0]:
            raise ValueError("weights2 must have shape (M,) matching other_coordinates.")
        if not np.all(np.isfinite(weights2)):
            raise ValueError("weights2 contains non-finite values.")


def _get_validated_tpcf_inputs(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert to float arrays and run validation, returning (coords1, coords2)."""
    coords1 = np.asarray(comoving_coordinates, dtype=float)
    coords2 = None if other_coordinates is None else np.asarray(other_coordinates, dtype=float)

    # Run validation on the raw inputs to keep error messages aligned with the public API
    _validate_tpcf_inputs(
        comoving_coordinates=coords1,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        boxsize=boxsize,
        weights=weights,
        other_coordinates=coords2,
        weights2=weights2,
    )

    return coords1, coords2


def _get_radial_bins(rmin: float, rmax: float, nbins: int) -> tuple[np.ndarray, ...]:
    """Get log-spaced radial bins between rmin and rmax."""
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r_centers = np.sqrt(rbins[:-1] * rbins[1:])
    return rbins, r_centers


def _get_randoms(box_size: float, N: int, lower_bound: float = 0.0) -> np.ndarray:
    """Get random points within a cubic box of given size."""
    rng = np.random.default_rng()
    N_int = int(max(1, int(np.ceil(float(N)))))
    return rng.uniform(lower_bound, box_size, size=(N_int, 3)).T  # (rx, ry, rz)


# Generalized weights parser for auto/cross
def _parse_weights(
        weights1: np.ndarray | None,
        weights2: np.ndarray | None,
        N1: int,
        N2: int,
        autocorr: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    """Normalize/validate weights for (auto|cross) correlation.
    Returns (w1, w2, rw1, rw2, weight_type) where w* are data weights and rw* are
    the corresponding random-catalog weights (ones) when weighted runs are requested.
    """
    w1 = None if weights1 is None else np.asarray(weights1, dtype=float)
    w2 = None if weights2 is None else np.asarray(weights2, dtype=float)

    if w1 is not None and w1.shape != (N1,):
        raise ValueError("weights1 must have shape (N1,).")
    if not autocorr and (w2 is not None) and (w2.shape != (N2,)):
        raise ValueError("weights2 must have shape (N2,) for cross-correlation.")

    # If any weights supplied, use pair-product scheme
    weight_type = (
        "pair_product" 
        if (w1 is not None or (w2 is not None and not autocorr)) else 
        None
    )

    # Random weights are all ones when weighted; None otherwise
    rw1 = (np.ones(N1, dtype=float) if weight_type is not None else None)
    rw2 = (np.ones((N1 if autocorr else N2), dtype=float) if weight_type is not None else None)

    return w1, (w1 if (autocorr and w2 is None) else w2), rw1, rw2, weight_type


def _effective_counts(rec: np.ndarray) -> np.ndarray:
    """Return effective pair counts per bin.
    If 'weightavg' exists and is nonzero anywhere, use npairs * weightavg.
    If 'weightavg' exists but is all zeros (Corrfunc w/o weights), return npairs.
    """
    npairs = rec["npairs"].astype(float)
    if rec.dtype.names and ("weightavg" in rec.dtype.names):
        wavg = rec["weightavg"].astype(float)
        return npairs if np.all(wavg == 0.0) else npairs * wavg
    return npairs



def _subsample_tracers(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        weights1: np.ndarray | None,
        weights2: np.ndarray | None,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        num_target_pairs: int,
        use_recommended_subsample: bool,
        use_cic_subsample: bool,
        cic_subsample_size: int,
        subsample_secondary: bool,
        subsample_rng_seed: int | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, int, int]:
    """
    Optionally subsample one or both catalogs to target a pair-count budget.
    Returns updated (coords1, coords2, weights1, weights2, N1, N2).
    """

    if not use_recommended_subsample:
        return coords1, coords2, weights1, weights2

    rng = np.random.default_rng(subsample_rng_seed)

    original_size_1 = coords1.shape[0]

    # Subsample primary catalog
    n_keep1 = get_tpcf_particle_subsample_size(
        num_particles_in_sim=coords1.shape[0], 
        box_size=boxsize, 
        min_radius=rmin, 
        max_radius=rmax, 
        num_radial_bins=nbins, 
        num_target_pairs=num_target_pairs
    )
    if n_keep1 < coords1.shape[0]:
        idxs1 = rng.choice(coords1.shape[0], size=n_keep1, replace=False)
        coords1 = coords1[idxs1]
        if weights1 is not None:
            weights1 = np.asarray(weights1, dtype=float)[idxs1]

    pct_kept_1 = 100.0 * float(n_keep1) / float(original_size_1)
    print(
        f"Subsampled primary catalog to {n_keep1:,} particles."
        f" ({pct_kept_1:.3f}% of original)"
    )

    # Subsample secondary catalog if requested
    if (coords2 is not None) and subsample_secondary:
        original_size_2 = coords2.shape[0]
        n_keep2 = get_tpcf_particle_subsample_size(
            num_particles_in_sim=coords2.shape[0], 
            box_size=boxsize, 
            min_radius=rmin, 
            max_radius=rmax, 
            num_radial_bins=nbins, 
            num_target_pairs=num_target_pairs
        )
        if n_keep2 < coords2.shape[0]:
            idxs2 = rng.choice(coords2.shape[0], size=n_keep2, replace=False)
            coords2 = coords2[idxs2]
            if weights2 is not None:
                weights2 = np.asarray(weights2, dtype=float)[idxs2]

        pct_kept_2 = 100.0 * float(n_keep2) / float(original_size_2)
        print(
            f"Subsampled secondary catalog to {n_keep2:,} particles."
            f" ({pct_kept_2:.3f}% of original)"
        )

    return coords1, coords2, weights1, weights2


def _get_tpcf_inputs(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        use_recommended_subsample: bool = False,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
        use_cic_subsample: bool = False,
        cic_subsample_size: int = DEFAULT_CIC_SUBSAMPLE_SIZE,
        subsample_secondary: bool = False,
        subsample_rng_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:

    # Validate and coerce inputs
    coords1, coords2 = _get_validated_tpcf_inputs(
        comoving_coordinates=comoving_coordinates,
        rmin=rmin, rmax=rmax, nbins=nbins, boxsize=boxsize,
        weights=weights, other_coordinates=other_coordinates, weights2=weights2,
    )

    if rmax > 0.5 * boxsize:
        print("[warn] rmax > boxsize/2; RR bins at the largest scales may be empty. Consider reducing rmax.")

    # ---- Subsampling step (if enabled) ----
    coords1, coords2, weights, weights2 = _subsample_tracers(
        coords1=coords1,
        coords2=coords2,
        weights1=weights,
        weights2=weights2,
        boxsize=boxsize,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        num_target_pairs=num_target_pairs,
        use_recommended_subsample=use_recommended_subsample,
        use_cic_subsample=use_cic_subsample,
        cic_subsample_size=cic_subsample_size,
        subsample_secondary=subsample_secondary,
        subsample_rng_seed=subsample_rng_seed,
    )

    return coords1, coords2, weights, weights2


# Inspect this to make sure it's returning the coords in the box!
def get_fold_mask(
        coords: np.ndarray,
        bounds: tuple[tuple[float, float], ...],
        include: bool = False,
    ) -> np.ndarray:
    """
    Return a boolean mask for selecting points either outside or inside the given bounds.
        Parameters
    ----------
    coords : (N, 3) array
    bounds : ((xlo,xhi),(ylo,yhi),(zlo,zhi))
    include : bool
        If False (default), return mask for points OUTSIDE the bounds (leave-one-out).
        If True, return mask for points INSIDE the bounds (single-subbox selection).

    Returns
    -------
    mask : (N,) boolean
    """
    b = np.asarray(bounds)
    inside = np.all((coords >= b[:, 0]) & (coords < b[:, 1]), axis=1)
    return inside if include else (~inside)


def _estimate_occupied_fraction( # Turn this off for right now.. histogramdd is slow
        coords: np.ndarray | None,
        boxsize: float,
        rmax: float,
        max_cells_per_axis: int = 64,
    ) -> float:
    """Fraction of (Corrfunc-like) link-cells that are occupied by data.
    We use a coarse 3D histogram with cell size ≈ rmax.
    """
    if coords is None or coords.size == 0:
        return 0.0
    cell_size = max(rmax, 1e-12)
    nb = max(2, min(max_cells_per_axis, int(np.ceil(boxsize / cell_size))))
    H, _ = np.histogramdd(
        coords, 
        bins=(nb, nb, nb),
        range=((0.0, boxsize), (0.0, boxsize), (0.0, boxsize))
    )
    occ = float(np.count_nonzero(H))
    tot = float(H.size)
    return 0.0 if tot == 0.0 else max(0.0, min(1.0, occ / tot))


def _compute_adaptive_random_count(
        n_data: int,
        coords: np.ndarray | None,
        boxsize: float,
        rmax: float,
        base_multiplier: int,
        adaptive: bool,
        max_randoms_per_fold: int | None,
        min_randoms_per_fold: int,
    ) -> int:
    """Pick per-fold randoms. If `adaptive` is True, scale down when the field
    is very uniform (large occupied-fraction) since RR/DR converge faster there.
    Scale is clamped to [0.3, 1.0] so you keep enough randoms for stability.
    """
    n = max(base_multiplier * max(1, n_data), 1)
    if adaptive and (coords is not None) and coords.size:
        f_occ = _estimate_occupied_fraction(coords, boxsize, rmax)
        # Decrease when f_occ→1: scale in [0.3, 1.0]
        scale = 1.0 - 0.7 * f_occ
        n = int(np.ceil(n * max(0.3, min(1.0, scale))))
    if max_randoms_per_fold is not None:
        n = min(n, int(max_randoms_per_fold))
    return max(n, min_randoms_per_fold)

def _randoms_inside_included(
        included_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        n_randoms: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
    """Generate uniform randoms inside the specified rectangular sub-box."""
    if n_randoms <= 0:
        return np.empty((0, 3), dtype=float)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = included_bounds
    N = int(max(1, int(np.ceil(float(n_randoms)))))
    return rng.uniform((xlo, ylo, zlo), (xhi, yhi, zhi), size=(N, 3))

def get_fold_arrays(
        coords: np.ndarray | None,
        weights: np.ndarray | None,
        boxsize: float,
        rmax: float,
        bounds: tuple[tuple[float, float], ...],
        random_multiplier: int = DEFAULT_RANDOM_MULTIPLIER,
        rng_seed: int | None = None,
        adaptive_randoms: bool = False,
        max_randoms_per_fold: int | None = None,
        min_randoms_per_fold: int = 1024,
        include_mode: str = "outside"
    ) -> tuple[np.ndarray | None, ...]:

    if (coords is None) or (coords.size == 0):
        return (None, None, None)

    include = (include_mode.lower().strip() == "inside")
    mask = get_fold_mask(coords, bounds, include=include)

    if not mask.any():
        raise ValueError("All coordinates are excluded from the fold.")

    coords_fold = coords[mask]
    weights_fold = weights[mask] if (weights is not None) else None

    num_random_fold = _compute_adaptive_random_count(
        n_data=coords_fold.shape[0],
        coords=coords_fold,
        boxsize=boxsize,
        rmax=rmax,
        base_multiplier=random_multiplier,
        adaptive=adaptive_randoms,
        max_randoms_per_fold=max_randoms_per_fold,
        min_randoms_per_fold=min_randoms_per_fold,
    )
    
    rng = np.random.default_rng(rng_seed)
    if include:
        randoms_fold = _randoms_inside_included(
            included_bounds=bounds,
            n_randoms=num_random_fold,
            rng=rng,
        )
    else:
        randoms_fold = _randoms_outside_excluded(
            boxsize=boxsize,
            excluded_bounds=bounds,
            n_randoms=num_random_fold,
            rng=rng,
        )

    return (coords_fold, weights_fold, randoms_fold)


def _get_tpcf_fold_inputs(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        weights1: np.ndarray | None,
        weights2: np.ndarray | None,
        radial_bins: np.ndarray,
        box_size: float,
        bounds: tuple[tuple[float, float], ...],
        random_multiplier: int = DEFAULT_RANDOM_MULTIPLIER,
        rng_seed: int | None = None,
        eps: float = DEFAULT_TOLERANCE,
        adaptive_randoms: bool = False,
        max_randoms_per_fold: int | None = None,
        min_randoms_per_fold: int = 1024,
        include_mode: str = "outside"
    ) -> dict[str, int | np.ndarray | None]:

    r_max = radial_bins[-1]

    try:
        coords1_fold, weights1_fold, randoms_fold1 = (
            get_fold_arrays(
                coords=coords1,
                weights=weights1,
                boxsize=box_size,
                rmax=r_max,
                bounds=bounds,
                random_multiplier=random_multiplier,
                rng_seed=rng_seed,
                adaptive_randoms=adaptive_randoms,
                max_randoms_per_fold=max_randoms_per_fold,
                min_randoms_per_fold=min_randoms_per_fold,
                include_mode=include_mode
            )
        )

    except ValueError as ve:
        raise ValueError(f"Error preparing fold inputs 1: {ve}") from ve
    
    try:
        coords2_fold, weights2_fold, randoms_fold2 = (
            get_fold_arrays(
                coords=coords2,
                weights=weights2,
                boxsize=box_size,
                rmax=r_max,
                bounds=bounds,
                random_multiplier=random_multiplier,
                rng_seed=rng_seed,
                adaptive_randoms=adaptive_randoms,
                max_randoms_per_fold=max_randoms_per_fold,
                min_randoms_per_fold=min_randoms_per_fold,
                include_mode=include_mode    
            )
        )
    except ValueError as ve:
        raise ValueError(f"Error preparing fold inputs 2: {ve}") from ve


    return {
        "coords1": coords1_fold,
        "coords2": coords2_fold,
        "N1": coords1_fold.shape[0], 
        "N2": (
            coords2_fold.shape[0] 
            if (coords2_fold is not None) else 
            coords1_fold.shape[0]
        ),
        "nthreads": _omp_nthreads(),
        "rbins": radial_bins,
        "w1": weights1_fold,
        "w2": weights2_fold,
        "rw1": (
            np.ones(randoms_fold1.shape[0], dtype=float) 
            if (weights1_fold is not None) else 
            None
        ),
        "rw2": (
            np.ones(randoms_fold2.shape[0], dtype=float) 
            if (weights2_fold is not None) else None
        ),
        "boxsize": box_size,
        "weight_type": ("pair_product" if (weights1_fold is not None) else None),
        "eps": eps,
        "rx1": randoms_fold1[:, 0] if (randoms_fold1 is not None) else None,
        "ry1": randoms_fold1[:, 1] if (randoms_fold1 is not None) else None,
        "rz1": randoms_fold1[:, 2] if (randoms_fold1 is not None) else None,
        "rx2": randoms_fold2[:, 0] if (randoms_fold2 is not None) else None,
        "ry2": randoms_fold2[:, 1] if (randoms_fold2 is not None) else None,
        "rz2": randoms_fold2[:, 2] if (randoms_fold2 is not None) else None
    }

def get_xi_fold_arr(
        fold_inputs: dict[str, int | np.ndarray | None],
        use_natural_estimator: bool = False
    ) -> np.ndarray:

    """
    Compute xi for a single fold from pre-built inputs. The contents of
    `fold_inputs` may correspond to either:
      • leave-one-out (outside the bounds), or
      • single-subbox (inside the bounds),
    depending on how `_get_tpcf_fold_inputs` was called.
    """

    return (
        _natural_estimator(**fold_inputs)
        if use_natural_estimator else
        _landy_szalay_estimator(**fold_inputs)
    )


def convert_jk_folds_to_results(
        xi_folds: list[np.ndarray],
        r_centers: np.ndarray,
        return_folds: bool = False
    ) -> dict[str, np.ndarray]:

    if len(xi_folds) < 2:
        raise ValueError("Not enough valid jackknife folds to estimate errors (need >= 2).")

    xi_est = jackknife_estimate(xi_folds)
    xi_err = jackknife_error(xi_folds)

    jackknifed_result = {
        "estimate": np.column_stack([r_centers, xi_est]),
        "errors":   np.column_stack([r_centers, xi_err]),
        "correlation_matrix": jackknife_correlation_matrix(xi_folds, xi_err)
    }

    if return_folds:
        # xi_folds is a list of 1D arrays with length nbins (one per fold).
        # Stack as (n_folds, nbins) then transpose to (nbins, n_folds)
        if xi_folds:
            fold_matrix = np.vstack([np.asarray(f, dtype=float) for f in xi_folds]).T
        else:
            fold_matrix = np.empty((r_centers.size, 0), dtype=float)

        # Prepend r_centers as the first column → shape = (nbins, n_folds + 1)
        jackknifed_result["folds"] = np.column_stack([r_centers.astype(float), fold_matrix])

    return jackknifed_result

def _ensure_random_catalogs(
        coords2: np.ndarray | None,
        boxsize: float,
        N1: int,
        N2: int,
        rx1: np.ndarray | None,
        ry1: np.ndarray | None,
        rz1: np.ndarray | None,
        rx2: np.ndarray | None,
        ry2: np.ndarray | None,
        rz2: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None,
               np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Ensure random catalogs exist for catalog 1 (always) and catalog 2 (if cross).
    Returns possibly-updated (rx1, ry1, rz1, rx2, ry2, rz2).
    """
    if rx1 is None or ry1 is None or rz1 is None:
        rx1, ry1, rz1 = _get_randoms(boxsize, N1)
    if (coords2 is not None) and (rx2 is None or ry2 is None or rz2 is None):
        rx2, ry2, rz2 = _get_randoms(boxsize, N2)
    return rx1, ry1, rz1, rx2, ry2, rz2


def _analytic_rr_unique_pairs(
        rbins: np.ndarray,
        boxsize: float,
        NR1: int,
        NR2: int | None = None,
        template_rec: np.ndarray | None = None,
        rweights1: np.ndarray | None = None,
        rweights2: np.ndarray | None = None,
    ) -> np.ndarray:
    """
    Build a Corrfunc-like record array for RR using the analytic expectation
    in a 3D periodic cube. Supports weighted runs (pair_product) by filling
    `weightavg` consistently with Corrfunc's definition.

    Autocorr (unique): <RR_i>_geom = 0.5 * N(N-1) * f_i
                       <sum_pair_weights_i> = 0.5 * (W^2 - Q) * f_i
                       => weightavg_i = (W^2 - Q) / (N(N-1))

    Cross:             <RR_i>_geom = N1 * N2 * f_i
                       <sum_pair_weights_i> = W1 * W2 * f_i
                       => weightavg_i = (W1 * W2) / (N1 * N2)

    where f_i = V_shell_i / boxsize^3, W = sum(w), Q = sum(w^2).
    """
    rlo, rhi = rbins[:-1], rbins[1:]
    V_shell = (4.0 * np.pi / 3.0) * (rhi**3 - rlo**3)
    V_box = boxsize**3
    frac = V_shell / V_box

    # ---------- Geometric (unweighted) expected pair counts ----------
    if NR2 is None:  # auto
        N = NR1
        npairs = 0.5 * N * (N - 1) * frac
    else:        # cross
        N1, N2 = NR1, int(NR2)
        npairs = float(N1) * float(N2) * frac

    npairs = np.asarray(npairs, dtype=float)

    # ---------- Expected mean pair-weight (weightavg) ----------
    # Default: no weights supplied => unit weights
    if NR2 is None:
        N = NR1
        if rweights1 is None:
            W = float(N)
            Q = float(N)
        else:
            w = np.asarray(rweights1, dtype=float)
            if w.shape[0] != N:
                raise ValueError("rweights1 length must match NR1")
            W = float(np.sum(w))
            Q = float(np.sum(w*w))
        denom = max(N*(N-1), 1)  # avoid divide-by-zero
        weightavg_val = (W**2 - Q) / denom
    else:
        N1, N2 = NR1, int(NR2)
        if rweights1 is None:
            W1 = float(N1)
        else:
            w1 = np.asarray(rweights1, dtype=float)
            if w1.shape[0] != N1:
                raise ValueError("rweights1 length must match NR1")
            W1 = float(np.sum(w1))
        if rweights2 is None:
            W2 = float(N2)
        else:
            w2 = np.asarray(rweights2, dtype=float)
            if w2.shape[0] != N2:
                raise ValueError("rweights2 length must match NR2")
            W2 = float(np.sum(w2))
        denom = max(N1*N2, 1)
        weightavg_val = (W1 * W2) / denom

    # ---------- Build record ----------
    if template_rec is None:
        # Minimal dtype with npairs; add weightavg, too, to be thorough
        rec = np.zeros(npairs.shape[0], dtype=[('npairs', 'f8'), ('weightavg', 'f8')])
        rec['npairs'] = npairs
        rec['weightavg'] = weightavg_val
        return rec

    rec = template_rec.copy()
    # Zero all numeric fields to be safe
    for name, dt in rec.dtype.fields.items():
        if np.issubdtype(dt[0], np.number):
            rec[name].fill(0.0)
    rec['npairs'] = npairs
    # If the template has weightavg (Corrfunc sets it when weights are in play), fill it.
    if 'weightavg' in rec.dtype.names:
        rec['weightavg'] = weightavg_val
    return rec


def _get_natural_pair_counts(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        nthreads: int,
        rbins: np.ndarray,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        boxsize: float,
        weight_type: str | None,
        rx1: np.ndarray | None,
        ry1: np.ndarray | None,
        rz1: np.ndarray | None,
        rx2: np.ndarray | None,
        ry2: np.ndarray | None,
        rz2: np.ndarray | None,
        is_periodic: bool = True,  # unused for natural but kept for symmetry
    ) -> tuple[np.ndarray, np.ndarray, None, None]:
    """
    Pair counts for the Natural estimator.
    Returns (dd, rr, None, None).
    """
    x1, y1, z1 = coords1.T
    autocorr = coords2 is None

    _assert_threads_match_env(nthreads, "DD/RR (Natural)")

    if autocorr:
        dd = _safe_corrfunc_run(
            nthreads, "DD Natural (auto): DD", DD,
            1, nthreads, rbins, x1, y1, z1,
            weights1=w1, boxsize=boxsize, weight_type=weight_type,
        )

        # Analytic RR (auto), weighted or unweighted
        NR1 = rx1.shape[0] if rx1 is not None else x1.shape[0]
        rr = _analytic_rr_unique_pairs(
            rbins=rbins, boxsize=boxsize, NR1=NR1, NR2=None,
            template_rec=dd,
            rweights1=rw1  # None => unit weights
        )
        return dd, rr, None, None

    x2, y2, z2 = coords2.T  # type: ignore[union-attr]
    # cross
    dd = _safe_corrfunc_run(
        nthreads, "DD Natural (cross): DD", DD,
        0, nthreads, rbins, x1, y1, z1,
        X2=x2, Y2=y2, Z2=z2,
        weights1=w1, weights2=w2,
        boxsize=boxsize, weight_type=weight_type,
    )

    # Analytic RR (cross), weighted or unweighted
    NR1 = rx1.shape[0] if rx1 is not None else x1.shape[0]
    NR2 = rx2.shape[0] if rx2 is not None else x2.shape[0]
    rr = _analytic_rr_unique_pairs(
        rbins=rbins, boxsize=boxsize, NR1=NR1, NR2=NR2,
        template_rec=dd,
        rweights1=rw1, rweights2=rw2
    )
    return dd, rr, None, None


def _get_ls_pair_counts(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        nthreads: int,
        rbins: np.ndarray,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        boxsize: float,
        weight_type: str | None,
        rx1: np.ndarray | None,
        ry1: np.ndarray | None,
        rz1: np.ndarray | None,
        rx2: np.ndarray | None,
        ry2: np.ndarray | None,
        rz2: np.ndarray | None,
        is_periodic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pair counts for the Landy–Szalay estimator.
    Returns (dd, rr, dr12, dr21).
    """
    x1, y1, z1 = coords1.T
    autocorr = coords2 is None

    _assert_threads_match_env(
        nthreads, f"DD/DR/RR (Landy–Szalay, {'auto' if autocorr else 'cross'})"
    )

    if autocorr:
        dd = _safe_corrfunc_run(
            nthreads, "DD LS (auto): DD", DD,
            1, nthreads, rbins, x1, y1, z1, periodic=is_periodic,
            weights1=w1, boxsize=boxsize, weight_type=weight_type,
        )
        dr12 = _safe_corrfunc_run(
            nthreads, "DD LS (auto): DR", DD,
            0, nthreads, rbins, x1, y1, z1,
            X2=rx1, Y2=ry1, Z2=rz1,
            weights1=w1, weights2=rw1, periodic=is_periodic,
            boxsize=boxsize, weight_type=weight_type,
        )
        dr21 = dr12

        # Analytic RR (auto), weighted or unweighted
        NR1 = rx1.shape[0] if rx1 is not None else x1.shape[0]
        rr = _analytic_rr_unique_pairs(
            rbins=rbins, boxsize=boxsize, NR1=NR1, NR2=None,
            template_rec=dd,
            rweights1=rw1
        )
        return dd, rr, dr12, dr21

    x2, y2, z2 = coords2.T  # type: ignore[union-attr]
        # cross
    dd = _safe_corrfunc_run(
        nthreads, "DD LS (cross): DD", DD,
        0, nthreads, rbins, x1, y1, z1,
        X2=x2, Y2=y2, Z2=z2,
        weights1=w1, weights2=w2, periodic=is_periodic,
        boxsize=boxsize, weight_type=weight_type,
    )
    dr12 = _safe_corrfunc_run(
        nthreads, "DD LS (cross): DR12", DD,
        0, nthreads, rbins, x1, y1, z1,
        X2=rx2, Y2=ry2, Z2=rz2,
        weights1=w1, weights2=rw2, periodic=is_periodic,
        boxsize=boxsize, weight_type=weight_type,
    )
    dr21 = _safe_corrfunc_run(
        nthreads, "DD LS (cross): DR21", DD,
        0, nthreads, rbins, x2, y2, z2,
        X2=rx1, Y2=ry1, Z2=rz1,
        weights1=w2, weights2=rw1, periodic=is_periodic,
        boxsize=boxsize, weight_type=weight_type,
    )

    # Analytic RR (cross), weighted or unweighted
    NR1 = rx1.shape[0] if rx1 is not None else x1.shape[0]
    NR2 = rx2.shape[0] if rx2 is not None else x2.shape[0]
    rr = _analytic_rr_unique_pairs(
        rbins=rbins, boxsize=boxsize, NR1=NR1, NR2=NR2,
        template_rec=dd,
        rweights1=rw1, rweights2=rw2
    )
    return dd, rr, dr12, dr21


def get_pair_counts(
        estimator: str,
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        nthreads: int,
        rbins: np.ndarray,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        boxsize: float,
        weight_type: str | None,
        rx1: np.ndarray | None,
        ry1: np.ndarray | None,
        rz1: np.ndarray | None,
        rx2: np.ndarray | None,
        ry2: np.ndarray | None,
        rz2: np.ndarray | None,
        is_periodic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Dispatcher for pair counts.
    estimator ∈ {'natural','ls'}.
    Returns (dd, rr, dr12, dr21). For 'natural', dr12 and dr21 are None.
    """
    est = estimator.lower().strip()
    if est in {"natural", "nat", "n"}:
        return _get_natural_pair_counts(
            coords1=coords1, coords2=coords2, nthreads=nthreads, rbins=rbins,
            w1=w1, w2=w2, rw1=rw1, rw2=rw2, boxsize=boxsize, weight_type=weight_type,
            rx1=rx1, ry1=ry1, rz1=rz1, rx2=rx2, ry2=ry2, rz2=rz2, is_periodic=is_periodic
        )
    if est in {"ls", "landy-szalay", "landy_szalay", "landy", "szalay"}:
        return _get_ls_pair_counts(
            coords1=coords1, coords2=coords2, nthreads=nthreads, rbins=rbins,
            w1=w1, w2=w2, rw1=rw1, rw2=rw2, boxsize=boxsize, weight_type=weight_type,
            rx1=rx1, ry1=ry1, rz1=rz1, rx2=rx2, ry2=ry2, rz2=rz2, is_periodic=is_periodic
        )
    raise ValueError(f"Unknown estimator '{estimator}'. Expected 'natural' or 'ls'.")

def _get_natural_normalized_counts(
        dd: np.ndarray,
        rr: np.ndarray,
        autocorr: bool,
        N1: int,
        N2: int,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        rx1: np.ndarray | None,
        rx2: np.ndarray | None,
        eps: float,
        weight_type: str | None,
    ) -> tuple[np.ndarray, np.ndarray, None, None]:
    """
    Normalize Natural-estimator pair counts.
    Returns (DDn, RRn, None, None).
    """
    DD_counts = _effective_counts(dd)
    RR_counts = _effective_counts(rr)

    if weight_type is None:
        NR1 = rx1.shape[0] if rx1 is not None else N1
        if autocorr:
            norm_DD = N1 * (N1 - 1) / 2.0
            norm_RR = NR1 * (NR1 - 1) / 2.0
        else:
            norm_DD = N1 * N2
            NR2 = rx2.shape[0] if rx2 is not None else N2
            norm_RR = NR1 * NR2
    elif autocorr:
        wd_sum = float(np.sum(w1)) if w1 is not None else float(N1)
        wd_sqsum = float(np.sum(np.square(w1))) if w1 is not None else float(N1)
        if rw1 is not None:
            wr_sum = float(np.sum(rw1))
            wr_sqsum = float(np.sum(np.square(rw1)))
        else:
            NR1 = rx1.shape[0] if rx1 is not None else N1
            wr_sum = float(NR1)
            wr_sqsum = float(NR1)
        norm_DD = 0.5 * (wd_sum**2 - wd_sqsum)
        norm_RR = 0.5 * (wr_sum**2 - wr_sqsum)
    else:
        wd1_sum = float(np.sum(w1)) if w1 is not None else float(N1)
        wd2_sum = float(np.sum(w2)) if w2 is not None else float(N2)
        if rw1 is not None:
            wr1_sum = float(np.sum(rw1))
        else:
            NR1 = rx1.shape[0] if rx1 is not None else N1
            wr1_sum = float(NR1)
        if rw2 is not None:
            wr2_sum = float(np.sum(rw2))
        else:
            NR2 = rx2.shape[0] if rx2 is not None else N2
            wr2_sum = float(NR2)
        norm_DD = wd1_sum * wd2_sum
        norm_RR = wr1_sum * wr2_sum

    DDn = DD_counts / max(norm_DD, eps)
    RRn = RR_counts / max(norm_RR, eps)
    return DDn, RRn, None, None


def _get_ls_normalized_counts(
        dd: np.ndarray,
        rr: np.ndarray,
        dr12: np.ndarray,
        dr21: np.ndarray,
        autocorr: bool,
        N1: int,
        N2: int,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        rx1: np.ndarray | None,
        rx2: np.ndarray | None,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize LS-estimator pair counts (weighted path).
    Returns (DDn, RRn, DR12n, DR21n).
    """
    DD_counts = _effective_counts(dd)
    DR12_counts = _effective_counts(dr12)
    DR21_counts = _effective_counts(dr21)
    RR_counts = _effective_counts(rr)

    if autocorr:
        wd_sum = float(np.sum(w1)) if w1 is not None else float(N1)
        wd_sqsum = float(np.sum(np.square(w1))) if w1 is not None else float(N1)
        if rw1 is not None:
            wr_sum = float(np.sum(rw1))
            wr_sqsum = float(np.sum(np.square(rw1)))
        else:
            NR1 = rx1.shape[0] if rx1 is not None else N1
            wr_sum = float(NR1)
            wr_sqsum = float(NR1)
        norm_DD = 0.5 * (wd_sum**2 - wd_sqsum)
        norm_RR = 0.5 * (wr_sum**2 - wr_sqsum)
        norm_DR12 = wd_sum * wr_sum
        norm_DR21 = wd_sum * wr_sum
    else:
        wd1_sum = float(np.sum(w1)) if w1 is not None else float(N1)
        wd2_sum = float(np.sum(w2)) if w2 is not None else float(N2)
        if rw1 is not None:
            wr1_sum = float(np.sum(rw1))
        else:
            NR1 = rx1.shape[0] if rx1 is not None else N1
            wr1_sum = float(NR1)
        if rw2 is not None:
            wr2_sum = float(np.sum(rw2))
        else:
            NR2 = rx2.shape[0] if rx2 is not None else N2
            wr2_sum = float(NR2)
        norm_DD = wd1_sum * wd2_sum
        norm_RR = wr1_sum * wr2_sum
        norm_DR12 = wd1_sum * wr2_sum
        norm_DR21 = wd2_sum * wr1_sum

    DDn = DD_counts / max(norm_DD, eps)
    DR12n = DR12_counts / max(norm_DR12, eps)
    DR21n = DR21_counts / max(norm_DR21, eps)
    RRn = RR_counts / max(norm_RR, eps)
    
    return DDn, RRn, DR12n, DR21n


def get_normalized_counts(
        estimator: str,
        dd: np.ndarray,
        rr: np.ndarray,
        dr12: np.ndarray | None,
        dr21: np.ndarray | None,
        autocorr: bool,
        N1: int,
        N2: int,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        rx1: np.ndarray | None,
        rx2: np.ndarray | None,
        eps: float,
        weight_type: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Dispatcher that returns normalized counts for the chosen estimator.
    For 'natural': returns (DDn, RRn, None, None).
    For 'ls': returns (DDn, RRn, DR12n, DR21n).
    """
    est = estimator.lower().strip()
    if est in {"natural", "nat", "n"}:
        return _get_natural_normalized_counts(
            dd=dd, rr=rr, autocorr=autocorr, N1=N1, N2=N2,
            w1=w1, w2=w2, rw1=rw1, rw2=rw2, rx1=rx1, rx2=rx2,
            eps=eps, weight_type=weight_type
        )
    if est in {"ls", "landy-szalay", "landy_szalay", "landy", "szalay"}:
        if dr12 is None or dr21 is None:
            raise ValueError("LS normalization requires dr12 and dr21 counts.")
        return _get_ls_normalized_counts(
            dd=dd, rr=rr, dr12=dr12, dr21=dr21, autocorr=autocorr,
            N1=N1, N2=N2, w1=w1, w2=w2, rw1=rw1, rw2=rw2, rx1=rx1, rx2=rx2, eps=eps
        )
    raise ValueError(f"Unknown estimator '{estimator}'. Expected 'natural' or 'ls'.")

def _natural_estimator(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        N1: int,
        N2: int,
        nthreads: int,
        rbins: np.ndarray,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        boxsize: float,
        weight_type: str | None,
        eps: float = DEFAULT_TOLERANCE,
        # optional explicit random catalogs (when provided, they override auto-generation)
        rx1: np.ndarray | None = None,
        ry1: np.ndarray | None = None,
        rz1: np.ndarray | None = None,
        rx2: np.ndarray | None = None,
        ry2: np.ndarray | None = None,
        rz2: np.ndarray | None = None,
    ) -> np.ndarray:
    """Natural estimator for auto- or cross-correlation.
    Auto:  xi = DD/RR - 1 using auto DD and auto RR.
    Cross: xi = DD12/RR12 - 1 using cross DD and cross RR.
    """
    autocorr = coords2 is None

    # Ensure random catalogs exist (needed both for counts and later normalizations)
    rx1, ry1, rz1, rx2, ry2, rz2 = _ensure_random_catalogs(
        coords2=coords2,
        boxsize=boxsize,
        N1=N1,
        N2=N2,
        rx1=rx1, ry1=ry1, rz1=rz1,
        rx2=rx2, ry2=ry2, rz2=rz2,
    )

    dd, rr, _, _ = get_pair_counts(
        estimator="natural",
        coords1=coords1,
        coords2=coords2,
        nthreads=nthreads,
        rbins=rbins,
        w1=w1,
        w2=w2,
        rw1=rw1,
        rw2=rw2,
        boxsize=boxsize,
        weight_type=weight_type,
        rx1=rx1, ry1=ry1, rz1=rz1,
        rx2=rx2, ry2=ry2, rz2=rz2,
        is_periodic=True,
    )

    DDn, RRn, _, _ = get_normalized_counts(
        estimator="natural",
        dd=dd,
        rr=rr,
        dr12=None,
        dr21=None,
        autocorr=autocorr,
        N1=N1,
        N2=N2,
        w1=w1,
        w2=w2,
        rw1=rw1,
        rw2=rw2,
        rx1=rx1,
        rx2=rx2,
        eps=eps,
        weight_type=weight_type
    )

    return DDn / np.maximum(RRn, eps) - 1.0


def _landy_szalay_estimator(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        N1: int,
        N2: int,
        nthreads: int,
        rbins: np.ndarray,
        w1: np.ndarray | None,
        w2: np.ndarray | None,
        rw1: np.ndarray | None,
        rw2: np.ndarray | None,
        boxsize: float,
        weight_type: str | None,
        eps: float = DEFAULT_TOLERANCE,
        # optional explicit random catalogs
        rx1: np.ndarray | None = None,
        ry1: np.ndarray | None = None,
        rz1: np.ndarray | None = None,
        rx2: np.ndarray | None = None,
        ry2: np.ndarray | None = None,
        rz2: np.ndarray | None = None,
        is_periodic: bool = True,
    ) -> np.ndarray:
    """Landy–Szalay estimator for auto- or cross-correlation.
    Auto  (coords2=None): uses DD_auto, DR (data–random1), RR_auto.
    Cross (coords2 provided): uses DD_cross, DR12 (D1–R2), DR21 (D2–R1), RR_cross.
    """
    autocorr = coords2 is None

    # Ensure random catalogs exist before calling pair counts
    rx1, ry1, rz1, rx2, ry2, rz2 = _ensure_random_catalogs(
        coords2=coords2,
        boxsize=boxsize,
        N1=N1,
        N2=N2,
        rx1=rx1, ry1=ry1, rz1=rz1,
        rx2=rx2, ry2=ry2, rz2=rz2,
    )

    dd, rr, dr12, dr21 = get_pair_counts(
        estimator="ls",
        coords1=coords1,
        coords2=coords2,
        nthreads=nthreads,
        rbins=rbins,
        w1=w1,
        w2=w2,
        rw1=rw1,
        rw2=rw2,
        boxsize=boxsize,
        weight_type=weight_type,
        rx1=rx1, ry1=ry1, rz1=rz1,
        rx2=rx2, ry2=ry2, rz2=rz2,
        is_periodic=is_periodic,
    )

    # Unweighted: use Corrfunc helper when possible
    if weight_type is None:
        ND1 = N1
        ND2 = (N1 if autocorr else N2)
        NR1 = (rx1.shape[0] if rx1 is not None else N1)
        NR2 = (rx1.shape[0] if autocorr else (rx2.shape[0] if rx2 is not None else N2))
        return convert_3d_counts_to_cf(ND1, ND2, NR1, NR2, dd, dr12, dr21, rr)

    DDn, RRn, DR12n, DR21n = get_normalized_counts(
        estimator="ls",
        dd=dd,
        rr=rr,
        dr12=dr12,
        dr21=dr21,
        autocorr=autocorr,
        N1=N1,
        N2=N2,
        w1=w1,
        w2=w2,
        rw1=rw1,
        rw2=rw2,
        rx1=rx1,
        rx2=rx2,
        eps=eps,
        weight_type=weight_type
    )

    return (DDn - DR12n - DR21n + RRn) / np.maximum(RRn, eps)


def compute_tpcf(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        eps: float = DEFAULT_TOLERANCE,
        use_natural: bool = True,  # True -> Natural; False -> Landy–Szalay
        # Subsampling controls:
        use_recommended_subsample: bool = False,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
        use_cic_subsample: bool = False,
        cic_subsample_size: int = DEFAULT_CIC_SUBSAMPLE_SIZE,
        subsample_secondary: bool = False,
        subsample_rng_seed: int | None = None,
    ) -> np.ndarray:
    """
    Two-point correlation for a periodic box using Corrfunc pair-counters.

    Supports both autocorrelation (single catalog) and cross-correlation (two catalogs).
    Returns (nbins, 2): [r_center, xi(r)].
    """
    coords1, coords2, weights, weights2 = _get_tpcf_inputs(
        comoving_coordinates=comoving_coordinates,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        boxsize=boxsize,
        weights=weights,
        other_coordinates=other_coordinates,
        weights2=weights2,
        use_recommended_subsample=use_recommended_subsample,
        num_target_pairs=num_target_pairs,
        use_cic_subsample=use_cic_subsample,
        cic_subsample_size=cic_subsample_size,
        subsample_secondary=subsample_secondary,
        subsample_rng_seed=subsample_rng_seed,
    )

    autocorr = coords2 is None
    N1 = coords1.shape[0]
    N2 = (N1 if autocorr else coords2.shape[0])

    rbins, r_centers = _get_radial_bins(rmin, rmax, nbins)
    nthreads = _omp_nthreads()

    # ---- Fast path: periodic-box natural estimator, unweighted autocorrelation ----
    # Safe & much faster than explicit DR/RR for this case.
    if autocorr and (weights is None) and use_natural:
        x, y, z = coords1.T
        _assert_threads_match_env(nthreads, "corrfunc_xi fast-path (autocorr, unweighted)")
        # Corrfunc xi returns a structured array with field 'xi'
        xi_rec = _safe_corrfunc_run(
            nthreads,
            "corrfunc_xi fast-path (autocorr, unweighted)",
            corrfunc_xi,
            boxsize, nthreads, rbins, x, y, z,
            periodic=True,
        )
        xi = np.asarray(xi_rec['xi'], dtype=float)
        return np.column_stack([r_centers, xi])

    # ---- General paths (weighted and/or cross, or Landy–Szalay) ----
    w1, w2, rw1, rw2, weight_type = _parse_weights(weights, weights2, N1, N2, autocorr)

    estimator_args = {
        "coords1": coords1,
        "coords2": coords2,
        "N1": N1,
        "N2": N2,
        "nthreads": nthreads,
        "rbins": rbins,
        "w1": w1,
        "w2": w2,
        "rw1": rw1,
        "rw2": rw2,
        "boxsize": boxsize,
        "weight_type": weight_type,
        "eps": eps,
    }

    xi = (
        _natural_estimator(**estimator_args)
        if use_natural else
        _landy_szalay_estimator(**estimator_args)
    )
    return np.column_stack([r_centers, xi])


def _extract_axis_bins(sub_box_info: dict[int, list[tuple[float, float]]]) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (xbins, ybins, zbins) *when* sub_box_info is an axis-bins mapping.

    Supported axis-bins schemas:
      • {0: [(x0,x1), ...], 1: [(y0,y1), ...], 2: [(z0,z1), ...]}
      • {'x': [...], 'y': [...], 'z': [...]}  (case-insensitive keys also accepted)

    If sub_box_info is not an axis-bins mapping (e.g., it maps fold_id -> bounds),
    this function raises a ValueError. Use `_iter_excluded_bounds` directly in that case.
    """
    if not isinstance(sub_box_info, dict):
        raise ValueError("sub_box_info must be a dict for axis-bins mapping.")

    # Numeric axis keys – require exactly {0,1,2}
    keyset = set(sub_box_info.keys())
    if keyset == {0, 1, 2}:
        return sub_box_info[0], sub_box_info[1], sub_box_info[2]

    # String axis keys – require exactly {'x','y','z'} (case-insensitive)
    lower_map = {str(k).lower(): k for k in sub_box_info}
    if set(lower_map.keys()) == {"x", "y", "z"}:
        kx, ky, kz = lower_map["x"], lower_map["y"], lower_map["z"]
        return sub_box_info[kx], sub_box_info[ky], sub_box_info[kz]

    raise ValueError(
        "sub_box_info does not look like an axis-bins mapping. Expected keys {0,1,2} or {'x','y','z'}."
    )


def _iter_excluded_bounds(
        sub_box_info: dict[int, list[tuple[float, float]]]
    ) -> Iterator[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """Yield (xlo,xhi),(ylo,yhi),(zlo,zhi) for each excluded sub-box (one per fold).

    Supported input schemas:
      A) Axis-bins mapping → cartesian product over x,y,z bins
         {0: [(x0,x1), ...], 1: [(y0,y1), ...], 2: [(z0,z1), ...]}
         or {'x': [...], 'y': [...], 'z': [...]}.

      B) Per-fold bounds → iterate entries directly:
         • dict: {fold_id: ((xlo,xhi),(ylo,yhi),(zlo,zhi)) or {0:(),1:(),2:()} or {'x':(),'y':(),'z':()}}
         • list/tuple: [ ((xlo,xhi),(ylo,yhi),(zlo,zhi)), ... ]
    """
    # Try axis-bins schema first
    try:
        xbins, ybins, zbins = _extract_axis_bins(sub_box_info)
    except Exception as e:
        # Treat as per-fold bounds
        entries = (
            sub_box_info.values() if isinstance(sub_box_info, dict) else sub_box_info
        )
        for b in entries:
            if isinstance(b, dict):
                # Accept dicts keyed by 0/1/2 or x/y/z (any case)
                if all(k in b for k in (0, 1, 2)):
                    yield tuple(b[0]), tuple(b[1]), tuple(b[2])
                else:
                    lk = {str(k).lower(): k for k in b.keys()}
                    if all(ax in lk for ax in ("x", "y", "z")):
                        yield tuple(b[lk["x"]]), tuple(b[lk["y"]]), tuple(b[lk["z"]])
                    else:
                        raise ValueError(
                            "Invalid per-fold bounds dict; expected keys {0,1,2} or {'x','y','z'}."
                        ) from e
            else:
                # Assume a 3-sequence of (lo,hi) tuples
                if (not hasattr(b, "__len__")) or len(b) != 3:
                    raise ValueError(
                        "Per-fold bounds must be a 3-sequence of (lo,hi) tuples."
                    ) from e
                xb, yb, zb = b
                yield tuple(xb), tuple(yb), tuple(zb)
        return

    # Axis-bins path: cartesian product
    for xb in xbins:
        for yb in ybins:
            for zb in zbins:
                yield tuple(xb), tuple(yb), tuple(zb)


def _randoms_outside_excluded(
        boxsize: float,
        excluded_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        n_randoms: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
    """Generate uniform randoms in the retained volume (box minus one rectangular sub-box).
    Uses simple rejection sampling; efficient because excluded fraction is small.
    Returns an array of shape (n_randoms, 3).
    """
    if n_randoms <= 0:
        return np.empty((0, 3), dtype=float)

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = excluded_bounds
    # Volume fraction retained helps choose batch size for rejection sampling
    vol_excl = max(0.0, (xhi - xlo) * (yhi - ylo) * (zhi - zlo))
    vol_box = boxsize**3
    retain_frac = max(1e-6, 1.0 - (vol_excl / max(vol_box, 1e-30)))

    accepted = []
    need = n_randoms
    # Slightly over-generate per batch to reduce loops
    batch = max(1024, int(need / retain_frac * 1.1))

    while need > 0:
        pts = rng.uniform(0.0, boxsize, size=(batch, 3))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        in_excl = (x >= xlo) & (x < xhi) & (y >= ylo) & (y < yhi) & (z >= zlo) & (z < zhi)
        keep = pts[~in_excl]
        if keep.size:
            if keep.shape[0] >= need:
                accepted.append(keep[:need])
                break
            else:
                accepted.append(keep)
                need -= keep.shape[0]
        # Adapt batch size if acceptance was unexpectedly low
        batch = max(1024, int((need / retain_frac) * 1.2))

    return np.vstack(accepted) if accepted else np.empty((0, 3), dtype=float)


def _region_tagged_jackknife_tpcf(
        coords1: np.ndarray,
        coords2: np.ndarray | None,
        weights1: np.ndarray | None,
        weights2: np.ndarray | None,
        rbins: np.ndarray,
        r_centers: np.ndarray,
        boxsize: float,
        sub_box_info: dict[int, list[tuple[float, float]]],
        eps: float = DEFAULT_TOLERANCE,
        random_multiplier: int = DEFAULT_RANDOM_MULTIPLIER,
        rng_seed: int | None = None,
    ) -> dict[str, np.ndarray]:
    """
    Jackknife TPCF using a Halotools-like *region-tagged* scheme.

    Current support:
      • Autocorrelation only (coords2 is None)
      • Unweighted Landy–Szalay (matches existing jackknife path)

    Strategy per fold k (with region bounds from sub_box_info):
        ALL   : counts with full data & randoms
        OUTOUT: counts with dataOUT vs dataOUT (and similarly DR, RR with OUT partitions)
        ININ  : counts with dataIN  vs dataIN  (and similarly DR, RR with IN partitions)
        IO    : ALL − OUTOUT − ININ
        effective counts = OUTOUT + 0.5 * IO      (0 * ININ)
        ND_out = number of OUT data, NR_out = number of OUT randoms
        xi_k   = convert_3d_counts_to_cf(ND_out, ND_out, NR_out, NR_out, DD_eff, DR_eff, DR_eff, RR_eff)
    """
    # guardrails
    if coords2 is not None:
        raise NotImplementedError("Region-tagged jackknife: cross-correlation not yet implemented.")
    if (weights1 is not None) or (weights2 is not None):
        raise NotImplementedError("Region-tagged jackknife: weighted pairs not yet implemented.")

    nthreads = _omp_nthreads()
    _assert_threads_match_env(nthreads, "Region-tagged JK precompute")

    # One global random catalog over the full box (reused for all folds)
    N1 = coords1.shape[0]
    Nran = int(max(1, np.ceil(float(random_multiplier) * float(N1))))
    rx, ry, rz = _get_randoms(boxsize, Nran)
    randoms_full = np.vstack([rx, ry, rz]).T

    # Precompute ALL counts once
    x1, y1, z1 = coords1.T
    xr, yr, zr = randoms_full.T

    dd_all = DD(1, nthreads, rbins, x1, y1, z1, boxsize=boxsize)
    dr_all = DD(0, nthreads, rbins, x1, y1, z1, X2=xr, Y2=yr, Z2=zr, boxsize=boxsize)
    rr_all = DD(1, nthreads, rbins, xr, yr, zr, boxsize=boxsize)

    xi_folds: list[np.ndarray] = []
    # regions = list(_iter_excluded_bounds(sub_box_info))

    for bounds in tqdm( # I think this was done incorrectly... 
        _iter_excluded_bounds(sub_box_info),
        unit="fold", total=len(sub_box_info),
        desc="Computing Jackknife TPCF folds"
    ):
        
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds

        # Partition data/randoms into OUT (keep) vs IN (excluded region)
        in_data = ((coords1[:, 0] >= xlo) & (coords1[:, 0] < xhi) &
                   (coords1[:, 1] >= ylo) & (coords1[:, 1] < yhi) &
                   (coords1[:, 2] >= zlo) & (coords1[:, 2] < zhi))
        out_data = ~in_data

        in_rand = ((randoms_full[:, 0] >= xlo) & (randoms_full[:, 0] < xhi) &
                   (randoms_full[:, 1] >= ylo) & (randoms_full[:, 1] < yhi) &
                   (randoms_full[:, 2] >= zlo) & (randoms_full[:, 2] < zhi))
        out_rand = ~in_rand

        fold_size = np.sum(out_data) / coords1.shape[0]
        print(f"Fold size: {fold_size:.3f}")

        D_out = coords1[out_data]
        D_in  = coords1[in_data]
        R_out = randoms_full[out_rand]
        R_in  = randoms_full[in_rand]

        # OUT-OUT
        if D_out.shape[0] >= 2:
            dd_out = DD(1, nthreads, rbins, *D_out.T, boxsize=boxsize)
        else:
            dd_out = dd_all.copy(); dd_out["npairs"][:] = 0.0
        if R_out.shape[0] >= 2:
            rr_out = DD(1, nthreads, rbins, *R_out.T, boxsize=boxsize)
        else:
            rr_out = rr_all.copy(); rr_out["npairs"][:] = 0.0
        if (D_out.shape[0] >= 1) and (R_out.shape[0] >= 1):
            dr_out = DD(0, nthreads, rbins, *D_out.T, X2=R_out[:,0], Y2=R_out[:,1], Z2=R_out[:,2], boxsize=boxsize)
        else:
            dr_out = dr_all.copy(); dr_out["npairs"][:] = 0.0

        # IN-IN
        if D_in.shape[0] >= 2:
            dd_in = DD(1, nthreads, rbins, *D_in.T, boxsize=boxsize)
        else:
            dd_in = dd_out.copy(); dd_in["npairs"][:] = 0.0
        if R_in.shape[0] >= 2:
            rr_in = DD(1, nthreads, rbins, *R_in.T, boxsize=boxsize)
        else:
            rr_in = rr_out.copy(); rr_in["npairs"][:] = 0.0
        if (D_in.shape[0] >= 1) and (R_in.shape[0] >= 1):
            dr_in = DD(0, nthreads, rbins, *D_in.T, X2=R_in[:,0], Y2=R_in[:,1], Z2=R_in[:,2], boxsize=boxsize)
        else:
            dr_in = dr_out.copy(); dr_in["npairs"][:] = 0.0

        # Helper: combine counts into effective per-fold counts
        def _eff(all_rec, out_rec, in_rec):
            a = _effective_counts(all_rec)
            o = _effective_counts(out_rec)
            i = _effective_counts(in_rec)
            eff_vals = o + 0.5 * (a - o - i)
            rec = out_rec.copy()
            rec["npairs"] = eff_vals.astype(rec["npairs"].dtype, copy=False)
            if "weightavg" in rec.dtype.names:
                rec["weightavg"][:] = 0.0
            return rec

        dd_eff = _eff(dd_all, dd_out, dd_in)
        dr_eff = _eff(dr_all, dr_out, dr_in)
        rr_eff = _eff(rr_all, rr_out, rr_in)

        # Normalizations for the "leave-k-out" sample
        ND = D_out.shape[0]
        NR = R_out.shape[0]

        # LS autocorr (unweighted) for this fold
        xi_k = convert_3d_counts_to_cf(ND, ND, NR, NR, dd_eff, dr_eff, dr_eff, rr_eff)
        xi_folds.append(np.asarray(xi_k, dtype=float))

    return convert_jk_folds_to_results(xi_folds, r_centers, return_folds=False)


# --- Small helper: covariance to correlation matrix ---
def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix with safe handling of zeros."""
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


# --- Helper to resolve Halotools Nsub from sub_box_info or explicit value ---
def _resolve_halotools_Nsub(
        sub_box_info: dict[int, list[tuple[float, float]]] | list | tuple | None,
        explicit_Nsub: int | tuple[int, int, int] | None = None
    ) -> np.ndarray:
    """Resolve (nx, ny, nz) for Halotools tpcf_jackknife from either explicit Nsub or sub_box_info.

    Rules:
      1) If explicit_Nsub is an int → (n,n,n); if 3-seq → as given.
      2) If sub_box_info is axis-bins mapping, use lengths per axis.
      3) Else, if sub_box_info is a per-fold container of length K, choose (nx,ny,nz)
         near-cubic with product ≥ K.
      4) Fallback to (5,5,5).
    """
    if explicit_Nsub is not None:
        if isinstance(explicit_Nsub, int):
            n = int(max(1, explicit_Nsub))
            return np.array([n, n, n], dtype=int)
        arr = np.asarray(explicit_Nsub, dtype=int).ravel()
        if arr.size == 3 and np.all(arr > 0):
            return arr

    if sub_box_info is not None:
        # Try axis-bins mapping
        try:
            xbins, ybins, zbins = _extract_axis_bins(sub_box_info)  # raises when not axis-bins
            return np.array([len(xbins), len(ybins), len(zbins)], dtype=int)
        except Exception:
            # Per-fold container
            try:
                K = len(sub_box_info)  # type: ignore[arg-type]
            except Exception:
                K = None
            if K and K > 0:
                c = int(round(K ** (1.0 / 3.0)))
                nx = ny = nz = max(1, c)
                while nx * ny * nz < K:
                    if nx <= ny and nx <= nz:
                        nx += 1
                    elif ny <= nx and ny <= nz:
                        ny += 1
                    else:
                        nz += 1
                return np.array([nx, ny, nz], dtype=int)

    return np.array([5, 5, 5], dtype=int)


def compute_jackknifed_tpcf(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        sub_box_info: dict[int, list[tuple[float, float]]],
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        eps: float = DEFAULT_TOLERANCE,
        use_natural: bool = True,
        random_multiplier: int = DEFAULT_RANDOM_MULTIPLIER,
        rng_seed: int | None = None,
        adaptive_randoms: bool = False,
        max_randoms_per_fold: int | None = None,
        min_randoms_per_fold: int = 1024,
        return_folds: bool = False,
        # Subsampling controls:
        use_recommended_subsample: bool = False,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
        use_cic_subsample: bool = False,
        cic_subsample_size: int = DEFAULT_CIC_SUBSAMPLE_SIZE,
        subsample_secondary: bool = False,
        subsample_rng_seed: int | None = None,
        use_region_tagged: bool = False, # Need to get this one to work...
        use_halotools: bool = False,
        halotools_Nsub: int | tuple[int, int, int] | None = None,
        include_inside: bool = True,
    ) -> dict[str, np.ndarray]:
    """
    Memory-efficient jackknife TPCF:
      • Builds masks and random catalogs *per fold*, on the fly (no giant lists kept in memory).
      • Generates randoms confined to the retained volume (box minus excluded sub-box) via rejection sampling.
    """
    # Validate + (optional) subsample once up front
    coords1, coords2, weights1, weights2 = _get_tpcf_inputs(
        comoving_coordinates=comoving_coordinates,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        boxsize=boxsize,
        weights=weights,
        other_coordinates=other_coordinates,
        weights2=weights2,
        use_recommended_subsample=use_recommended_subsample,
        num_target_pairs=num_target_pairs,
        use_cic_subsample=use_cic_subsample,
        cic_subsample_size=cic_subsample_size,
        subsample_secondary=subsample_secondary,
        subsample_rng_seed=subsample_rng_seed,
    ) 

    
    rbins, r_centers = _get_radial_bins(rmin, rmax, nbins)

    # --- Optional: Halotools benchmark path ---
    if use_halotools:
        if not _HAS_HALOTOOLS:
            raise ImportError(
                "Halotools not installed. Please `pip install halotools` to use the Halotools benchmark path."
            )
        else:
            print("Using Halotools jackknife TPCF implementation.")
        # Resolve Nsub (nx,ny,nz) for Halotools from either explicit input or sub_box_info
        Nsub_arr = _resolve_halotools_Nsub(sub_box_info=sub_box_info, explicit_Nsub=halotools_Nsub)
        # Halotools supports both Natural and Landy–Szalay; keep estimator consistent with flag
        est_map = {
            "natural": "Natural",
            "landy": "Landy-Szalay",
            "landy-szalay": "Landy-Szalay",
            "landy_szalay": "Landy-Szalay",
        }
        est_name = "Natural" if use_natural else "Landy-Szalay"
        # Randoms: Halotools can generate internally when randoms is np.array([N]) and period provided
        Nran = int(max(1, np.ceil(float(random_multiplier) * float(coords1.shape[0]))))
        nthreads = _omp_nthreads()

        if coords2 is None:
            xi_full, xi_cov = ht_tpcf_jackknife(
                sample1=coords1,
                randoms=np.array([Nran]),
                rbins=rbins,
                Nsub=Nsub_arr,
                sample2=None,
                period=boxsize,
                do_auto=True,
                do_cross=False,
                estimator=est_name,
                num_threads=nthreads,
                seed=rng_seed,
            )
            xi = np.asarray(xi_full, dtype=float)
            cov = np.asarray(xi_cov, dtype=float)
        else:
            # Halotools returns multiple outputs; use xi_11 and its cov to match our single-output API
            xi_11, xi_12, xi_22, cov_11, cov_12, cov_22 = ht_tpcf_jackknife(
                sample1=coords1,
                randoms=np.array([Nran]),
                rbins=rbins,
                Nsub=Nsub_arr,
                sample2=coords2,
                period=boxsize,
                do_auto=True,
                do_cross=True,
                estimator=est_name,
                num_threads=nthreads,
                seed=rng_seed,
            )
            xi = np.asarray(xi_11, dtype=float)
            cov = np.asarray(cov_11, dtype=float)

        err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        corr = _cov_to_corr(cov)
        out = {
            "estimate": np.column_stack([r_centers, xi]),
            "errors":   np.column_stack([r_centers, err]),
            "correlation_matrix": corr,
        }
        if return_folds:
            # Halotools doesn't return per-fold xi; leave empty with correct shape
            out["folds"] = np.column_stack([r_centers.astype(float), np.empty((r_centers.size, 0), dtype=float)])
        return out

    if use_region_tagged:
        return _region_tagged_jackknife_tpcf(
            coords1=coords1,
            coords2=coords2,
            weights1=weights1,
            weights2=weights2,
            rbins=rbins,
            r_centers=r_centers,
            boxsize=boxsize,
            sub_box_info=sub_box_info,
            eps=eps,
            random_multiplier=random_multiplier,
            rng_seed=rng_seed,
        )

    xi_folds: list[np.ndarray] = []

    for fold_idx, bounds in tqdm(
        enumerate(_iter_excluded_bounds(sub_box_info)),
        unit="fold", total=len(sub_box_info),
        desc="Computing Jackknife TPCF folds"
    ):
        try:
            fold_inputs = _get_tpcf_fold_inputs(
                coords1=coords1,
                coords2=coords2,
                weights1=weights1,
                weights2=weights2,
                radial_bins=rbins,
                box_size=boxsize,
                eps=eps,
                bounds=bounds,
                random_multiplier=random_multiplier,
                rng_seed=rng_seed,
                adaptive_randoms=adaptive_randoms,
                max_randoms_per_fold=max_randoms_per_fold,
                min_randoms_per_fold=min_randoms_per_fold,
                include_mode="inside" if include_inside else "outside",
            )
        except ValueError as ve:
            print(f"[warn] Skipping fold {fold_idx} due to error: {ve}")
            continue

        xi_arr = get_xi_fold_arr(fold_inputs, use_natural_estimator=use_natural)
        xi_folds.append(np.asarray(xi_arr, dtype=float))

    return convert_jk_folds_to_results(xi_folds, r_centers, return_folds=return_folds)

def compute_tpcf_treecorr(
    coords1: np.ndarray,
    coords2: np.ndarray | None,
    rmin: float,
    rmax: float,
    nbins: int,
    boxsize: float,
    var_method: str | None = None,
    bin_type: str = "Log",
) -> dict[str, np.ndarray]:
    """
    Compute two-point correlation using TreeCorr's NNCorrelation.

    Returns the same dict format as your existing compute_tpcf / get_tpcf:
      {"estimate": ..., "errors": ..., "correlation_matrix": ...}
    If var_method is given (e.g. 'jackknife') then errors and covariance are enabled.
    """
    # Build catalogs
    cat1 = treecorr.Catalog(
        x=coords1[:, 0],  x_units='unitless',
        y=coords1[:, 1],  y_units='unitless',
        z=coords1[:, 2],  z_units='unitless'
    )
    if coords2 is not None:
        cat2 = treecorr.Catalog(
            x=coords2[:, 0],  x_units='unitless',
            y=coords2[:, 1],  y_units='unitless',
            z=coords2[:, 2],  z_units='unitless'
        )
    else:
        cat2 = None

    # Build NNCorrelation instance
    nn = treecorr.NNCorrelation(
        min_sep=rmin, max_sep=rmax, nbins=nbins,
        bin_type=bin_type,
        var_method=var_method,
        metric="Periodic",  # important for periodic box
        period=boxsize,
    )

    # Process auto- or cross-correlation
    if cat2 is None:
        nn.process(cat1)
    else:
        nn.process(cat1, cat2)

    # After processing, get xi and variance / covariance
    r = np.exp(nn.logr)  # bin centers in physical units
    xi = nn.xi

    # Setup outputs
    estimate = np.column_stack([r, xi])

    if var_method is not None:
        # varxi is the diagonal variances
        err = np.sqrt(nn.varxi)
        errors = np.column_stack([r, err])
        cov = nn.cov  # full covariance matrix
    else:
        errors = np.full((nbins, 2), np.nan)
        cov = np.eye(nbins)

    return {
        "estimate": estimate,
        "errors": errors,
        "correlation_matrix": cov,
    }


# def compute_jackknifed_tpcf_treecorr(
#         coords: np.ndarray,
#         rmin: float,
#         rmax: float,
#         nbins: int,
#         boxsize: float,
#         n_patches: int = 8,
#         bin_type: str = "Log",
#     ) -> dict[str, np.ndarray]:
#     """
#     Compute jackknife-estimated two-point correlation using TreeCorr.

#     This wraps compute_tpcf_treecorr with var_method='jackknife' and
#     assigns patches automatically via a grid in the box.
#     """
#     # Build catalog with patch assignment:
#     # We divide the domain into n_patches^(1/3) per axis (approx) or use 1D partitioning.
#     # Here’s a simple cubic grid partition:
#     nside = int(round(n_patches ** (1/3)))
#     if nside ** 3 != n_patches:
#         # Fallback: choose nside such that nside^3 >= n_patches
#         nside = int(np.ceil(n_patches ** (1/3)))
#     # compute patch indices for each point
#     coords_mod = coords % boxsize
#     frac = coords_mod / boxsize  # in [0,1) in each axis
#     ix = np.floor(frac[:, 0] * nside).astype(int)
#     iy = np.floor(frac[:, 1] * nside).astype(int)
#     iz = np.floor(frac[:, 2] * nside).astype(int)
#     patch = ix * (nside**2) + iy * nside + iz  # id 0..nside^3-1

#     cat = treecorr.Catalog(
#         x=coords[:, 0], x_units='unitless',
#         y=coords[:, 1], y_units='unitless',
#         z=coords[:, 2], z_units='unitless',
#         patch=patch
#     )

#     nn = treecorr.NNCorrelation(
#         min_sep=rmin, 
#         max_sep=rmax, 
#         nbins=nbins,
#         bin_type=bin_type,
#         var_method='jackknife',
#         metric="Periodic",
#         period=boxsize,
#     )

#     nn.process(cat)

#     r = np.exp(nn.logr)
#     xi = nn.xi
#     err = np.sqrt(nn.varxi)
#     cov = nn.cov

#     return {
#         "estimate": np.column_stack([r, xi]),
#         "errors": np.column_stack([r, err]),
#         "correlation_matrix": cov,
#     }

def get_tpcf(
        comoving_coordinates: np.ndarray,
        rmin: float,
        rmax: float,
        nbins: int,
        boxsize: float,
        weights: np.ndarray | None = None,
        other_coordinates: np.ndarray | None = None,
        weights2: np.ndarray | None = None,
        sub_box_info: dict[int, list[tuple[float, float]]] | None = None,
        eps: float = DEFAULT_TOLERANCE,
        use_natural: bool = True,
        run_jackknife: bool = False,
        # jackknife controls (optional):
        random_multiplier: int | None = None,
        rng_seed: int | None = None,
        return_folds: bool = False,
        # Subsampling controls (optional):
        use_recommended_subsample: bool = False,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS,
        use_cic_subsample: bool = False,
        cic_subsample_size: int = DEFAULT_CIC_SUBSAMPLE_SIZE,
        subsample_secondary: bool = False,
        subsample_rng_seed: int | None = None,
        # Adaptive randoms controls (jackknife only):
        adaptive_randoms: bool = False,
        max_randoms_per_fold: int | None = None,
        min_randoms_per_fold: int = 1024,
        include_inside: bool = True,
    ) -> dict[str, np.ndarray]:
    """
    Main API for two-point correlation function.

    Returns a dict with keys:
      - "estimate": (nbins, 2) array of [r_center, xi]
      - "errors":   (nbins, 2) array of [r_center, sigma_xi] (NaNs for non-jackknife)
      - "correlation_matrix": (nbins, nbins) array; identity for non-jackknife

    If `run_jackknife=True`, this calls `compute_jackknifed_tpcf` (using fold-matched
    random catalogs inside the retained volume). You may override the randoms density
    and RNG via `random_multiplier` and `rng_seed`.
    """
    if rmax > 0.5 * boxsize:
        print("[warn] rmax > boxsize/2; RR bins at the largest scales may be empty. Consider reducing rmax.")
    if run_jackknife:
        if sub_box_info is None:
            raise ValueError("sub_box_info must be provided when run_jackknife=True")
        return compute_jackknifed_tpcf(
            comoving_coordinates=comoving_coordinates,
            rmin=rmin, rmax=rmax, nbins=nbins, boxsize=boxsize,
            weights=weights,
            other_coordinates=other_coordinates,
            weights2=weights2,
            sub_box_info=sub_box_info,
            eps=eps,
            use_natural=False, # always use LS for jackknife
            random_multiplier=(random_multiplier if random_multiplier is not None else 10),
            rng_seed=rng_seed,
            return_folds=return_folds,
            use_recommended_subsample=use_recommended_subsample,
            num_target_pairs=num_target_pairs,
            use_cic_subsample=use_cic_subsample,
            cic_subsample_size=cic_subsample_size,
            subsample_secondary=subsample_secondary,
            subsample_rng_seed=subsample_rng_seed,
            adaptive_randoms=adaptive_randoms,
            max_randoms_per_fold=max_randoms_per_fold,
            min_randoms_per_fold=min_randoms_per_fold,
            include_inside=include_inside
        )

    # Non-jackknife path
    return {
        "estimate": compute_tpcf(
            comoving_coordinates=comoving_coordinates,
            rmin=rmin, rmax=rmax, nbins=nbins, boxsize=boxsize,
            weights=weights,
            other_coordinates=other_coordinates,
            weights2=weights2,
            eps=eps,
            use_natural=use_natural,
            use_recommended_subsample=use_recommended_subsample,
            num_target_pairs=num_target_pairs,
            use_cic_subsample=use_cic_subsample,
            cic_subsample_size=cic_subsample_size,
            subsample_secondary=subsample_secondary,
            subsample_rng_seed=subsample_rng_seed,
        ),
        "errors": np.full((nbins, 2), np.nan, dtype=float),
        "correlation_matrix": np.eye(nbins, dtype=float),
    }


def recommended_particle_fraction(
        num_particles_in_sim: int,
        box_size: float,
        min_radius: float,
        max_radius: float,
        num_radial_bins: int,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS
    ) -> float:
    # Validate inputs
    if (
        min_radius <= 0
        or max_radius <= min_radius
        or num_radial_bins < 1
        or box_size <= 0
        or num_particles_in_sim < 2
    ):
        return 1.0
    dlnr = np.log(max_radius / min_radius) / float(num_radial_bins)
    pbin = 4.0 * np.pi * dlnr * (min_radius / box_size) ** 3
    denom = float(num_particles_in_sim**2) * pbin
    if denom <= 0:
        return 1.0
    f = np.sqrt(max(0.0, 2.0 * num_target_pairs / denom))
    return float(min(1.0, f))

def get_tpcf_particle_subsample_size(
        num_particles_in_sim: int,
        box_size: float,
        min_radius: float,
        max_radius: float,
        num_radial_bins: int,
        num_target_pairs: int = DEFAULT_NUM_TARGET_PAIRS
    ) -> int:
    fraction = recommended_particle_fraction(
        num_particles_in_sim=num_particles_in_sim,
        box_size=box_size,
        min_radius=min_radius,
        max_radius=max_radius,
        num_radial_bins=num_radial_bins,
        num_target_pairs=num_target_pairs
    )
    return max(2, min(num_particles_in_sim, int(np.floor(fraction * num_particles_in_sim))))

