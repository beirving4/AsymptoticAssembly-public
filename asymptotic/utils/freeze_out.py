import numpy as np

from functools import partial
from pathlib import Path
from scipy.interpolate import interp1d


# ------------------------------------------------------------
# NEW: build a shared high-resolution scale-factor grid
# ------------------------------------------------------------
def _make_hi_res_grid(a_min: float, a_max: float, n: int = 1000) -> np.ndarray:
    """
    Generate a high-resolution log-spaced scale-factor grid.
    Ensures strictly increasing positive endpoints.
    """
    if a_min <= 0:
        raise ValueError("Scale factors must be positive for logspace grid.")

    return np.logspace(np.log10(a_min), np.log10(a_max), n)


# ------------------------------------------------------------
# UPDATED: compute jackknife mean + std on a high-resolution grid
# ------------------------------------------------------------
def _compute_jackknife_stats(
        fold_values: list[np.ndarray],
        scale_factors: np.ndarray,
        hi_res_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given per–fold abundance histories y_i(a) (all on the same original
    scale_factors grid), interpolate each to hi_res_a and return
    the jackknife mean and std.

    Parameters
    ----------
    fold_values : list of arrays, each shape (N,)
        Per-fold abundance curves.
    scale_factors : array, shape (N,)
        Original a-grid shared by all folds.
    hi_res_a : array, shape (N_hi,)
        High-resolution grid.

    Returns
    -------
    mean : array, shape (N_hi,)
    std  : array, shape (N_hi,)
    """
    interpolated = []
    for y in fold_values:
        f = interp1d(scale_factors, y, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        interpolated.append(f(hi_res_a))

    arr = np.vstack(interpolated)  # (N_folds, N_hi)
    mean = np.mean(arr, axis=0)
    std  = np.std(arr, axis=0, ddof=1)
    return mean, std


# ------------------------------------------------------------
# NEW: compute jackknife rate statistics on a hi-res grid
# ------------------------------------------------------------
def _compute_jackknife_rate_stats(
        fold_values: list[np.ndarray],
        scale_factors: np.ndarray,
        hi_res_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Jackknife stats for the logarithmic growth rate d ln y / d ln a.

    Returns
    -------
    rate_mean : array, shape (N_hi,)
    rate_std  : array, shape (N_hi,)
    """
    rate_curves = []

    for y in fold_values:
        f = interp1d(scale_factors, y, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        y_hi = f(hi_res_a)

        y_clip = np.clip(y_hi, 1e-300, None)
        ln_y   = np.log(y_clip)
        ln_a   = np.log(hi_res_a)
        rate   = np.gradient(ln_y, ln_a)
        rate_curves.append(rate)

    arr = np.vstack(rate_curves)  # (N_folds, N_hi)
    rate_mean = np.mean(arr, axis=0)
    rate_std  = np.std(arr, axis=0, ddof=1)
    return rate_mean, rate_std


# ------------------------------------------------------------
# Freeze-out detection remains unchanged
# ------------------------------------------------------------
def _find_freeze_out_time(
        a: np.ndarray,
        y: np.ndarray,
        tol: float = 2.0,
        y_err: np.ndarray | None = None,
    ) -> float | None:

    y_inf = y[-1]
    N = len(a)

    if y_err is None:
        scale = abs(y_inf) if abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y, 0.02 * scale)

    for j in range(N):
        if np.abs(y[j] - y_inf) > tol * y_err[j]:
            continue

        ok = True
        for m in range(j, N):

            if np.abs(y[m] - y_inf) > tol * y_err[m]:
                ok = False
                break

            if (m < (N - 1)) and (np.abs(y[m] - y[m+1]) > tol * y_err[m]):
                ok = False
                break

        if ok:
            return a[j]

    return None

# ------------------------------------------------------------
# NEW: Freeze-out finder based on the growth rate of abundance
# ------------------------------------------------------------
def _find_freeze_out_time_by_rate(
    a, 
    y, 
    y_err=None, 
    abs_tol_frac=0.05, 
    rate_tol=0.01
):
    """
    Freeze-out time defined by:
      1) amplitude is within abs_tol_frac of asymptote
      2) |dy/dln a| <= rate_tol
    y_err is optional — used only for peak bracketing by perturbation.
    """

    # --- Compute asymptote (final value) ---
    y_inf = y[-1]
    abs_tol = abs_tol_frac * abs(y_inf)

    # --- If no error model is provided, inject a minimal synthetic one ---
    if y_err is None:
        y_err = np.full_like(y, 0.02 * max(1e-12, abs(y_inf)))

    # --- Compute growth rate dy/d ln a ---
    loga = np.log(a)
    dloga = np.diff(loga)
    dy = np.diff(y)
    rate = np.concatenate([[dy[0] / dloga[0]], dy / dloga])

    # --- Loop to find freeze-out ---
    N = len(a)
    for j in range(N):
        amp_ok  = abs(y[j] - y_inf) <= abs_tol
        rate_ok = abs(rate[j])    <= rate_tol

        if not (amp_ok and rate_ok):
            continue

        ok = True
        for m in range(j, N):
            amp_m  = abs(y[m] - y_inf) <= abs_tol
            rate_m = abs(rate[m])      <= rate_tol
            if not (amp_m and rate_m):
                ok = False
                break

        if ok:
            return a[j]

    return None

# ------------------------------------------------------------
# NEW: Freeze-out finder based on finite-difference window
# ------------------------------------------------------------
def _find_freeze_out_time_by_window(
        a: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray | None = None,
        tol: float = 2.0,
        amp_tol_frac: float = 0.05,
        diff_window: float = 0.3,
        diff_tol: float = 0.01,
        K: int = 3,
    ) -> float | None:
    """Freeze-out using a finite-width window in ln a.

    We declare freeze-out at the earliest scale factor a_j where:
      1) |y(a_j) - y_inf| and |y(a_end) - y_inf| are both within
         amp_tol_frac * |y_inf| (amplitude tolerance), where a_end is the
         first point with ln(a_end) - ln(a_j) >= diff_window;
      2) The fractional change over that window is small:
            |y(a_end) - y(a_j)| / max(|y_inf|, 1) < diff_tol;
      3) Conditions (1) and (2) are satisfied for K consecutive starting
         indices j, j+1, ..., j+K-1.

    This uses finite differences instead of an instantaneous derivative and
    is more robust to noise and very slow late-time evolution.
    """

    N = len(a)
    if N == 0:
        return None

    ln_a = np.log(a)
    y_inf = y[-1]
    amp_scale = max(abs(y_inf), 1.0)
    amp_tol = amp_tol_frac * amp_scale

    # Boolean flags for each starting index j
    ok = np.zeros(N, dtype=bool)

    for j in range(N):
        # Find the minimal index j_end such that ln(a_end) - ln(a_j) >= diff_window
        j_end = j
        while j_end < N and (ln_a[j_end] - ln_a[j]) < diff_window:
            j_end += 1
        if j_end >= N:
            break  # no full window available from this j to the end

        # Amplitude close to asymptote at both ends of the window
        if abs(y[j] - y_inf) > amp_tol:
            continue
        if abs(y[j_end] - y_inf) > amp_tol:
            continue

        # Fractional change over the window relative to asymptotic scale
        frac_change = abs(y[j_end] - y[j]) / amp_scale
        if frac_change > diff_tol:
            continue

        ok[j] = True

    # Require K consecutive windows satisfying the above conditions
    if K <= 1:
        # If K == 1, freeze-out is just the first ok[j]
        idx = np.argmax(ok) if ok.any() else None
        return a[idx] if idx is not None else None

    return next(
        (a[j] for j in range(N - K + 1) if ok[j] and ok[j : j + K].all()),
        None,
    )


def _find_peak_time(
        a: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        tol: float = 2.0
    ) -> float | None:

    y_inf = y[-1]
    idx = np.argmax(y)
    delta = y[idx] - y_inf
    sigma_delta = np.sqrt(y_err[idx]**2 + y_err[-1]**2)
    return a[idx] if (delta > tol * sigma_delta) else None

# ------------------------------------------------------------
# Helper: Compute central and bracketing estimates for peak/freeze
# ------------------------------------------------------------
def _bracket_peak_and_freeze_times(
        a: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray,
        freeze_func=_find_freeze_out_time,
        tol: float = 2.0,
        **freeze_kwargs,
    ) -> tuple[float | None, ...]:
    """
    Compute central and bracketing (low/high) estimates for a_peak and a_frz
    by perturbing the abundance history by ± y_err.

    Parameters
    ----------
    freeze_func : callable
        Function used to determine the freeze-out time. Signature must be
        freeze_func(a, y, y_err, tol=..., **freeze_kwargs).

    Returns
    -------
    a_peak, a_frz, a_peak_lo, a_peak_hi, a_frz_lo, a_frz_hi
    """
    # Central values
    a_frz = freeze_func(a, y, **freeze_kwargs)
    a_peak_candidate = _find_peak_time(a, y, y_err, tol=tol)
    if a_peak_candidate is None:
        a_peak = a_frz
    else:
        a_peak = a_peak_candidate
        if a_frz is not None and a_peak > a_frz:
            a_peak = a_frz

    # If we somehow failed to find any central times, bail early.
    if a_frz is None and a_peak is None:
        return a_peak, a_frz, None, None, None, None

    # Build ±1σ abundance curves.
    y_minus = y - y_err
    y_plus = y + y_err

    # Freeze-out bracketing
    a_frz_minus = freeze_func(a, y_minus, **freeze_kwargs)
    a_frz_plus = freeze_func(a, y_plus, **freeze_kwargs)

    # Fallbacks if the perturbed curves don't yield valid times.
    if a_frz_minus is None:
        a_frz_minus = a_frz
    if a_frz_plus is None:
        a_frz_plus = a_frz

    # Peak bracketing
    a_peak_minus = _find_peak_time(a, y_minus, y_err, tol=tol)
    a_peak_plus = _find_peak_time(a, y_plus, y_err, tol=tol)

    if a_peak_minus is None:
        a_peak_minus = a_frz_minus if a_frz_minus is not None else a_peak
    if a_peak_plus is None:
        a_peak_plus = a_frz_plus if a_frz_plus is not None else a_peak

    # Ensure low <= central <= high for any non-None entries.
    def _ordered_triplet(central, low, high):
        vals = [v for v in (central, low, high) if v is not None]
        if not vals:
            return None, None, None
        lo = min(vals)
        hi = max(vals)
        return central, lo, hi

    a_frz, a_frz_lo, a_frz_hi = _ordered_triplet(a_frz, a_frz_minus, a_frz_plus)
    a_peak, a_peak_lo, a_peak_hi = _ordered_triplet(a_peak, a_peak_minus, a_peak_plus)

    return a_peak, a_frz, a_peak_lo, a_peak_hi, a_frz_lo, a_frz_hi


# ------------------------------------------------------------
# FULL UPDATED MAIN FUNCTION (HIGH-RESOLUTION VERSION)
# ------------------------------------------------------------
def compute_times_by_evolution(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
    ) -> dict[str, float | np.ndarray]:

    # High-res a-grid
    hi_res_a = _make_hi_res_grid(
        scale_factors.min(), 
        scale_factors.max(),
        n=n_hi_res
    )

    # Interpolate main curve
    f_full = interp1d(
        scale_factors, full_evolution,
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    y_full_hi = f_full(hi_res_a)

    # Errors: jackknife if folds, else fixed frac
    if fold_id_to_evolution is not None and len(fold_id_to_evolution) > 0:
        folds = list(fold_id_to_evolution.values())
        _, y_std_jk_hi = _compute_jackknife_stats(
            folds, scale_factors, hi_res_a
        )
        y_err = y_std_jk_hi
    else:
        y_inf = y_full_hi[-1]
        scale = abs(y_inf) if abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y_full_hi, rel_tol_no_folds * scale)

    (
        a_peak,
        a_frz,
        a_peak_lo,
        a_peak_hi,
        a_frz_lo,
        a_frz_hi,
    ) = _bracket_peak_and_freeze_times(
        hi_res_a, y_full_hi, y_err,
        freeze_func=partial(_find_freeze_out_time, tol=tol),
        tol=tol,
    )

    def _get_max_diff(central, low, high):
        if central is None:
            return None
        diffs: list[float] = []
        if low is not None:
            diffs.append(central - low)
        if high is not None:
            diffs.append(high - central)
        return max(diffs) if diffs else None

    out: dict[str, float | np.ndarray] = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": a_peak_lo,
        "a_peak_hi": a_peak_hi,
        "a_frz_lo": a_frz_lo,
        "a_frz_hi": a_frz_hi,
        "a_peak_err": _get_max_diff(a_peak, a_peak_lo, a_peak_hi),
        "a_frz_err": _get_max_diff(a_frz, a_frz_lo, a_frz_hi),
    }

    if return_full_grid:
        out["a_grid"] = hi_res_a
        out["y_grid"] = y_full_hi
        out["y_err"]  = y_err

    return out

# ------------------------------------------------------------
# NEW: Compute times using the rate-based freeze-out finder
# ------------------------------------------------------------
def compute_times_by_rate(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        abs_tol_frac: float = 0.05,
        rate_tol: float = 0.01,
    ) -> dict[str, float | np.ndarray]:

    hi_res_a = _make_hi_res_grid(
        scale_factors.min(), 
        scale_factors.max(),
        n=n_hi_res
    )

    f_full = interp1d(
        scale_factors, full_evolution,
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    y_full_hi = f_full(hi_res_a)

    if fold_id_to_evolution is not None and len(fold_id_to_evolution) > 0:
        folds = list(fold_id_to_evolution.values())
        _, y_std_jk_hi = _compute_jackknife_stats(
            folds, scale_factors, hi_res_a
        )
        y_err = y_std_jk_hi
    else:
        y_inf = y_full_hi[-1]
        scale = abs(y_inf) if abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y_full_hi, rel_tol_no_folds * scale)

    (
        a_peak,
        a_frz,
        a_peak_lo,
        a_peak_hi,
        a_frz_lo,
        a_frz_hi,
    ) = _bracket_peak_and_freeze_times(
        hi_res_a,
        y_full_hi,
        y_err,
        freeze_func=_find_freeze_out_time_by_rate,
        tol=tol,
        abs_tol_frac=abs_tol_frac,
        rate_tol=rate_tol,
    )

    def _get_max_diff(central, low, high):
        if central is None:
            return None
        diffs: list[float] = []
        if low is not None:
            diffs.append(central - low)
        if high is not None:
            diffs.append(high - central)
        return max(diffs) if diffs else None

    out: dict[str, float | np.ndarray] = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": a_peak_lo,
        "a_peak_hi": a_peak_hi,
        "a_frz_lo": a_frz_lo,
        "a_frz_hi": a_frz_hi,
        "a_peak_err": _get_max_diff(a_peak, a_peak_lo, a_peak_hi),
        "a_frz_err": _get_max_diff(a_frz, a_frz_lo, a_frz_hi),
    }

    if return_full_grid:
        out["a_grid"] = hi_res_a
        out["y_grid"] = y_full_hi
        out["y_err"]  = y_err

    return out




# ------------------------------------------------------------
# NEW: Main interface for computing peak and freeze-out times
# ------------------------------------------------------------

# ------------------------------------------------------------
# NEW: Hybrid freeze-out definition combining amplitude and rate
# ------------------------------------------------------------
def compute_times_by_hybrid(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        abs_tol_frac: float = 0.05,
        rate_tol: float = 0.01,
        rate_sigma_factor: float = 2.0,
    ) -> dict[str, float | np.ndarray]:

    hi_res_a = _make_hi_res_grid(
        scale_factors.min(), 
        scale_factors.max(),
        n=n_hi_res
    )

    f_full = interp1d(
        scale_factors, full_evolution,
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    y_hi = f_full(hi_res_a)

    has_folds = fold_id_to_evolution is not None and len(fold_id_to_evolution) > 0

    if has_folds:
        folds = list(fold_id_to_evolution.values())
        _, y_err = _compute_jackknife_stats(folds, scale_factors, hi_res_a)
    else:
        y_inf = y_hi[-1]
        scale = abs(y_inf) if abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y_hi, rel_tol_no_folds * scale)

    # growth rate on hi-res grid
    y_clip = np.clip(y_hi, 1e-300, None)
    ln_a   = np.log(hi_res_a)
    ln_y   = np.log(y_clip)
    rate   = np.gradient(ln_y, ln_a)

    if has_folds:
        folds = list(fold_id_to_evolution.values())
        rate_mean_hi, rate_err_hi = _compute_jackknife_rate_stats(
            folds, scale_factors, hi_res_a
        )
    else:
        rate_err_hi = np.full_like(rate, rate_tol)

    y_inf  = y_hi[-1]
    abs_tol = abs_tol_frac * max(1.0, abs(y_inf))

    def _hybrid_frz(a_arr: np.ndarray, y_arr: np.ndarray) -> float | None:
        N = len(a_arr)
        for j in range(N):
            amp_ok  = abs(y_arr[j] - y_inf) <= abs_tol
            rate_ok = abs(rate[j]) <= rate_tol
            stat_ok = abs(rate[j]) <= rate_sigma_factor * rate_err_hi[j]

            if not (amp_ok and rate_ok and stat_ok):
                continue

            ok = True
            for m in range(j, N):
                amp_m  = abs(y_arr[m] - y_inf) <= abs_tol
                rate_m = abs(rate[m]) <= rate_tol
                stat_m = abs(rate[m]) <= rate_sigma_factor * rate_err_hi[m]
                if not (amp_m and rate_m and stat_m):
                    ok = False
                    break

            if ok:
                return a_arr[j]

        return None
    
    (
        a_peak,
        a_frz,
        a_peak_lo,
        a_peak_hi,
        a_frz_lo,
        a_frz_hi,
    ) = _bracket_peak_and_freeze_times(
        hi_res_a,
        y_hi,
        y_err,
        freeze_func=_hybrid_frz,
        tol=tol,
    )

    def _get_max_diff(central, low, high):
        if central is None:
            return None
        diffs = []
        if low is not None:
            diffs.append(central - low)
        if high is not None:
            diffs.append(high - central)
        return max(diffs) if diffs else None

    out: dict[str, float | np.ndarray] = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": a_peak_lo,
        "a_peak_hi": a_peak_hi,
        "a_frz_lo": a_frz_lo,
        "a_frz_hi": a_frz_hi,
        "a_peak_err": _get_max_diff(a_peak, a_peak_lo, a_peak_hi),
        "a_frz_err": _get_max_diff(a_frz, a_frz_lo, a_frz_hi),
    }

    if return_full_grid:
        out["a_grid"]    = hi_res_a
        out["y_grid"]    = y_hi
        out["y_err"]     = y_err
        out["rate"]      = rate
        out["rate_err"]  = rate_err_hi

    return out


# ------------------------------------------------------------
# HYBRID STRATEGY B — "MAX OF AMP AND RATE"
# ------------------------------------------------------------
def compute_times_by_hybrid_max(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        abs_tol_frac: float = 0.05,
        rate_tol: float = 0.01,
    ) -> dict[str, float | np.ndarray]:
    """
    Hybrid Strategy B:
    1) Compute freeze-out by evolution-based criterion.
    2) Compute freeze-out by rate-based criterion.
    3) Hybrid freeze-out = max(evolution_frz, rate_frz)
       (ensures consistency across diagnostics).
    """
    evo = compute_times_by_evolution(
        scale_factors, full_evolution, fold_id_to_evolution,
        tol, rel_tol_no_folds, n_hi_res, return_full_grid=False,
    )
    rate = compute_times_by_rate(
        scale_factors, full_evolution, fold_id_to_evolution,
        tol, rel_tol_no_folds, n_hi_res, return_full_grid=False,
        abs_tol_frac=abs_tol_frac, rate_tol=rate_tol
    )

    a_frz_e = evo["a_frz"]
    a_frz_r = rate["a_frz"]

    if a_frz_e is None and a_frz_r is None:
        a_frz = None
    elif a_frz_e is None:
        a_frz = a_frz_r
    elif a_frz_r is None:
        a_frz = a_frz_e
    else:
        a_frz = max(a_frz_e, a_frz_r)

    a_peak_e = evo["a_peak"]
    a_peak_r = rate["a_peak"]
    a_peak = a_peak_e if a_peak_e is not None else a_peak_r

    out = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": evo["a_peak_lo"],
        "a_peak_hi": evo["a_peak_hi"],
        "a_frz_lo": rate["a_frz_lo"],
        "a_frz_hi": rate["a_frz_hi"],
        "a_peak_err": evo["a_peak_err"],
        "a_frz_err": rate["a_frz_err"],
    }

    if return_full_grid:
        out["a_grid"] = evo.get("a_grid")
        out["y_grid"] = evo.get("y_grid")
        out["y_err"] = evo.get("y_err")

    return out


# ------------------------------------------------------------
# HYBRID STRATEGY C — STATISTICAL SIGNIFICANCE TEST
# ------------------------------------------------------------
def compute_times_by_hybrid_stat(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        abs_tol_frac: float = 0.05,
        rate_tol: float = 0.01,
        T_crit: float = 2.0,
        K: int = 5,
    ) -> dict[str, float | np.ndarray]:
    """
    Hybrid Strategy C:
    Freeze-out occurs when the joint significance metric
        T(a) = sqrt(((y - y_inf)/y_err)^2 + (rate/rate_err)^2)
    drops below T_crit and stays below for K future steps.
    """
    hi_res_a = _make_hi_res_grid(
        scale_factors.min(), scale_factors.max(), n=n_hi_res
    )
    f = interp1d(
        scale_factors, 
        full_evolution, 
        kind="linear", 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    y_hi = f(hi_res_a)

    has_folds = fold_id_to_evolution is not None and len(fold_id_to_evolution) > 0
    if has_folds:
        folds = list(fold_id_to_evolution.values())
        _, y_err = _compute_jackknife_stats(folds, scale_factors, hi_res_a)
        rate_mean_hi, rate_err = _compute_jackknife_rate_stats(folds, scale_factors, hi_res_a)
    else:
        y_inf = y_hi[-1]
        scale = np.abs(y_inf) if np.abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y_hi, rel_tol_no_folds * scale)
        rate_err = np.full_like(y_hi, rate_tol)

    y_clip = np.clip(y_hi, 1e-300, None)
    ln_a = np.log(hi_res_a)
    ln_y = np.log(y_clip)
    rate = np.gradient(ln_y, ln_a)
    y_inf = y_hi[-1]

    T = np.sqrt(((y_hi - y_inf) / y_err) ** 2 + (rate / rate_err) ** 2)
    N = len(T)

    a_frz = None
    for j in range(N - K):
        if T[j] < T_crit and np.all(T[j:j+K] < T_crit):
            a_frz = hi_res_a[j]
            break

    a_peak = _find_peak_time(hi_res_a, y_hi, y_err, tol=tol)
    if a_peak is None:
        a_peak = a_frz

    out = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": None,
        "a_peak_hi": None,
        "a_frz_lo": None,
        "a_frz_hi": None,
        "a_peak_err": None,
        "a_frz_err": None,
    }

    if return_full_grid:
        out["a_grid"] = hi_res_a
        out["y_grid"] = y_hi
        out["y_err"] = y_err
        out["T"] = T


    return out

# ------------------------------------------------------------
# HYBRID STRATEGY D — FINITE-WINDOW AMPLITUDE + SLOPE
# ------------------------------------------------------------
def compute_times_by_hybrid_window(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        amp_tol_frac: float = 0.05,
        diff_window: float = 0.3,
        diff_tol: float = 0.01,
        K: int = 3,
    ) -> dict[str, float | np.ndarray]:
    """Hybrid Strategy D: finite-window freeze-out.

    Combines:
      - amplitude closeness to the asymptotic value y_inf, and
      - small finite-difference change over a window in ln a,
    with a requirement that these conditions hold for K consecutive
    starting points.
    """

    # High-resolution grid
    hi_res_a = _make_hi_res_grid(scale_factors.min(), scale_factors.max(), n=n_hi_res)
    interp_f = interp1d(
        scale_factors, full_evolution, kind="linear",
        bounds_error=False, fill_value="extrapolate"
    )
    y_hi = interp_f(hi_res_a)

    # Amplitude errors
    has_folds = fold_id_to_evolution is not None and len(fold_id_to_evolution) > 0
    if has_folds:
        folds = list(fold_id_to_evolution.values())
        _, y_std_hi = _compute_jackknife_stats(folds, scale_factors, hi_res_a)
        y_err = y_std_hi
    else:
        y_inf = y_hi[-1]
        scale = np.abs(y_inf) if np.abs(y_inf) > 0 else 1.0
        y_err = np.full_like(y_hi, rel_tol_no_folds * scale)

    # Use the bracketing helper with the window-based freeze finder
    (
        a_peak,
        a_frz,
        a_peak_lo,
        a_peak_hi,
        a_frz_lo,
        a_frz_hi,
    ) = _bracket_peak_and_freeze_times(
        hi_res_a,
        y_hi,
        y_err,
        freeze_func=_find_freeze_out_time_by_window,
        tol=tol,
        amp_tol_frac=amp_tol_frac,
        diff_window=diff_window,
        diff_tol=diff_tol,
        K=K,
    )

    def _get_max_diff(central: float | None, low: float | None, high: float | None) -> float | None:
        if central is None:
            return None
        diffs: list[float] = []
        if low is not None:
            diffs.append(central - low)
        if high is not None:
            diffs.append(high - central)
        return max(diffs) if diffs else None

    out: dict[str, float | np.ndarray] = {
        "a_peak": a_peak,
        "a_frz": a_frz,
        "a_peak_lo": a_peak_lo,
        "a_peak_hi": a_peak_hi,
        "a_frz_lo": a_frz_lo,
        "a_frz_hi": a_frz_hi,
        "a_peak_err": _get_max_diff(a_peak, a_peak_lo, a_peak_hi),
        "a_frz_err": _get_max_diff(a_frz, a_frz_lo, a_frz_hi),
        "a_frz_err_hi": a_frz_hi - a_frz if ((a_frz is not None) and (a_frz_hi is not None)) else np.nan,
        "a_frz_err_lo": a_frz - a_frz_lo if ((a_frz is not None) and (a_frz_lo is not None)) else np.nan,
    }

    if return_full_grid:
        out["a_grid"] = hi_res_a
        out["y_grid"] = y_hi
        out["y_err"] = y_err

    return out


def compute_peak_and_freeze_times(
        scale_factors: np.ndarray,
        full_evolution: np.ndarray,
        fold_id_to_evolution: dict[int, np.ndarray] | None = None,
        tol: float = 2.0,
        rel_tol_no_folds: float = 0.02,
        n_hi_res: int = 1000,
        return_full_grid: bool = False,
        method: str = "evolution",
        **kwargs,
    ) -> dict[str, float | np.ndarray]:

    """
    Unified interface for computing peak and freeze‑out times.

    All sub‑methods are called explicitly with
    keyword arguments for clarity and traceability.
    """

    method = method.lower()

    # Extract commonly shared arguments
    base_args = dict(
        scale_factors=scale_factors,
        full_evolution=full_evolution,
        fold_id_to_evolution=fold_id_to_evolution,
        tol=tol,
        rel_tol_no_folds=rel_tol_no_folds,
        n_hi_res=n_hi_res,
        return_full_grid=return_full_grid,
    )

    # --- Evolution-based method ---
    if method == "evolution":
        return compute_times_by_evolution(**base_args)

    # --- Rate-based method ---
    elif method == "rate":
        return compute_times_by_rate(
            **base_args,
            abs_tol_frac=kwargs.get("abs_tol_frac", 0.05),
            rate_tol=kwargs.get("rate_tol", 0.01),
        )

    # --- Hybrid Strategy A (amp + rate consistency check) ---
    elif method == "hybrid":
        return compute_times_by_hybrid(
            **base_args,
            abs_tol_frac=kwargs.get("abs_tol_frac", 0.05),
            rate_tol=kwargs.get("rate_tol", 0.01),
            rate_sigma_factor=kwargs.get("rate_sigma_factor", 2.0),
        )

    # --- Hybrid Strategy B (max of evolution and rate) ---
    elif method == "hybrid_max":
        return compute_times_by_hybrid_max(
            **base_args,
            abs_tol_frac=kwargs.get("abs_tol_frac", 0.05),
            rate_tol=kwargs.get("rate_tol", 0.01),
        )

    # --- Hybrid Strategy C (joint significance metric) ---
    elif method == "hybrid_stat":
        return compute_times_by_hybrid_stat(
            **base_args,
            abs_tol_frac=kwargs.get("abs_tol_frac", 0.05),
            rate_tol=kwargs.get("rate_tol", 0.01),
            T_crit=kwargs.get("T_crit", 2.0),
            K=kwargs.get("K", 5),
        )

    # --- Hybrid Strategy D (finite-window amplitude + slope test) ---
    elif method == "hybrid_window":
        return compute_times_by_hybrid_window(
            **base_args,
            amp_tol_frac=kwargs.get("amp_tol_frac", 0.05),
            diff_window=kwargs.get("diff_window", 0.3),
            diff_tol=kwargs.get("diff_tol", 0.01),
            K=kwargs.get("K", 3),
        )

    else:
        raise ValueError(f"Unknown freeze-out method: {method}")




def get_freeze_out_times():
    pass  # TODO: build array versions here to call them later in AccumulationHistory

# Do this for the option to return freeze, peak, or both times. Then call in AccumulationHistory.


def print_freeze_out_time(results: dict[str, float | np.ndarray]) -> None:

    if (a_frz := results["a_frz"]) is None:
        print("  Freeze-out scale factor: Not detected")
        return

    if (results["a_frz_err"] is None):
        print(f"  Freeze-out scale factor: {a_frz:.4f} (no error estimate)")
    else:
        a_frz_error_lo = a_frz - results["a_frz_lo"] if results["a_frz_lo"] is not None else 0.0
        a_frz_error_hi = results["a_frz_hi"] - a_frz if results["a_frz_hi"] is not None else 0.0

        print(
            f" Freeze-out scale factor: {a_frz:.4f} "
            f"+{a_frz_error_hi:.4f}/-{a_frz_error_lo:.4f}"
        )


def print_peak_time(results: dict[str, float | np.ndarray]) -> None:
    if (a_peak := results["a_peak"]) is None:
        print("  Peak scale factor: Not detected")
        return

    if (results["a_peak_err"] is None):
        print(f"  Peak scale factor: {a_peak:.4f} (no error estimate)")
    else:
        a_peak_error_lo = a_peak - results["a_peak_lo"] if results["a_peak_lo"] is not None else 0.0
        a_peak_error_hi = results["a_peak_hi"] - a_peak if results["a_peak_hi"] is not None else 0.0

        print(
            f" Peak scale factor: {a_peak:.4f} "
            f"+{a_peak_error_hi:.4f}/-{a_peak_error_lo:.4f}"
        )

def print_peak_and_freeze_times(results: dict[str, float | np.ndarray]) -> None:
    print_peak_time(results)
    print_freeze_out_time(results)


def save_time_results_to_hdf5(
        time_results_array: dict[str, dict[str, np.ndarray]],
        file_path: Path
    ) -> None:
    """
    Save time results arrays to HDF5 file.
    """
    with h5py.File(file_path, "w") as hf:
        for mass_def, results in time_results_array.items():
            grp = hf.create_group(mass_def)
            for key, arr in results.items():
                grp.create_dataset(key, data=arr)

def load_time_results_from_hdf5(file_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """
    Load time results arrays from HDF5 file.
    """
    final = {}
    with h5py.File(file_path, "r") as hf:
        for mass_def in hf.keys():
            grp = hf[mass_def]
            results = {key: grp[key][()] for key in grp.keys()}
            final[mass_def] = results
    return final