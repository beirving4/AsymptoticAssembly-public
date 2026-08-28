import numpy as np

from scipy.interpolate import interp1d, UnivariateSpline

from ..cosmo.model import Cosmology
from ..simulation.moments import Moment
from .power import MatterPowerSpectrumEvoData
from ..particles.masses import get_particle_mass
from .knn import kNNDistributionEvoData, get_estimated_two_point_mass_range
from .correlation import (
    MatterTwoPointCorrelationData,
    TwoPointCorrelation,
    MatterTwoPointCorrelation,
    MatterTwoPointCorrelationData
)
from .treecorr_stitching import (
    StitchPlan,
    build_stitched_treecorr_bundle,
    recompute_dd_xi_and_cov,
)

def _pcdf_curve_for_box_at_snapshot(
        evo: kNNDistributionEvoData, 
        snap_idx: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:

    """
    Return (r, pcdf) for 2NN (k=2) from a kNNDistributionEvoData at a snapshot.
    Returns (None, None) if snapshot not present.
    """
    # kNNDistributionEvoData stores data: OrderedDict[int, kNNDistributionData]
    data = getattr(evo, "data", None)
    if data is None or snap_idx not in data:
        return None, None
    # kNNDistributionData.__getitem__(2) -> NNDistributionData (estimate/errors)
    if not data[snap_idx].is_k_available(2):
        return None, None
    nn2 = data[snap_idx][2]
    r = np.asarray(nn2.estimate.radii, dtype=float)
    pcdf = np.asarray(nn2.estimate.peaked_cdf, dtype=float)
    if r.size == 0 or pcdf.size == 0:
        return None, None
    # enforce strictly increasing r for interpolation robustness
    keep = np.isfinite(r) & np.isfinite(pcdf)
    r, pcdf = r[keep], pcdf[keep]
    if r.size == 0:
        return None, None
    order = np.argsort(r)
    return r[order], pcdf[order]

def _cic_prob_curve_for_box_at_snapshot(
        evo: kNNDistributionEvoData, 
        snap_idx: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:

    """
    Return (r, pcdf) for 2NN (k=2) from a kNNDistributionEvoData at a snapshot.
    Returns (None, None) if snapshot not present.
    """
    # kNNDistributionEvoData stores data: OrderedDict[int, kNNDistributionData]
    data = getattr(evo, "data", None)
    if data is None or snap_idx not in data:
        return None, None
    # kNNDistributionData.__getitem__(2) -> NNDistributionData (estimate/errors)
    if not data[snap_idx].is_k_available(2):
        return None, None
    nn2 = data[snap_idx][2]
    r = np.asarray(nn2.estimate.radii, dtype=float)
    cic_prob = np.asarray(nn2.estimate.cic_prob, dtype=float)
    if r.size == 0 or cic_prob.size == 0:
        return None, None
    # enforce strictly increasing r for interpolation robustness
    keep = np.isfinite(r) & np.isfinite(cic_prob)
    r, cic_prob = r[keep], cic_prob[keep]
    if r.size == 0:
        return None, None
    order = np.argsort(r)
    return r[order], cic_prob[order]


def _interp_y_of_logr(r: np.ndarray, y: np.ndarray):
    """Return y(r) interpolator on log(r) domain; NaN outside data range."""
    x = np.log(r)
    return interp1d(
        x, y, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True
    )

def _intersection_r_on_interval(
        r1: np.ndarray,
        y1: np.ndarray,
        r2: np.ndarray,
        y2: np.ndarray,
        r_lo: float,
        r_hi: float,
    ) -> float | None:
    """Like _intersection_r, but restricted to [r_lo, r_hi] (in r, not log r).

    Returns the first intersection within the closed interval if a sign change
    exists; otherwise None.
    """
    if not (np.isfinite(r_lo) and np.isfinite(r_hi) and (r_hi > r_lo)):
        return None

    lo = max(r1.min(), r2.min(), r_lo)
    hi = min(r1.max(), r2.max(), r_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi <= lo):
        return None

    f1 = _interp_y_of_logr(r1, y1)
    f2 = _interp_y_of_logr(r2, y2)

    xs = np.linspace(np.log(lo), np.log(hi), num=1025)
    d = f1(xs) - f2(xs)
    valid = np.isfinite(d)
    xs, d = xs[valid], d[valid]
    if d.size < 2:
        return None

    sign = np.sign(d)
    idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if idx.size == 0:
        return None

    i = int(idx[0])
    a, b = xs[i], xs[i + 1]
    fa, fb = d[i], d[i + 1]
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa == 0:
        return float(np.exp(a))
    if fb == 0:
        return float(np.exp(b))

    for _ in range(50):
        m = 0.5 * (a + b)
        fm = f1(m) - f2(m)
        if not np.isfinite(fm):
            a = np.nextafter(a, b)
            b = np.nextafter(b, a)
            continue
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
        if abs(b - a) < 1e-12:
            break
    return float(np.exp(0.5 * (a + b)))

def _threshold_crossings(
        r: np.ndarray, 
        pcdf: np.ndarray, 
        thr: float
    ) -> tuple[float | None, float | None]:
    """
    Return (r_low, r_high): smallest and largest r with pcdf(r) >= thr.
    Returns None for a bound if the threshold is never reached on that side.
    """
    mask = np.isfinite(pcdf)
    if not mask.any():
        return None, None
    r, y = r[mask], pcdf[mask]

    above = y >= thr
    if not above.any():
        return None, None

    idx = np.where(above)[0]
    i0, i1 = int(idx[0]), int(idx[-1])

    # linearly interpolate around crossings if interior
    def interp_cross(i_left: int, i_right: int) -> float:
        x0, y0 = r[i_left], y[i_left]
        x1, y1 = r[i_right], y[i_right]
        if not np.isfinite(y0) or not np.isfinite(y1) or y0 == y1:
            return float(x0)
        t = (thr - y0) / (y1 - y0)
        t = float(np.clip(t, 0.0, 1.0))
        return float(x0 + t * (x1 - x0))

    # low crossing: just before i0 is below (if exists), otherwise use r[i0]
    if i0 > 0 and y[i0 - 1] < thr <= y[i0]:
        r_low = interp_cross(i0 - 1, i0)
    else:
        r_low = float(r[i0])

    # high crossing: just after i1 is below (if exists), otherwise use r[i1]
    if i1 < y.size - 1 and y[i1] >= thr > y[i1 + 1]:
        r_high = interp_cross(i1, i1 + 1)
    else:
        r_high = float(r[i1])

    return r_low, r_high


def _threshold_bounds_for_curves(
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        pcdf_threshold: float,
    ) -> tuple[list[int], dict[int, tuple[float | None, float | None]]]:
    """Compute per-box threshold crossings and return valid boxes.

    Returns a tuple (valid_Ls, thr_bounds) where:
      - valid_Ls is a list of box sizes L that reach the threshold (lo, hi finite and hi > lo)
      - thr_bounds maps L -> (r_lo, r_hi) where PCDF >= threshold
    """
    Ls = sorted(curves.keys())
    valid: list[int] = []
    thr_bounds: dict[int, tuple[float | None, float | None]] = {}

    for L in Ls:
        r, y = curves[L]
        lo, hi = _threshold_crossings(r, y, pcdf_threshold)
        thr_bounds[L] = (lo, hi)
        if (lo is not None) and (hi is not None) and (hi > lo):
            valid.append(L)

    return valid, thr_bounds

def _has_above_threshold_overlap(
        La: int, Lb: int,
        thr_bounds: dict[int, tuple[float | None, float | None]],
    ) -> bool:
    lo_a, hi_a = thr_bounds[La]
    lo_b, hi_b = thr_bounds[Lb]
    overlap_lo = max(lo_a if lo_a is not None else -np.inf,
                     lo_b if lo_b is not None else -np.inf)
    overlap_hi = min(hi_a if hi_a is not None else np.inf,
                     hi_b if hi_b is not None else np.inf)
    return np.isfinite(overlap_lo) and np.isfinite(overlap_hi) and (overlap_hi > overlap_lo)


essential_drop_note = None


def _drop_for_pair(
        La: int, Lb: int,
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        thr_bounds: dict[int, tuple[float | None, float | None]],
        pcdf_threshold: float,
    ) -> int | None:
    """Return which box to drop for the adjacent pair (La, Lb), or None to keep both.

    Policy: only ever drop the lower-resolution (larger box) Lb, and only if there is
    an above-threshold overlap AND either there is no intersection or the intersection
    lies below the threshold.
    """
    if not _has_above_threshold_overlap(La, Lb, thr_bounds):
        return None

    ra, ya = curves[La]
    rb, yb = curves[Lb]
        # Restrict intersection search to the above-threshold overlap only
    lo_a, hi_a = thr_bounds[La]
    lo_b, hi_b = thr_bounds[Lb]
    overlap_lo = max(lo_a if lo_a is not None else -np.inf,
                     lo_b if lo_b is not None else -np.inf)
    overlap_hi = min(hi_a if hi_a is not None else np.inf,
                     hi_b if hi_b is not None else np.inf)
    rc = _intersection_r_on_interval(ra, ya, rb, yb, overlap_lo, overlap_hi)
    if rc is None:
        return Lb

    fa = _interp_y_of_logr(ra, ya)(np.log(rc))
    return Lb if not np.isfinite(fa) or float(fa) < pcdf_threshold else None


def _prune_until_adjacent_pairs_intersect(
        Ls: list[int],
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        thr_bounds: dict[int, tuple[float | None, float | None]],
        pcdf_threshold: float,
    ) -> list[int]:
    """Iteratively prune larger boxes until each adjacent pair intersects above threshold."""
    Ls = list(Ls)
    changed = True
    while changed and len(Ls) >= 2:
        changed = False
        drops: list[int] = []
        for i in range(len(Ls) - 1):
            La, Lb = Ls[i], Ls[i + 1]
            to_drop = _drop_for_pair(La, Lb, curves, thr_bounds, pcdf_threshold)
            if to_drop is not None:
                drops.append(to_drop)
        if drops:
            changed = True
            to_drop = set(drops)
            Ls = [L for L in Ls if L not in to_drop]
    return Ls

def _bounds_for_single_box(L: int, thr_bounds: dict[int, tuple[float | None, float | None]]) -> dict[int, tuple[float, float]]:
    """Return bounds dict for a single surviving box using its threshold crossings."""
    rlo, rhi = thr_bounds[L]
    if (rlo is not None) and (rhi is not None) and (rhi > rlo):
        return {L: (float(rlo), float(rhi))}
    return {}

def _compute_valid_intersections(
        Ls: list[int],
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        thr_bounds: dict[int, tuple[float | None, float | None]],
        pcdf_threshold: float,
    ) -> dict[tuple[int, int], float]:
    """Return {(La, Lb): r_intersect} for adjacent pairs with valid, above-threshold intersections.

    Only records intersections for consecutive boxes in `Ls` if:
      - The above-threshold regions overlap, and
      - The curves intersect within their shared radial support, and
      - The PCDF value at the intersection is ≥ `pcdf_threshold`.
    """
    intersections: dict[tuple[int, int], float] = {}
    for i in range(len(Ls) - 1):
        La, Lb = Ls[i], Ls[i + 1]
        ra, ya = curves[La]
        rb, yb = curves[Lb]

        lo_a, hi_a = thr_bounds[La]
        lo_b, hi_b = thr_bounds[Lb]
        overlap_lo = max(lo_a if lo_a is not None else -np.inf,
                         lo_b if lo_b is not None else -np.inf)
        overlap_hi = min(hi_a if hi_a is not None else np.inf,
                         hi_b if hi_b is not None else np.inf)
        has_overlap = np.isfinite(overlap_lo) and np.isfinite(overlap_hi) and (overlap_hi > overlap_lo)
        if not has_overlap:
            continue

        rc = _intersection_r_on_interval(ra, ya, rb, yb, overlap_lo, overlap_hi)
        if rc is None:
            continue
        fa = _interp_y_of_logr(ra, ya)(np.log(rc))
        if not np.isfinite(fa) or float(fa) < pcdf_threshold:
            continue
        intersections[(La, Lb)] = float(rc)
    return intersections

def _build_per_box_radii_bounds(
        Ls: list[int],
        thr_bounds: dict[int, tuple[float | None, float | None]],
        intersections: dict[tuple[int, int], float],
    ) -> dict[int, tuple[float, float]]:
    """Build per-box (r_min, r_max) bounds using threshold crossings and neighbor intersections.

    For each surviving box L in order:
      - start with threshold-defined [r_lo(L), r_hi(L)]
      - if a valid left intersection exists with the previous box, replace r_min
      - if a valid right intersection exists with the next box, replace r_max
      - keep the box only if r_max > r_min
    """
    bounds: dict[int, tuple[float, float]] = {}
    for i, L in enumerate(Ls):
        rlo, rhi = thr_bounds[L]
        if (rlo is None) or (rhi is None) or not (rhi > rlo):
            continue

        # Default to threshold-defined bounds
        r_min = float(rlo)
        r_max = float(rhi)

        # Use left intersection if valid
        if i > 0:
            rc_left = intersections.get((Ls[i - 1], L))
            if rc_left is not None:
                r_min = float(rc_left)

        # Use right intersection if valid
        if i < len(Ls) - 1:
            rc_right = intersections.get((L, Ls[i + 1]))
            if rc_right is not None:
                r_max = float(rc_right)

        if r_max > r_min:
            bounds[L] = (r_min, r_max)
    return bounds

def _build_bounds_for_snapshot(
        curves: dict[int, tuple[np.ndarray, np.ndarray]],  # L -> (r, pcdf)
        pcdf_threshold: float,
    ) -> dict[int, tuple[float, float]]:
    """
    Given curves for a single snapshot (box_size -> (r, pcdf)),
    iteratively prune lower-res boxes whose intersection with the next larger box
    is invalid or below threshold; then compute (r_min, r_max) for each remaining box.

    Returns: {L: (r_min, r_max)} for valid boxes only.
    """
    if not curves: return {}

    Ls, thr_bounds = _threshold_bounds_for_curves(curves, pcdf_threshold)

    if not Ls: return {}

    # iteratively prune until every adjacent pair intersects above threshold
    Ls = _prune_until_adjacent_pairs_intersect(
        Ls, curves, thr_bounds, pcdf_threshold
    )

    if not Ls: return {}

    # With a clean set of consecutive boxes, compute bounds
    if len(Ls) == 1: 
        return _bounds_for_single_box(Ls[0], thr_bounds)

    # Precompute consecutive intersections
    intersections = _compute_valid_intersections(Ls, curves, thr_bounds, pcdf_threshold)

    # Build per-box radii bounds
    return _build_per_box_radii_bounds(Ls, thr_bounds, intersections)



def get_convergence_bounds(
        sample_distributions: dict[int, kNNDistributionEvoData],
        particle_count: int, 
        threshold: float,
        cosmo: Cosmology,
        use_cic_probability: bool = False,
        use_uniform_sphere: bool = False
    ) -> dict[int, dict[int, dict[str, tuple[float, float]]]]:
    """
    For each snapshot, compute per-box (r_min, r_max) from intersections of 2NN Peaked CDFs
    (pruning lower-res boxes if intersections are below `pcdf_threshold`). Then provide
    wavenumber and Lagrangian mass bounds derived from those radii.

    Returns:
        {
          snap_id: {
            L: {
              "radii": (r_min, r_max),
              "wavenumber": (k_min, k_max),
              "mass": (m_min, m_max),
            }, ...
          }, ...
        }
    """
    # Collect union of snapshot ids across the provided boxes
    all_snaps: set[int] = set()
    for evo in sample_distributions.values():
        all_snaps.update(getattr(evo, "data", {}).keys())

    out: dict[int, dict[int, dict[str, tuple[float, float]]]] = {}

    for snap in sorted(all_snaps):
        # Build the curves (L -> (r, pcdf)) available at this snapshot
        curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for L, evo in sample_distributions.items():
            if use_cic_probability:
                r, curve = _cic_prob_curve_for_box_at_snapshot(evo, snap)
            else:
                r, curve = _pcdf_curve_for_box_at_snapshot(evo, snap)
            if r is not None:
                curves[L] = (r, curve)

        # Compute radii bounds (may prune some L)
        radii_bounds = _build_bounds_for_snapshot(curves, threshold)

        if not radii_bounds:
            # No usable bounds for this snapshot
            continue

        # Convert to wavenumbers and masses
        result_for_snap: dict[int, dict[str, tuple[float, float]]] = {}
        for L, (r_min, r_max) in radii_bounds.items():
            # k_min = 2π / r_max; k_max = 2π / r_min
            k_min = 2.0 * np.pi / r_max
            k_max = 2.0 * np.pi / r_min

            # Lagrangian mass in physical units; radii are comoving ⇒ pass a=1
            m_min, m_max = get_estimated_two_point_mass_range(
                r_min=r_min, 
                r_max=r_max, 
                cosmo=cosmo, 
                use_uniform_sphere=use_uniform_sphere
            )

            particle_mass = get_particle_mass(
                cosmo_name=cosmo.name,
                box_size=L,
                num_particles=particle_count, 
            )

            result_for_snap[L] = {
                "radii": (float(r_min), float(r_max)),
                "wavenumber": (float(k_min), float(k_max)),
                "mass": (m_min, m_max),
                "num_particles": (int(m_min / particle_mass), int(m_max / particle_mass)),
            }

        out[snap] = result_for_snap

    return out




# ================================================================
# Quantitative transition selection from precomputed power spectra
# ================================================================

def _logspace_overlap(k1: np.ndarray, k2: np.ndarray) -> tuple[float, float] | None:
    """Return (kmin, kmax) for the positive-overlap of two k-arrays in log-space."""
    k1 = np.asarray(k1, float)
    k2 = np.asarray(k2, float)
    if k1.size == 0 or k2.size == 0:
        return None
    kmin = max(np.nanmin(k1), np.nanmin(k2))
    kmax = min(np.nanmax(k1), np.nanmax(k2))
    if not (np.isfinite(kmin) and np.isfinite(kmax)):
        return None
    
    return None if (kmax <= kmin) else (float(kmin), float(kmax))


def _interp_logy_of_logx(x: np.ndarray, y: np.ndarray):
    """Interpolator for log(y) as a function of log(x) with NaN outside range."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return None
    order = np.argsort(x)
    lx = np.log(x[order])
    ly = np.log(y[order])
    return interp1d(
        lx, ly, 
        kind="linear", 
        bounds_error=False, 
        fill_value=np.nan, 
        assume_sorted=True
    )


def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    """Simple centered rolling mean with odd window length w; pads with NaN at edges."""
    y = np.asarray(y, float)
    if w <= 1:
        return y
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = np.ones(w) / float(w)
    z = np.convolve(np.where(np.isfinite(y), y, 0.0), kernel, mode='same')
    n = np.convolve(np.isfinite(y).astype(float), kernel, mode='same')
    out = z / np.where(n > 0, n, np.nan)
    # enforce NaN at boundaries where support is incomplete
    out[:pad] = np.nan
    out[-pad:] = np.nan
    return out


def estimate_nmodes(k: np.ndarray, box_size: float) -> np.ndarray:
    """Estimate number of Fourier modes per k-bin for a cubic box of side L.

    Assumes spherically averaged bins with widths taken from neighboring centers in log-space.
    N_modes(k) ≈ V/(2π^2) * k^2 * Δk.
    """
    k = np.asarray(k, float)
    if k.size < 2:
        return np.full_like(k, np.nan)
    # Δk from neighbors in linear space (more stable for this estimate)
    dk = np.empty_like(k)
    dk[1:-1] = 0.5 * (k[2:] - k[:-2])
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    V = box_size ** 3
    Nm = (V / (2.0 * np.pi**2)) * (k**2) * dk
    Nm[~np.isfinite(Nm) | (Nm <= 0)] = np.nan
    return Nm


def standardized_logdiff(k: np.ndarray, P1: np.ndarray, P2: np.ndarray, Nm1: np.ndarray, Nm2: np.ndarray) -> np.ndarray:
    """Return S(k) = |log P1 - log P2| / sqrt( 2/Nm1 + 2/Nm2 )."""
    k = np.asarray(k, float)
    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)
    Nm1 = np.asarray(Nm1, float)
    Nm2 = np.asarray(Nm2, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        lp1 = np.log(P1)
        lp2 = np.log(P2)
        var1 = 2.0 / Nm1
        var2 = 2.0 / Nm2
        S = np.abs(lp1 - lp2) / np.sqrt(var1 + var2)
    S[~np.isfinite(S)] = np.nan
    return S


def _slope_loglog(logk: np.ndarray, logP: np.ndarray) -> np.ndarray:
    """Finite-difference slope d logP / d logk with simple NaN handling."""
    s = np.gradient(logP, logk, edge_order=2)
    s[~np.isfinite(s)] = np.nan
    return s


def find_ps_transition(
        k_small: np.ndarray, P_small: np.ndarray, L_small: float,
        k_large: np.ndarray, P_large: np.ndarray, L_large: float,
        k_nyq_worse: float | None = None,
        S_thr: float = 2.0,
        tau: float = 0.2,
        smooth_w: int = 7,
    ) -> float:
    """Find quantitative transition wavenumber between two precomputed power spectra.

    The rule mirrors the kNN recipe: choose the largest k in the overlap where a short
    window satisfies both a standardized-difference threshold (S <= S_thr) and a
    slope-consistency threshold (|Δα| <= tau). Optionally cap at a fraction of the
    worse Nyquist via `k_nyq_worse` supplied by caller.

    Returns k_trans (np.nan if no valid overlap passes the criteria).
    """
    # Overlap domain
    ov = _logspace_overlap(k_small, k_large)
    if ov is None:
        return float('nan')
    kmin, kmax = ov

    if k_nyq_worse is not None and np.isfinite(k_nyq_worse):
        kmax = min(kmax, 0.5 * float(k_nyq_worse))
    if not (kmax > kmin):
        return float('nan')

    # Common grid in log k
    grid = np.logspace(np.log10(kmin), np.log10(kmax), num=512)
    f1 = _interp_logy_of_logx(k_small, P_small)
    f2 = _interp_logy_of_logx(k_large, P_large)
    if (f1 is None) or (f2 is None):
        return float('nan')
    lkg = np.log(grid)
    lp1 = f1(lkg)
    lp2 = f2(lkg)

    # Smooth logP
    lp1_s = _rolling_mean(lp1, smooth_w)
    lp2_s = _rolling_mean(lp2, smooth_w)

    # Slopes
    a1 = _slope_loglog(lkg, lp1_s)
    a2 = _slope_loglog(lkg, lp2_s)
    dalpha = np.abs(a1 - a2)

    # Standardized difference S(k)
    Nm1 = estimate_nmodes(grid, L_small)
    Nm2 = estimate_nmodes(grid, L_large)
    S = standardized_logdiff(grid, np.exp(lp1_s), np.exp(lp2_s), Nm1, Nm2)

    # Valid bins
    finite = np.isfinite(S) & np.isfinite(dalpha)
    if not finite.any():
        return float('nan')

    # Run-length rule: last index where a small window passes both thresholds
    w = max(3, smooth_w)
    if w % 2 == 0:
        w += 1
    half = w // 2
    ok = (S <= S_thr) & (dalpha <= tau) & finite

    # Rolling all-true window
    ok_int = ok.astype(int)
    rolling = np.convolve(ok_int, np.ones(w, dtype=int), mode='same')
    good = rolling >= w
    idx = np.where(good)[0]
    return float('nan') if idx.size == 0 else grid[idx.max()]



def get_ps_k_bounds(
        sample_distributions: dict[int, MatterPowerSpectrumEvoData],
        use_linear: bool = False,
        S_thr: float = 2.0,
        tau: float = 0.2,
        smooth_w: int = 7,
        k_nyq_cap_frac: float | None = 0.5,
    ) -> dict[int, dict[int, tuple[float, float]]]:
    """
    For each snapshot, compute per-box (k_min, k_max) using pairwise power-spectrum
    agreement between adjacent box sizes.

    Input
    -----
    sample_distributions : dict[int, MatterPowerSpectrumEvoData]
        Mapping {L : MatterPowerSpectrumEvoData} for multiple box sizes.
    use_linear : bool, optional
        If True, use linear P(k) curves; otherwise use nonlinear.
    S_thr, tau, smooth_w : float, int
        Parameters passed to `find_ps_transition`.
    k_nyq_cap_frac : float or None
        If provided, cap the upper bound for each box to `k_nyq_cap_frac * k_max_available`
        where `k_max_available` is the max k present in that box's spectrum at that snapshot.

    Output
    ------
    dict[int, dict[int, tuple[float, float]]]
        {snap_id: {L: (k_min, k_max)}}
    """
    # Collect union of snapshot ids across provided boxes
    all_snaps: set[int] = set()
    for evo in sample_distributions.values():
        all_snaps.update(getattr(evo, "data", {}).keys())

    out: dict[int, dict[int, tuple[float, float]]] = {}

    # Process per snapshot
    for snap in sorted(all_snaps):
        # Gather spectra for this snapshot
        curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        k_max_avail: dict[int, float] = {}
        k_min_avail: dict[int, float] = {}
        for L, evo in sample_distributions.items():
            data_dict = getattr(evo, "data", {})
            if snap not in data_dict:
                continue
            mpsd = data_dict[snap]  # MatterPowerSpectrumData
            ps = (mpsd.estimate.linear if use_linear else mpsd.estimate.nonlinear)
            k = np.asarray(ps.wavenumbers, dtype=float)
            P = np.asarray(ps.amplitudes, dtype=float)
            if k.size < 2 or not np.isfinite(k).any() or not np.isfinite(P).any():
                continue
            # Ensure strictly increasing k for interpolation
            mask = np.isfinite(k) & np.isfinite(P) & (k > 0) & (P > 0)
            if not mask.any():
                continue
            k, P = k[mask], P[mask]
            order = np.argsort(k)
            k, P = k[order], P[order]
            curves[L] = (k, P)
            k_min_avail[L] = float(k.min())
            k_max_avail[L] = float(k.max())

        if not curves:
            continue

        # Sort boxes by L (ascending): larger physical size later
        Ls = sorted(curves.keys())
        if len(Ls) == 1:
            L = Ls[0]
            k_lo = k_min_avail[L]
            k_hi = k_max_avail[L]
            if k_nyq_cap_frac is not None:
                k_hi = min(k_hi, k_nyq_cap_frac * k_hi)
            out.setdefault(snap, {})[L] = (float(k_lo), float(k_hi))
            continue

        # Compute transitions for adjacent pairs
        transitions: dict[tuple[int, int], float] = {}
        for i in range(len(Ls) - 1):
            La, Lb = Ls[i], Ls[i + 1]
            k_a, P_a = curves[La]
            k_b, P_b = curves[Lb]
            # worse Nyquist cap: use the smaller of the two max-k values
            k_nyq_worse = min(k_max_avail[La], k_max_avail[Lb]) if k_nyq_cap_frac is not None else None
            k_trans = find_ps_transition(
                k_small=k_a, P_small=P_a, L_small=float(La),
                k_large=k_b, P_large=P_b, L_large=float(Lb),
                k_nyq_worse=(k_nyq_worse if k_nyq_cap_frac is None else k_nyq_cap_frac * k_nyq_worse),
                S_thr=S_thr, tau=tau, smooth_w=smooth_w,
            )
            if np.isfinite(k_trans):
                transitions[(La, Lb)] = float(k_trans)

        # Assign bounds per box using transitions and available ranges
        bounds: dict[int, tuple[float, float]] = {}
        for i, L in enumerate(Ls):
            if i == 0:  # smallest L (finest kmax)
                # low-k bound from availability; upper from transition with next
                k_min = k_min_avail[L]
                k_max = transitions.get((L, Ls[i + 1]), k_max_avail[L])
            elif i == len(Ls) - 1:  # largest L (best low-k)
                # lower from transition with previous; high-k from availability
                k_min = transitions.get((Ls[i - 1], L), k_min_avail[L])
                k_max = k_max_avail[L]
            else:
                k_min = transitions.get((Ls[i - 1], L), k_min_avail[L])
                k_max = transitions.get((L, Ls[i + 1]), k_max_avail[L])

            # Enforce availability and optional Nyquist cap
            k_min = max(k_min, k_min_avail[L])
            k_max_cap = k_max_avail[L]
            if k_nyq_cap_frac is not None:
                k_max_cap = min(k_max_cap, k_nyq_cap_frac * k_max_cap)
            k_max = min(k_max, k_max_cap)

            if np.isfinite(k_min) and np.isfinite(k_max) and (k_max > k_min):
                bounds[L] = (float(k_min), float(k_max))

        if bounds:
            out[snap] = bounds

    return out


def _compute_local_overlap(
        r_target: float,
        all_boxes_r: list[np.ndarray],
        log_window: float = 0.1
    ) -> int:
    """
    Count how many boxes have data within a log-space window of a target radius.
    
    This measures local overlap at a specific radius, which indicates confidence
    in measurements at that location. Higher overlap suggests better convergence.
    
    Parameters
    ----------
    r_target : float
        Target radius to check overlap around
    all_boxes_r : list[np.ndarray]
        List of radius arrays from all boxes
    log_window : float, optional
        Half-width of log10 window for counting overlap (default: 0.1)
        Window of 0.1 corresponds to ~26% in linear space
    
    Returns
    -------
    int
        Number of boxes with data within log_window of r_target
    """
    if r_target <= 0:
        return 0
    
    log_r_target = np.log10(r_target)
    overlap_count = 0
    
    for r_box in all_boxes_r:
        # Check if this box has valid data near the target radius
        valid = (r_box > 0) & np.isfinite(r_box)
        if valid.any():
            log_r_box = np.log10(r_box[valid])
            # Count this box if it has any data within the window
            if np.any(np.abs(log_r_box - log_r_target) < log_window):
                overlap_count += 1
    
    return overlap_count


def _find_transition_point(
        r_large: np.ndarray,
        xi_large: np.ndarray,
        r_small: np.ndarray,
        xi_small: np.ndarray,
        deviation_threshold: float = 0.95,
        min_points: int = 5,
        use_absolute_difference: bool = False,
        abs_diff_threshold: float = 0.1,
        all_boxes_r: list[np.ndarray] | None = None,
        overlap_weight: float = 0.3,
    ) -> float:
    """
    Find the transition point where the larger box starts to deviate 
    from the smaller box.
    
    Uses adaptive comparison method:
    - For |xi| > 0.1: uses log-space differences (works across orders of magnitude)
    - For |xi| < 0.1 or negative values: uses absolute differences
    
    Optionally weights transition selection by local overlap count to prefer
    transitions in regions where more boxes agree (higher confidence).
    
    Parameters
    ----------
    r_large : np.ndarray
        Radii from the larger box
    xi_large : np.ndarray
        Correlation function from the larger box
    r_small : np.ndarray
        Radii from the smaller box
    xi_small : np.ndarray
        Correlation function from the smaller box
    deviation_threshold : float, optional
        Threshold for detecting deviation in log space
        Default is 0.95 (corresponding to ~5% in log space)
    min_points : int, optional
        Minimum number of points to check for consistent agreement
    use_absolute_difference : bool, optional
        Force use of absolute difference instead of adaptive method
    abs_diff_threshold : float, optional
        Threshold for absolute difference method (default 0.1)
    all_boxes_r : list[np.ndarray] | None, optional
        List of radius arrays from all boxes for computing overlap.
        If None, overlap weighting is disabled.
    overlap_weight : float, optional
        Weight for overlap preference (0.0-1.0). Higher values more strongly
        prefer transitions in high-overlap regions. Default is 0.3.
        
    Returns
    -------
    float
        The radius at which to transition from large to small box
    """
    # Filter out NaN and inf values from both arrays
    valid_large = ~(np.isnan(r_large) | np.isnan(xi_large) | np.isinf(r_large) | np.isinf(xi_large))
    valid_small = ~(np.isnan(r_small) | np.isnan(xi_small) | np.isinf(r_small) | np.isinf(xi_small))
    
    r_large = r_large[valid_large]
    xi_large = xi_large[valid_large]
    r_small = r_small[valid_small]
    xi_small = xi_small[valid_small]
    
    # Check if we have enough valid data points
    if len(r_large) < min_points or len(r_small) < min_points:
        # Not enough valid points, return midpoint of available data
        if len(r_large) > 0 and len(r_small) > 0:
            r_min = max(r_large.min(), r_small.min())
            r_max = min(r_large.max(), r_small.max())
            if r_max > r_min:
                return np.sqrt(r_min * r_max)
        if len(r_large) > 0:
            return r_large[len(r_large) // 2]
        elif len(r_small) > 0:
            return r_small[len(r_small) // 2]
        else:
            return 0.0
    
    # Find overlapping region
    r_min = max(r_large.min(), r_small.min())
    r_max = min(r_large.max(), r_small.max())
    
    if r_max <= r_min:
        # No overlap
        return np.sqrt(r_min * r_max) if r_min > 0 else 0.0
    
    # Get points in overlapping region from both arrays
    mask_large = (r_large >= r_min) & (r_large <= r_max)
    mask_small = (r_small >= r_min) & (r_small <= r_max)
    
    r_large_overlap = r_large[mask_large]
    xi_large_overlap = xi_large[mask_large]
    r_small_overlap = r_small[mask_small]
    xi_small_overlap = xi_small[mask_small]
    
    if len(r_large_overlap) < min_points:
        return np.sqrt(r_min * r_max)
    
    # Interpolate smaller box onto larger box radii in log-space for smoother comparison
    log_r_small = np.log(r_small_overlap)
    log_r_large = np.log(r_large_overlap)
    
    xi_small_interp_func = interp1d(
        log_r_small,
        xi_small_overlap,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    xi_small_interp = xi_small_interp_func(log_r_large)
    
    # Remove points where interpolation failed
    valid = np.isfinite(xi_small_interp)
    if not valid.any() or valid.sum() < min_points:
        return np.sqrt(r_min * r_max)
    
    r_compare = r_large_overlap[valid]
    xi_large_compare = xi_large_overlap[valid]
    xi_small_compare = xi_small_interp[valid]
    
    # Decide on comparison method based on typical |xi| values in overlap
    typical_abs_xi = np.median(np.abs([xi_large_compare, xi_small_compare]))
    has_negatives = (xi_large_compare < 0).any() or (xi_small_compare < 0).any()
    use_abs = use_absolute_difference or has_negatives or (typical_abs_xi < 0.1)
    
    if use_abs:
        # Use absolute difference for small or negative correlations
        diff = np.abs(xi_large_compare - xi_small_compare)
        # Adaptive threshold: scale with typical amplitude
        threshold = abs_diff_threshold * max(1.0, typical_abs_xi)
        agreement = diff <= threshold
    else:
        # Use log-space comparison for positive correlations with large values
        # This handles multiple orders of magnitude well
        with np.errstate(divide='ignore', invalid='ignore'):
            # Check both arrays are positive
            pos_mask = (xi_large_compare > 0) & (xi_small_compare > 0)
            if not pos_mask.any() or pos_mask.sum() < min_points:
                # Fall back to absolute difference
                diff = np.abs(xi_large_compare - xi_small_compare)
                threshold = abs_diff_threshold * max(1.0, typical_abs_xi)
                agreement = diff <= threshold
            else:
                # Compute log difference
                r_compare = r_compare[pos_mask]
                xi_large_compare = xi_large_compare[pos_mask]
                xi_small_compare = xi_small_compare[pos_mask]
                
                log_diff = np.abs(np.log(xi_large_compare) - np.log(xi_small_compare))
                # Convert deviation_threshold (e.g., 0.95) to log-space threshold
                # 0.95 ratio ≈ 0.05 in log space
                log_threshold = -np.log(deviation_threshold)
                agreement = log_diff <= log_threshold
    
    # Find the largest radius where we have consistent agreement
    # Use a sliding window to find regions of consistent agreement
    if len(agreement) < min_points:
        return np.sqrt(r_min * r_max)
    
    # Convolve with window to find runs of agreement
    window = np.ones(min_points, dtype=int)
    agreement_count = np.convolve(agreement.astype(int), window, mode='valid')
    
    # Find indices where all min_points in window agree
    good_regions = agreement_count >= min_points
    
    if good_regions.any():
        # Find the region of consistent agreement
        good_indices = np.where(good_regions)[0]
        
        # Within the good region, find the point of BEST agreement
        # Optionally weight by overlap count to prefer high-overlap regions
        best_agreement_idx = None
        min_score = np.inf
        
        # If overlap weighting is enabled, compute overlap for all candidates
        use_overlap = all_boxes_r is not None and overlap_weight > 0
        overlap_counts = []
        diff_metrics = []
        candidate_indices = []
        
        for idx in good_indices:
            # This index corresponds to the START of a good window
            # The actual transition point is at idx + min_points - 1
            transition_idx = idx + min_points - 1
            
            if transition_idx < len(r_compare):
                # Compute agreement metric at this point
                if use_abs:
                    metric = diff[transition_idx]
                else:
                    if transition_idx < len(agreement):
                        # For log-space, use the log difference
                        r_idx = transition_idx
                        if r_idx < len(xi_large_compare) and r_idx < len(xi_small_compare):
                            if xi_large_compare[r_idx] > 0 and xi_small_compare[r_idx] > 0:
                                metric = abs(np.log(xi_large_compare[r_idx]) - np.log(xi_small_compare[r_idx]))
                            else:
                                metric = abs(xi_large_compare[r_idx] - xi_small_compare[r_idx])
                        else:
                            continue
                    else:
                        continue
                
                diff_metrics.append(metric)
                candidate_indices.append(transition_idx)
                
                if use_overlap:
                    # Compute local overlap at this radius
                    overlap = _compute_local_overlap(
                        r_compare[transition_idx],
                        all_boxes_r,
                        log_window=0.1
                    )
                    overlap_counts.append(overlap)
        
        if len(candidate_indices) > 0:
            if use_overlap and len(overlap_counts) == len(diff_metrics):
                # Normalize both metrics to [0, 1]
                diff_array = np.array(diff_metrics)
                overlap_array = np.array(overlap_counts)
                
                diff_norm = (diff_array - diff_array.min()) / (diff_array.max() - diff_array.min() + 1e-10)
                overlap_norm = (overlap_array - overlap_array.min()) / (overlap_array.max() - overlap_array.min() + 1e-10)
                
                # Combined score: minimize difference, maximize overlap
                # Lower score is better
                scores = diff_norm - overlap_weight * overlap_norm
                best_idx = np.argmin(scores)
                best_agreement_idx = candidate_indices[best_idx]
            else:
                # No overlap weighting: use minimum difference
                best_idx = np.argmin(diff_metrics)
                best_agreement_idx = candidate_indices[best_idx]
            
            return float(r_compare[best_agreement_idx])
        
        # Fallback: use the largest radius with consistent agreement
        last_good_idx = good_indices[-1] + min_points - 1
        return float(r_compare[last_good_idx])
    
    # No consistent agreement found
    # Check if disagreement is at large or small scales
    if len(agreement) >= 2 * min_points:
        # Check if small scales agree better
        small_scale_agree = np.mean(agreement[-min_points:])
        large_scale_agree = np.mean(agreement[:min_points])
        
        if small_scale_agree > large_scale_agree:
            # Better agreement at small scales, be conservative
            # Use a point further into the overlap
            idx = min(len(r_compare) - 1, len(r_compare) * 2 // 3)
            return float(r_compare[idx])
    
    # Default: use geometric mean of overlap
    return np.sqrt(r_min * r_max)


def _stitch_arrays(
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        variable_name: str,
        transition_points: dict[int, tuple],
        return_box_labels: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Stitch together arrays from different boxes using transition points.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Dictionary of box_size -> variable_name -> array
    variable_name : str
        Name of the variable to stitch (e.g., 'xi', 'xi_err')
    transition_points : dict
        Dictionary of box_size -> (r_min, r_max) defining the range to use
    return_box_labels : bool, optional
        If True, also return array indicating which box each point came from
        
    Returns
    -------
    np.ndarray or tuple
        Stitched array, or (stitched_array, box_labels) if return_box_labels=True
    """
    stitched_segments = []
    box_labels = []
    
    for box_size in sorted(transition_points.keys(), reverse=True):
        r_min, r_max = transition_points[box_size]
        
        r = sample_tpcfs[box_size].nonlinear.estimate.radii

        match variable_name:
            case 'radii':
                variable = sample_tpcfs[box_size].nonlinear.estimate.radii
            case 'xi':
                variable = sample_tpcfs[box_size].nonlinear.estimate.correlation
            case 'xi_err':
                variable = sample_tpcfs[box_size].nonlinear.error.correlation

        # Select points in the range for this box
        mask = (r >= r_min) & (r <= r_max)
        n_points = np.sum(mask)
        
        stitched_segments.append(variable[mask])
        box_labels.extend([box_size] * n_points)
    
    if return_box_labels:
        return np.concatenate(stitched_segments), np.array(box_labels)
    
    return np.concatenate(stitched_segments)


def _compute_correlation_matrix(
    xi_err: np.ndarray,
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
    stitched_radii: np.ndarray,
    transition_points: dict[int, tuple]
) -> np.ndarray:
    """
    Compute correlation matrix for stitched arrays.
    
    For each stitched point, we use the correlation structure from the 
    original box it came from. Off-diagonal elements between different 
    boxes are set to zero (assuming independence).
    
    Parameters
    ----------
    xi_err : np.ndarray
        Stitched error array
    sample_tpcfs : dict
        Dictionary of box_size -> variable_name -> array
    stitched_radii : np.ndarray
        Stitched radii array
    transition_points : dict
        Dictionary of box_size -> (r_min, r_max) defining ranges
        
    Returns
    -------
    np.ndarray
        Correlation matrix for the stitched data
    """
    n_points = len(stitched_radii)
    corr_matrix = np.zeros((n_points, n_points))
    
    current_idx = 0
    
    for box_size in sorted(transition_points.keys(), reverse=True):

        r_min, r_max = transition_points[box_size]
        tpcf_data = sample_tpcfs[box_size].nonlinear

        r = tpcf_data.estimate.radii
        mask = (r >= r_min) & (r <= r_max)
        n_points_box = np.sum(mask)
        
        if n_points_box == 0:
            continue

        if tpcf_data.correlation_matrix is not None:
            indices = np.where(mask)[0]
            corr_submatrix = tpcf_data.correlation_matrix[np.ix_(indices, indices)]
        else:
            corr_submatrix = np.eye(n_points_box)
        
        # Insert into the full correlation matrix
        end_idx = current_idx + n_points_box
        corr_matrix[current_idx:end_idx, current_idx:end_idx] = corr_submatrix
        current_idx = end_idx
    
    return corr_matrix

def print_stitch_summary(
        stitched_tpcf: MatterTwoPointCorrelation,
        transition_points: dict[int, tuple],
        n_points_per_box: dict[int, int],
        box_labels: np.ndarray
    ) -> None:
    """
    Print a summary of the stitching results.
    
    Parameters
    ----------
    stitched_tpcf : MatterTwoPointCorrelation
        Stitched two-point correlation function data
    transition_points : dict[int, tuple]
        Dictionary of transition points for each box size
    """
    print("\n" + "="*60)
    print("TPCF STITCHING SUMMARY")
    print("="*60)
    
    # Transition points
    print("\nTransition Points (r_min, r_max):")
    print("-" * 40)
    for box_size, (r_min, r_max) in sorted(
        transition_points.items(), key=lambda x: x[0], reverse=True
    ):
        r_min_str = f"{r_min:.4f}" if r_min > 0 else "0.0"
        r_max_str = f"{r_max:.4f}" if r_max != np.inf else "inf"
        print(f"  L = {box_size:4d}: ({r_min_str:>10s}, {r_max_str:>10s})")
    
    # Points per box
    print("\nPoints Used Per Box:")
    print("-" * 40)
    total_points = len(stitched_tpcf.nonlinear.radii)
    for box_size, n_points in sorted(
        n_points_per_box.items(), key=lambda x: x[0], reverse=True
    ):
        percentage = 100 * n_points / total_points
        print(f"  L = {box_size:4d}: {n_points:4d} points ({percentage:5.1f}%)")
    
    print(f"\n  Total: {total_points} points")
    
    # Radii range for each box
    print("\nRadii Range Per Box (in stitched data):")
    print("-" * 40)
    for box_size in sorted(n_points_per_box, reverse=True):
        mask = box_labels == box_size
        if np.any(mask):
            r_box = stitched_tpcf.nonlinear.radii[mask]
            print(f"  L = {box_size:4d}: r = [{r_box.min():.4f}, {r_box.max():.4f}]")
    
    print("="*60 + "\n")


def _collect_tpcf_box_curves(sample_tpcfs, *, drop_tail_frac=0.0):
    """Collect per-box (r, xi, xi_err) arrays, sorted in r, with optional tail drop."""
    proc = []
    for box_size in sorted(sample_tpcfs.keys()):
        tpcf = sample_tpcfs[box_size]
        r = np.asarray(tpcf.nonlinear.estimate.radii, dtype=float)
        xi = np.asarray(tpcf.nonlinear.estimate.correlation, dtype=float)
        xi_err = np.asarray(tpcf.nonlinear.error.correlation, dtype=float)

        valid = np.isfinite(r) & np.isfinite(xi) & np.isfinite(xi_err) & (r > 0)
        r, xi, xi_err = r[valid], xi[valid], xi_err[valid]
        if r.size == 0:
            continue

        order = np.argsort(r)
        r, xi, xi_err = r[order], xi[order], xi_err[order]

        if drop_tail_frac and drop_tail_frac > 0:
            keep = int(np.floor((1.0 - drop_tail_frac) * len(r)))
            keep = max(keep, 1)
            r, xi, xi_err = r[:keep], xi[:keep], xi_err[:keep]

        proc.append({"box_size": int(box_size), "r": r, "xi": xi, "xi_err": xi_err})
    return proc


def _build_common_log_grid(proc, num_radii, r_range):
    """Return log-spaced r grid spanning all input curves (or provided range)."""
    if not proc:
        raise ValueError("No valid data found in any box")

    all_r = np.concatenate([np.asarray(p["r"], float) for p in proc])
    r_min, r_max = (np.min(all_r), np.max(all_r)) if r_range is None else r_range
    if not (np.isfinite(r_min) and np.isfinite(r_max) and (r_max > r_min)):
        raise ValueError("Invalid r_range for stitching")

    return np.logspace(np.log10(float(r_min)), np.log10(float(r_max)), num=int(num_radii))


def _interp_boxes_to_grid(proc, r_grid):
    """Interpolate each box's xi onto r_grid (log-log when xi>0, else linear)."""
    r_grid = np.asarray(r_grid, float)
    all_xi = []

    for p in proc:
        r = np.asarray(p["r"], float)
        xi = np.asarray(p["xi"], float)

        if (xi > 0).sum() >= 3:
            pos = xi > 0
            r_pos, xi_pos = r[pos], xi[pos]
            f = interp1d(
                np.log10(r_pos),
                np.log10(xi_pos),
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            all_xi.append(10.0 ** f(np.log10(r_grid)))
        else:
            f = interp1d(
                r,
                xi,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            all_xi.append(f(r_grid))

    return np.vstack(all_xi) if all_xi else np.empty((0, r_grid.size), dtype=float)


def _robust_upper_envelope(all_xi, *, clip_sigma, use_percentile, min_valid, overlap_power):
    """Return (xi_robust, weights, good_mask) computed per separation."""
    if all_xi.size == 0:
        raise ValueError("No interpolated data available for stitching")

    overlap_count = np.sum(np.isfinite(all_xi), axis=0)

    def mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))

    xi_robust = np.full(all_xi.shape[1], np.nan, dtype=float)
    quality_weights = np.zeros(all_xi.shape[1], dtype=float)

    for j in range(all_xi.shape[1]):
        col = all_xi[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue

        med = np.nanmedian(col)
        s = mad(col)
        if np.isfinite(s) and s > 0:
            col = col[np.abs(col - med) <= clip_sigma * s]

        if col.size >= min_valid:
            xi_robust[j] = np.nanpercentile(col, use_percentile)
            quality_weights[j] = float(overlap_count[j]) ** float(overlap_power)
        else:
            xi_robust[j] = med
            quality_weights[j] = 0.5

    good = np.isfinite(xi_robust) & np.isfinite(quality_weights) & (quality_weights > 0)
    return xi_robust, quality_weights, good


def _smooth_envelope_on_grid(r_grid, xi_robust, weights, good, *, smoothing_factor=0.0):
    """Return smoothed envelope xi_smooth defined on r_grid."""
    r_grid = np.asarray(r_grid, float)
    log_r = np.log10(r_grid)

    if int(np.sum(good)) < 3:
        raise ValueError("Insufficient valid data after filtering")

    w = np.asarray(weights[good], float)
    w = w / np.nanmax(w)

    if (xi_robust[good] > 0).sum() >= 3:
        spline = UnivariateSpline(
            x=log_r[good],
            y=np.log10(xi_robust[good]),
            w=w,
            s=(smoothing_factor if smoothing_factor > 0 else None),
        )
        return 10.0 ** spline(log_r)

    spline = UnivariateSpline(
        x=log_r[good],
        y=np.asarray(xi_robust[good], float),
        w=w,
        s=(smoothing_factor if smoothing_factor > 0 else None),
    )
    return spline(log_r)


def _flatten_original_points(proc):
    """Return concatenated (r, xi, xi_err, box_labels) arrays from proc."""
    rs, xs, es, bs = [], [], [], []
    for p in proc:
        L = int(p["box_size"])
        r = np.asarray(p["r"], float)
        xi = np.asarray(p["xi"], float)
        err = np.asarray(p["xi_err"], float)
        rs.extend(r.tolist())
        xs.extend(xi.tolist())
        es.extend(err.tolist())
        bs.extend([L] * int(r.size))
    return np.asarray(rs, float), np.asarray(xs, float), np.asarray(es, float), np.asarray(bs, int)


def _median_log_spacing(r, default=0.05):
    """Robust median spacing of log10(r)."""
    r = np.asarray(r, float)
    lr = np.log10(r[np.isfinite(r) & (r > 0)])
    lr = np.sort(lr)
    if lr.size < 5:
        return float(default)
    d = np.diff(lr)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float(default)
    m = float(np.nanmedian(d))
    return float(default) if (not np.isfinite(m) or m <= 0) else m


def _select_points_near_envelope(
    *,
    r_targets,
    xi_targets,
    all_r,
    all_xi,
    per_box_r,                 # list of per-box r arrays
    selection_overlap_weight,
    adapt_selection_window,
    selection_log_window_dex,
    selection_log_window_min_dex,
    selection_log_window_max_dex,
    med_dlog,
):
    """
    Select indices of original points that best trace the smooth envelope.

    This is the main place to fix the L=32 vs L=128 issue:
    - adapt the search window to the native r sampling (med_dlog)
    - avoid overly wide windows that let large-box points dominate transitions
    """
    selected = []
    use_overlap = selection_overlap_weight > 0

    for r_t, xi_t in zip(np.asarray(r_targets, float), np.asarray(xi_targets, float)):
        r_t = float(r_t)
        xi_t = float(xi_t)

        if adapt_selection_window:
            win = max(selection_log_window_min_dex, 3.0 * float(med_dlog))
            win = min(max(win, selection_log_window_dex), selection_log_window_max_dex)
        else:
            win = selection_log_window_dex

        mask = np.abs(np.log10(all_r) - np.log10(r_t)) < float(win)
        if not mask.any():
            continue

        # distance in (log r, log xi) when possible
        if (xi_t > 0) and (all_xi[mask] > 0).any():
            pos_mask = mask & (all_xi > 0)
            if not pos_mask.any():
                continue
            cand = np.where(pos_mask)[0]
            dist = np.sqrt(
                (np.log10(all_r[cand]) - np.log10(r_t)) ** 2
                + (np.log10(all_xi[cand]) - np.log10(xi_t)) ** 2
            )
        else:
            cand = np.where(mask)[0]
            r_norm = (all_r[cand] - r_t) / (r_t + 1e-12)
            denom = np.abs(xi_t) + 1e-12
            xi_norm = (all_xi[cand] - xi_t) / denom
            dist = np.sqrt(r_norm**2 + xi_norm**2)

        if use_overlap:
            # assumes you already have _compute_local_overlap(r_i, per_box_r, log_window=...)
            dist_norm = dist / (np.nanmax(dist) + 1e-12)
            ov = np.array([_compute_local_overlap(all_r[i], per_box_r, log_window=0.1) for i in cand], float)
            ov[~np.isfinite(ov)] = 0.0
            ov_pen = 1.0 - (ov / (np.nanmax(ov) + 1e-12))
            score = dist_norm + float(selection_overlap_weight) * ov_pen
            chosen = int(cand[np.nanargmin(score)])
        else:
            chosen = int(cand[np.nanargmin(dist)])

        selected.append(chosen)

    if not selected:
        raise ValueError("No points selected during contour stitching")

    return np.array(sorted(set(selected)), dtype=int)


# def _enforce_strictly_increasing_r(r, xi, err, box):
#     """Drop non-increasing r entries (including duplicates) to ensure strict monotonicity."""
#     r = np.asarray(r, float)
#     xi = np.asarray(xi, float)
#     err = np.asarray(err, float)
#     box = np.asarray(box, int)

#     if r.size < 2:
#         return r, xi, err, box

#     keep = np.ones_like(r, dtype=bool)
#     keep[1:] = np.diff(r) > 0
#     return r[keep], xi[keep], err[keep], box[keep]

def _enforce_strictly_increasing_r(r, xi, err, box, *, rtol=0.0, atol=0.0):
    """
    Enforce strictly increasing r.
    If duplicates exist, keep the one with the smallest fractional error |err/xi|.
    Optionally treat near-duplicates as duplicates via rtol/atol.
    """
    r   = np.asarray(r, float)
    xi  = np.asarray(xi, float)
    err = np.asarray(err, float)
    box = np.asarray(box, int)

    if r.size < 2:
        return r, xi, err, box

    # identify "same-r" groups
    # if rtol/atol are 0, this is exact duplicates; otherwise it's tolerant
    groups = []
    start = 0
    for i in range(1, r.size):
        same = np.isclose(r[i], r[i-1], rtol=rtol, atol=atol)
        if not same:
            groups.append((start, i))
            start = i
    groups.append((start, r.size))

    keep_idx = []
    eps = 1e-12
    for a, b in groups:
        if b - a == 1:
            keep_idx.append(a)
            continue
        # pick best in the duplicate group
        frac = np.abs(err[a:b]) / (np.abs(xi[a:b]) + eps)
        k = a + int(np.nanargmin(frac))
        keep_idx.append(k)

    keep_idx = np.array(keep_idx, dtype=int)

    # enforce strict increasing after collapsing
    order = np.argsort(r[keep_idx])
    keep_idx = keep_idx[order]

    return r[keep_idx], xi[keep_idx], err[keep_idx], box[keep_idx]


def _extract_stitched_folds(
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
    stitched_r: np.ndarray,
    box_labels: np.ndarray,
) -> np.ndarray | None:
    """Build a stitched folds array by looking up each stitched point's fold row.

    Parameters
    ----------
    sample_tpcfs : dict
        Dictionary of box_size -> MatterTwoPointCorrelationData (source boxes).
    stitched_r : np.ndarray
        Final stitched radii array (length N), after monotonicity enforcement.
    box_labels : np.ndarray
        Array indicating which box each stitched point came from (length N).

    Returns
    -------
    np.ndarray or None
        Shape ``(N, 1 + n_folds)`` where column 0 is ``stitched_r`` and columns
        ``1..n_folds`` are the per-fold xi values, or ``None`` if any source box
        lacks folds data or fold counts are inconsistent.

    Notes
    -----
    Each stitched point is a native (un-interpolated) radial bin from one specific
    box, so we locate it in that box's folds array via nearest-radius matching.
    The folds column 0 is the box's native radii; columns 1..n_folds are per-fold
    xi values.  We take the minimum fold count across all *used* boxes to handle
    the (unlikely) case of mismatched sub-box counts.
    """
    stitched_r = np.asarray(stitched_r, dtype=float)
    box_labels = np.asarray(box_labels, dtype=int)

    used_boxes = set(int(b) for b in box_labels)

    # --- Determine fold count and validate availability ------------------
    n_folds_per_box: dict[int, int] = {}
    for L in used_boxes:
        folds = sample_tpcfs[L].folds
        if folds is None:
            return None                       # can't stitch folds if any box lacks them
        n_folds_per_box[L] = int(folds.shape[1]) - 1   # subtract radii column

    if not n_folds_per_box:
        return None

    n_folds = min(n_folds_per_box.values())   # use minimum across boxes
    if n_folds < 1:
        return None

    # --- Pre-build per-box radii arrays for fast lookup -------------------
    box_folds_r: dict[int, np.ndarray] = {}
    box_folds_xi: dict[int, np.ndarray] = {}
    for L in used_boxes:
        folds = sample_tpcfs[L].folds
        box_folds_r[L] = np.asarray(folds[:, 0], dtype=float)
        box_folds_xi[L] = np.asarray(folds[:, 1 : 1 + n_folds], dtype=float)

    # --- Build the stitched folds array -----------------------------------
    out_xi = np.full((stitched_r.size, n_folds), np.nan, dtype=float)

    for i, (r_val, L) in enumerate(zip(stitched_r, box_labels)):
        L = int(L)
        # Nearest-radius match (exact for native points; tiny fp tolerance)
        idx = int(np.argmin(np.abs(box_folds_r[L] - r_val)))
        out_xi[i, :] = box_folds_xi[L][idx, :]

    return np.column_stack([stitched_r, out_xi])


def _print_contour_stitch_debug(stitched_r, box_labels):
    stitched_r = np.asarray(stitched_r, float)
    box_labels = np.asarray(box_labels, int)
    counts = {L: int(np.sum(box_labels == L)) for L in sorted(set(box_labels.tolist()))}

    print("\n" + "=" * 60)
    print("CONTOUR-BASED STITCHING SUMMARY")
    print("=" * 60)
    print(f"Total selected points: {int(stitched_r.size)}")
    print(f"Radius range: [{float(np.nanmin(stitched_r)):.4f}, {float(np.nanmax(stitched_r)):.4f}]")
    print("Points per box:")
    for L in sorted(counts.keys(), reverse=True):
        print(f"  L={int(L):4d}: {counts[L]:4d} points")
    print("=" * 60 + "\n")

def _anchor_tail_indices(
        all_r: np.ndarray,
        all_box: np.ndarray,
        *,
        L_min: int,
        L_max: int,
        tail_frac: float = 0.02,
        min_tail: int = 2,
    ) -> set[int]:
    """Return indices that force-include small-/large-r endpoints in the stitched selection.

    We force-include:
      - a small-r tail from the smallest (highest-resolution) box L_min
      - a large-r tail from the largest (highest-volume) box L_max

    This prevents the contour selector from accidentally dropping endpoints when the
    envelope-selection window misses the extreme bins.
    """
    all_r = np.asarray(all_r, dtype=float)
    all_box = np.asarray(all_box, dtype=int)

    forced: set[int] = set()

    def _take_tail(mask: np.ndarray, *, from_low: bool) -> None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        # sort these indices by radius
        order = np.argsort(all_r[idx])
        idx = idx[order]
        n = idx.size
        k = int(np.ceil(float(tail_frac) * n))
        k = max(int(min_tail), k)
        k = min(k, n)
        take = idx[:k] if from_low else idx[-k:]
        forced.update(int(i) for i in take)

    _take_tail(all_box == int(L_min), from_low=True)
    _take_tail(all_box == int(L_max), from_low=False)

    return forced

def _nearest_index_sorted(x_sorted: np.ndarray, x: float) -> int:
    """Return index of nearest value in a 1D sorted array."""
    x_sorted = np.asarray(x_sorted, dtype=float)
    if x_sorted.size == 0:
        return 0
    j = int(np.searchsorted(x_sorted, x, side="left"))
    if j <= 0:
        return 0
    if j >= x_sorted.size:
        return int(x_sorted.size - 1)
    return int(j - 1 if abs(x_sorted[j - 1] - x) <= abs(x_sorted[j] - x) else j)


def _box_corr_lookup(
    box_r: np.ndarray,
    box_corr: np.ndarray,
    r_i: float,
    r_j: float,
) -> tuple[float | None, float, float]:
    """
    Look up corr at (r_i, r_j) in a box by nearest-radius mapping.

    Returns (corr, mis_i, mis_j) where mis_* are absolute log-radius mismatches.
    If lookup is invalid, returns (None, inf, inf).
    """
    box_r = np.asarray(box_r, dtype=float)
    if box_r.size == 0 or box_corr is None:
        return None, float("inf"), float("inf")

    if not (np.isfinite(r_i) and np.isfinite(r_j)):
        return None, float("inf"), float("inf")
    if (r_i < box_r[0]) or (r_i > box_r[-1]) or (r_j < box_r[0]) or (r_j > box_r[-1]):
        return None, float("inf"), float("inf")

    ii = _nearest_index_sorted(box_r, float(r_i))
    jj = _nearest_index_sorted(box_r, float(r_j))

    try:
        c = float(box_corr[ii, jj])
    except Exception:
        return None, float("inf"), float("inf")
    if not np.isfinite(c):
        return None, float("inf"), float("inf")

    mi = abs(np.log(float(r_i)) - np.log(float(box_r[ii])))
    mj = abs(np.log(float(r_j)) - np.log(float(box_r[jj])))
    return c, float(mi), float(mj)


def build_correlation_matrix_from_boxes_blended(
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
    stitched_radii: np.ndarray,
    box_labels: np.ndarray,
    *,
    prefer_larger_box: bool = True,
    clip_corr: bool = True,
) -> np.ndarray:
    """
    Build a stitched correlation matrix by blending source-box correlation matrices.

    - If both points are from the same box and that box has a correlation matrix,
      use it directly.
    - Otherwise, consider boxes that cover both radii and blend their correlations
      using weights based on how close the stitched radii are to each box's native bins.
    - Optionally prefer the larger (or smaller) of the two contributing boxes.
    """
    stitched_radii = np.asarray(stitched_radii, dtype=float)
    box_labels = np.asarray(box_labels, dtype=int)

    n = int(stitched_radii.size)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # Gather per-box radii + correlation matrices
    box_r: dict[int, np.ndarray] = {}
    box_corr: dict[int, np.ndarray] = {}

    for L, tpcf in sample_tpcfs.items():
        try:
            rL = np.asarray(tpcf.nonlinear.estimate.radii, dtype=float)
        except Exception:
            rL = np.asarray(getattr(tpcf, "radii", []), dtype=float)

        C = getattr(tpcf, "correlation_matrix", None)
        if C is None:
            C = getattr(tpcf, "corr", None)

        if rL.size < 2 or C is None:
            continue

        order = np.argsort(rL)
        rL = rL[order]
        C = np.asarray(C, dtype=float)

        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            continue
        if C.shape[0] != order.size:
            continue

        C = C[np.ix_(order, order)]
        box_r[int(L)] = rL
        box_corr[int(L)] = C

    # If no usable matrices, return identity (best-effort)
    if not box_r:
        return np.eye(n, dtype=float)

    # Which boxes cover each stitched radius (simple range check)
    covers: list[list[int]] = []
    for r in stitched_radii:
        if not np.isfinite(r):
            covers.append([])
            continue
        eligible = [L for L, rL in box_r.items() if (r >= rL[0]) and (r <= rL[-1])]
        covers.append(sorted(eligible))

    out = np.zeros((n, n), dtype=float)
    np.fill_diagonal(out, 1.0)

    eps = 1e-12
    for i in range(n):
        ri = float(stitched_radii[i])
        Li = int(box_labels[i])

        for j in range(i + 1, n):
            rj = float(stitched_radii[j])
            Lj = int(box_labels[j])

            cand = set(covers[i]).intersection(covers[j])
            if not cand:
                out[i, j] = out[j, i] = 0.0
                continue

            if Li == Lj and Li in cand and Li in box_corr:
                cij, _, _ = _box_corr_lookup(box_r[Li], box_corr[Li], ri, rj)
                out[i, j] = out[j, i] = float(0.0 if cij is None else cij)
                continue

            preferred = max(Li, Lj) if prefer_larger_box else min(Li, Lj)
            if preferred not in cand:
                preferred = max(cand) if prefer_larger_box else min(cand)

            vals: list[float] = []
            wts: list[float] = []
            for L in sorted(cand):
                if L not in box_corr:
                    continue
                c, mi, mj = _box_corr_lookup(box_r[L], box_corr[L], ri, rj)
                if c is None:
                    continue
                w = 1.0 / (mi + mj + eps)
                if L == preferred:
                    w *= 1.5
                vals.append(float(c))
                wts.append(float(w))

            if not vals:
                cij = 0.0
            else:
                w = np.asarray(wts, dtype=float)
                v = np.asarray(vals, dtype=float)
                cij = float(np.sum(w * v) / np.sum(w))

            out[i, j] = out[j, i] = cij

    if clip_corr:
        out = np.clip(out, -1.0, 1.0)
        np.fill_diagonal(out, 1.0)

    return out


def _stitch_by_contour(
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
    cosmo: Cosmology,
    moment: Moment,
    num_radii: int = 50,
    r_range: tuple[float, float] | None = None,
    clip_sigma: float = 3.0,
    use_percentile: float = 95.0,
    min_valid: int = 2,
    drop_tail_frac: float = 0.05,
    smoothing_factor: float = 0.0,
    spline_overlap_power: float = 1.5,
    selection_overlap_weight: float = 0.4,
    num_subboxes: int = 8,
    print_debug_info: bool = True,
    *,
    # --- selection tuning knobs (helps L=32↔128 transitions) ---
    selection_log_window_dex: float = 0.15,
    selection_log_window_min_dex: float = 0.06,
    selection_log_window_max_dex: float = 0.35,
    adapt_selection_window: bool = True,
    # --- tail retention knobs ---
    anchor_endpoints: bool = True,
    anchor_tail_frac: float = 0.02,
    anchor_min_tail: int = 2,
    # --- corr-matrix behavior ---
    prefer_larger_box_for_corr: bool = True,
) -> MatterTwoPointCorrelationData:

    # 1) collect curves
    proc = _collect_tpcf_box_curves(sample_tpcfs, drop_tail_frac=drop_tail_frac)
    if not proc:
        raise ValueError("No valid data found in any box")

    # 2) common r grid + interpolate (for envelope finding only)
    r_span = _build_common_log_grid(proc, num_radii=num_radii, r_range=r_range)
    all_xi = _interp_boxes_to_grid(proc, r_span)

    # 3) robust envelope
    xi_robust, qweights, good = _robust_upper_envelope(
        all_xi,
        clip_sigma=clip_sigma,
        use_percentile=use_percentile,
        min_valid=min_valid,
        overlap_power=spline_overlap_power,
    )

    # 4) smooth envelope
    xi_smooth = _smooth_envelope_on_grid(
        r_span, xi_robust, qweights, good, smoothing_factor=smoothing_factor
    )

    # 5) select original points closest to envelope (preserves native errors!)
    all_r, all_x, all_e, all_box = _flatten_original_points(proc)
    per_box_r = [np.asarray(p["r"], float) for p in proc]
    med_dlog = _median_log_spacing(all_r, default=0.05)

    sel = _select_points_near_envelope(
        r_targets=np.asarray(r_span[good], float),
        xi_targets=np.asarray(xi_smooth[good], float),
        all_r=all_r,
        all_xi=all_x,
        per_box_r=per_box_r,
        selection_overlap_weight=selection_overlap_weight,
        adapt_selection_window=adapt_selection_window,
        selection_log_window_dex=selection_log_window_dex,
        selection_log_window_min_dex=selection_log_window_min_dex,
        selection_log_window_max_dex=selection_log_window_max_dex,
        med_dlog=float(med_dlog),
    )

    # 6) endpoint anchoring: prevent accidental tail clipping
    if anchor_endpoints:
        # expects you already have _anchor_tail_indices(all_r, all_box, ...)
        forced = _anchor_tail_indices(
            all_r,
            all_box,
            L_min=int(np.min(all_box)),
            L_max=int(np.max(all_box)),
            tail_frac=float(anchor_tail_frac),
            min_tail=int(anchor_min_tail),
        )
        if forced:
            sel = np.array(sorted(set(sel.tolist()) | forced), dtype=int)

    # sort selected points by r
    sel = np.asarray(sel, dtype=int)
    sel = sel[np.argsort(all_r[sel])]

    stitched_r = all_r[sel]
    stitched_x = all_x[sel]
    stitched_e = all_e[sel]
    box_labels = all_box[sel]

    # enforce monotonic r
    stitched_r, stitched_x, stitched_e, box_labels = _enforce_strictly_increasing_r(
        stitched_r, stitched_x, stitched_e, box_labels
    )

    # 6b) stitch folds (if all source boxes carry them)
    stitched_folds = _extract_stitched_folds(
        sample_tpcfs, stitched_r, box_labels,
    )

    # 7) correlation matrix
    # expects you already have build_correlation_matrix_from_boxes_blended(...)
    corr = build_correlation_matrix_from_boxes_blended(
        sample_tpcfs,
        stitched_r,
        box_labels,
        prefer_larger_box=prefer_larger_box_for_corr,
        clip_corr=True,
    )

    # 8) linear reference
    linear_radii = np.logspace(
        np.log10(max(float(np.nanmin(stitched_r)), 1e-3)),
        np.log10(min(float(np.nanmax(stitched_r)), 500.0)),
        1000,
    )

    estimate = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=cosmo.linear_matter_correlation(r=linear_radii, z=moment.redshift),
            is_linear=True,
            in_comoving=False,
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_r,
            correlation=stitched_x,
            is_linear=False,
            in_comoving=False,
        ),
    )

    error = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=np.zeros_like(linear_radii),
            is_linear=True,
            in_comoving=False,
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_r,
            correlation=stitched_e,
            is_linear=False,
            in_comoving=False,
        ),
    )

    if print_debug_info:
        _print_contour_stitch_debug(stitched_r, box_labels)

    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=error,
        correlation_matrix=corr,
        in_comoving=False,
        folds=stitched_folds,
    )


def check_transition_quality(
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        transition_points: dict[int, tuple],
        verbose: bool = True
    ) -> dict[str, float]:
    """
    Check the quality of transitions between boxes.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Input dictionary passed to get_joint_tpcf
    results : dict
        Output from get_joint_tpcf with return_debug_info=True
    verbose : bool, optional
        If True, print detailed information
        
    Returns
    -------
    dict
        Dictionary with transition quality metrics
    """
    box_sizes = sorted(sample_tpcfs.keys(), reverse=True)
    transition_quality = {}

    if verbose:
        print("\n" + "="*60)
        print("TRANSITION QUALITY CHECK")
        print("="*60)

    for i in range(len(box_sizes) - 1):
        large_box = box_sizes[i]
        small_box = box_sizes[i + 1]

        transition_r = transition_points[large_box][0]

        # Get xi values near transition from both boxes
        r_large = sample_tpcfs[large_box].estimate.nonlinear.radii
        xi_large = sample_tpcfs[large_box].estimate.nonlinear.correlation
        r_small = sample_tpcfs[small_box].estimate.nonlinear.radii
        xi_small = sample_tpcfs[small_box].estimate.nonlinear.correlation

        # Find closest points to transition
        idx_large = np.argmin(np.abs(r_large - transition_r))
        idx_small = np.argmin(np.abs(r_small - transition_r))

        # Compare values
        xi_large_at_trans = xi_large[idx_large]
        xi_small_at_trans = xi_small[idx_small]
        ratio = xi_large_at_trans / xi_small_at_trans

        transition_quality[f'{large_box}->{small_box}'] = {
            'transition_r': transition_r,
            'xi_large': xi_large_at_trans,
            'xi_small': xi_small_at_trans,
            'ratio': ratio,
        }

        if verbose:
            print(f"\nTransition: L={large_box} -> L={small_box}")
            print(f"  r = {transition_r:.4f}")
            print(f"  xi_large = {xi_large_at_trans:.4f}")
            print(f"  xi_small = {xi_small_at_trans:.4f}")
            print(f"  ratio = {ratio:.4f}")
            if ratio < 0.95:
                print(f"  ⚠ Large box underestimating by {(1-ratio)*100:.1f}%")
            elif ratio > 1.05:
                print(f"  ⚠ Large box overestimating by {(ratio-1)*100:.1f}%")
            else:
                print("  ✓ Good agreement")

    if verbose:
        print("="*60 + "\n")

    return transition_quality


def _stitch_by_deviation(
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        cosmo: Cosmology, 
        moment: Moment,
        deviation_threshold: float = 0.95,
        min_points: int = 5,
        num_subboxes: int = 8,
        print_debug_info: bool = True,
        use_absolute_difference: bool = False,
        abs_diff_threshold: float = 0.1,
        max_scale_fraction: float = 0.25,
        transition_overlap_weight: float = 1.0,
    ) -> MatterTwoPointCorrelationData:
    """
    Stitch correlation functions using deviation-based transition detection.
    
    This function combines TPCF measurements from simulations with different box sizes,
    transitioning from larger to smaller boxes as the spatial scale decreases. The 
    transition points are automatically detected based on where larger boxes begin to 
    deviate (underestimate) relative to smaller boxes.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Nested dictionary with structure: box_size -> MatterTwoPointCorrelationData
    deviation_threshold : float, optional
        Threshold for detecting when larger box underestimates (default: 0.95)
        The ratio xi_large/xi_small dropping below this triggers a transition
    min_points : int, optional
        Minimum consecutive points showing deviation before transitioning (default: 5)
    num_subboxes : int, optional
        Number of subboxes used in jackknife error estimation (default: 8)
    print_debug_info : bool, optional
        If True, print debugging information (default: True)
    use_absolute_difference : bool, optional
        Force use of absolute difference for all comparisons instead of adaptive method (default: False)
    abs_diff_threshold : float, optional
        Threshold for absolute difference method (default: 0.1)
    max_scale_fraction : float, optional
        Maximum fraction of box size to use for smallest box (default: 0.33)
        Prevents finite box size effects. Set to np.inf to disable.
    transition_overlap_weight : float, optional
        Weight for overlap preference in transition selection (0.0-1.0, default: 0.3)
        Higher values more strongly prefer transitions in high-overlap regions.
        Set to 0.0 to disable overlap weighting.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'radii': stitched radii array
        - 'xi': stitched correlation function
        - 'xi_err': stitched error array
        - 'correlation_matrix': correlation matrix for stitched data
        - 'transition_points': dict of box_size -> (r_min, r_max) [if return_debug_info=True]
        - 'box_labels': array indicating which box each point came from [if return_debug_info=True]
        - 'n_points_per_box': dict of box_size -> number of points used [if return_debug_info=True]
        
    Examples
    --------
    >>> sample_tpcfs = {
    ...     32: MatterTwoPointCorrelationData(radii=r32, xi=xi32, xi_err=err32, corr=corr32),
    ...     128: MatterTwoPointCorrelationData(radii=r128, xi=xi128, xi_err=err128, corr=corr128),
    ... }
    >>> results = get_joint_tpcf(sample_tpcfs)
    >>> plt.errorbar(
    ...     results.nonlinear.estimate.radii, 
    ...     results.nonlinear.estimate.correlation, 
            yerr=results.nonlinear.error.correlation
        )
    """
    # Sort box sizes from largest to smallest
    box_sizes = sorted(sample_tpcfs.keys(), reverse=True)
    
    if len(box_sizes) < 2:
        # Only one box, return as-is
        box_size = box_sizes[0]
        return sample_tpcfs[box_size]
    
    # Collect all radii arrays for overlap computation
    all_boxes_r = [sample_tpcfs[box].nonlinear.estimate.radii for box in box_sizes]
    
    # Find transition points between adjacent boxes
    transitions = []  # List of transition radii
    
    for i in range(len(box_sizes) - 1):
        
        large_box = box_sizes[i]
        small_box = box_sizes[i + 1]
        
        r_large = sample_tpcfs[large_box].nonlinear.estimate.radii
        xi_large = sample_tpcfs[large_box].nonlinear.estimate.correlation
        r_small = sample_tpcfs[small_box].nonlinear.estimate.radii
        xi_small = sample_tpcfs[small_box].nonlinear.estimate.correlation
        
        transition_r = _find_transition_point(
            r_large, xi_large, r_small, xi_small,
            deviation_threshold, min_points,
            use_absolute_difference, abs_diff_threshold,
            all_boxes_r=all_boxes_r if transition_overlap_weight > 0 else None,
            overlap_weight=transition_overlap_weight
        )
        
        # Enforce monotonic ordering: each transition should be smaller than previous
        # (going from large to small scales)
        if transitions and transition_r > transitions[-1]:
            # This transition is at a larger scale than the previous one
            # This means the middle box is unreliable or has poor overlap
            # Use the previous transition as upper bound
            transition_r = transitions[-1] * 0.95  # Slightly below to ensure valid range
        
        transitions.append(transition_r)
    
    # Build contiguous ranges for each box
    # Transitions are shared boundaries - ensures no gaps!
    transition_points = {}
    
    for i, box_size in enumerate(box_sizes):
        if i == 0:
            # Largest box: [transition[0], inf)
            r_min = transitions[0]
            r_max = np.inf
        elif i == len(box_sizes) - 1:
            # Smallest box: [0, transition[-1])
            r_min = 0.0
            r_max_original = transitions[-1]
            r_max = r_max_original
            
            # Additional validation for smallest box: detect divergence
            r_small = sample_tpcfs[box_size].nonlinear.estimate.radii
            xi_small = sample_tpcfs[box_size].nonlinear.estimate.correlation
            
            # Find where correlation starts trending upward (finite box effect)
            valid_mask = np.isfinite(r_small) & np.isfinite(xi_small) & (r_small > 0) & (r_small < r_max)
            if valid_mask.sum() >= 5:
                r_valid = r_small[valid_mask]
                xi_valid = xi_small[valid_mask]
                sort_idx = np.argsort(r_valid)
                r_sorted = r_valid[sort_idx]
                xi_sorted = xi_valid[sort_idx]
                
                # Only check positive correlation values
                if (xi_sorted > 0).sum() >= 5:
                    pos_mask = xi_sorted > 0
                    r_pos = r_sorted[pos_mask]
                    xi_pos = xi_sorted[pos_mask]
                    
                    # Check slope in windows
                    window = max(3, len(r_pos) // 5)
                    for idx in range(window, len(r_pos)):
                        if idx >= len(r_pos):
                            break
                        r_window = r_pos[max(0, idx-window):idx+1]
                        xi_window = xi_pos[max(0, idx-window):idx+1]
                        
                        if len(r_window) >= 3:
                            log_r = np.log(r_window)
                            log_xi = np.log(xi_window)
                            slope = (log_xi[-1] - log_xi[0]) / (log_r[-1] - log_r[0])
                            
                            # Cut off if trending significantly upward
                            if slope > 0.2:
                                cutoff_r = float(r_pos[max(0, idx-window)])
                                r_max = min(r_max, cutoff_r)
                                break
            
            # Apply physical constraint
            if np.isfinite(max_scale_fraction) and max_scale_fraction > 0:
                max_physical_r = box_size * max_scale_fraction
                r_max = min(r_max, max_physical_r)
            
            # If we constrained the smallest box, update the last transition to avoid gaps
            if r_max < r_max_original:
                transitions[-1] = r_max
                # Also need to update the second-smallest box's lower bound
                if len(box_sizes) > 2:
                    # Update the previous box's range (it's already been set)
                    prev_box = box_sizes[i - 1]
                    if prev_box in transition_points:
                        r_min_prev, r_max_prev = transition_points[prev_box]
                        transition_points[prev_box] = (r_max, r_max_prev)
        else:
            # Middle boxes: [transition[i], transition[i-1])
            r_min = transitions[i]
            r_max = transitions[i - 1]
            
            # Safety check: ensure valid range for middle boxes
            if r_max <= r_min:
                # Invalid range - give this box a narrow valid range
                # Use geometric mean of neighboring transitions
                if i < len(transitions):
                    r_mid = np.sqrt(abs(transitions[i-1] * transitions[i]))
                    r_min = r_mid * 0.98
                    r_max = r_mid * 1.02
                else:
                    # Fallback: tiny range
                    r_min = transitions[i]
                    r_max = r_min * 1.01
        
        transition_points[box_size] = (r_min, r_max)


    
    # Stitch together the arrays
    stitched_radii, box_labels = _stitch_arrays(
        sample_tpcfs, 
        'radii', 
        transition_points, 
        return_box_labels=True
    )

    if stitched_radii.size == 0:
        raise ValueError(
            "Stitched radii array is empty. Check transition points and input data."
        )
    
    stitched_xi = _stitch_arrays(sample_tpcfs, 'xi', transition_points)
    stitched_xi_err = _stitch_arrays(sample_tpcfs, 'xi_err', transition_points)

    # Build correlation matrix
    # Use improved correlation matrix that preserves structure from source boxes
    correlation_matrix = build_correlation_matrix_from_boxes(
        sample_tpcfs, stitched_radii, box_labels
    )

    linear_radii = np.logspace(
        np.log10(max(stitched_radii.min(), 1e-3)), 
        np.log10(min(stitched_radii.max(), 500.0)), 
        1000
    )

    estimate = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=cosmo.linear_matter_correlation(
                r=linear_radii,
                z=moment.redshift
            ),
            is_linear=True,
            in_comoving=False
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_radii,
            correlation=stitched_xi,
            is_linear=False,
            in_comoving=False
        )
    )

    error = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=np.zeros_like(linear_radii),
            is_linear=True,
            in_comoving=False
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_radii,
            correlation=stitched_xi_err,
            is_linear=False,
            in_comoving=False
        )
    )

    if print_debug_info:
        # Count points per box
        n_points_per_box = {}
        for box_size in sorted(sample_tpcfs.keys()):
            n_points_per_box[box_size] = np.sum(box_labels == box_size)

        print_stitch_summary(
            stitched_tpcf=estimate,
            transition_points=transition_points,
            n_points_per_box=n_points_per_box,
            box_labels=box_labels
        )

    
    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=error,
        correlation_matrix=correlation_matrix,
        in_comoving=False
    )


def _stitch_by_overlap(
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        cosmo: Cosmology,
        moment: Moment,
        num_radii: int = 50,
        r_range: tuple[float, float] | None = None,
        clip_sigma: float = 3.0,
        min_overlap: int = 2,
        drop_tail_frac: float = 0.05,
        overlap_weight_power: float = 2.0,
        smoothing_factor: float = 0.0,
        num_subboxes: int = 8,
        print_debug_info: bool = True,
    ) -> MatterTwoPointCorrelationData:
    """
    Stitch correlation functions by preferentially selecting regions with more box overlap.
    
    This method explicitly weights each radius by the number of boxes that have valid
    measurements there. Regions where multiple boxes agree indicate convergence and
    should be preferred.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Dictionary of box_size -> MatterTwoPointCorrelationData
    cosmo : Cosmology
        Cosmology object for linear correlation
    moment : Moment
        Moment object containing redshift information
    num_radii : int, optional
        Number of points in common radii grid (default: 500)
    r_range : tuple[float, float] | None, optional
        (r_min, r_max) range for stitching. If None, uses full range
    clip_sigma : float, optional
        MAD-based sigma clipping per radius (default: 3.0)
    min_overlap : int, optional
        Minimum number of boxes required for a radius to be included (default: 2)
    drop_tail_frac : float, optional
        Fraction of high-r tail to drop from each box (default: 0.05)
    overlap_weight_power : float, optional
        Exponent for overlap weighting (default: 2.0)
        Higher values more strongly prefer high-overlap regions
    smoothing_factor : float, optional
        Spline smoothing factor (default: 0.0)
    num_subboxes : int, optional
        Number of subboxes for error estimation (default: 8)
    print_debug_info : bool, optional
        Whether to print debug information (default: True)
    
    Returns
    -------
    MatterTwoPointCorrelationData
        Stitched correlation function data
    """
    # 1) Collect and process data from each box
    box_sizes = sorted(sample_tpcfs.keys())
    proc = []
    
    for box_size in box_sizes:
        r = sample_tpcfs[box_size].nonlinear.estimate.radii
        xi = sample_tpcfs[box_size].nonlinear.estimate.correlation
        xi_err = sample_tpcfs[box_size].nonlinear.error.correlation
        
        # Filter valid data
        valid = np.isfinite(r) & np.isfinite(xi) & (r > 0)
        r, xi, xi_err = r[valid], xi[valid], xi_err[valid]
        
        if len(r) == 0:
            continue
            
        # Sort by radius
        sort_idx = np.argsort(r)
        r, xi, xi_err = r[sort_idx], xi[sort_idx], xi_err[sort_idx]
        
        # Drop high-r tail
        if drop_tail_frac > 0:
            keep = int(np.floor((1.0 - drop_tail_frac) * len(r)))
            keep = max(keep, 1)
            r, xi, xi_err = r[:keep], xi[:keep], xi_err[:keep]
        
        proc.append({'box_size': box_size, 'r': r, 'xi': xi, 'xi_err': xi_err})
    
    if not proc:
        raise ValueError("No valid data found in any box")
    
    # 2) Build common radii grid
    all_r = np.concatenate([p['r'] for p in proc])
    r_min, r_max = (np.min(all_r), np.max(all_r)) if r_range is None else r_range
    r_span = np.logspace(np.log10(r_min), np.log10(r_max), num=num_radii)
    
    # 3) Interpolate all boxes onto common grid
    all_xi = []
    for p in proc:
        if (p['xi'] > 0).sum() >= 3:
            # Use log-log for positive correlations
            pos_mask = p['xi'] > 0
            r_pos, xi_pos = p['r'][pos_mask], p['xi'][pos_mask]
            
            f = interp1d(
                np.log10(r_pos),
                np.log10(xi_pos),
                kind='linear',
                bounds_error=False,
                fill_value=np.nan,
            )
            all_xi.append(10.0 ** f(np.log10(r_span)))
        else:
            # Use linear for negative correlations
            f = interp1d(
                p['r'],
                p['xi'],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan,
            )
            all_xi.append(f(r_span))
    
    all_xi = np.vstack(all_xi)  # shape: (n_boxes, num_radii)
    
    # 4) Compute overlap count at each radius (number of valid boxes)
    overlap_count = np.sum(np.isfinite(all_xi), axis=0)
    
    # 5) MAD-based clipping and weighted aggregation
    def mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))
    
    xi_weighted = np.full_like(r_span, np.nan, dtype=float)
    quality_score = np.zeros_like(r_span, dtype=float)
    
    for j in range(all_xi.shape[1]):
        col = all_xi[:, j]
        col = col[np.isfinite(col)]
        
        if col.size < min_overlap:
            continue
        
        # Sigma clipping
        med = np.nanmedian(col)
        s = mad(col)
        if np.isfinite(s) and s > 0:
            keep = np.abs(col - med) <= clip_sigma * s
            col = col[keep]
        
        if col.size < min_overlap:
            xi_weighted[j] = med
            quality_score[j] = 0.5  # Low quality
        else:
            # Weight by overlap count: more boxes agreeing = higher weight
            n_overlap = len(col)
            weight = n_overlap ** overlap_weight_power
            
            # Use median (robust) but track quality
            xi_weighted[j] = np.nanmedian(col)
            quality_score[j] = weight
    
    # 6) Smooth the weighted curve with quality-weighted spline
    log_r = np.log10(r_span)
    good = np.isfinite(xi_weighted) & (quality_score > 0)
    
    if good.sum() < 3:
        raise ValueError("Insufficient valid data after filtering")
    
    # Use quality scores as weights for spline fitting
    weights = quality_score[good]
    weights = weights / weights.max()  # Normalize to [0, 1]
    
    if (xi_weighted[good] > 0).sum() >= 3:
        # Log-log spline
        log_xi = np.log10(xi_weighted[good])
        spline = UnivariateSpline(
            x=log_r[good],
            y=log_xi,
            w=weights,
            s=(smoothing_factor if smoothing_factor > 0 else None),
        )
        xi_smooth = 10 ** spline(log_r)
    else:
        # Linear spline
        spline = UnivariateSpline(
            x=log_r[good],
            y=xi_weighted[good],
            w=weights,
            s=(smoothing_factor if smoothing_factor > 0 else None),
        )
        xi_smooth = spline(log_r)
    
    # 7) Select actual data points closest to smooth curve, preferring high-overlap regions
    all_data_r = []
    all_data_xi = []
    all_data_xi_err = []
    all_data_box = []
    all_data_overlap = []  # Track local overlap count for each point
    
    for p in proc:
        for r_val, xi_val, xi_err_val in zip(p['r'], p['xi'], p['xi_err']):
            # Count how many other boxes have data near this radius
            local_overlap = 0
            for p2 in proc:
                # Check if this box has data within 20% (in log space)
                if np.any(np.abs(np.log10(p2['r']) - np.log10(r_val)) < 0.1):
                    local_overlap += 1
            
            all_data_r.append(r_val)
            all_data_xi.append(xi_val)
            all_data_xi_err.append(xi_err_val)
            all_data_box.append(p['box_size'])
            all_data_overlap.append(local_overlap)
    
    all_data_r = np.array(all_data_r)
    all_data_xi = np.array(all_data_xi)
    all_data_xi_err = np.array(all_data_xi_err)
    all_data_box = np.array(all_data_box)
    all_data_overlap = np.array(all_data_overlap)
    
    # Select points closest to smooth curve, with overlap-based scoring
    selected_indices = []
    
    for i, (r_target, xi_target) in enumerate(zip(r_span[good], xi_smooth[good])):
        r_window = 0.15  # log10 window
        mask = (np.abs(np.log10(all_data_r) - np.log10(r_target)) < r_window)
        
        if not mask.any():
            continue
        
        # Compute combined score: distance + overlap bonus
        if xi_target > 0 and (all_data_xi[mask] > 0).any():
            pos_mask = mask & (all_data_xi > 0)
            if pos_mask.any():
                log_dist = np.sqrt(
                    (np.log10(all_data_r[pos_mask]) - np.log10(r_target))**2 +
                    (np.log10(all_data_xi[pos_mask]) - np.log10(xi_target))**2
                )
                # Normalize distance to [0, 1]
                log_dist_norm = log_dist / (log_dist.max() + 1e-10)
                
                # Overlap bonus: higher overlap = lower score
                overlap_bonus = 1.0 / (all_data_overlap[pos_mask] ** 0.5 + 1e-10)
                overlap_bonus = overlap_bonus / (overlap_bonus.max() + 1e-10)
                
                # Combined score: distance - overlap benefit
                score = log_dist_norm - 0.5 * (1 - overlap_bonus)  # Lower is better
                closest_idx = np.where(pos_mask)[0][np.argmin(score)]
            else:
                continue
        else:
            # Linear distance
            r_norm = (all_data_r[mask] - r_target) / (r_target + 1e-10)
            xi_norm = (all_data_xi[mask] - xi_target) / (abs(xi_target) + 1e-10)
            dist = np.sqrt(r_norm**2 + xi_norm**2)
            dist_norm = dist / (dist.max() + 1e-10)
            
            overlap_bonus = 1.0 / (all_data_overlap[mask] ** 0.5 + 1e-10)
            overlap_bonus = overlap_bonus / (overlap_bonus.max() + 1e-10)
            
            score = dist_norm - 0.5 * (1 - overlap_bonus)
            closest_idx = np.where(mask)[0][np.argmin(score)]
        
        if closest_idx not in selected_indices:
            selected_indices.append(closest_idx)
    
    # Sort by radius
    selected_indices = np.array(selected_indices)
    sort_order = np.argsort(all_data_r[selected_indices])
    selected_indices = selected_indices[sort_order]
    
    # Extract stitched data
    stitched_radii = all_data_r[selected_indices]
    stitched_xi = all_data_xi[selected_indices]
    stitched_xi_err = all_data_xi_err[selected_indices]
    box_labels = all_data_box[selected_indices]
    overlap_labels = all_data_overlap[selected_indices]
    
    # 8) Build correlation matrix
    # Use improved correlation matrix that preserves structure from source boxes
    correlation_matrix = build_correlation_matrix_from_boxes(
        sample_tpcfs, stitched_radii, box_labels
    )
    
    # 9) Create results
    linear_radii = np.logspace(
        np.log10(max(stitched_radii.min(), 1e-3)),
        np.log10(min(stitched_radii.max(), 500.0)),
        1000
    )
    
    estimate = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=cosmo.linear_matter_correlation(
                r=linear_radii,
                z=moment.redshift
            ),
            is_linear=True,
            in_comoving=False
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_radii,
            correlation=stitched_xi,
            is_linear=False,
            in_comoving=False
        )
    )
    
    error = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=linear_radii,
            correlation=np.zeros_like(linear_radii),
            is_linear=True,
            in_comoving=False
        ),
        nonlinear=TwoPointCorrelation(
            radii=stitched_radii,
            correlation=stitched_xi_err,
            is_linear=False,
            in_comoving=False
        )
    )
    
    if print_debug_info:
        n_points_per_box = {}
        for box_size in box_sizes:
            n_points_per_box[box_size] = np.sum(box_labels == box_size)
        
        print("\n" + "="*60)
        print("OVERLAP-WEIGHTED STITCHING SUMMARY")
        print("="*60)
        print(f"\nTotal selected points: {len(stitched_radii)}")
        print(f"Radius range: [{stitched_radii.min():.4f}, {stitched_radii.max():.4f}]")
        print(f"\nAverage overlap count: {overlap_labels.mean():.1f} boxes")
        print(f"Max overlap: {overlap_labels.max():.0f} boxes")
        print("\nPoints per box:")
        for box_size in sorted(box_sizes, reverse=True):
            n = n_points_per_box[box_size]
            pct = 100 * n / len(stitched_radii)
            print(f"  L = {box_size:4d}: {n:4d} points ({pct:5.1f}%)")
        print("="*60 + "\n")
    
    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=error,
        correlation_matrix=correlation_matrix,
        in_comoving=False
    )


def build_stitched_correlation_matrix(
    sample_tpcfs: dict,
    stitched_radii: np.ndarray,
    box_labels: np.ndarray,
    selected_indices_per_box: dict[int, np.ndarray] | None = None
) -> np.ndarray:
    """
    Build correlation matrix for stitched TPCF by extracting from source boxes.
    
    This properly preserves the correlation structure from each box's correlation
    matrix, rather than assuming independent errors.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Dictionary of box_size -> MatterTwoPointCorrelationData
    stitched_radii : np.ndarray
        Array of stitched radii (length N)
    box_labels : np.ndarray
        Array indicating which box each point came from (length N)
    selected_indices_per_box : dict[int, np.ndarray] | None
        Dictionary mapping box_size -> array of original indices in that box.
        If None, will attempt to reconstruct from stitched_radii and box_labels.
    
    Returns
    -------
    correlation_matrix : np.ndarray
        N x N correlation matrix for the stitched data
        
    Notes
    -----
    The correlation matrix has three types of elements:
    
    1. **Same box, same-scale correlations**: 
       If points i and j both come from box L, extract corr[i,j] directly from
       box L's correlation matrix.
    
    2. **Same box, different-scale correlations**:
       Also extracted from the box's correlation matrix (often strong anti-correlations
       visible in your plots).
    
    3. **Cross-box correlations**:
       Points from different boxes measuring similar scales should be correlated.
       Use a model based on:
       - Radial separation (closer separations → higher correlation)
       - Box size ratio (closer sizes → higher correlation)  
       - Sign of original correlations
       
    For now, we implement a conservative approach:
    - Within-box: use full correlation matrix
    - Cross-box: use a damped correlation based on scale similarity
    """
    n_points = len(stitched_radii)
    corr_matrix = np.zeros((n_points, n_points))
    
    # Get unique box sizes
    box_sizes = np.unique(box_labels)
    
    # If indices not provided, try to reconstruct them
    if selected_indices_per_box is None:
        selected_indices_per_box = _reconstruct_original_indices(
            sample_tpcfs, stitched_radii, box_labels
        )
    
    # Build index mapping: stitched_idx -> (box_size, original_idx)
    point_info = []
    for i in range(n_points):
        box_size = box_labels[i]
        if box_size in selected_indices_per_box:
            orig_idx = selected_indices_per_box[box_size][np.sum(box_labels[:i+1] == box_size) - 1]
        else:
            orig_idx = None
        point_info.append((box_size, orig_idx))
    
    # Fill correlation matrix
    for i in range(n_points):
        box_i, idx_i = point_info[i]
        
        for j in range(i, n_points):  # Only upper triangle (symmetric)
            box_j, idx_j = point_info[j]
            
            if box_i == box_j:
                # Same box - extract from source correlation matrix
                corr_value = _get_within_box_correlation(
                    sample_tpcfs[box_i], idx_i, idx_j
                )
            else:
                # Different boxes - model cross-box correlation
                corr_value = _estimate_cross_box_correlation(
                    sample_tpcfs, box_i, box_j, 
                    stitched_radii[i], stitched_radii[j],
                    idx_i, idx_j
                )
            
            corr_matrix[i, j] = corr_value
            corr_matrix[j, i] = corr_value  # Symmetric
    
    return corr_matrix


def _reconstruct_original_indices(
    sample_tpcfs: dict,
    stitched_radii: np.ndarray,
    box_labels: np.ndarray
) -> dict[int, np.ndarray]:
    """
    Reconstruct which original indices were selected from each box.
    
    This is done by matching stitched radii back to original radii arrays.
    """
    selected_indices = {}
    
    for box_size in np.unique(box_labels):
        # Get stitched points from this box
        mask = box_labels == box_size
        stitched_r_from_box = stitched_radii[mask]
        
        # Get original radii from this box
        original_r = sample_tpcfs[box_size].nonlinear.estimate.radii
        
        # Find matching indices (allowing small numerical tolerance)
        indices = []
        for r_stitch in stitched_r_from_box:
            idx = np.argmin(np.abs(original_r - r_stitch))
            # Verify it's a close match
            if np.abs(original_r[idx] - r_stitch) < 1e-6 * r_stitch:
                indices.append(idx)
            else:
                # No exact match - this shouldn't happen
                indices.append(-1)  # Flag as invalid
        
        selected_indices[box_size] = np.array(indices)
    
    return selected_indices


def _get_within_box_correlation(tpcf_data, idx_i: int, idx_j: int) -> float:
    """
    Extract correlation between two points from the same box.
    """
    if tpcf_data.correlation_matrix is None:
        # No correlation matrix available - assume independent
        return 1.0 if idx_i == idx_j else 0.0
    
    if idx_i < 0 or idx_j < 0:
        # Invalid indices
        return 1.0 if idx_i == idx_j else 0.0
    
    # Extract from correlation matrix
    corr_mat = tpcf_data.correlation_matrix
    
    if idx_i >= corr_mat.shape[0] or idx_j >= corr_mat.shape[1]:
        # Out of bounds
        return 1.0 if idx_i == idx_j else 0.0
    
    return corr_mat[idx_i, idx_j]


def _estimate_cross_box_correlation(
    sample_tpcfs: dict,
    box_i: int, box_j: int,
    r_i: float, r_j: float,
    idx_i: int, idx_j: int
) -> float:
    """
    Estimate correlation between points from different boxes.
    
    This is the tricky part. When box L1 and box L2 measure at similar radii,
    they're probing the same physical scales and should be correlated.
    
    Physical intuition:
    - Large-scale modes are shared between boxes (cosmic variance)
    - Small-scale modes are independent (shot noise, resolution effects)
    - The transition depends on the box sizes
    
    Conservative model:
    1. Compute scale similarity: exp(-|log(r_i/r_j)|^2 / σ^2)
    2. Damp by box size ratio: smaller box/larger box
    3. Reduce based on typical correlation strength in each box
    """
    # Scale similarity (how close are the radii?)
    log_r_diff = np.abs(np.log(r_i) - np.log(r_j))
    scale_similarity = np.exp(-log_r_diff**2 / 0.5)  # σ = 0.5 in log space
    
    if scale_similarity < 0.01:
        # Very different scales - independent
        return 0.0
    
    # Box size damping: smaller boxes have more independent noise
    L_min = min(box_i, box_j)
    L_max = max(box_i, box_j)
    box_damping = (L_min / L_max) ** 0.5
    
    # Get typical correlation strength from each box
    # (look at auto-correlation at similar lag)
    strength_i = _get_typical_correlation_strength(sample_tpcfs[box_i], idx_i)
    strength_j = _get_typical_correlation_strength(sample_tpcfs[box_j], idx_j)
    typical_strength = np.sqrt(strength_i * strength_j)
    
    # Combined cross-box correlation
    cross_corr = scale_similarity * box_damping * typical_strength
    
    # Clamp to reasonable range
    return np.clip(cross_corr, -0.5, 0.5)


def _get_typical_correlation_strength(tpcf_data, idx: int) -> float:
    """
    Get typical correlation strength at a given index.
    
    This is estimated by looking at nearby off-diagonal elements
    in the correlation matrix.
    """
    if tpcf_data.correlation_matrix is None or idx < 0:
        return 0.1  # Default weak correlation
    
    corr_mat = tpcf_data.correlation_matrix
    n = corr_mat.shape[0]
    
    if idx >= n:
        return 0.1
    
    # Look at correlations with nearby points (lag 1-5)
    nearby_corr = []
    for lag in range(1, min(6, n - idx)):
        if idx + lag < n:
            nearby_corr.append(abs(corr_mat[idx, idx + lag]))
    
    if len(nearby_corr) == 0:
        return 0.1
    
    # Return median absolute correlation
    return np.median(nearby_corr)


# Simple wrapper function that can be dropped into existing code
def build_correlation_matrix_from_boxes(
    sample_tpcfs: dict,
    stitched_radii: np.ndarray,
    box_labels: np.ndarray
) -> np.ndarray:
    """
    Convenience function matching the interface of existing code.
    
    Usage:
    ------
    Replace:
        cov_matrix = (1 - 1/num_subboxes) * stitched_xi_err.T @ stitched_xi_err
        correlation_matrix = cov_matrix / np.outer(stitched_xi_err, stitched_xi_err)
    
    With:
        correlation_matrix = build_correlation_matrix_from_boxes(
            sample_tpcfs, stitched_radii, box_labels
        )
    """
    return build_stitched_correlation_matrix(
        sample_tpcfs, stitched_radii, box_labels
    )


# --- Linear tail blend helper ---
def _apply_linear_tail_blend(
    *,
    stitched: MatterTwoPointCorrelationData,
    cosmo: Cosmology,
    moment: Moment,
    L_max: float | None,
    blend_start: float | None,
    blend_end: float | None,
    blend_start_frac: float = 0.04,
    blend_end_frac: float = 0.09,
    fit_offset: bool = True,
    fit_window: tuple[float, float] | None = None,
    use_log_space: bool = True,
) -> MatterTwoPointCorrelationData:
    """Post-process a stitched TPCF by smoothly blending its large-s tail to the linear prediction.

    Notes
    -----
    - This modifies ONLY the nonlinear *estimate* curve (i.e., mean). It preserves the
      stitched radii, the stitched nonlinear errors, and the stitched correlation matrix.
    - The blend weight ramps from 0 -> 1 between (blend_start, blend_end) using a smoothstep.
    - Optionally fits a constant offset C so that xi_lin(s)+C matches xi_stitched(s) on a
      pre-blend overlap window, which is often a proxy for an integral-constraint-like shift.
    """
    # Access the stitched nonlinear arrays
    r = stitched.nonlinear.estimate.radii
    xi = stitched.nonlinear.estimate.correlation

    # Defensive checks
    if r is None or xi is None or len(r) == 0:
        return stitched

    r = np.asarray(r, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if r.size != xi.size:
        return stitched

    # Determine L_max (largest box) if not provided
    if L_max is None or not np.isfinite(L_max):
        # fall back: use the largest separation scale in stitched curve as a scale proxy
        L_max = float(np.nanmax(r) / 0.49) if np.isfinite(np.nanmax(r)) else None

    # Compute default start/end from L_max, with a cap at 0.49 L_max
    if L_max is not None and np.isfinite(L_max):
        r_cap = 0.49 * float(L_max)
        s1 = float(blend_start) if blend_start is not None else float(blend_start_frac) * float(L_max)
        s2 = float(blend_end) if blend_end is not None else float(blend_end_frac) * float(L_max)
        s1 = min(s1, r_cap)
        s2 = min(s2, r_cap)
    else:
        # no box-size anchor; pick a conservative default based on r extent
        r_max = float(np.nanmax(r))
        s1 = float(blend_start) if blend_start is not None else 0.25 * r_max
        s2 = float(blend_end) if blend_end is not None else 0.50 * r_max

    # If the window is degenerate, do nothing
    if not (np.isfinite(s1) and np.isfinite(s2)):
        return stitched
    if s2 <= s1:
        return stitched

    # Linear prediction evaluated on the stitched radii (same bins)
    xi_lin = np.asarray(
        cosmo.linear_matter_correlation(r=r, z=moment.redshift),
        dtype=float,
    )

    # Optional constant offset fit on a pre-blend overlap window
    C = 0.0
    if fit_offset:
        if fit_window is None:
            lo = max(float(np.nanmin(r)), 0.6 * s1)
            hi = s1
            fit_window = (lo, hi)
        lo, hi = float(fit_window[0]), float(fit_window[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            m = (r >= lo) & (r <= hi) & np.isfinite(xi) & np.isfinite(xi_lin)
            if np.count_nonzero(m) >= 3:
                # robust location estimate
                C = float(np.nanmedian(xi[m] - xi_lin[m]))

    xi_target = xi_lin + C

    # Smoothstep weight from 0->1 between s1 and s2
    w = np.zeros_like(r, dtype=float)
    if use_log_space:
        lr = np.log10(np.clip(r, 1e-300, np.inf))
        l1 = np.log10(max(s1, 1e-300))
        l2 = np.log10(max(s2, 1e-300))
        t = (lr - l1) / (l2 - l1)
    else:
        t = (r - s1) / (s2 - s1)

    t = np.clip(t, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)  # smoothstep

    # Blend only where both are finite
    xi_new = np.array(xi, copy=True)
    good = np.isfinite(xi) & np.isfinite(xi_target) & np.isfinite(w)
    xi_new[good] = (1.0 - w[good]) * xi[good] + w[good] * xi_target[good]

    # Rebuild the estimate object while preserving errors and correlation matrix
    estimate = MatterTwoPointCorrelation(
        linear=stitched.estimate.linear,
        nonlinear=TwoPointCorrelation(
            radii=r,
            correlation=xi_new,
            is_linear=False,
            in_comoving=False,
        ),
    )

    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=stitched.error,
        correlation_matrix=stitched.correlation_matrix,
        in_comoving=False,
        folds=stitched.folds,
    )


def _apply_max_sep_cut(
    tpcf: MatterTwoPointCorrelationData,
    *,
    max_sep: float | None = None,
    max_sep_frac_of_box: float | None = None,
    box_size_for_frac: int | None = None,
    keep_at_least: int = 8,
    print_debug_info: bool = False,
) -> MatterTwoPointCorrelationData:
    """
    Hard-trim the *nonlinear* stitched TPCF beyond a maximum separation.

    This is a simple alternative to linear-tail blending:
      - keep nonlinear (measured) xi(s) up to s_max
      - discard bins at s > s_max (often where ringing / integral-constraint / finite-volume effects appear)
      - slice the correlation matrix consistently

    Cutoff choice:
      - Provide max_sep directly, OR
      - Provide max_sep_frac_of_box * box_size_for_frac (e.g. 0.49 * L_max)
    """
    if tpcf is None:
        raise ValueError("tpcf is None")

    # Determine s_max
    s_max: float | None = None
    if max_sep is not None:
        s_max = float(max_sep)
    elif (max_sep_frac_of_box is not None) and (box_size_for_frac is not None):
        s_max = float(max_sep_frac_of_box) * float(box_size_for_frac)

    # Nothing to do
    if s_max is None or not np.isfinite(s_max):
        return tpcf

    # Pull nonlinear arrays
    r_nl = np.asarray(tpcf.estimate.nonlinear.radii, dtype=float)
    xi_nl = np.asarray(tpcf.estimate.nonlinear.correlation, dtype=float)
    err_nl = np.asarray(tpcf.error.nonlinear.correlation, dtype=float)

    valid = np.isfinite(r_nl) & np.isfinite(xi_nl) & np.isfinite(err_nl) & (r_nl > 0)

    # Apply cutoff
    keep = valid & (r_nl <= s_max)

    # Ensure we don’t accidentally keep too few bins
    if int(np.sum(keep)) < int(keep_at_least):
        order = np.argsort(r_nl[valid])
        idx_valid = np.where(valid)[0][order]
        k = min(int(keep_at_least), int(idx_valid.size))
        keep = np.zeros_like(r_nl, dtype=bool)
        keep[idx_valid[:k]] = True

    # If no actual cut occurred, return as-is
    if np.all(keep[valid]):
        return tpcf

    # Slice nonlinear arrays
    r_new = r_nl[keep]
    xi_new = xi_nl[keep]
    err_new = err_nl[keep]

    # Slice correlation matrix if it matches nonlinear length
    C = getattr(tpcf, "correlation_matrix", None)
    C_new = None
    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.ndim == 2 and C.shape[0] == C.shape[1] and C.shape[0] == r_nl.size:
            idx = np.where(keep)[0]
            C_new = C[np.ix_(idx, idx)]

    # Trim linear arrays for plotting consistency (optional)
    r_lin = np.asarray(tpcf.estimate.linear.radii, dtype=float)
    xi_lin = np.asarray(tpcf.estimate.linear.correlation, dtype=float)
    lin_keep = (
        np.isfinite(r_lin) & np.isfinite(xi_lin) & (r_lin > 0)
        & (r_lin <= float(np.nanmax(r_new)))
    )
    if int(np.sum(lin_keep)) >= 3:
        r_lin_new = r_lin[lin_keep]
        xi_lin_new = xi_lin[lin_keep]
    else:
        r_lin_new = r_lin
        xi_lin_new = xi_lin

    estimate = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=r_lin_new,
            correlation=xi_lin_new,
            is_linear=True,
            in_comoving=tpcf.estimate.linear.in_comoving,
        ),
        nonlinear=TwoPointCorrelation(
            radii=r_new,
            correlation=xi_new,
            is_linear=False,
            in_comoving=tpcf.estimate.nonlinear.in_comoving,
        ),
    )

    error = MatterTwoPointCorrelation(
        linear=TwoPointCorrelation(
            radii=r_lin_new,
            correlation=np.zeros_like(r_lin_new),
            is_linear=True,
            in_comoving=tpcf.error.linear.in_comoving,
        ),
        nonlinear=TwoPointCorrelation(
            radii=r_new,
            correlation=err_new,
            is_linear=False,
            in_comoving=tpcf.error.nonlinear.in_comoving,
        ),
    )

    # Slice folds if present
    folds_new = None
    if tpcf.folds is not None:
        folds_arr = np.asarray(tpcf.folds, dtype=float)
        if folds_arr.shape[0] == r_nl.size:
            folds_new = folds_arr[keep, :]

    if print_debug_info:
        print("\n" + "=" * 60)
        print("MAX-SEPARATION CUTOFF POST-PROCESSING")
        print("=" * 60)
        print(f"Requested cutoff s_max = {s_max:.4g}")
        print(f"Nonlinear bins before: {r_nl.size} | after: {r_new.size}")
        print(f"r range before: [{np.nanmin(r_nl):.4g}, {np.nanmax(r_nl):.4g}]")
        print(f"r range after : [{np.nanmin(r_new):.4g}, {np.nanmax(r_new):.4g}]")
        print("=" * 60 + "\n")

    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=error,
        correlation_matrix=(C_new if C_new is not None else tpcf.correlation_matrix),
        in_comoving=tpcf.in_comoving,
        folds=folds_new,
    )


def _extract_r_x_e_from_tpcf_obj(tpcf_obj):
    """Best-effort extraction of (r, xi, err) arrays from a TPCF-like object.

    Supports (duck-typed):
      - MatterTwoPointCorrelationData: uses .nonlinear (TwoPointCorrelationData) when available
      - TwoPointCorrelationData: uses .radii/.xi/.sigma
      - estimate/error containers with .radii/.correlation (or .xi)
      - legacy .to_numpy fallback (either (3,N) or (N,3))
    """
    import numpy as np

    def _pick(o, names):
        for n in names:
            v = getattr(o, n, None)
            if v is not None:
                return v
        return None

    # -------------------------------
    # 0) Prefer "nonlinear" data path
    # -------------------------------
    nl = getattr(tpcf_obj, "nonlinear", None)
    if nl is not None:
        # TwoPointCorrelationData-style
        r = _pick(nl, ("radii", "separations", "s"))
        x = _pick(nl, ("xi", "correlation", "values", "amplitudes"))
        e = _pick(nl, ("sigma", "err", "errors"))

        # Or: TwoPointCorrelationData has .estimate/.error
        if r is None and hasattr(nl, "estimate"):
            r = _pick(nl.estimate, ("radii", "separations", "s"))
            x = _pick(nl.estimate, ("correlation", "xi", "values", "amplitudes"))
        if e is None and hasattr(nl, "error"):
            e = _pick(nl.error, ("correlation", "xi", "values", "amplitudes"))

        if r is not None and x is not None:
            r = np.asarray(r, dtype=float)
            x = np.asarray(x, dtype=float)
            e = None if e is None else np.asarray(e, dtype=float)

            m = np.isfinite(r) & np.isfinite(x) & (r > 0)
            if e is not None:
                m &= np.isfinite(e)

            r, x = r[m], x[m]
            if e is not None:
                e = e[m]

            if r.size == 0:
                raise ValueError("TPCF arrays are empty after finite/positive filtering.")

            order = np.argsort(r)
            r, x = r[order], x[order]
            if e is not None:
                e = e[order]

            return r, x, e

    # ------------------------------------------
    # 1) estimate/error containers (generic path)
    # ------------------------------------------
    est = getattr(tpcf_obj, "estimate", None)

    # prefer error, then errors (some outputs use one or the other)
    err = getattr(tpcf_obj, "error", None)
    if err is None:
        err = getattr(tpcf_obj, "errors", None)

    r = x = e = None

    if est is not None:
        # Case: MatterTwoPointCorrelation (linear/nonlinear)
        if hasattr(est, "nonlinear"):
            nl_est = getattr(est, "nonlinear")
            r = _pick(nl_est, ("radii", "separations", "s"))
            x = _pick(nl_est, ("correlation", "xi", "values", "amplitudes"))
        else:
            r = _pick(est, ("radii", "separations", "s"))
            x = _pick(est, ("correlation", "xi", "values", "amplitudes"))
    else:
        r = _pick(tpcf_obj, ("radii", "separations", "s"))
        x = _pick(tpcf_obj, ("correlation", "xi", "values", "amplitudes"))

    if err is not None:
        # Case: MatterTwoPointCorrelation error container (linear/nonlinear)
        if hasattr(err, "nonlinear"):
            nl_err = getattr(err, "nonlinear")
            e = _pick(nl_err, ("correlation", "xi", "values", "amplitudes", "sigma"))
        else:
            e = _pick(err, ("correlation", "xi", "values", "amplitudes", "sigma"))
    else:
        e = _pick(tpcf_obj, ("sigma", "errors", "xi_err", "err"))

    # -------------------------
    # 2) fallback: .to_numpy()
    # -------------------------
    if (r is None) or (x is None):
        arr = getattr(tpcf_obj, "to_numpy", None)
        if arr is not None:
            a = np.asarray(arr, dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 2:
                # allow either (rows: r,xi,err) OR (cols: r,xi,err)
                if a.shape[0] <= a.shape[1]:
                    r = a[0]
                    x = a[1]
                    if a.shape[0] >= 3:
                        e = a[2]
                else:
                    r = a[:, 0]
                    x = a[:, 1]
                    if a.shape[1] >= 3:
                        e = a[:, 2]

    if r is None or x is None:
        raise ValueError(
            "Unable to extract radii/correlation arrays from TPCF object; "
            "expected MatterTwoPointCorrelationData(.nonlinear), "
            "or .estimate/.error containers, or .to_numpy."
        )

    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    e = None if e is None else np.asarray(e, dtype=float)

    m = np.isfinite(r) & np.isfinite(x) & (r > 0)
    if e is not None:
        m &= np.isfinite(e)

    r, x = r[m], x[m]
    if e is not None:
        e = e[m]

    if r.size == 0:
        raise ValueError("TPCF arrays are empty after finite/positive filtering.")

    order = np.argsort(r)
    r, x = r[order], x[order]
    if e is not None:
        e = e[order]

    return r, x, e


def _adaptive_max_sep_from_largest_box(
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
    *,
    snr_min: float = 1.0,
    relerr_max: float | None = None,
    smooth_window: int = 7,
    min_keep: int = 25,
    max_sep_cap: float | None = None,
    buffer_frac: float = 0.0,
) -> float:
    """Choose max separation from the largest-box tail where the measurement is still trustworthy."""
    if not sample_tpcfs:
        raise ValueError("sample_tpcfs is empty; cannot compute adaptive cutoff.")

    Lmax = max(sample_tpcfs.keys())
    r, x, e = _extract_r_x_e_from_tpcf_obj(sample_tpcfs[Lmax])

    # No errors => can't be adaptive; fall back to the largest available r.
    if e is None:
        r_max = float(np.nanmax(r))
        return min(r_max, float(max_sep_cap)) if max_sep_cap is not None else r_max

    eps = np.finfo(float).tiny
    absx = np.abs(x)
    abse = np.abs(e)

    if relerr_max is not None:
        diag = abse / np.maximum(absx, eps)   # relative error
        passes = diag <= float(relerr_max)
    else:
        diag = absx / np.maximum(abse, eps)   # SNR
        passes = diag >= float(snr_min)

    w = int(max(1, smooth_window))
    if w % 2 == 0:
        w += 1
    if w > 1 and passes.size >= w:
        kernel = np.ones(w, dtype=float)
        frac = np.convolve(passes.astype(float), kernel, mode="same") / float(w)
        passes_s = frac >= 0.5
    else:
        passes_s = passes

    idx = np.where(passes_s)[0]
    if idx.size == 0:
        i_keep = int(min(max(min_keep - 1, 0), r.size - 1))
        r_max = float(r[i_keep])
    else:
        i_keep = int(idx.max())
        if i_keep + 1 < min_keep:
            i_keep = int(min(max(min_keep - 1, 0), r.size - 1))
        r_max = float(r[i_keep])

    if buffer_frac and buffer_frac > 0:
        r_max *= (1.0 + float(buffer_frac))

    r_max = min(r_max, float(np.nanmax(r)))
    if max_sep_cap is not None:
        r_max = min(r_max, float(max_sep_cap))

    return float(r_max)


def _infer_stitch_plan_from_stitched_result(
    *,
    stitched: MatterTwoPointCorrelationData,
    sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
) -> StitchPlan:
    """
    Infer a TreeCorr StitchPlan for the *nonlinear* stitched bins by matching stitched xi(r)
    to the best-fitting source box xi(r), then mapping to nearest TreeCorr bin index.
    """
    r_out = np.asarray(stitched.estimate.nonlinear.radii, dtype=float)
    xi_out = np.asarray(stitched.estimate.nonlinear.correlation, dtype=float)
    if r_out.size == 0:
        raise ValueError("Stitched result has empty nonlinear radii.")
    if r_out.size != xi_out.size:
        raise ValueError("Stitched nonlinear radii/correlation have mismatched lengths.")

    box_sizes = sorted(sample_tpcfs.keys())
    if not box_sizes:
        raise ValueError("sample_tpcfs is empty.")

    # Score matrix: (n_boxes, n_bins)
    all_scores = np.full((len(box_sizes), r_out.size), np.inf, dtype=float)

    for bi, L in enumerate(box_sizes):
        t = sample_tpcfs[L]
        r_src = np.asarray(t.estimate.nonlinear.radii, dtype=float)
        xi_src = np.asarray(t.estimate.nonlinear.correlation, dtype=float)
        m = np.isfinite(r_src) & np.isfinite(xi_src) & (r_src > 0)
        if np.count_nonzero(m) < 2:
            continue

        r_src = r_src[m]
        xi_src = xi_src[m]
        order = np.argsort(r_src)
        r_src = r_src[order]
        xi_src = xi_src[order]

        f = interp1d(
            np.log10(r_src),
            xi_src,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        xi_i = f(np.log10(np.clip(r_out, 1e-300, np.inf)))

        ok = np.isfinite(xi_i) & np.isfinite(xi_out)
        if not np.any(ok):
            continue

        # Prefer log distance when both positive (scale-free), otherwise linear distance
        pos = ok & (xi_i > 0) & (xi_out > 0)
        if np.any(pos):
            all_scores[bi, pos] = np.abs(np.log10(xi_i[pos]) - np.log10(xi_out[pos]))

        lin = ok & ~pos
        if np.any(lin):
            all_scores[bi, lin] = np.abs(xi_i[lin] - xi_out[lin])

    src_box_idx = np.argmin(all_scores, axis=0)

    # Guard: bins that no box covered remain inf
    bad = ~np.isfinite(all_scores[src_box_idx, np.arange(r_out.size)])
    if np.any(bad):
        raise ValueError("Unable to infer stitch source for some bins (no source box covers those radii).")

    source_index = src_box_idx.astype(int)
    source_bin_index = np.empty_like(source_index)

    for k in range(r_out.size):
        L = box_sizes[int(source_index[k])]
        t = sample_tpcfs[L]
        dd = getattr(t, "treecorr_nn", None)
        if dd is None:
            raise ValueError(f"Missing treecorr_nn for box L={L}.")

        meanr = np.asarray(dd.meanr, dtype=float)
        r = float(r_out[k])

        if r > 0 and np.all(meanr > 0):
            j = int(np.nanargmin(np.abs(np.log10(meanr) - np.log10(r))))
        else:
            j = int(np.nanargmin(np.abs(meanr - r)))

        source_bin_index[k] = j

    return StitchPlan(source_index=source_index, source_bin_index=source_bin_index)


def _maybe_recompute_cov_with_treecorr(
        *,
        stitched: MatterTwoPointCorrelationData,
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        template_index: int = 0,
        rtol_tot: float = 5e-6,
        verbose: bool = False,
    ) -> MatterTwoPointCorrelationData:
    """
    If the per-box TPCF objects carry TreeCorr NNCorrelation internals, rebuild the nonlinear
    xi + varxi + covariance consistently on the stitched bins.

    When `verbose=True`, this will:
      - announce which TreeCorr components (RR/DR/RD) are available
      - print when TreeCorr sanity checks (config + tot-ratio checks) pass
      - print a summary of the recomputed covariance, plus a preview (or full matrix if small)
    """

    def _safe_get_arr(obj, public_name: str, private_name: str) -> np.ndarray | None:
        """Robustly fetch an array from TreeCorr objects across versions."""
        try:
            val = getattr(obj, public_name)
        except Exception:
            val = getattr(obj, private_name, None)
        return None if val is None else np.asarray(val, dtype=float)

    def _safe_get_cov(obj) -> np.ndarray | None:
        cov = getattr(obj, "cov", None)
        if cov is None:
            cov = getattr(obj, "_cov", None)
        return None if cov is None else np.asarray(cov, dtype=float)



    box_sizes = sorted(sample_tpcfs.keys())

    dd_list, rr_list = [], []
    dr_list, rd_list = [], []
    have_dr, have_rd = True, True

    if verbose:
        print("[treecorr-stitch] Attempting TreeCorr covariance recomputation")
        print(f"[treecorr-stitch] Sources (box sizes): {box_sizes}")

    for L in box_sizes:
        t = sample_tpcfs[L]
        dd = getattr(t, "treecorr_nn", None)
        if dd is None:
            if verbose:
                print(f"[treecorr-stitch] L={L}: missing treecorr_nn -> skip")
            return stitched

        rr = getattr(dd, "_rr", None)
        if rr is None:
            if verbose:
                print(f"[treecorr-stitch] L={L}: dd._rr is None (run calculateXi first?) -> skip")
            return stitched

        dd_list.append(dd)
        rr_list.append(rr)

        dr = getattr(dd, "_dr", None)
        rd = getattr(dd, "_rd", None)
        if dr is None:
            have_dr = False
        if rd is None:
            have_rd = False
        dr_list.append(dr)
        rd_list.append(rd)

    use_dr = dr_list if have_dr else None
    use_rd = rd_list if have_rd else None

    if verbose:
        print("[treecorr-stitch] Found TreeCorr RR for all sources: yes")
        print(f"[treecorr-stitch] Found TreeCorr DR for all sources: {'yes' if have_dr else 'no'}")
        print(f"[treecorr-stitch] Found TreeCorr RD for all sources: {'yes' if have_rd else 'no'}")

    plan = _infer_stitch_plan_from_stitched_result(
        stitched=stitched,
        sample_tpcfs=sample_tpcfs,
    )

    # Build stitched DD/RR/(DR/RD) and run TreeCorr's internal sanity checks.
    if verbose:
        print("[treecorr-stitch] Building stitched TreeCorr bundle (includes sanity checks)...")
    try:
        bundle = build_stitched_treecorr_bundle(
            dd_list=dd_list,
            rr_list=rr_list,
            dr_list=use_dr,
            rd_list=use_rd,
            plan=plan,
            template_index=template_index,
            rtol_tot=rtol_tot,
        )
    except Exception as err:
        if verbose:
            print(f"[treecorr-stitch] Sanity checks FAILED -> skip. Reason: {err}")
        return stitched

    if verbose:
        print("[treecorr-stitch] Sanity checks PASSED (config + tot-ratio consistency)")

    # Recompute xi + varxi + cov on stitched objects
    try:
        dd_stitched = recompute_dd_xi_and_cov(bundle)
    except Exception as err:
        if verbose:
            print(f"[treecorr-stitch] calculateXi FAILED -> skip. Reason: {err}")
        return stitched

    r_nl = np.asarray(dd_stitched.meanr, dtype=float)
    xi_nl = _safe_get_arr(dd_stitched, "xi", "_xi")
    var_nl = _safe_get_arr(dd_stitched, "varxi", "_varxi")
    if xi_nl is None or var_nl is None:
        if verbose:
            print(
                "[treecorr-stitch] Missing xi/varxi after calculateXi -> skip. "
                f"have_xi={xi_nl is not None} have_varxi={var_nl is not None}"
            )
        return stitched
    sigma_nl = np.sqrt(var_nl)

    cov = _safe_get_cov(dd_stitched)
    if verbose:
        if cov is None:
            print("[treecorr-stitch] Recompute completed but covariance is missing (cov/_cov is None).")
        else:
            m = cov.shape[0]
            print(f"[treecorr-stitch] TreeCorr recompute SUCCESS. cov.shape={cov.shape}")
            if m <= 15:
                print("[treecorr-stitch] cov =\n", cov)
            else:
                k = 10
                print(f"[treecorr-stitch] cov preview (top-left {k}x{k}) =\n", cov[:k, :k])

    if cov is None:
        if verbose:
            print("[treecorr-stitch] Recompute ran, but no covariance found on stitched DD (dd.cov/dd._cov is None) -> returning stitched")
        return stitched

    cov = np.asarray(cov, dtype=float)

    # Guard: if TreeCorr fell back to an effectively diagonal covariance, do not overwrite
    # the stitched correlation matrix with an identity-like matrix.
    if cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
        off = cov.copy()
        np.fill_diagonal(off, 0.0)
        off_max = float(np.nanmax(np.abs(off))) if off.size else 0.0
        diag = np.diag(cov)
        diag_max = float(np.nanmax(np.abs(diag))) if diag.size else 0.0
        rel = off_max / (diag_max if diag_max > 0 else 1.0)
        if verbose:
            print(f"[treecorr-stitch] cov offdiag max={off_max:.6g}, diag max={diag_max:.6g}, off/diag={rel:.3e}")
        if rel < 1e-8:
            if verbose:
                print("[treecorr-stitch] Covariance is effectively diagonal; keeping original stitched correlation matrix.")
            return stitched

    # Build correlation matrix (optional; stored on MatterTwoPointCorrelationData)
    corr = None
    denom = np.outer(sigma_nl, sigma_nl)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / denom
    if corr is not None and corr.ndim == 2 and corr.shape[0] == corr.shape[1]:
        np.fill_diagonal(corr, 1.0)

    if verbose:
        n = int(cov.shape[0])
        finite = np.isfinite(cov)
        frac_finite = float(np.count_nonzero(finite)) / float(cov.size) if cov.size else 0.0
        diag = np.diag(cov) if cov.ndim == 2 and cov.shape[0] == cov.shape[1] else None

        print("[treecorr-stitch] SUCCESS: recomputed stitched covariance via TreeCorr")
        print(f"[treecorr-stitch] cov shape = {cov.shape}, finite fraction = {frac_finite:.4f}")
        if diag is not None and diag.size:
            print(
                "[treecorr-stitch] cov diag: "
                f"min={float(np.nanmin(diag)):.6g} "
                f"median={float(np.nanmedian(diag)):.6g} "
                f"max={float(np.nanmax(diag)):.6g}"
            )

        # Print the covariance matrix (full if small, else preview)
        # with np.printoptions(precision=6, suppress=False, linewidth=140, threshold=2000):
        #     if n <= 30:
        #         print("[treecorr-stitch] cov (full):")
        #         print(cov)
        #     else:
        #         k = 10
        #         print(f"[treecorr-stitch] cov (preview {k}x{k} of {n}x{n}):")
        #         print(cov[:k, :k])

    estimate = MatterTwoPointCorrelation(
        linear=stitched.estimate.linear,
        nonlinear=TwoPointCorrelation(
            radii=r_nl,
            correlation=xi_nl,
            is_linear=False,
            in_comoving=stitched.estimate.nonlinear.in_comoving,
        ),
    )
    error = MatterTwoPointCorrelation(
        linear=stitched.error.linear,
        nonlinear=TwoPointCorrelation(
            radii=r_nl,
            correlation=sigma_nl,
            is_linear=False,
            in_comoving=stitched.error.nonlinear.in_comoving,
        ),
    )

    return MatterTwoPointCorrelationData(
        estimate=estimate,
        error=error,
        correlation_matrix=(corr if corr is not None else stitched.correlation_matrix),
        in_comoving=stitched.in_comoving,
        folds=stitched.folds,
    )


def stitch_tpcf(
        sample_tpcfs: dict[int, MatterTwoPointCorrelationData],
        cosmo: Cosmology,
        moment: Moment,
        method: str = 'deviation',
        **kwargs
    ) -> MatterTwoPointCorrelationData:
    """Stitch together two-point correlation functions from multiple simulation boxes.

    Parameters
    ----------
    sample_tpcfs : dict
        Dictionary of box_size -> MatterTwoPointCorrelationData
    cosmo : Cosmology
        Cosmology object for linear correlation
    moment : Moment
        Moment object containing redshift information
    method : str, optional
        Stitching method to use (default: 'deviation')
        - 'deviation': Transition-based method using deviation detection
        - 'contour': Upper envelope method following power spectrum approach
        - 'overlap': Overlap-weighted method preferring regions where boxes agree
        - 'contour_linear': same as 'contour' then blends the large-s tail toward linear

    Other Parameters
    ----------------
    apply_linear_tail_blend : bool, optional
        If True, post-process the stitched nonlinear estimate by blending its large-s tail
        toward the linear prediction evaluated at the same radii. (default: False)
    linear_blend_start : float | None, optional
        Start of the blend window in separation units. If None, uses linear_blend_start_frac*L_max.
    linear_blend_end : float | None, optional
        End of the blend window in separation units. If None, uses linear_blend_end_frac*L_max.
    linear_blend_start_frac : float, optional
        Default blend start as a fraction of the largest box size L_max (default: 0.04).
    linear_blend_end_frac : float, optional
        Default blend end as a fraction of the largest box size L_max (default: 0.09).
    linear_blend_fit_offset : bool, optional
        If True, fit a constant offset C so xi_lin(s)+C matches the stitched curve on a
        pre-blend window (robust median). (default: True)
    linear_blend_fit_window : tuple[float, float] | None, optional
        Explicit (s_lo, s_hi) window for fitting C. If None, uses (0.6*s1, s1).
    linear_blend_use_log_space : bool, optional
        If True, the blend ramp is linear in log10(s). (default: True)

    Notes
    -----
    The linear-tail blending modifies ONLY the nonlinear estimate curve. It preserves
    the stitched radii, the stitched error bars, and the stitched correlation matrix.
    """

    # --- pop linear-tail blending knobs so they don't get forwarded to stitch engines ---
    apply_linear_tail_blend = bool(kwargs.pop('apply_linear_tail_blend', False))
    linear_blend_start = kwargs.pop('linear_blend_start', None)
    linear_blend_end = kwargs.pop('linear_blend_end', None)
    linear_blend_start_frac = float(kwargs.pop('linear_blend_start_frac', 0.04))
    linear_blend_end_frac = float(kwargs.pop('linear_blend_end_frac', 0.09))
    linear_blend_fit_offset = bool(kwargs.pop('linear_blend_fit_offset', True))
    linear_blend_fit_window = kwargs.pop('linear_blend_fit_window', None)
    linear_blend_use_log_space = bool(kwargs.pop('linear_blend_use_log_space', True))

    # --- TreeCorr covariance recomputation knobs (popped so they don't get forwarded) ---
    recompute_cov_with_treecorr = bool(kwargs.pop("recompute_cov_with_treecorr", False))
    treecorr_template_index = int(kwargs.pop("treecorr_template_index", 0))
    treecorr_rtol_tot = float(kwargs.pop("treecorr_rtol_tot", 5e-6))
    treecorr_verbose = bool(kwargs.pop("treecorr_verbose", False))

    # Alias method: contour + linear-tail blend
    if method == 'contour_linear':
        apply_linear_tail_blend = True
        method = 'contour'

    match method:
        case 'deviation':
            result = _stitch_by_deviation(
                sample_tpcfs=sample_tpcfs,
                cosmo=cosmo,
                moment=moment,
                **kwargs,
            )
        case 'contour':
            result = _stitch_by_contour(
                sample_tpcfs=sample_tpcfs,
                cosmo=cosmo,
                moment=moment,
                **kwargs,
            )
        case 'overlap':
            result = _stitch_by_overlap(
                sample_tpcfs=sample_tpcfs,
                cosmo=cosmo,
                moment=moment,
                **kwargs,
            )
        case "contour_cutoff":
            # Extract cutoff-only kwargs so they don't get passed into _stitch_by_contour(...)
            cut_kwargs = dict(
                max_sep=kwargs.pop("max_sep", None),
                max_sep_frac_of_box=kwargs.pop("max_sep_frac_of_box", None),
                box_size_for_frac=kwargs.pop("box_size_for_frac", None),
                keep_at_least=kwargs.pop("keep_at_least", 8),
                print_debug_info=kwargs.pop("cut_print_debug_info", False),
            )
            base = _stitch_by_contour(
                sample_tpcfs=sample_tpcfs,
                cosmo=cosmo,
                moment=moment,
                **kwargs,
            )
            result = _apply_max_sep_cut(base, **cut_kwargs)
        case "contour_cutoff_adaptive":
            snr_min = kwargs.pop("snr_min", 1.0)
            relerr_max = kwargs.pop("relerr_max", None)
            smooth_window = kwargs.pop("smooth_window", 7)
            min_keep = kwargs.pop("min_keep", 25)
            max_sep_cap = kwargs.pop("max_sep_cap", None)
            buffer_frac = kwargs.pop("buffer_frac", 0.0)

            # allow manual override; otherwise compute adaptively
            max_sep = kwargs.pop("max_sep", None)
            if max_sep is None:
                max_sep = _adaptive_max_sep_from_largest_box(
                    sample_tpcfs,
                    snr_min=snr_min,
                    relerr_max=relerr_max,
                    smooth_window=smooth_window,
                    min_keep=min_keep,
                    max_sep_cap=max_sep_cap,
                    buffer_frac=buffer_frac,
                )

            # respect user r_range lower bound if provided
            r_range = kwargs.pop("r_range", None)
            if r_range is None:
                rmins = []
                for _t in sample_tpcfs.values():
                    try:
                        rr, _, _ = _extract_r_x_e_from_tpcf_obj(_t)
                        rmins.append(float(np.nanmin(rr)))
                    except Exception:
                        continue
                r_min = float(np.nanmin(rmins)) if rmins else 1e-3
                r_range = (r_min, float(max_sep))
            else:
                r_range = (float(r_range[0]), float(max_sep))

            result = _stitch_by_contour(
                sample_tpcfs=sample_tpcfs,
                cosmo=cosmo,
                moment=moment,
                r_range=r_range,
                **kwargs,
            )
        case _:
            raise ValueError(
                f"Unknown stitching method: '{method}'. "
                "Valid options are 'deviation', 'contour', 'contour_linear', 'overlap', 'contour_cutoff', or 'contour_cutoff_adaptive'."
            )

    # Optional: recompute nonlinear xi + covariance on the stitched bins using TreeCorr internals.
    # This is done *before* any linear-tail blending, since blending should not affect covariance.
    if recompute_cov_with_treecorr:
        try:
            result = _maybe_recompute_cov_with_treecorr(
                stitched=result,
                sample_tpcfs=sample_tpcfs,
                template_index=treecorr_template_index,
                rtol_tot=treecorr_rtol_tot,
                verbose=treecorr_verbose,
            )
        except Exception as err:
            if treecorr_verbose:
                print(f"[treecorr-stitch] Falling back to existing covariance due to: {err}")

    if apply_linear_tail_blend:
        L_max = float(max(sample_tpcfs.keys())) if sample_tpcfs else None
        result = _apply_linear_tail_blend(
            stitched=result,
            cosmo=cosmo,
            moment=moment,
            L_max=L_max,
            blend_start=linear_blend_start,
            blend_end=linear_blend_end,
            blend_start_frac=linear_blend_start_frac,
            blend_end_frac=linear_blend_end_frac,
            fit_offset=linear_blend_fit_offset,
            fit_window=linear_blend_fit_window,
            use_log_space=linear_blend_use_log_space,
        )

    return result
