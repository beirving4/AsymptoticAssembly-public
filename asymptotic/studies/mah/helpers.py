import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import lognorm, rv_continuous

def get_accretion_rate(scale_factors: np.ndarray, mah: np.ndarray) -> np.ndarray: 
    return np.gradient(np.log(mah), np.log(scale_factors))

def get_scale_factor_interp(scale_factors: np.ndarray, mah: np.ndarray) -> interp1d:
    return interp1d(np.log10(mah), np.log10(scale_factors))

def get_formation_time(scale_factors: np.ndarray, mah: np.ndarray) -> float: 
    log10_interp = interp1d(np.log10(mah), np.log10(scale_factors))
    return 10.0**log10_interp(np.log10(0.5))

def get_present_formation_time(scale_factors: np.ndarray, mah: np.ndarray) -> float: 
    scale_to_mah_interp = interp1d(np.log10(scale_factors), np.log10(mah))
    log10_interp = interp1d(np.log10(mah), np.log10(scale_factors))
    present_day_mass = 10.0**scale_to_mah_interp(np.log10(1.0))
    if (np.nanmin(mah) > (present_day_mass / 2.0)):
        return 0.0
    else:
        return 10.0**log10_interp(np.log10(present_day_mass / 2.0))

def robustify_mah(
    scale_factors: np.ndarray, mah: np.ndarray, m_final: float, spike_factor: float = 5.0
) -> tuple[np.ndarray, int, float]:
    """Mask isolated extreme upward excursions before formation-time estimation.

    Splashback/bound radial-boundary estimators occasionally emit late-time mass
    spikes that are hundreds of times the converged (descendant) mass; left in,
    they would dominate a cumulative-max envelope and corrupt formation times.
    A tracked point is treated as such an artifact when it exceeds
    ``spike_factor * m_final`` (the final mass is the converged reference, NOT the
    raw maximum). Returns ``(mah_robust, n_extreme, raw_max_over_final)`` where
    spiked points are set to NaN in ``mah_robust``.
    """
    a = np.asarray(scale_factors, dtype=float)
    m = np.asarray(mah, dtype=float)
    valid = np.isfinite(m) & (m > 0)
    raw_max = float(np.nanmax(m[valid])) if valid.any() else np.nan
    ratio = raw_max / m_final if (np.isfinite(m_final) and m_final > 0) else np.nan
    m_rob = m.copy()
    n_extreme = 0
    if np.isfinite(m_final) and m_final > 0:
        spike = valid & (m > spike_factor * m_final)
        m_rob[spike] = np.nan
        n_extreme = int(spike.sum())
    return m_rob, n_extreme, ratio


def interp_mass_at(scale_factors: np.ndarray, mah: np.ndarray, a_target: float) -> float:
    """Mass at ``a_target`` via log-log interpolation on the a-sorted finite MAH.

    Returns NaN if ``a_target`` falls outside the tracked range or <2 finite points.
    """
    a = np.asarray(scale_factors, dtype=float)
    m = np.asarray(mah, dtype=float)
    finite = np.isfinite(m) & (m > 0) & np.isfinite(a)
    if finite.sum() < 2:
        return np.nan
    aa, mm = a[finite], m[finite]
    order = np.argsort(aa)
    aa, mm = aa[order], mm[order]
    if a_target < aa[0] or a_target > aa[-1]:
        return np.nan
    return float(10.0 ** np.interp(np.log10(a_target), np.log10(aa), np.log10(mm)))


def formation_time_envelope(
    scale_factors: np.ndarray,
    mah: np.ndarray,
    threshold: float,
    a_lo: float | None = None,
    a_hi: float | None = None,
) -> float:
    """Scale factor where the cumulative-max envelope of ``mah`` first reaches ``threshold``.

    The envelope (most mass ever assembled by a) is monotonic non-decreasing, so
    the half-mass crossing is single-valued even when the raw MAH dips after a peak.
    Search is optionally restricted to ``a in [a_lo, a_hi]`` (the envelope is still
    built from the full history, i.e. it reflects all mass assembled up to each a).

    Returns NaN for <2 finite points or if ``threshold`` is never reached in-domain;
    returns ``0.0`` as a sentinel when the envelope already exceeds ``threshold`` at
    the first in-domain epoch (formed before/at the window edge).
    """
    a = np.asarray(scale_factors, dtype=float)
    m = np.asarray(mah, dtype=float)
    finite = np.isfinite(m) & (m > 0) & np.isfinite(a)
    if finite.sum() < 2 or not (np.isfinite(threshold) and threshold > 0):
        return np.nan
    aa, mm = a[finite], m[finite]
    order = np.argsort(aa)
    aa, mm = aa[order], mm[order]
    env = np.maximum.accumulate(mm)
    in_dom = np.ones(aa.shape, dtype=bool)
    if a_lo is not None:
        in_dom &= aa >= a_lo
    if a_hi is not None:
        in_dom &= aa <= a_hi
    if in_dom.sum() < 1:
        return np.nan
    aw, ew = aa[in_dom], env[in_dom]
    if ew[0] >= threshold:
        return 0.0                       # already above threshold at window start
    if ew[-1] < threshold:
        return np.nan                    # never reaches threshold in-domain
    j = int(np.searchsorted(ew, threshold, side="left"))  # ew non-decreasing -> j>=1
    a0, a1, e0, e1 = aw[j - 1], aw[j], ew[j - 1], ew[j]
    if e1 <= e0:
        return float(a1)
    t = (np.log10(threshold) - np.log10(e0)) / (np.log10(e1) - np.log10(e0))
    return float(10.0 ** (np.log10(a0) + t * (np.log10(a1) - np.log10(a0))))


def get_freeze_out_time(accretion_rate: np.ndarray, scale_factors: np.ndarray) -> float:
    acc_to_scale_interp = interp1d(accretion_rate, np.log10(scale_factors))
    return 10.0**acc_to_scale_interp(1E-4)

def get_mass_accumulation_time(
        scale_factors: np.ndarray, mah: np.ndarray, pct: float = 0.98
    ) -> float:
    log10_interp = interp1d(np.log10(mah), np.log10(scale_factors))
    return 10.0**log10_interp(np.log10(pct))

def get_mass_accumulation_times(
        scale_factors: np.ndarray, mah: np.ndarray, pcts: np.ndarray
    ) -> dict[float, float]:
    log10_interp = interp1d(np.log10(mah), np.log10(scale_factors))
    return {pct : 10.0**log10_interp(np.log10(pct)) for pct in pcts}

def get_formation_times(histories: np.ndarray) -> np.ndarray: 
    try:
        times = np.vstack([
            [i, get_formation_time(mah[:, 0], mah[:, 1])] 
            for i, mah in enumerate(histories) if np.nanmin(mah[:, 1]) < 0.5
        ])
    except ValueError as e:
        return np.empty_like(histories[:, 0, 0])

    valid_time_mask = (times[:, 1] != 0) & (~np.isnan(times[:, 1]))

    return times[valid_time_mask, :]

def get_present_formation_times(histories: np.ndarray) -> np.ndarray: 
    try:
        times = np.vstack([
            [i, get_present_formation_time(mah[:, 0], mah[:, 1])]
            for i, mah in enumerate(histories) if np.nanmin(mah[:, 0]) < 0.5
        ])
    except ValueError as e:
        return np.empty_like(histories[:, 0, 0])

    valid_time_mask = (times[:, 1] != 0) & (~np.isnan(times[:, 1]))

    return times[valid_time_mask, :]

def get_formation_distribution(times: np.ndarray, num_bins: int) -> dict[str, np.ndarray]:
    valid_times = times[~np.isnan(times)]
    hist, time_bins = np.histogram(np.log10(valid_times), num_bins)
    return {
        "counts" : hist, 
        "bins" : time_bins,
        "bin_centers" : 0.5 * (time_bins[1:] + time_bins[:-1]),
    }

def fit_to_lognormal_distribution(times: np.ndarray)-> rv_continuous:
    return lognorm(*lognorm.fit(np.log10(times)))

def get_common_formation_assemblies(
        formation_times: np.ndarray, 
        present_day_formation_times: np.ndarray,
    ) -> dict[str, np.ndarray]:

    _, present_common, final_common = np.intersect1d(
        present_day_formation_times[:, 0],
        formation_times[:, 0], 
        assume_unique=True,
        return_indices=True
    )

    common_times = np.column_stack((
        present_day_formation_times[present_common, 1],
        formation_times[final_common, 1]
    ))

    non_zero_idxs = np.nonzero(common_times[:, 0])[0]

    return {
        "times" : common_times[non_zero_idxs, :],
        "present_form_idxs" : np.asarray(present_common[non_zero_idxs], dtype=int),
        "final_form_idxs" : np.asarray(final_common[non_zero_idxs], dtype=int)
    }