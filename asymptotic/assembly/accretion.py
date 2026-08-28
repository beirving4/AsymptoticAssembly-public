"""Accretion / evolution rates and assembly scalars.

- Mass keeps the MORIA-style log accretion rate ``Gamma = d ln M / d ln a`` over
  instantaneous / 0.5 t_dyn / 1 t_dyn baselines (``compute_accretion_rates``).
- Every tracked quantity also gets a generic linear evolution rate
  ``d X / d ln a`` (instantaneous + over 1 t_dyn) via ``compute_quantity_rate``.
- Assembly scalars (formation times, peaks) reuse the proven log-log interpolation
  in ``studies/mah/helpers.py`` and use GROUP values only (``v_peak`` from a group
  circular velocity, derived where ``Group_V`` is absent).
"""
from __future__ import annotations

import astropy.constants as const
import numpy as np
from attrs import define

from ..studies.mah import helpers
from .cleaning import PointStatus

# G in km^2 Mpc / (Msun s^2): V_c[km/s] = sqrt(G * M[Msun] / R[Mpc]).
_G_KM2_MPC_PER_MSUN_S2 = const.G.to("km2 Mpc / (Msun s2)").value


def circular_velocity_kms(group_m: np.ndarray, group_r: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Group circular velocity V_c = sqrt(G M / R) [km/s].

    ``group_m`` in 1e10 Msun/h, ``group_r`` comoving Mpc/h, ``a`` scale factor.
    ``h`` cancels in the M/R ratio, so we only apply the 1e10 Msun factor and the
    comoving->physical radius scaling (R_phys = R * a). Verified against the
    catalog's own ``Group_V`` on apertures that carry it.
    """
    m = np.asarray(group_m, dtype=float) * 1e10
    r = np.asarray(group_r, dtype=float) * np.asarray(a, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(_G_KM2_MPC_PER_MSUN_S2 * m / r)


@define(slots=True)
class AccretionRates:
    gamma_inst: np.ndarray        # [n_halos, n_snaps] d ln M / d ln a (gradient)
    gamma_half_tdyn: np.ndarray   # over 0.5 t_dyn baseline
    gamma_tdyn: np.ndarray        # over 1 t_dyn baseline (primary)
    status: np.ndarray            # [n_halos, n_snaps] uint8 PointStatus


@define(slots=True)
class QuantityRates:
    rate_inst: np.ndarray         # [n_halos, n_snaps] d X / d ln a (gradient)
    rate_tdyn: np.ndarray         # over 1 t_dyn baseline
    status: np.ndarray            # [n_halos, n_snaps] uint8 PointStatus


@define(slots=True)
class AssemblyScalars:
    m_final: np.ndarray
    m_peak: np.ndarray
    a_at_mpeak: np.ndarray
    v_peak: np.ndarray
    a_at_vpeak: np.ndarray
    a_half: np.ndarray                  # half of FINAL mass (asymptotic); a >= a_half_present
    a_half_present: np.ndarray          # half of a=1 (present-day) mass; in (0, 1]
    a_half_present_status: np.ndarray   # [n_halos] uint8 PointStatus
    a_freeze_out: np.ndarray
    a_mass_accum: np.ndarray
    a_last_major_merger: np.ndarray
    # --- robust-MAH formation diagnostics ---
    a_half_raw: np.ndarray              # half FINAL mass from RAW (uncleaned) MAH
    a_half_present_raw: np.ndarray      # half a=1 mass from RAW (uncleaned) MAH
    raw_max_over_final: np.ndarray      # max_a M(a) / M_final (spike indicator)
    n_extreme_jumps: np.ndarray         # [n_halos] count of masked spike points
    was_clipped: np.ndarray             # [n_halos] bool: any spike masked


def _baseline_indices(ages: np.ndarray, dynamical_times: np.ndarray, n_tdyn: float) -> np.ndarray:
    """For each snapshot, the index ~``n_tdyn`` dynamical times earlier (in age)."""
    target_age = ages - n_tdyn * dynamical_times
    idx = np.searchsorted(ages, target_age, side="left")
    return np.clip(idx, 0, len(ages) - 1)


def baseline_indices(ages: np.ndarray, dynamical_times: np.ndarray, n_tdyn: float) -> np.ndarray:
    """Public alias: per-snapshot index ~``n_tdyn`` dynamical times away (signed).

    ``n_tdyn > 0`` looks back in age (earlier snapshot); ``n_tdyn < 0`` looks
    forward (later snapshot). Used to bracket a merger by +/- a fraction of a
    dynamical time when measuring how a property jumps across the event.
    """
    target_age = ages - n_tdyn * dynamical_times
    idx = np.searchsorted(ages, target_age, side="left")
    return np.clip(idx, 0, len(ages) - 1)


def _finite_span_gradient(y: np.ndarray, ln_a: np.ndarray) -> np.ndarray:
    """Per-row d y / d ln a on each row's finite span (avoids NaN-neighbour loss)."""
    out = np.full(y.shape, np.nan)
    for i in range(y.shape[0]):
        fin = np.isfinite(y[i])
        if fin.sum() >= 2:
            out[i, fin] = np.gradient(y[i, fin], ln_a[fin])
    return out


def _diff_over_baseline(y: np.ndarray, ln_a: np.ndarray, base_idx: np.ndarray) -> np.ndarray:
    """(y_j - y_base) / (ln a_j - ln a_base), vectorized over halos (linear)."""
    dy = y - y[:, base_idx]
    dln_a = (ln_a - ln_a[base_idx])[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = dy / dln_a
    rate[:, dln_a[0] == 0] = np.nan
    return rate


def _rate_status(values: np.ndarray, ages: np.ndarray, full_idx: np.ndarray) -> np.ndarray:
    status = np.full(values.shape, PointStatus.SUCCESS, dtype=np.uint8)
    status[~np.isfinite(values)] = PointStatus.INVALID
    reduced_cols = full_idx == np.arange(len(ages))
    status[:, reduced_cols] = np.where(
        status[:, reduced_cols] == PointStatus.INVALID,
        PointStatus.INVALID, PointStatus.REDUCED_INTERVAL,
    )
    return status


def compute_accretion_rates(
    a: np.ndarray, mah: np.ndarray, ages: np.ndarray, dynamical_times: np.ndarray,
) -> AccretionRates:
    """Mass log accretion rate Gamma = d ln M / d ln a (inst + 0.5/1.0 t_dyn)."""
    a = np.asarray(a, dtype=float)
    mah = np.asarray(mah, dtype=float)
    ln_a = np.log(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_m = np.log(np.where(mah > 0, mah, np.nan))

    gamma_inst = _finite_span_gradient(ln_m, ln_a)
    half_idx = _baseline_indices(ages, dynamical_times, 0.5)
    full_idx = _baseline_indices(ages, dynamical_times, 1.0)
    gamma_half = _diff_over_baseline(ln_m, ln_a, half_idx)
    gamma_full = _diff_over_baseline(ln_m, ln_a, full_idx)
    return AccretionRates(gamma_inst, gamma_half, gamma_full, _rate_status(mah, ages, full_idx))


def compute_quantity_rate(
    values: np.ndarray, a: np.ndarray, ages: np.ndarray, dynamical_times: np.ndarray,
) -> QuantityRates:
    """Generic linear evolution rate d X / d ln a (instantaneous + over 1 t_dyn).

    Works for signed/bounded quantities (beta, energy, shape ratios) where a log
    rate is undefined.
    """
    values = np.asarray(values, dtype=float)
    ln_a = np.log(np.asarray(a, dtype=float))
    rate_inst = _finite_span_gradient(values, ln_a)
    full_idx = _baseline_indices(ages, dynamical_times, 1.0)
    rate_tdyn = _diff_over_baseline(values, ln_a, full_idx)
    return QuantityRates(rate_inst, rate_tdyn, _rate_status(values, ages, full_idx))


def compute_quantity_peak(values: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak (max) value of each row + the scale factor at which it occurs."""
    a = np.asarray(a, dtype=float)
    v = np.where(np.isfinite(values), values, np.nan)
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(v, axis=1)
    a_at = np.array([
        a[np.nanargmax(v[i])] if np.isfinite(v[i]).any() else np.nan for i in range(v.shape[0])
    ])
    return peak, a_at


def _safe(fn, *args) -> float:
    try:
        val = float(fn(*args))
        return val if np.isfinite(val) else np.nan
    except (ValueError, IndexError, ZeroDivisionError):
        return np.nan


def compute_assembly_scalars(
    a: np.ndarray,
    mah: np.ndarray,
    v: np.ndarray | None,
    rates: AccretionRates,
    desc_col: int,
    a_last_major_merger: np.ndarray | None = None,
    half_frac: float = 0.5,
    a_present_ref: float = 1.0,
    spike_factor: float = 5.0,
) -> AssemblyScalars:
    """Per-halo assembly scalars. ``v`` is the GROUP circular-velocity trajectory.

    Formation times use a *robust* MAH (extreme radial-boundary spikes masked,
    see ``helpers.robustify_mah``) and a monotonic cumulative-max envelope so the
    half-mass crossing is single-valued. Two bounded definitions:

    - **present-day** ``a_half_present``: where the envelope first reaches
      ``half_frac * M(a=1)``, searched only over ``a <= a_present_ref`` -> in (0, 1];
    - **asymptotic** ``a_half``: where the envelope first reaches
      ``half_frac * M(a_final)`` (the descendant mass, not the raw max), searched
      over ``a >= a_half_present`` -> ``a_half >= a_half_present``.

    The raw (uncleaned, legacy-interp) formation times and spike diagnostics are
    returned alongside for validation.
    """
    a = np.asarray(a, dtype=float)
    mah = np.asarray(mah, dtype=float)
    n = mah.shape[0]

    m_final = mah[:, desc_col]
    m_peak, a_at_mpeak = compute_quantity_peak(mah, a)
    if v is not None:
        v_peak, a_at_vpeak = compute_quantity_peak(v, a)
    else:
        v_peak, a_at_vpeak = np.full(n, np.nan), np.full(n, np.nan)

    a_half = np.full(n, np.nan)
    a_present = np.full(n, np.nan)
    a_present_status = np.full(n, PointStatus.INVALID, dtype=np.uint8)
    a_freeze = np.full(n, np.nan)
    a_accum = np.full(n, np.nan)
    # diagnostics
    a_half_raw = np.full(n, np.nan)
    a_present_raw = np.full(n, np.nan)
    raw_max_over_final = np.full(n, np.nan)
    n_extreme = np.zeros(n, dtype=np.int32)
    clipped = np.zeros(n, dtype=bool)
    for i in range(n):
        finite = np.isfinite(mah[i]) & (mah[i] > 0)
        if finite.sum() < 2:
            continue
        ai, mi = a[finite], mah[i, finite]
        mf = m_final[i]

        # --- raw (legacy) formation times, kept purely as a diagnostic ---
        a_present_raw[i] = _safe(helpers.get_present_formation_time, ai, mi)
        if np.isfinite(mf) and mf > 0:
            a_half_raw[i] = _safe(helpers.get_formation_time, ai, mi / mf)

        # --- robust MAH: mask extreme upward spikes, then use cummax envelope ---
        m_rob, n_ext, ratio = helpers.robustify_mah(a, mah[i], mf, spike_factor)
        n_extreme[i] = n_ext
        clipped[i] = n_ext > 0
        raw_max_over_final[i] = ratio

        # present-day formation: threshold = frac * M(a=1), domain a <= 1
        m_at_present = helpers.interp_mass_at(a, m_rob, a_present_ref)
        if np.isfinite(m_at_present) and m_at_present > 0:
            val = helpers.formation_time_envelope(
                a, m_rob, half_frac * m_at_present, a_hi=a_present_ref
            )
            if val == 0.0:
                a_present_status[i] = PointStatus.FORMED_BEFORE_TRACKING
            elif np.isfinite(val):
                a_present[i], a_present_status[i] = val, PointStatus.SUCCESS

        # asymptotic formation: threshold = frac * M_final, domain a >= a_present
        if np.isfinite(mf) and mf > 0:
            a_lo = a_present[i] if np.isfinite(a_present[i]) else None
            af = helpers.formation_time_envelope(a, m_rob, half_frac * mf, a_lo=a_lo)
            if af == 0.0:                       # half-final already met at the floor
                a_half[i] = a_lo if a_lo is not None else np.nan
            elif np.isfinite(af):
                a_half[i] = af
            if np.isfinite(a_half[i]) and np.isfinite(a_present[i]):
                a_half[i] = max(a_half[i], a_present[i])   # enforce a_half >= a_present
            a_accum[i] = _safe(helpers.get_mass_accumulation_time, ai, mi / mf)

        g = rates.gamma_inst[i, finite]
        gf = np.isfinite(g)
        if gf.sum() >= 2:
            a_freeze[i] = _safe(helpers.get_freeze_out_time, g[gf], ai[gf])

    return AssemblyScalars(
        m_final=m_final, m_peak=m_peak, a_at_mpeak=a_at_mpeak,
        v_peak=v_peak, a_at_vpeak=a_at_vpeak,
        a_half=a_half, a_half_present=a_present, a_half_present_status=a_present_status,
        a_freeze_out=a_freeze, a_mass_accum=a_accum,
        a_last_major_merger=(
            a_last_major_merger if a_last_major_merger is not None else np.full(n, np.nan)
        ),
        a_half_raw=a_half_raw, a_half_present_raw=a_present_raw,
        raw_max_over_final=raw_max_over_final, n_extreme_jumps=n_extreme, was_clipped=clipped,
    )
