"""Kinematic outer-boundary milestones (lambda / turnaround / inf / hs /
busha05) extracted from per-halo radial-velocity profiles.

Originally co-located with the solo profile pipeline as private helpers
in postprocessing/save_solo_profiles_phusis.py; lifted here so the
validator (postprocessing/validate_solo_milestones_phusis.py) can re-run
the same logic on already-written solo files without dragging in the
solo driver's argparse / IOConfig side-effects.

Public API
----------
``smooth_v_total(v_total, n_r)``
    Per-halo Savitzky-Golay smoothing along r. Window auto-scaled to
    ~10% of n_r (min 5, forced odd). NaN-tolerant via linear-interp fill
    before SavGol then re-mask.

``compute_outer_boundary_milestones(profiles_stack, property_profiles_stack, ...)``
    Always emits ``lambda_*`` (closed-form max-turnaround from
    ⟨ρ_enc⟩ = 2·ρ_Λ — Pavlidou-Tomaras 2014). When per-halo
    ``catalog_R`` and ``catalog_V`` are supplied AND the property
    profile carries ``mean_vr``, the kinematic outer-boundary
    framework also emits ``inf_*``, ``hs_*``, ``turnaround_*``, and
    ``busha05_*``. See the function's own docstring for the full
    algorithm description.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import brentq
from scipy.signal import savgol_filter


def smooth_v_total(
    v_total: np.ndarray, n_r: int,
) -> np.ndarray:
    """Per-halo Savitzky-Golay smoothing of v_total along r. NaN-tolerant
    via linear interp fill before SavGol (then re-mask), so individual-
    halo profiles with sparse outer bins don't break smoothing. Window
    auto-scaled to ~10% of n_r (min 5) and forced odd. Returns array of
    same shape as input; halos with too few finite bins pass through
    unsmoothed.
    """
    out = v_total.copy()
    win = max(5, n_r // 10)
    if win % 2 == 0:
        win += 1
    if win > n_r:
        win = n_r if n_r % 2 == 1 else n_r - 1
    if win < 5:
        return out  # too few bins for SavGol — skip
    n_halos = v_total.shape[0]
    for h in range(n_halos):
        vt = v_total[h]
        ok = np.isfinite(vt)
        if int(ok.sum()) < win:
            continue
        if ok.all():
            out[h] = savgol_filter(vt, window_length=win, polyorder=2)
        else:
            xs = np.arange(n_r, dtype=np.float64)
            filled = vt.copy()
            filled[~ok] = np.interp(xs[~ok], xs[ok], vt[ok])
            sm = savgol_filter(filled, window_length=win, polyorder=2)
            out[h] = np.where(ok, sm, np.nan)
    return out


def compute_outer_boundary_milestones(
    profiles_stack: dict,
    property_profiles_stack: dict,
    Omega_m: float,
    Omega_L: float,
    a_scale_factor: float,
    use_physical_units: bool,
    catalog_R: np.ndarray | None = None,
    catalog_V: np.ndarray | None = None,
    catalog_M: np.ndarray | None = None,
    lambda_so_R: np.ndarray | None = None,
    eta_hs: float = 0.01,
    eta_hs_floor_kms: float = 20.0,
    accept_inf: float = 0.05,
    r_inner_frac_inf: float = 1.0,
    r_inner_frac_other: float = 0.3,
    busha05_tolerance_dex: float = 0.05,
) -> dict:
    """Outer-boundary cosmological milestones not extracted by phusis's
    profile engine. Always emits ``lambda_*`` (closed-form max-turnaround
    from ⟨ρ_enc⟩ = 2·ρ_Λ — Pavlidou-Tomaras 2014).

    When per-halo ``catalog_R`` and ``catalog_V`` are supplied AND the
    property profile carries ``mean_vr``, the kinematic outer-boundary
    framework also emits, anchored on the global minimum of v_total in
    the search window:

        ``inf_*``       Infall radius r_inf — global min of smoothed
                        v_total = ⟨v_r,act⟩ (already includes Hubble
                        flow via use_actual_velocities=True) in
                        ``[1.0·R_norm, 0.95·max(r)]``. Acceptance:
                        ``v_total(r_inf) < -accept_inf · V_norm``;
                        otherwise no real infall zone is present and
                        inf_/hs_/turnaround_ all stay NaN (the merged-
                        boundary case is handled below).

        ``hs_*``        Hydrostatic radius r_hs (Busha 2005, Abdullah 2025)
                        — outermost-from-r_inf radius where smoothed
                        v_total crosses up through
                        ``-max(eta_hs·V_norm, eta_hs_floor_kms)``,
                        defaulting to ``-max(0.01·V_norm, 20 km/s)``.
                        Significantly tighter than Busha's original
                        0.1·V_norm so r_hs lands at the actual
                        departure of v_smooth from the hydrostatic
                        core (~ where v_smooth crosses zero) rather
                        than 10% into the dip. The km/s floor
                        prevents the threshold from collapsing below
                        the SavGol shot-noise level for low-V_norm
                        halos. Inner edge of the infall zone.

        ``turnaround_*`` Outer edge of the infall zone — first outward-
                        from-r_inf zero crossing of smoothed v_total.

        ``busha05_*``   Busha 2005 merged-boundary scale. Two firing
                        paths into the same field, both reporting the
                        same physical scale (the single zero-velocity
                        surface that hs and ta merge into):

                        1. CONVERGENCE — when r_inf is accepted but
                           r_hs and r_ta have closed to within
                           ``|log10(r_ta/r_hs)| < busha05_tolerance_dex``
                           (default 0.05 dex). Set to r_ta. Catches
                           the late-transition regime where accretion
                           is winding down but a faint dip is still
                           detectable.

                        2. LATE-TIME ASSUMPTION — when r_inf is
                           rejected (no infall zone) but a single
                           zero crossing of v_smooth exists in the
                           wide search window. At that point r_hs
                           and r_ta are assumed merged; the zero
                           crossing IS the merged surface. Set to
                           that radius. ``hs_R`` and ``turnaround_R``
                           are aliased to the same value so downstream
                           consumers see a single "merged" boundary.

    Without ``catalog_R/V``, only lambda is emitted (legacy fallback).

    Smoothing: per-halo Savitzky-Golay (window auto-scaled to n_r,
    polyorder 2, NaN-tolerant) before feature extraction so individual-
    halo profiles aren't dominated by per-shell shot noise.

    Physics-prior constraint enforced by construction:

        r_hs < r_inf < r_ta ≤ r_lambda    (active path)
        r_busha05 ≤ r_lambda              (late-time merged path)

    r_hs < r_inf < r_ta follows from the bracket geometry (hs from the
    pre-r_inf scan, ta from the post-r_inf scan). r_ta ≤ r_lambda is
    enforced by capping the turnaround search at
    ``min(0.95·r_max, r_lambda)`` — the Pavlidou-Tomaras max-turnaround
    radius is a hard physical upper bound on any turnaround surface.
    Late-time busha05 inherits the same ceiling because hs / ta / b05
    are aliased to the merged surface in that path.

    Source of r_lambda for the ceiling (priority order):

    1. ``lambda_so_R`` argument when provided per-halo. This is the
       SO-bisection result computed on the underlying particle data
       by ``compute_so_catalog_phusis`` and is the authoritative
       downstream value — strictly more accurate than the profile
       interp-based ``lambda_R`` because it doesn't sample-bin the
       (ρ_enc) curve before solving for the threshold crossing.
    2. The profile-derived ``lambda_R`` computed in this function
       (used per-halo when ``lambda_so_R`` is None or NaN for that
       halo).
    3. ``0.95·r_max_valid`` fallback when neither lambda source is
       available (rare — only when ρ_enc never crosses 2·ρ_Λ on the
       on-file profile *and* no SO sidecar was passed in).

    R fields in NATIVE units; M linear Msun/h; V derived via
    sqrt(G·M/r_phys).
    """
    G_COSMO = 4.30091e-9
    H0_h = 100.0  # km/s/(Mpc/h) — h-independent in h-units.
    rho_lambda_h = 3.0 * H0_h * H0_h * float(Omega_L) / (8.0 * np.pi * G_COSMO)
    H_a = H0_h * np.sqrt(
        float(Omega_m) / float(a_scale_factor) ** 3 + float(Omega_L)
    )
    log10_threshold = np.log10(2.0 * rho_lambda_h)

    r_native = np.asarray(profiles_stack["r"], dtype=np.float64)
    M_enc = np.asarray(profiles_stack["M_enc"], dtype=np.float64)
    n_halos, n_r = r_native.shape

    a_to_phys = 1.0 if use_physical_units else float(a_scale_factor)
    r_phys = r_native * a_to_phys

    with np.errstate(invalid="ignore", divide="ignore"):
        rho_phys_enc = np.where(
            r_phys > 0,
            M_enc / ((4.0 / 3.0) * np.pi * r_phys ** 3),
            np.nan,
        )

    lambda_R = np.full(n_halos, np.nan, dtype=np.float64)
    lambda_M = np.full(n_halos, np.nan, dtype=np.float64)
    turnaround_R = np.full(n_halos, np.nan, dtype=np.float64)
    turnaround_M = np.full(n_halos, np.nan, dtype=np.float64)
    inf_R = np.full(n_halos, np.nan, dtype=np.float64)
    inf_M = np.full(n_halos, np.nan, dtype=np.float64)
    hs_R = np.full(n_halos, np.nan, dtype=np.float64)
    hs_M = np.full(n_halos, np.nan, dtype=np.float64)
    busha05_R = np.full(n_halos, np.nan, dtype=np.float64)
    busha05_M = np.full(n_halos, np.nan, dtype=np.float64)

    mean_vr = property_profiles_stack.get("mean_vr")
    if mean_vr is not None:
        mean_vr = np.asarray(mean_vr, dtype=np.float64)
        # mean_vr is the per-shell mean of the ACTUAL radial velocity
        # (peculiar + Hubble flow), set by ``use_actual_velocities=True``
        # in compute_properties_batch. Build:
        #   v_total = actual velocity (peculiar + Hubble) — used to
        #             extract milestone *values* in the active branch.
        #   v_pec   = peculiar component (Hubble subtracted) — used to
        #             *detect* the active vs late-time regime, and to
        #             estimate v_floor for the late-time analytical knee.
        v_total = mean_vr
        v_pec = mean_vr - H_a * r_phys
        v_smooth = smooth_v_total(v_total, n_r)
        v_pec_smooth = smooth_v_total(v_pec, n_r)
    else:
        v_total = None
        v_pec = None
        v_smooth = None
        v_pec_smooth = None

    have_kin_norm = (
        v_smooth is not None and catalog_R is not None
        and (catalog_V is not None or catalog_M is not None)
    )
    if catalog_R is not None:
        catalog_R = np.asarray(catalog_R, dtype=np.float64)
    if catalog_V is not None:
        catalog_V = np.asarray(catalog_V, dtype=np.float64)
    if catalog_M is not None:
        catalog_M_arr = np.asarray(catalog_M, dtype=np.float64)
    else:
        catalog_M_arr = None
    if lambda_so_R is not None:
        lambda_so_R_arr = np.asarray(lambda_so_R, dtype=np.float64)
    else:
        lambda_so_R_arr = None

    for h in range(n_halos):
        ok = (
            np.isfinite(r_native[h]) & (r_native[h] > 0)
            & np.isfinite(M_enc[h]) & (M_enc[h] > 0)
            & np.isfinite(rho_phys_enc[h]) & (rho_phys_enc[h] > 0)
        )
        if int(ok.sum()) < 2:
            continue
        log10_r = np.log10(r_native[h, ok])
        log10_M = np.log10(M_enc[h, ok])
        log10_rho = np.log10(rho_phys_enc[h, ok])

        f_logM_from_logr = interp1d(
            log10_r, log10_M, kind="linear",
            bounds_error=False, fill_value=np.nan,
        )

        # ---- r_Λ via inverse interp on (log10 ρ_enc → log10 r) ----
        order = np.argsort(log10_rho)
        log10_rho_s = log10_rho[order]
        log10_r_s = log10_r[order]
        keep = np.concatenate([[True], np.diff(log10_rho_s) > 0])
        log10_rho_s = log10_rho_s[keep]
        log10_r_s = log10_r_s[keep]
        if (log10_rho_s.size >= 2
                and log10_rho_s[0] <= log10_threshold <= log10_rho_s[-1]):
            f_logr_from_logrho = interp1d(
                log10_rho_s, log10_r_s, kind="linear",
                bounds_error=False, fill_value=np.nan,
            )
            log10_r_lambda = float(f_logr_from_logrho(log10_threshold))
            if np.isfinite(log10_r_lambda):
                lambda_R[h] = 10.0 ** log10_r_lambda
                log10_M_lambda = float(f_logM_from_logr(log10_r_lambda))
                if np.isfinite(log10_M_lambda):
                    lambda_M[h] = 10.0 ** log10_M_lambda

        # ---- kinematic milestones (inf, hs, turnaround, halo) ----
        # Anchor everything on r_inf so a halo either gets a coherent
        # (hs, inf, ta) triple or nothing — never a quiet partial answer.
        if not have_kin_norm:
            continue
        R_norm = float(catalog_R[h])
        if not (np.isfinite(R_norm) and R_norm > 0):
            continue
        V_norm = float(catalog_V[h]) if catalog_V is not None else 0.0
        # V_circ at R_norm. catalog_V is sometimes empty-zeroed in the
        # AA pipeline (the SO V wasn't populated at catalog-load time);
        # fall back to sqrt(G·M/r_phys) from catalog_M when available.
        if not (np.isfinite(V_norm) and V_norm > 0) and catalog_M_arr is not None:
            M_norm = float(catalog_M_arr[h])
            R_phys_norm = R_norm if use_physical_units else R_norm * float(a_scale_factor)
            if (np.isfinite(M_norm) and M_norm > 0
                    and np.isfinite(R_phys_norm) and R_phys_norm > 0):
                V_norm = float(np.sqrt(G_COSMO * M_norm / R_phys_norm))
        if not (np.isfinite(V_norm) and V_norm > 0):
            continue
        vs = v_smooth[h]
        vs_pec = v_pec_smooth[h] if v_pec_smooth is not None else None
        valid = ok & np.isfinite(vs)
        if int(valid.sum()) < 5:
            continue
        rn = r_native[h]

        # Two-stage search window. r_inf needs a TIGHT inner bound
        # (default 1·R_norm) to skip the virialized core where individual
        # halos have noise blips that fake an "infall" minimum. r_hs and
        # busha05 use a WIDER inner bound (default 0.3·R_norm) so the
        # inner edge of the infall zone (which can extend below R_200)
        # is reachable. The outer bound is shared.
        valid_idx = np.where(valid)[0]
        r_max_valid = float(rn[valid_idx[-1]])
        r_outer_search = 0.95 * r_max_valid
        # Physical ceiling for the turnaround / busha05 surfaces:
        # r_ta ≤ r_lambda (Pavlidou-Tomaras max-turnaround is a hard
        # upper bound on any turnaround surface — and r_busha05 is the
        # aliased ta in the late-time merged path, so it inherits the
        # bound). Source preference:
        #   1. lambda_so_R (SO bisection on particles — authoritative)
        #   2. lambda_R    (profile-interp — internal fallback)
        #   3. 0.95·r_max  (when neither lambda is available)
        # The hs/inf searches are unchanged — those live inside r_ta
        # by construction.
        r_lambda_ceiling = float("nan")
        if (lambda_so_R_arr is not None
                and np.isfinite(lambda_so_R_arr[h])
                and lambda_so_R_arr[h] > 0):
            r_lambda_ceiling = float(lambda_so_R_arr[h])
        elif np.isfinite(lambda_R[h]) and lambda_R[h] > 0:
            r_lambda_ceiling = float(lambda_R[h])
        if np.isfinite(r_lambda_ceiling):
            r_outer_ta = min(r_outer_search, r_lambda_ceiling)
        else:
            r_outer_ta = r_outer_search
        r_inner_inf = r_inner_frac_inf * R_norm
        r_inner_other = r_inner_frac_other * R_norm
        if r_inner_inf >= r_outer_search or r_inner_other >= r_outer_search:
            continue
        inf_mask = valid & (rn >= r_inner_inf) & (rn <= r_outer_search)
        wide_mask = valid & (rn >= r_inner_other) & (rn <= r_outer_search)
        inf_idx = np.where(inf_mask)[0]
        wide_idx = np.where(wide_mask)[0]
        if len(inf_idx) < 3 or len(wide_idx) < 3:
            continue

        # Dispatch on PECULIAR profile dip (the kinematic-feature carrier
        # independent of Hubble flow); extract milestone *values* from
        # v_total per the Busha 2005 / Abdullah 2025 definition.
        if vs_pec is None:
            continue
        log10_r_v = np.log10(rn[valid])
        vs_v = vs[valid]
        f_v = interp1d(
            log10_r_v, vs_v, kind="linear",
            bounds_error=False, fill_value=np.nan,
        )

        accept_thresh = -accept_inf * V_norm
        eta_thresh_kms = max(eta_hs * V_norm, float(eta_hs_floor_kms))
        eta_thresh = -eta_thresh_kms

        # Peculiar-min dispatches; r_inf comes from v_total argmin
        v_pec_in = vs_pec[inf_idx]
        if np.isfinite(v_pec_in).any():
            v_pec_min = float(np.nanmin(v_pec_in))
        else:
            v_pec_min = float("inf")
        v_in = vs[inf_idx]
        i_min_local = int(np.argmin(v_in))
        i_min = int(inf_idx[i_min_local])
        v_min = float(vs[i_min])
        search_idx = inf_idx  # alias for the parabolic-vertex code below

        if v_pec_min < accept_thresh:
            # Real infall zone — proceed with the three-milestone path.
            # Sub-bin r_inf via brentq on the derivative of an
            # InterpolatedUnivariateSpline through the smoothed bins —
            # uniform with the brentq-on-interp1d pattern used for r_hs
            # and r_ta. Empirically equivalent in accuracy to a 3-point
            # parabolic vertex (both ~0.015 dex bias dominated by the
            # SavGol smoothing window, not the interpolation kind), but
            # consistent with the rest of the algorithm.
            R_inf_native = float(rn[i_min])
            if 0 < i_min_local < len(search_idx) - 1:
                ia = int(search_idx[i_min_local - 1])
                ic = int(search_idx[i_min_local + 1])
                if rn[ia] > 0 and rn[ic] > 0 and rn[i_min] > 0:
                    try:
                        spl = InterpolatedUnivariateSpline(
                            log10_r_v, vs_v, k=3,
                        )
                        dspl = spl.derivative()
                        log10_r_inf = brentq(
                            dspl, np.log10(rn[ia]), np.log10(rn[ic]),
                        )
                        R_inf_native = 10.0 ** log10_r_inf
                    except (ValueError, RuntimeError):
                        pass  # fall back to integer-bin r_inf
            inf_R[h] = R_inf_native
            log10_M_inf = float(f_logM_from_logr(np.log10(R_inf_native)))
            if np.isfinite(log10_M_inf):
                inf_M[h] = 10.0 ** log10_M_inf

            # 2. r_hs: scan r < r_inf for the outward crossing
            # vs(r) = eta_thresh (going from > eta_thresh to < eta_thresh
            # as r grows, since vs is becoming more negative toward
            # r_inf). Outermost such bracket is the inner edge of the
            # infall zone — the radius at which the virialized core
            # transitions into coherent infall.  Uses the WIDE window
            # so r_hs can reach below R_norm where the inner edge often
            # lives.
            pre_inf = wide_mask & (rn < R_inf_native)
            pre_inf_idx = np.where(pre_inf)[0]
            if len(pre_inf_idx) >= 2:
                bracket = None
                for k in range(len(pre_inf_idx) - 1, 0, -1):
                    ia, ib = int(pre_inf_idx[k - 1]), int(pre_inf_idx[k])
                    if (vs[ia] >= eta_thresh > vs[ib]
                            and vs[ia] != vs[ib]):
                        bracket = (ia, ib)
                        break
                if bracket is not None:
                    f_eta = interp1d(
                        log10_r_v, vs_v - eta_thresh, kind="linear",
                        bounds_error=False, fill_value=np.nan,
                    )
                    try:
                        log10_R_hs = brentq(
                            f_eta,
                            np.log10(rn[bracket[0]]),
                            np.log10(rn[bracket[1]]),
                        )
                        hs_R[h] = 10.0 ** log10_R_hs
                        log10_M_hs = float(f_logM_from_logr(log10_R_hs))
                        if np.isfinite(log10_M_hs):
                            hs_M[h] = 10.0 ** log10_M_hs
                    except (ValueError, RuntimeError):
                        pass

            # 3. r_ta: scan r > r_inf for the first zero crossing
            # vs(r) = 0 (going from negative back to positive). Anchored
            # on r_inf so we get the outer edge of the infall zone, not
            # any zero crossing.  Uses the inf (tight) window since the
            # outer edge sits in the r > R_norm regime by construction.
            # Upper-bounded by r_outer_ta = min(r_outer_search, r_lambda)
            # so the turnaround can never violate the Pavlidou-Tomaras
            # constraint r_ta ≤ r_lambda.
            post_inf = inf_mask & (rn > R_inf_native) & (rn <= r_outer_ta)
            post_inf_idx = np.where(post_inf)[0]
            if len(post_inf_idx) >= 2:
                bracket = None
                for k in range(len(post_inf_idx) - 1):
                    ia, ib = int(post_inf_idx[k]), int(post_inf_idx[k + 1])
                    if vs[ia] <= 0.0 < vs[ib] and vs[ia] != vs[ib]:
                        bracket = (ia, ib)
                        break
                if bracket is not None:
                    try:
                        log10_R_ta = brentq(
                            f_v,
                            np.log10(rn[bracket[0]]),
                            np.log10(rn[bracket[1]]),
                        )
                        turnaround_R[h] = 10.0 ** log10_R_ta
                        log10_M_ta = float(f_logM_from_logr(log10_R_ta))
                        if np.isfinite(log10_M_ta):
                            turnaround_M[h] = 10.0 ** log10_M_ta
                    except (ValueError, RuntimeError):
                        pass

            # busha05 convergence indicator (active-accretion path):
            # r_hs and r_ta merge into a single zero-velocity surface
            # as accretion winds down. Set busha05 = r_ta when
            # |log10(r_ta/r_hs)| < tolerance.
            if (np.isfinite(hs_R[h]) and np.isfinite(turnaround_R[h])
                    and hs_R[h] > 0 and turnaround_R[h] > 0):
                if abs(np.log10(turnaround_R[h] / hs_R[h])) < busha05_tolerance_dex:
                    busha05_R[h] = turnaround_R[h]
                    busha05_M[h] = turnaround_M[h]

            # Active dispatched on v_pec but v_total didn't fully
            # cooperate (intermediate / quasi-decoupled regime). Fall
            # through to the late-time analytical knee for a merged
            # hs=ta=busha05 surface.
            active_complete = (np.isfinite(hs_R[h])
                               and np.isfinite(turnaround_R[h]))
            if not active_complete:
                hs_R[h] = np.nan; hs_M[h] = np.nan
                turnaround_R[h] = np.nan; turnaround_M[h] = np.nan
                inf_R[h] = np.nan; inf_M[h] = np.nan
                busha05_R[h] = np.nan; busha05_M[h] = np.nan
                _try_late_time = True
            else:
                _try_late_time = False
        else:
            _try_late_time = True

        if _try_late_time:
            # Late-time / quiescent regime. Merged hs/ta/busha05 surface
            # = outermost upward crossing of v_smooth through
            # (plateau + eta_thresh_floor), where plateau = median
            # v_smooth in [0.05·R_norm, 0.3·R_norm]. Threshold is
            # RELATIVE to the inner plateau so very-late epochs (where
            # the composite has a small positive offset at small r)
            # don't get caught by an absolute floor too early.
            plateau_lo = 0.05 * R_norm
            plateau_hi = 0.3 * R_norm
            plateau_mask = valid & (rn >= plateau_lo) & (rn <= plateau_hi)
            plateau_arr = vs[plateau_mask]
            plateau_arr = plateau_arr[np.isfinite(plateau_arr)]
            plateau = (float(np.median(plateau_arr))
                       if plateau_arr.size >= 2 else 0.0)
            pos_thresh = plateau + float(eta_hs_floor_kms)
            late_inner = 0.1 * R_norm
            # busha05 inherits the r_outer_ta ceiling: in late-time mode
            # hs / ta / busha05 are aliased to the same merged surface,
            # so the r_ta ≤ r_lambda constraint applies here too.
            late_mask = valid & (rn >= late_inner) & (rn <= r_outer_ta)
            late_idx = np.where(late_mask)[0]
            bracket = None
            for k in range(len(late_idx) - 1, 0, -1):
                ia, ib = int(late_idx[k - 1]), int(late_idx[k])
                if (vs[ia] < pos_thresh <= vs[ib]
                        and vs[ia] != vs[ib]):
                    bracket = (ia, ib)
                    break
            if bracket is not None:
                try:
                    f_target = interp1d(
                        log10_r_v, vs_v - pos_thresh, kind="linear",
                        bounds_error=False, fill_value=np.nan,
                    )
                    log10_R_busha = brentq(
                        f_target,
                        np.log10(rn[bracket[0]]),
                        np.log10(rn[bracket[1]]),
                    )
                    R_b = 10.0 ** log10_R_busha
                    if np.isfinite(R_b) and R_b > 0:
                        log10_M_busha = float(
                            f_logM_from_logr(np.log10(R_b))
                        )
                        M_b = (10.0 ** log10_M_busha
                               if np.isfinite(log10_M_busha)
                               else float("nan"))
                        busha05_R[h] = R_b; busha05_M[h] = M_b
                        hs_R[h]      = R_b; hs_M[h]      = M_b
                        turnaround_R[h] = R_b; turnaround_M[h] = M_b
                except (ValueError, RuntimeError):
                    pass

    def _v_circ(M: np.ndarray, R_native: np.ndarray) -> np.ndarray:
        R_phys = R_native * a_to_phys
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                (R_phys > 0) & (M > 0),
                np.sqrt(G_COSMO * M / R_phys),
                np.nan,
            )

    return {
        "lambda_M":     lambda_M,
        "lambda_R":     lambda_R,
        "lambda_V":     _v_circ(lambda_M, lambda_R),
        "turnaround_M": turnaround_M,
        "turnaround_R": turnaround_R,
        "turnaround_V": _v_circ(turnaround_M, turnaround_R),
        "inf_M":        inf_M,
        "inf_R":        inf_R,
        "inf_V":        _v_circ(inf_M, inf_R),
        "hs_M":         hs_M,
        "hs_R":         hs_R,
        "hs_V":         _v_circ(hs_M, hs_R),
        "busha05_M":    busha05_M,
        "busha05_R":    busha05_R,
        "busha05_V":    _v_circ(busha05_M, busha05_R),
    }
