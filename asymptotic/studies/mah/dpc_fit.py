"""Milestone-anchored DPC(u=2) splashback-MAH fitting form + finite-reference fitter.

Faithful extraction of the canonical fit engine developed over 16 passes in
``notebooks/chapter5_assembly/explore_mah_dpc_jackknife.py`` (full development
record: ``Thesis Notes/mah_fitting_form_development_notes.md``). Pure fit math +
scipy — no project-specific imports — so it is reused by the standalone
``postprocessing/save_mah_fits.py`` driver and by the notebook.

Target quantity: ``Psihat = Psi/Psi(a_ref) = M(a)/M(a_ref)`` — the DATA's
finite-reference convention, with ``M_inf = M(a_ref)`` and ``a_ref`` DERIVED from
the data (the final snapshot's scale factor, ~100 for the primary suite; 1e4 for
the extended sample). The model prediction for that quantity is

    ln Psihat = -K (B(a) - B(a_ref)) + O(a) - O(a_ref)
    B(a) = (1-phi)(a/a_f)^-g1 + phi (a/a_f)^-g2                  (B(a_f) = 1)
    O(a) = 2h / [ (a/a_pk)^-2 + (a/a_pk)^v ]                     (u = 2 transient)
    K    = [ln2 + O(a_f) - O(a_ref)] / (1 - B(a_ref))

Anchors exact for EVERY theta: ``Psihat(0)=0``, ``Psihat(a_f)=1/2`` (same
convention as the data a_50), ``Psihat(a_ref)=1`` (so excluding the a_ref snapshot
from the fit is rigorous). The rise is provably monotone (both core amplitudes
>= 0); the overshoot comes solely from the transient. Nesting is exact:
``h=0, phi=0`` -> Busha+07 Eq. 9 in its own a0-normalized form (a0=a_ref) with
``a_c = (g1 A / S)^(1/g1)``, ``A = K a_f^g1``; additionally ``g1=1`` -> Wechsler+02.

The residual relaxation ``eps = Psi(a_ref)-1 = exp(O(a_ref) - K B(a_ref)) - 1`` is
a DERIVED prediction (M_sp(a_ref) sits ~0.1-1% above the true asymptote),
extrapolation-dominated — quote with caution. A soft regularizer
``O(a_ref) <= 0.005`` in lnPsi keeps the beyond-reference tail (a flat direction of
the finite-reference likelihood) from drifting into broad-bump degeneracies.

Internal parameter vector (fit space, keeps g1 > g2 and positivity by
construction): ``theta = (ln a_f, phi, delta = g1 - g2, g2, h, ln a_pk, v)``.
Physical params: ``(a_f, phi, gamma1 = g2 + delta, gamma2, h, a_pk, v)`` + eps.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

U_FIX = 2.0              # transient rise index (u=2, fixed — pass 4)
SIG_FLOOR = 0.002        # SE-of-median floor (relative)
SYS_FLOOR = 0.005        # systematic error floor added in quadrature (pass 7)
S_CONV = 2.0             # Wechsler a_c rate convention dlnM/dlna(a_c) = S
O_REF_CAP, O_REF_SIG = 0.005, 0.002   # transient-decay regularizer (pass 9)
DEFAULT_A_REF = 100.0    # default only; drivers pass the data-derived a_ref

# theta = (ln a_f, phi, delta=g1-g2, g2, h, ln a_pk, v)
LB = [np.log(0.05), 0.0, 0.05, 0.35, 0.0, np.log(2.0), 0.15]
UB = [np.log(5.0), 0.98, 8.0, 4.0, 0.12, np.log(60.0), 4.0]
PKEYS = ("a_f", "phi", "gamma1", "gamma2", "h", "a_pk", "v", "eps")


def bump_u2(a, h, apk, v):
    """Double-power transient O(a) = 2h / [(a/a_pk)^-2 + (a/a_pk)^v]."""
    x = np.asarray(a, float) / apk
    return 2.0 * h / (x ** (-U_FIX) + x ** v)


def sys_weights(se_rel):
    """Fit weights from the relative SE-of-median: 1/sqrt(SE^2 + SYS_FLOOR^2)."""
    return 1.0 / np.sqrt(np.maximum(se_rel, SIG_FLOOR) ** 2 + SYS_FLOOR ** 2)


def _kb(th, a_ref):
    laf, phi, dlt, g2, h, lapk, v = th
    g1 = g2 + dlt
    af_, apk = np.exp(laf), np.exp(lapk)
    b_ref = (1 - phi) * (a_ref / af_) ** (-g1) + phi * (a_ref / af_) ** (-g2)
    o_ref = bump_u2(a_ref, h, apk, v)
    o_af = bump_u2(af_, h, apk, v)
    k = (np.log(2.0) + o_af - o_ref) / (1.0 - b_ref)
    return k, b_ref, o_ref, af_, apk, g1, g2, phi, h, v


def model_lnpsihat(a, th, a_ref=DEFAULT_A_REF):
    """ln Psihat(a) in the finite-reference convention (Psihat(a_ref)=1)."""
    k, b_ref, o_ref, af_, apk, g1, g2, phi, h, v = _kb(th, a_ref)
    x = np.asarray(a, float) / af_
    b = (1 - phi) * x ** (-g1) + phi * x ** (-g2)
    return -k * (b - b_ref) + bump_u2(a, h, apk, v) - o_ref


def eps_of(th, a_ref=DEFAULT_A_REF):
    """Derived residual relaxation eps = Psi(a_ref) - 1."""
    k, b_ref, o_ref, *_ = _kb(th, a_ref)
    return float(np.exp(o_ref - k * b_ref) - 1.0)


def _resid(th, a, y, w, a_ref):
    r = (y - model_lnpsihat(a, th, a_ref)) * w
    o_ref = bump_u2(a_ref, th[4], np.exp(th[5]), th[6])
    return np.append(r, max(0.0, o_ref - O_REF_CAP) / O_REF_SIG)


def fit_dpc(a, y, w, af0, bump0=(0.03, np.log(6.0), 0.9), a_ref=DEFAULT_A_REF):
    """Multistart finite-reference DPC fit. ``y = ln p50`` (median lnPsi),
    ``w`` = weights. ``af0`` = initial a_f guess (data a_50). Returns
    ``dict(theta, ymod, rss)`` — the best over the (phi0 x (g2,delta)) starts."""
    best = None
    for phi0 in (0.1, 0.4, 0.7):
        for g20, dlt0 in ((0.6, 1.2), (1.0, 0.8), (0.5, 2.0)):
            th0 = np.clip([np.log(af0), phi0, dlt0, g20, *bump0], LB, UB)
            try:
                sol = least_squares(_resid, th0, args=(a, y, w, a_ref),
                                    bounds=(LB, UB), method="trf", max_nfev=15000)
                c2 = float(np.sum(sol.fun ** 2))
                if best is None or c2 < best[0]:
                    best = (c2, sol.x)
            except Exception:
                continue
    if best is None:
        raise RuntimeError("DPC fit failed on all starts")
    c2, th = best
    return dict(theta=th, ymod=model_lnpsihat(a, th, a_ref), rss=c2)


def warm_fit(a, y, w, th0, a_ref=DEFAULT_A_REF):
    """Warm-started refit from ``th0`` (jackknife/bootstrap folds)."""
    sol = least_squares(_resid, np.clip(th0, LB, UB), args=(a, y, w, a_ref),
                        bounds=(LB, UB), method="trf", max_nfev=8000)
    return sol.x


def theta_to_params(th):
    laf, phi, dlt, g2, h, lapk, v = th
    return dict(a_f=float(np.exp(laf)), phi=float(phi), gamma1=float(g2 + dlt),
                gamma2=float(g2), h=float(h), a_pk=float(np.exp(lapk)), v=float(v))


def model_overshoot(theta, a_lo, a_hi, a_ref=DEFAULT_A_REF):
    """Peak overshoot (percent above the reference) and the epoch it occurs."""
    ad = np.logspace(np.log10(a_lo), np.log10(a_hi), 600)
    ln = model_lnpsihat(ad, theta, a_ref)
    i = int(np.argmax(ln))
    return 100.0 * (np.exp(ln[i]) - 1.0), float(ad[i])


def a_c_busha(theta, a_ref=DEFAULT_A_REF):
    """Busha+07 a_c via the h,phi->0 bridge: a_c = (g1 A / S)^(1/g1), A = K a_f^g1.
    Physically meaningful mainly where the fit reduces toward the Busha family
    (h, phi small); returned for every bin as a diagnostic."""
    k, b_ref, o_ref, af_, apk, g1, g2, phi, h, v = _kb(theta, a_ref)
    A = k * af_ ** g1
    return float((g1 * A / S_CONV) ** (1.0 / g1)) if A > 0 else np.nan
