"""Merger-rate statistics binned/integrated from the flat event catalog.

Operates purely on a :class:`~.mergers.MergerCatalog` (no tree access). Reproduces
the standard merger-rate estimators from the literature:

- ``dN/dxi/dz`` per descendant -- Fakhouri & Ma (2008), Fakhouri+BK (2010):
  ``B(M0, xi, z) = A (M0/1e12)^alpha xi^beta exp(gamma/xi) (1+z)^y``.
- ``dN/dxi/dt`` per descendant [Gyr^-1] -- Rodriguez-Gomez et al. (2015).
- The universal ``xi`` shape ``A xi^beta exp(gamma/xi)`` fit via curve_fit.
- Cumulative major mergers per descendant since a scale factor.
- A mass-growth join (Amoura-style AHG / LGS building blocks) with the
  per-aperture assembly-history files, matched by descendant node.

All rates are *specific* (per descendant): event counts are divided by the number
of descendants in the relevant mass bin, then by the bin widths.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from attrs import define
from scipy.optimize import curve_fit

from .cleaning import PointStatus

PIVOT_MASS_CODE = 100.0  # 1e12 Msun/h in code units of 1e10 Msun/h


@define(slots=True)
class RateGrids:
    """Binned specific merger rates + the bin axes."""

    mass_edges: np.ndarray        # (nm+1,) log10 M0 [code units, 1e10 Msun/h]
    xi_edges: np.ndarray          # (nx+1,) log-spaced mass ratio
    z_edges: np.ndarray           # (nz+1,) redshift
    n_desc_per_mass: np.ndarray   # (nm,) descendants in each M0 bin
    counts: np.ndarray            # (nm, nx, nz) raw event counts
    dN_dxi_dz: np.ndarray         # (nm, nx, nz) per descendant per dxi per dz
    dN_dxi_dt: np.ndarray         # (nm, nx, nz) per descendant per dxi per dt[Gyr]
    dN_dxi: np.ndarray            # (nx,) per descendant per dxi (all mass, all z)

    @property
    def xi_centers(self) -> np.ndarray:
        return np.sqrt(self.xi_edges[:-1] * self.xi_edges[1:])  # log-mid


def _valid(catalog) -> np.ndarray:
    """Events usable for rate statistics (finite ratio, resolved secondary)."""
    return (catalog.status != PointStatus.INVALID) & (catalog.status != PointStatus.FLOORED) \
        & np.isfinite(catalog.xi_peak)


def compute_rate_grids(
    catalog,
    desc_M0_code: np.ndarray,
    moments,
    mass_bins: int | np.ndarray = 6,
    xi_bins: int | np.ndarray = 20,
    z_bins: int | np.ndarray = 8,
    xi_min: float = 0.01,
) -> RateGrids:
    """Bin events into specific ``dN/dxi/dz`` and ``dN/dxi/dt`` grids.

    ``desc_M0_code`` is the per-descendant M0 (Group_M_Crit200, code units), used
    to count descendants per mass bin (the normalization). Events are binned by
    their own ``M0_Crit200``, ``xi_peak`` and ``z_merger``.
    """
    sel = _valid(catalog)
    log_m0_desc = np.log10(desc_M0_code[desc_M0_code > 0])
    ev_logm0 = np.log10(np.where(catalog.M0_Crit200 > 0, catalog.M0_Crit200, np.nan))[sel]
    ev_xi = catalog.xi_peak[sel]
    ev_z = catalog.z_merger[sel]

    if np.isscalar(mass_bins):
        mass_edges = np.linspace(np.nanmin(log_m0_desc), np.nanmax(log_m0_desc) + 1e-6, mass_bins + 1)
    else:
        mass_edges = np.asarray(mass_bins, float)
    if np.isscalar(xi_bins):
        xi_edges = np.logspace(np.log10(xi_min), 0.0, xi_bins + 1)
    else:
        xi_edges = np.asarray(xi_bins, float)
    if np.isscalar(z_bins):
        z_edges = np.linspace(0.0, np.nanmax(ev_z) + 1e-6, z_bins + 1)
    else:
        z_edges = np.asarray(z_bins, float)

    n_desc_per_mass, _ = np.histogram(log_m0_desc, bins=mass_edges)
    counts, _ = np.histogramdd(
        np.column_stack([ev_logm0, ev_xi, ev_z]), bins=[mass_edges, xi_edges, z_edges]
    )

    dxi = np.diff(xi_edges)
    dz = np.diff(z_edges)
    # dt per z bin from the cosmic-age axis (interp age at the z-bin edges)
    z_axis, age_axis = moments.redshifts, moments.ages
    order = np.argsort(z_axis)
    age_at = np.interp(z_edges, z_axis[order], age_axis[order])
    dt = np.abs(np.diff(age_at))  # Gyr per z bin

    norm = n_desc_per_mass[:, None, None].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_desc = counts / np.where(norm > 0, norm, np.nan)
        dN_dxi_dz = per_desc / (dxi[None, :, None] * dz[None, None, :])
        dN_dxi_dt = per_desc / (dxi[None, :, None] * np.where(dt > 0, dt, np.nan)[None, None, :])

    n_desc_total = max(int((desc_M0_code > 0).sum()), 1)
    counts_xi, _ = np.histogram(ev_xi, bins=xi_edges)
    dN_dxi = counts_xi / (n_desc_total * dxi)

    return RateGrids(
        mass_edges=mass_edges, xi_edges=xi_edges, z_edges=z_edges,
        n_desc_per_mass=n_desc_per_mass, counts=counts,
        dN_dxi_dz=dN_dxi_dz, dN_dxi_dt=dN_dxi_dt, dN_dxi=dN_dxi,
    )


def _xi_shape(xi: np.ndarray, A: float, beta: float, gamma: float) -> np.ndarray:
    """Universal mass-ratio shape ``A xi^beta exp(gamma/xi)`` (Fakhouri & Ma 2008)."""
    return A * xi ** beta * np.exp(gamma / xi)


def completeness_xi(min_sec_particles: int, min_num_particles: int) -> float:
    """xi below which the rate is incomplete: a secondary at ratio xi of the
    smallest resolved descendant falls below the ``min_sec_particles`` floor.

    ``xi_complete ~ min_sec_particles / min_num_particles`` (e.g. 100/1000 = 0.1).
    Below this, sub-floor secondaries are FLOORED out and the measured ``dN/dxi``
    turns over -- so the universal slope must be fit only above it.
    """
    return float(min_sec_particles) / float(max(min_num_particles, 1))


def fit_xi_shape(xi_centers: np.ndarray, dN_dxi: np.ndarray,
                 xi_fit_min: float | None = None, with_cutoff: bool = False) -> dict:
    """Fit the marginal ``dN/dxi`` in LOG space above the completeness limit.

    By default a pure power law ``A xi^beta`` (the interpretable diagnostic): the
    fit is on ``log10(dN/dxi)`` so every decade weighs equally (a linear
    least-squares is swamped by the high-amplitude small-xi points and ignores the
    steep high-xi tail), and is restricted to ``xi >= xi_fit_min`` (the resolution
    completeness limit; below it the rate turns over -- see
    :func:`completeness_xi`). ``with_cutoff=True`` adds the literature
    ``exp(gamma/xi)`` term, but for a high-xi-declining sample that term is
    degenerate with ``beta``, so the power law is the default.

    Returns ``A, beta, gamma`` (+ 1-sigma errors), ``xi_fit_min`` and ``n_points``.
    Note: for a narrow, massive, single-snapshot descendant set the marginal
    ``beta`` is steeper than the universal ~-2 (selection + small-N at high xi);
    the mass/z-resolved ``dN/dxi/dz`` grid is the object for a true comparison.
    """
    good = np.isfinite(dN_dxi) & (dN_dxi > 0) & np.isfinite(xi_centers)
    out = {"A": np.nan, "beta": np.nan, "gamma": 0.0, "A_err": np.nan,
           "beta_err": np.nan, "gamma_err": np.nan, "n_points": 0, "xi_fit_min": np.nan}
    if good.sum() < 3:
        return out
    xc, yc = xi_centers[good], dN_dxi[good]

    if xi_fit_min is None:
        xi_fit_min = float(xc[int(np.argmax(yc))])   # fall back to the peak bin
    keep = xc >= xi_fit_min
    if keep.sum() >= 3:
        xc, yc = xc[keep], yc[keep]
    out["xi_fit_min"] = float(xc.min())
    out["n_points"] = int(len(xc))
    ln10 = np.log(10.0)

    try:
        if with_cutoff:
            def model(xi, log_a, beta, gamma):
                return log_a + beta * np.log10(xi) + (gamma / xi) / ln10
            p0 = (np.log10(yc.max()), -2.0, 0.0)
        else:
            def model(xi, log_a, beta):
                return log_a + beta * np.log10(xi)
            p0 = (np.log10(yc.max()), -2.0)
        popt, pcov = curve_fit(model, xc, np.log10(yc), p0=p0, maxfev=20000)
        err = np.sqrt(np.diag(pcov))
        out.update(A=10.0 ** popt[0], beta=popt[1],
                   A_err=10.0 ** popt[0] * ln10 * err[0], beta_err=err[1])
        if with_cutoff:
            out.update(gamma=popt[2], gamma_err=err[2])
    except (RuntimeError, ValueError):
        pass
    return out


def specific_rate_by_mass(
    catalog,
    h: float,
    a_targets=(1.0, 10.0, 100.0),
    host_len_bins=(1e3, 1e4, 1e5, 1e8),
    min_sec_particles: int = 100,
    xi_bins: int = 24,
    xi_min: float = 0.003,
    a_window_dex: float = 0.5,
) -> dict:
    """Specific merger rate ``dN/dxi`` resolved by HOST mass + epoch.

    The mass-ratio function ``dN/dxi`` should trace the progenitor/halo mass
    function (Dong et al. 2022, ApJ 929, 120): ``~ xi^{-1.9}`` rising toward small
    xi, with a cutoff near xi=1. Recovering that shape REQUIRES binning by host
    mass and respecting the host-dependent resolution limit: the tree resolves
    secondaries only above ``min_sec_particles``, so the smallest measurable ratio
    in a host of ``N_host`` particles is ``xi_min(host) ~ min_sec_particles/N_host``.
    A single marginal stacked over all hosts is flattened by this limit.

    Events are binned by the main progenitor's peak (infall) particle count
    (``main_peak_len``); each bin carries a completeness ``xi_complete`` =
    ``min_sec_particles / host_len_lo`` below which it is incomplete. Rates are per
    contributing descendant, per unit xi, accumulated in a +/- ``a_window_dex``
    (log a) window about each target scale factor. ``host_mass_msun`` gives the
    approximate log10 Msun range of each host bin (from the data's particle mass).
    """
    status = catalog.status
    base = (status != PointStatus.INVALID) & np.isfinite(catalog.xi_peak) \
        & (catalog.sec_peak_len >= min_sec_particles) & (catalog.main_peak_len > 0)

    # particle mass in code units (1e10 Msun/h) from M_main_peak / N_host
    ok_mp = base & (catalog.M_main_peak > 0)
    m_p_code = float(np.median(catalog.M_main_peak[ok_mp] / catalog.main_peak_len[ok_mp]))

    host_edges = np.asarray(host_len_bins, float)
    n_host_bins = len(host_edges) - 1
    xi_edges = np.logspace(np.log10(xi_min), 0.0, xi_bins + 1)
    xi_centers = np.sqrt(xi_edges[:-1] * xi_edges[1:])
    dxi = np.diff(xi_edges)
    xi_complete = min_sec_particles / host_edges[:-1]              # per host bin
    host_mass_msun = np.log10(host_edges * m_p_code) + 10.0 - np.log10(h)

    xi = catalog.xi_peak
    am = catalog.a_merger
    hl = catalog.main_peak_len.astype(float)
    dn = catalog.desc_node
    host_bin = np.digitize(hl, host_edges) - 1                     # -1 below, n_host_bins above

    by_a = {}
    for at in a_targets:
        lo, hi = at * 10.0 ** (-a_window_dex), at * 10.0 ** (a_window_dex)
        win = base & (am >= lo) & (am < hi)
        rate = np.full((n_host_bins, len(xi_centers)), np.nan)
        n_desc_bin = np.zeros(n_host_bins, dtype=int)
        for hb in range(n_host_bins):
            m = win & (host_bin == hb)
            n_desc = len(np.unique(dn[m]))
            n_desc_bin[hb] = n_desc
            if n_desc == 0:
                continue
            cnt, _ = np.histogram(xi[m], bins=xi_edges)
            rate[hb] = cnt / (n_desc * dxi)
        by_a[at] = {"rate": rate, "n_in": int(win.sum()), "n_desc_bin": n_desc_bin,
                    "window": (lo, hi)}

    return {"xi_centers": xi_centers, "xi_edges": xi_edges, "host_edges": host_edges,
            "host_mass_msun": host_mass_msun, "xi_complete": xi_complete,
            "m_p_code": m_p_code, "a_targets": tuple(a_targets), "by_a": by_a}


def dong_eq2(xi: np.ndarray) -> np.ndarray:
    """Dong et al. (2022) Eq. 2 double-Schechter specific merger rate at z=0 (L1).

    ``dN/dξ/dlogM = (a1 ξ^b1 + a2 ξ^b2) exp(c ξ^d)`` with the published best-fit.
    """
    a1, a2, b1, b2, c, d = 0.467, 17.362, -1.717, 0.212, -3.682, 0.937
    xi = np.asarray(xi, float)
    return (a1 * xi ** b1 + a2 * xi ** b2) * np.exp(c * xi ** d)


def dong_specific_rate(
    M_sat: np.ndarray,
    M_host: np.ndarray,
    is_splashout: np.ndarray,
    infall_a: np.ndarray,
    growth_sum: np.ndarray,            # [fine_mass_bin, snap] Σ Δlog10 M_host
    growth_mass_edges: np.ndarray,     # (n_fine+1,) log10 M [Msun/h]
    growth_a: np.ndarray,              # (n_snap,) scale factor per growth snapshot column
    host_mass_bins: np.ndarray,        # (nb+1,) log10 M [Msun/h] for the output bins
    xi_bins=24,
    xi_min: float = 1e-3,
    a_range: tuple[float, float] | None = None,
) -> dict:
    """Reconstruct Dong's ``dN/dξ/dlogM`` from the stored infall + host-growth tables.

    Numerator: infall mergers (``~is_splashout``) of ratio ``ξ = M_sat/M_host``,
    binned by host mass. Denominator: the host log-mass-growth ``Σ Δlog M_host``
    summed over the fine mass bins inside each output host-mass bin (Dong Eq. 1).
    ``a_range`` optionally restricts both numerator and denominator to an epoch.
    Compare the result to :func:`dong_eq2`.
    """
    M_sat = np.asarray(M_sat, float); M_host = np.asarray(M_host, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = M_sat / M_host
        logMh = np.log10(M_host) + 10.0
    infall = (~np.asarray(is_splashout, bool)) & np.isfinite(xi) & (xi > 0) & (xi <= 1) \
        & np.isfinite(logMh)
    if a_range is not None:
        infall &= (infall_a >= a_range[0]) & (infall_a < a_range[1])

    xi_edges = np.logspace(np.log10(xi_min), 0.0, xi_bins + 1)
    xi_centers = np.sqrt(xi_edges[:-1] * xi_edges[1:])
    dxi = np.diff(xi_edges)
    host_mass_bins = np.asarray(host_mass_bins, float)
    nb = len(host_mass_bins) - 1

    num, _, _ = np.histogram2d(logMh[infall], xi[infall], bins=[host_mass_bins, xi_edges])

    # denominator: assign each fine growth-mass bin to an output bin; sum over the
    # selected snapshots; then sum the fine bins inside each output bin.
    fine_centers = 0.5 * (growth_mass_edges[:-1] + growth_mass_edges[1:])
    fine_to_out = np.digitize(fine_centers, host_mass_bins) - 1
    snap_sel = np.ones(growth_sum.shape[1], dtype=bool)
    if a_range is not None:
        snap_sel = (growth_a >= a_range[0]) & (growth_a < a_range[1])
    fine_sum = growth_sum[:, snap_sel].sum(axis=1)            # (n_fine,) Σ over snaps
    denom = np.zeros(nb)
    for ob in range(nb):
        denom[ob] = fine_sum[fine_to_out == ob].sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = num / np.where(denom > 0, denom, np.nan)[:, None] / dxi[None, :]
    return {"rate": rate, "xi_centers": xi_centers, "xi_edges": xi_edges,
            "host_mass_bins": host_mass_bins, "denom": denom, "counts": num}


def cumulative_major_per_descendant(catalog, n_halos: int, xi_min: float = 1.0 / 3.0,
                                    a_min: float = 0.0) -> float:
    """Mean number of mergers with ``xi_peak >= xi_min`` since ``a_min``, per descendant."""
    sel = _valid(catalog) & (catalog.xi_peak >= xi_min) & (catalog.a_merger >= a_min)
    return float(sel.sum()) / max(n_halos, 1)


@define(slots=True)
class GrowthTest:
    """Per-descendant mass-growth-from-mergers building blocks (Amoura AHG/LGS)."""

    desc_node: np.ndarray         # (n,)
    f_merger_major: np.ndarray    # (n,) sum(M_sec_peak | major) / M0
    f_merger_all: np.ndarray      # (n,) sum(M_sec_peak | resolved) / M0
    had_major: np.ndarray         # (n,) bool, any major merger
    n_major: np.ndarray           # (n,)


def assembly_growth_test(catalog, histories_path: str | Path,
                         a_min: float = 0.0) -> GrowthTest:
    """Join the catalog to descendant M0 (assembly-history file) for AHG/LGS.

    ``histories_path`` is a per-aperture assembly-history HDF5 (Crit200). Matched
    by descendant node. ``f_merger_*`` is the fraction of the final mass supplied
    by (major / all resolved) mergers -- the integrand of the average-halo-growth
    (AHG) test; ``had_major`` underlies the large-growth-systems (LGS) fraction.
    """
    with h5py.File(histories_path, "r") as f:
        desc_node = f["group/desc_index"][:]
        masses_log = f["group/masses"][:]  # log10 M_final, code units
    m0 = np.where(np.isfinite(masses_log), 10.0 ** masses_log, np.nan)
    node_to_row = {int(d): i for i, d in enumerate(desc_node)}

    n = len(desc_node)
    sum_major = np.zeros(n)
    sum_all = np.zeros(n)
    n_major = np.zeros(n, dtype=int)

    sel = _valid(catalog) & (catalog.a_merger >= a_min)
    for k in np.flatnonzero(sel):
        row = node_to_row.get(int(catalog.desc_node[k]))
        if row is None:
            continue
        ms = catalog.M_sec_peak[k]
        if not np.isfinite(ms):
            continue
        sum_all[row] += ms
        if catalog.is_major[k]:
            sum_major[row] += ms
            n_major[row] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        f_major = sum_major / m0
        f_all = sum_all / m0
    return GrowthTest(desc_node=desc_node, f_merger_major=f_major, f_merger_all=f_all,
                      had_major=n_major > 0, n_major=n_major)
