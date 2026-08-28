"""Fakhouri-Ma / Amoura FOF halo merger rate ``B/n(M0, xi, z)`` (Product 1).

This is the **primary cosmological-test product** (Amoura et al. 2024, Ω_m–σ₈).
It is the FOF/halo merger rate, NOT the subhalo-coalescence rate that
``mergers.detect_mergers_forest`` measures. The distinction is the merger
*timing*: Fakhouri & Ma (2008) / Fakhouri+BK (2010) and Amoura define a merger at
the **infall / FOF-linking / virial-crossing** moment ("counted as soon as a
progenitor's particles are included in a descendant… roughly when the virial
radii first overlap"), which in an HBT subhalo tree is a central→satellite
transition -- exactly the infall events from :func:`mergers.detect_infalls`.

Estimator (FM+2010 §2.3): every descendant FOF halo at every z_d has N_p FOF
progenitors; count N_p−1 mergers, each ξ = M_i/M_1 with M_1 the most massive
(host) progenitor. The per-halo rate is

    B/n(M0, ξ, z) = dN_m/dξ/dz = N_merge(M0, ξ, z) / n(M0, z) / Δξ / Δz,

where n(M0, z) is the number of ALL resolved central FOF haloes in that (M0, z)
bin -- not just those with a merger. We bin natively per snapshot transition
(z_p → z_d) and convert counts to dz with the snapshot redshift step.

The marginal dN/dξ follows the universal FM+2010 Eq. 1 shape; :func:`fit_bn`
fits it. :func:`ahg_lgs` integrates B/n over the last dynamical time for the
average-halo-growth (AHG, Eq. 7) and large-growth-system fraction (LGS, Eq. 8).
"""
from __future__ import annotations

import numpy as np
from attrs import define, field
from scipy.optimize import curve_fit

# Fakhouri+BK 2010 resolution cuts (descendant ≥1000 part, progenitor ≥40 part).
DEFAULT_MIN_DESC_PARTICLES = 1000
DEFAULT_MIN_PROG_PARTICLES = 40
DEFAULT_MASS_DEF = "Virial"        # Bryan-Norman Δ_vir, as in Dong+2022 / EPS
# Comoving critical density in code units (1e10 Msun/h) per (Mpc/h)^3, h-independent.
RHO_CRIT_CODE = 27.754


def particle_mass_code(omega_m: float, box_size: float, num_particles: int) -> float:
    """Dark-matter particle mass in code units (1e10 Msun/h).

    ``m_p = Ω_m · ρ_crit,0 · (L_box/N)^3`` with L in Mpc/h. h-independent in these
    units. Used to convert a FOF ``GroupMass`` to a FOF particle count for the
    Fakhouri resolution cut (these trees store no GroupLen column).
    """
    return float(omega_m) * RHO_CRIT_CODE * (float(box_size) / float(num_particles)) ** 3


@define(slots=True)
class FoFMergerRate:
    """Forest-wide merger rate ``B/n`` on a (mass, ξ, snapshot) grid.

    Used for both products: ``kind="fof_infall"`` (Fakhouri/Amoura FOF mergers,
    the primary cosmological-test product) and ``kind="coalescence"`` (the
    Rodriguez-Gomez subhalo-coalescence comparison product). Same estimator
    shape; the two differ only in which events feed it.
    """

    kind: str
    mass_def: str
    mass_edges: np.ndarray        # (nm+1,) log10 M0 [Msun/h]
    xi_edges: np.ndarray          # (nx+1,) log-spaced mass ratio
    snapshot_ids: np.ndarray      # (ns,)
    a_of_snap: np.ndarray         # (ns,) descendant scale factor per snapshot
    z_of_snap: np.ndarray         # (ns,) descendant redshift per snapshot
    step_of_snap: np.ndarray      # (ns,) the |Δa| or |Δz| step (prog→desc) divided out
    rate_axis: str                # "a" (B/n = dN_m/dξ/da) or "z" (dN_m/dξ/dz)
    n_desc: np.ndarray            # (nm, ns) census: resolved central FOF haloes
    counts: np.ndarray            # (nm, nx, ns) net infall-merger counts
    bn: np.ndarray                # (nm, nx, ns) B/n = dN_m/dξ/d(axis) per descendant
    len_source: str               # "fof" (GroupMass/m_p) or "subhalo" (SubhaloLen)
    min_desc_particles: int
    min_prog_particles: int
    net: bool

    @property
    def xi_centers(self) -> np.ndarray:
        return np.sqrt(self.xi_edges[:-1] * self.xi_edges[1:])

    @property
    def mass_centers(self) -> np.ndarray:
        return 0.5 * (self.mass_edges[:-1] + self.mass_edges[1:])

    def marginal_dN_dxi(self, mass_range=None, z_range=None) -> np.ndarray:
        """Σ over mass+snapshot of B/n·Δz / (#descendants) → dN/dξ per halo.

        Sums per-snapshot per-descendant merger counts (already integrated over
        each snapshot's dz) over the selected mass and redshift range, then
        divides by Δξ. Optional ``mass_range``/``z_range`` are (lo, hi) in the
        grid's own units (log10 Msun/h, redshift).
        """
        mm = self._mass_mask(mass_range)
        zm = self._z_mask(z_range)
        per_desc = _safe_div(self.counts, self.n_desc[:, None, :])  # (nm,nx,ns)
        # nansum: empty (mass,snap) cells are 0/0 -> nan and must not poison the sum
        sel = np.nansum(per_desc[mm][:, :, zm], axis=(0, 2))        # (nx,)
        dxi = np.diff(self.xi_edges)
        return sel / dxi

    def cumulative_vs_a(self, xi_mins=(1.0 / 3.0, 0.1, 0.01, 0.001), mass_range=None,
                        per_step: bool = True) -> dict:
        """Cumulative merger rate ``N(>ξ_min)`` per descendant vs scale factor.

        For each ``ξ_min`` threshold, sum ``counts/n_desc`` over ξ ≥ ξ_min and the
        selected mass range, per snapshot → per-descendant cumulative count at each
        a. With ``per_step`` it is divided by the snapshot's Δ(axis) to give a rate
        ``dN(>ξ_min)/d(axis)`` (Fakhouri Fig 3 / Dong Fig 1-right style). Returns
        ``a``, ``z``, and a ``curves`` dict keyed by ξ_min.
        """
        mm = self._mass_mask(mass_range)
        per_desc = _safe_div(self.counts, self.n_desc[:, None, :])      # (nm,nx,ns)
        xi_c = self.xi_centers
        step = np.where(self.step_of_snap > 0, self.step_of_snap, np.nan)
        curves = {}
        for xm in xi_mins:
            sel = xi_c >= xm
            cum = np.nansum(per_desc[mm][:, sel, :], axis=(0, 1))        # (ns,) per descendant
            curves[float(xm)] = cum / step if per_step else cum
        return {"a": self.a_of_snap, "z": self.z_of_snap, "rate_axis": self.rate_axis,
                "per_step": per_step, "curves": curves}

    def _mass_mask(self, rng):
        c = self.mass_centers
        return np.ones_like(c, bool) if rng is None else (c >= rng[0]) & (c < rng[1])

    def _z_mask(self, rng):
        z = self.z_of_snap
        return np.ones_like(z, bool) if rng is None else (z >= rng[0]) & (z < rng[1])

    def to_hdf5(self, group) -> None:
        for name in ("mass_edges", "xi_edges", "snapshot_ids", "a_of_snap",
                     "z_of_snap", "step_of_snap", "n_desc", "counts", "bn"):
            group.create_dataset(name, data=getattr(self, name))
        group.create_dataset("xi_centers", data=self.xi_centers)
        group.create_dataset("mass_centers", data=self.mass_centers)
        group.create_dataset("dN_dxi", data=self.marginal_dN_dxi())   # all-mass, all-z marginal
        group.attrs["kind"] = self.kind
        group.attrs["mass_def"] = self.mass_def
        group.attrs["rate_axis"] = self.rate_axis
        group.attrs["len_source"] = self.len_source
        group.attrs["min_desc_particles"] = self.min_desc_particles
        group.attrs["min_prog_particles"] = self.min_prog_particles
        group.attrs["net"] = self.net


def _safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / np.where(den > 0, den, np.nan)


def _backward_step(x: np.ndarray) -> np.ndarray:
    """|Δx| from the previous snapshot (progenitor→descendant); first reuses next."""
    d = np.empty_like(x, dtype=float)
    if len(x) > 1:
        d[1:] = np.abs(x[:-1] - x[1:])
        d[0] = d[1]
    else:
        d[:] = 1.0
    return d


def _snapshot_axes(moments, rate_axis: str = "a"):
    """(snapshot_ids, a_of_snap, z_of_snap, step_of_snap) on the ascending-snap grid.

    ``step`` is the |Δa| (``rate_axis="a"``) or |Δz| (``rate_axis="z"``) from the
    progenitor snapshot to this one -- the differential the rate is divided by. For
    a future-extended box (a>1, z<0) only the scale-factor axis is meaningful late;
    the redshift axis is for verification against the literature up to a=1.
    """
    snaps = np.asarray(moments.snapshot_ids)
    a = np.asarray(moments.scale_factors, dtype=float)
    z = np.asarray(moments.redshifts, dtype=float)
    step = _backward_step(a if rate_axis == "a" else z)
    return snaps, a, z, step


def _fof_particle_count(store, rows, particle_mass: float) -> np.ndarray:
    """FOF particle count = GroupMass / m_p (these trees store no GroupLen)."""
    gm = store.column("GroupMass").astype(float)[rows]
    return gm / particle_mass


def descendant_census(
    store, moments, backend, mass_def: str, mass_edges: np.ndarray,
    min_desc_particles: int = DEFAULT_MIN_DESC_PARTICLES,
    len_source: str = "fof", particle_mass: float | None = None,
) -> np.ndarray:
    """``n(M0, snap)``: count of resolved central FOF haloes per (mass, snapshot).

    The B/n normalization denominator. Every central FOF node passing the
    resolution cut is one descendant halo, binned by its FOF mass
    ``Group_M_<mass_def>`` (log10 Msun/h) and ``SnapNum``. ``len_source="fof"``
    (default, Fakhouri) cuts on the FOF particle count ``GroupMass/m_p``;
    ``"subhalo"`` cuts on the central's ``SubhaloLen`` (for cross-checking against
    HBT-mass reference papers).
    """
    snaps, _, _, _ = _snapshot_axes(moments)
    col = backend.resolve_column("M", mass_def)
    if col is None:
        raise KeyError(f"mass def {mass_def!r} not available in {store.path.name}")
    rows = np.arange(store.n_rows)
    central = store.is_central(rows)
    if len_source == "fof":
        if particle_mass is None:
            raise ValueError("len_source='fof' needs particle_mass")
        resolved = _fof_particle_count(store, rows, particle_mass) >= min_desc_particles
    else:
        resolved = store.column("SubhaloLen") >= min_desc_particles
    m = central & resolved
    mass = store.column(col).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logM = np.log10(np.where(mass > 0, mass, np.nan)) + 10.0     # Msun/h
    snap = store.column("SnapNum")
    ok = m & np.isfinite(logM)
    snap_edges = np.append(snaps - 0.5, snaps[-1] + 0.5)
    n_desc, _, _ = np.histogram2d(logM[ok], snap[ok], bins=[mass_edges, snap_edges])
    return n_desc       # (nm, ns)


def rate_from_events(
    logM0, xi, snap, weights, store, moments, backend, *,
    kind: str,
    mass_def: str,
    mass_edges: np.ndarray | None = None,
    xi_edges: np.ndarray | None = None,
    xi_bins: int = 24,
    xi_min: float = 1e-3,
    rate_axis: str = "a",
    len_source: str = "fof",
    particle_mass: float | None = None,
    min_desc_particles: int = DEFAULT_MIN_DESC_PARTICLES,
    min_prog_particles: int = DEFAULT_MIN_PROG_PARTICLES,
    net: bool = True,
) -> FoFMergerRate:
    """Generic ``B/n(M0, ξ, axis)`` core shared by both products.

    Bins pre-computed per-event ``(logM0 [Msun/h], xi, snap)`` with ``weights``
    into a (mass, ξ, snapshot) count grid, divides by the forest-wide central-FOF
    census ``n(M0, snap)`` (:func:`descendant_census`) and by Δξ·Δ(axis), where the
    axis is scale factor (``rate_axis="a"``, default) or redshift (``"z"``). The
    caller supplies the events; only the substrate (infall vs coalescence) differs.
    """
    snaps, a_of_snap, z_of_snap, step = _snapshot_axes(moments, rate_axis)
    if mass_edges is None:
        mass_edges = np.arange(10.0, 15.51, 0.25)        # log10 Msun/h
    mass_edges = np.asarray(mass_edges, float)
    if xi_edges is None:
        xi_edges = np.logspace(np.log10(xi_min), 0.0, xi_bins + 1)
    xi_edges = np.asarray(xi_edges, float)

    snap_edges = np.append(snaps - 0.5, snaps[-1] + 0.5)
    sample = np.column_stack([np.asarray(logM0, float), np.asarray(xi, float),
                              np.asarray(snap, float)])
    counts, _ = np.histogramdd(sample, bins=[mass_edges, xi_edges, snap_edges],
                               weights=np.asarray(weights, float))
    n_desc = descendant_census(store, moments, backend, mass_def, mass_edges,
                               min_desc_particles, len_source, particle_mass)
    dxi = np.diff(xi_edges)
    per_desc = _safe_div(counts, n_desc[:, None, :])         # mergers/descendant
    bn = _safe_div(per_desc, (dxi[None, :, None] * step[None, None, :]))
    return FoFMergerRate(
        kind=kind, mass_def=mass_def, mass_edges=mass_edges, xi_edges=xi_edges,
        snapshot_ids=snaps, a_of_snap=a_of_snap, z_of_snap=z_of_snap,
        step_of_snap=step, rate_axis=rate_axis,
        n_desc=n_desc, counts=counts, bn=bn, len_source=len_source,
        min_desc_particles=min_desc_particles, min_prog_particles=min_prog_particles,
        net=net,
    )


def compute_fof_merger_rate(
    infall, store, moments, backend, *,
    mass_def: str = DEFAULT_MASS_DEF,
    mass_edges: np.ndarray | None = None,
    xi_bins: int = 24,
    xi_min: float = 1e-3,
    net: bool = True,
    rate_axis: str = "a",
    len_source: str = "fof",
    particle_mass: float | None = None,
    min_desc_particles: int = DEFAULT_MIN_DESC_PARTICLES,
    min_prog_particles: int = DEFAULT_MIN_PROG_PARTICLES,
) -> FoFMergerRate:
    """Build the FOF ``B/n(M0, ξ, axis)`` from the infall catalog (Product 1).

    Numerator: infall events (``~is_splashout``; ``net`` also subtracts splashouts,
    Dong's backsplash correction), ratio ξ = M_sat / M_host_prog (pre-merger host
    mass), binned by descendant mass M0 = M_host (post-merger). The resolution cut
    is on the FOF particle count (``len_source="fof"``: M_FoF/m_p, Fakhouri) or the
    HBT ``SubhaloLen`` (``"subhalo"``, for reference cross-checks).
    """
    M_sat = np.asarray(infall.masses[f"M_sat_{mass_def}"], float)
    M_host = np.asarray(infall.masses[f"M_host_{mass_def}"], float)            # M0
    M_host_prog = np.asarray(infall.masses[f"M_host_prog_{mass_def}"], float)  # M1
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = M_sat / M_host_prog
        logM0 = np.log10(np.where(M_host > 0, M_host, np.nan)) + 10.0          # Msun/h

    # resolution cut: FOF particle count (= M_FoF/m_p) or HBT SubhaloLen
    if len_source == "fof":
        if particle_mass is None:
            raise ValueError("len_source='fof' needs particle_mass")
        prog_n = np.asarray(infall.masses[f"M_sat_FoF"], float) / particle_mass
        desc_n = np.asarray(infall.masses[f"M_host_FoF"], float) / particle_mass
    else:
        prog_n, desc_n = infall.sat_len, infall.host_len
    keep = (np.isfinite(xi) & (xi > 0) & (xi <= 1) & np.isfinite(logM0)
            & (prog_n >= min_prog_particles) & (desc_n >= min_desc_particles))
    if not net:                                   # infalls only (drop splashouts)
        keep = keep & (~infall.is_splashout)
        weights = np.ones(int(keep.sum()))
    else:                                         # net: +1 infall, −1 splashout
        weights = np.where(infall.is_splashout[keep], -1.0, 1.0)

    return rate_from_events(
        logM0[keep], xi[keep], infall.snap[keep], weights, store, moments, backend,
        kind="fof_infall", mass_def=mass_def, mass_edges=mass_edges,
        xi_bins=xi_bins, xi_min=xi_min, net=net, rate_axis=rate_axis,
        len_source=len_source, particle_mass=particle_mass,
        min_desc_particles=min_desc_particles, min_prog_particles=min_prog_particles,
    )


def compute_coalescence_rate(
    catalog, store, moments, backend, *,
    mass_def: str = "Crit200",
    mass_edges: np.ndarray | None = None,
    xi_bins: int = 24,
    xi_min: float = 1e-3,
    rate_axis: str = "a",
    len_source: str = "fof",
    particle_mass: float | None = None,
    min_desc_particles: int = DEFAULT_MIN_DESC_PARTICLES,
    min_prog_particles: int = DEFAULT_MIN_PROG_PARTICLES,
) -> FoFMergerRate:
    """Build the subhalo-coalescence ``B/n(M0, ξ, axis)`` from the catalog (Product 2).

    Same estimator as Product 1 but fed by ``mergers.detect_mergers_forest``
    events: M0 = descendant FOF mass (``M0_<mass_def>``), ξ = ``xi_peak`` (peak
    SubhaloMass ratio), snapshot = ``snap_merger``. Comparison product -- lands
    at lower z (later) than the FOF/infall rate. The secondary cut uses the peak
    SubhaloLen (``sec_peak_len``); descendant completeness is enforced by the
    shared census cut (``len_source``).
    """
    from .cleaning import PointStatus
    m0 = catalog.M0_Crit200 if mass_def == "Crit200" else \
        catalog.masses.get(f"M0_{mass_def}", catalog.M0_Crit200)
    m0 = np.asarray(m0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logM0 = np.log10(np.where(m0 > 0, m0, np.nan)) + 10.0          # Msun/h
    valid = ((catalog.status != PointStatus.INVALID)
             & (catalog.status != PointStatus.FLOORED)
             & np.isfinite(catalog.xi_peak) & (catalog.xi_peak > 0)
             & (catalog.xi_peak <= 1) & np.isfinite(logM0)
             & (catalog.sec_peak_len >= min_prog_particles))
    weights = np.ones(int(valid.sum()))
    return rate_from_events(
        logM0[valid], catalog.xi_peak[valid], catalog.snap_merger[valid], weights,
        store, moments, backend, kind="coalescence", mass_def=mass_def,
        mass_edges=mass_edges, xi_bins=xi_bins, xi_min=xi_min, rate_axis=rate_axis,
        len_source=len_source, particle_mass=particle_mass,
        min_desc_particles=min_desc_particles, min_prog_particles=min_prog_particles,
    )


def _fm_shape(xi, A, beta, gamma, xi_tilde):
    """FM+2010 Eq. 1 ξ-shape ``A ξ^β exp[(ξ/ξ̃)^γ]`` (z- and M-marginal)."""
    return A * xi ** beta * np.exp((xi / xi_tilde) ** gamma)


def fit_bn(rate: FoFMergerRate, mass_range=None, z_range=None,
           xi_fit_min: float | None = None) -> dict:
    """Fit the marginal ``dN/dξ`` to a power law in log space (FM+2010 β≈−2).

    Restricted to ``ξ >= xi_fit_min`` (resolution completeness). Returns
    ``A, beta`` (+1σ) and ``n_points``. The full Eq.1 exp-cutoff term is
    degenerate for a single marginal and is left to a joint (M,z) fit.
    """
    xc = rate.xi_centers
    yc = rate.marginal_dN_dxi(mass_range, z_range)
    good = np.isfinite(yc) & (yc > 0) & np.isfinite(xc)
    out = {"A": np.nan, "beta": np.nan, "A_err": np.nan, "beta_err": np.nan,
           "n_points": 0, "xi_fit_min": np.nan}
    if good.sum() < 3:
        return out
    xg, yg = xc[good], yc[good]
    if xi_fit_min is not None:
        keep = xg >= xi_fit_min
        if keep.sum() >= 3:
            xg, yg = xg[keep], yg[keep]
    out["xi_fit_min"] = float(xg.min())
    out["n_points"] = int(len(xg))
    ln10 = np.log(10.0)
    try:
        def model(xi, log_a, beta):
            return log_a + beta * np.log10(xi)
        popt, pcov = curve_fit(model, xg, np.log10(yg), p0=(np.log10(yg.max()), -2.0),
                               maxfev=20000)
        err = np.sqrt(np.diag(pcov))
        out.update(A=10.0 ** popt[0], beta=popt[1],
                   A_err=10.0 ** popt[0] * ln10 * err[0], beta_err=err[1])
    except (RuntimeError, ValueError):
        pass
    return out


def ahg_lgs(rate: FoFMergerRate, moments, z0: float = 0.0,
            major_ratio: float = 1.0 / 3.0) -> dict:
    """Average halo growth (AHG, Amoura Eq.7) + large-growth fraction (LGS, Eq.8).

    Integrates B/n over one dynamical time before ``z0``:
    ``AHG(M0) = Σ_window Σ_ξ (N_merge/n_desc)·ξ/(1+ξ)`` and
    ``LGS(M0) = Σ_window Σ_{ξ>major_ratio} (N_merge/n_desc)`` (≈ expected number
    of >major mergers per halo in the last t_dyn). Per-descendant merger counts
    are already integrated over each snapshot's dz, so the window sum IS ∫dz∫dξ.
    """
    snaps = rate.snapshot_ids
    ages = np.asarray(moments.ages, float)
    tdyn = np.asarray(moments.dynamical_times, float)
    z = rate.z_of_snap
    # target snapshot: nearest to z0
    j0 = int(np.argmin(np.abs(z - z0)))
    window = (ages >= ages[j0] - tdyn[j0]) & (ages <= ages[j0] + 1e-9)
    per_desc = _safe_div(rate.counts, rate.n_desc[:, None, :])    # (nm,nx,ns)
    xi_c = rate.xi_centers
    wfac = xi_c / (1.0 + xi_c)
    ahg = np.nansum(per_desc[:, :, window] * wfac[None, :, None], axis=(1, 2))
    major = xi_c >= major_ratio
    lgs = np.nansum(per_desc[:, major][:, :, window], axis=(1, 2))
    return {"mass_centers": rate.mass_centers, "AHG": ahg, "LGS": lgs,
            "z0": float(z[j0]), "snap0": int(snaps[j0]),
            "t_dyn": float(tdyn[j0]), "major_ratio": float(major_ratio)}
