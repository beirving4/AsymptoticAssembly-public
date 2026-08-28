"""Diagnostic visualization for assembly-history outputs.

``plot_aperture_assembly`` reads one per-aperture assembly-history HDF5 file,
selects descendant halos in a given mass bin (log10 Msun, with the per-aperture
descendant-snapshot mass), and makes a 3x3 panel of property-evolution tracks
(median + 25-75 band + faint individual halos) plus a formation-time histogram.

Designed to run post-hoc on the written files (incl. on the cluster before
download) — it reads the HDF5 directly, no in-memory objects needed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..studies.mah import helpers  # noqa: E402

# (quantity, axis label, transform) for the 8 evolution panels. transform:
#   "mass"     -> log10(value * 1e10 / h)   [Msun]
#   "phys_log" -> log10(value * a / h)       [physical Mpc] (comoving cMpc/h -> pMpc)
#   "log"      -> log10(value)               (positive quantities, log-log)
#   None       -> raw value
_PANELS: tuple[tuple[str, str, str | None], ...] = (
    ("M",             r"$\log_{10}\,M\ [{\rm M_\odot}]$",      "mass"),
    ("R",             r"$\log_{10}\,R\ [{\rm pMpc}]$",          "phys_log"),
    ("SpinBullock",   r"$\log_{10}\,\lambda_{\rm Bullock}$",    "log"),
    ("Concentration", r"$c$",                                   None),
    ("Beta",          r"$\beta$",                               None),
    ("Triaxiality",   r"$T$",                                   None),
    ("ShapeQ",        r"$q$ (axis ratio)",                      None),
    ("ShapeS",        r"$s$ (axis ratio)",                      None),
)


def select_mass_bin(
    masses_log10_code: np.ndarray, mass_bin: float, half_width: float, h: float
) -> np.ndarray:
    """Boolean mask of halos within ``mass_bin +/- half_width`` (log10 Msun).

    ``masses_log10_code`` is the stored log10(M_final) in GADGET units
    (1e10 Msun/h); convert to log10 Msun via ``+ 10 - log10(h)``.
    """
    log_msun = masses_log10_code + 10.0 - np.log10(h)
    return np.abs(log_msun - mass_bin) <= half_width


def _evolution_panel(ax, a, values, label, transform, h, rasterize=True):
    y = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if transform == "mass":
            y = np.log10(y * 1e10 / h)                       # 1e10 Msun/h -> log10 Msun
        elif transform == "phys_log":
            y = np.log10(np.where(y > 0, y, np.nan) * a[None, :] / h)  # cMpc/h -> log10 pMpc
        elif transform == "log":
            y = np.log10(np.where(y > 0, y, np.nan))
    if rasterize:
        # flatten the dense gray background (zorder<1) into ONE raster layer in
        # vector output; axes/text/median/IQR (zorder>=2) stay vector. Cuts the
        # 3x3 PDFs from ~20 MB (thousands of vector paths) to a few hundred KB.
        ax.set_rasterization_zorder(1)
    for row in y:
        ax.semilogx(a, row, color="grey", alpha=0.08, lw=0.5, zorder=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(y, axis=0)
        p25 = np.nanpercentile(y, 25, axis=0)
        p75 = np.nanpercentile(y, 75, axis=0)
    ax.fill_between(a, p25, p75, color="C0", alpha=0.25, zorder=2)
    ax.semilogx(a, med, color="C0", lw=2.0, zorder=3)
    ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)  # a=1 reference
    ax.set_xlabel(r"Scale factor $a$")
    ax.set_ylabel(label)


def _na_panel(ax, label):
    ax.text(0.5, 0.5, f"{label}\n(N/A)", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _population_curve_formation(a, m_curve, spike_factor=5.0):
    """Formation times of a single (median) MAH curve, same defs as the per-halo path.

    Returns (a_present_pop, a_half_pop): half-M(a=1) crossing in (0,1] and the
    half-M_final crossing floored at a_present_pop -- the population-curve analogue
    to validate the per-halo distribution medians against.
    """
    finite = np.isfinite(m_curve) & (m_curve > 0)
    if finite.sum() < 2:
        return np.nan, np.nan
    m_final = m_curve[finite][-1]                      # last finite = final epoch
    m_rob, _, _ = helpers.robustify_mah(a, m_curve, m_final, spike_factor)
    m1 = helpers.interp_mass_at(a, m_rob, 1.0)
    a_present = np.nan
    if np.isfinite(m1) and m1 > 0:
        v = helpers.formation_time_envelope(a, m_rob, 0.5 * m1, a_hi=1.0)
        a_present = v if (np.isfinite(v) and v > 0) else np.nan
    a_lo = a_present if np.isfinite(a_present) else None
    vh = helpers.formation_time_envelope(a, m_rob, 0.5 * m_final, a_lo=a_lo)
    if vh == 0.0:                                       # half-final met at the floor
        a_half = a_lo if a_lo is not None else np.nan
    elif np.isfinite(vh):
        a_half = vh
    else:
        a_half = np.nan
    if np.isfinite(a_half) and np.isfinite(a_present):
        a_half = max(a_half, a_present)
    return a_present, a_half


def _formation_panel(ax, a_half, a_half_present, a=None, m_traj=None):
    af = a_half[np.isfinite(a_half) & (a_half > 0)]
    ap = a_half_present[np.isfinite(a_half_present) & (a_half_present > 0)]
    pooled = np.concatenate([af, ap]) if (af.size or ap.size) else np.array([1.0])
    # log-spaced bins clipped to the 1-99 pct range so late-forming outliers
    # don't stretch the axis; step outline + fill so both histograms stay visible.
    lo = max(float(np.nanpercentile(pooled, 1)), 1e-2)
    hi = float(np.nanpercentile(pooled, 99))
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 1.1)), 30)
    # population-curve formation times from the median MAH (validation reference)
    pop_present = pop_half = np.nan
    if a is not None and m_traj is not None and m_traj.size:
        med_mah = np.nanmedian(m_traj, axis=0)
        pop_present, pop_half = _population_curve_formation(np.asarray(a, float), med_mah)
    handles = []
    for data, color, name, pop in (
        (ap, "C1", r"half $M(a{=}1)$", pop_present),
        (af, "C0", r"half $M_{\rm final}$", pop_half),
    ):
        if not data.size:
            continue
        med = float(np.nanmedian(data))
        ax.hist(data, bins=bins, alpha=0.45, color=color)
        ax.hist(data, bins=bins, histtype="step", color=color, lw=1.5)
        ax.axvline(med, color=color, ls="--", lw=1.3)            # distribution median
        if np.isfinite(pop):
            ax.axvline(pop, color=color, ls="-", lw=1.8, alpha=0.9)  # population-curve value
        lab = rf"{name}: med={med:.2f}" + (f", pop={pop:.2f}" if np.isfinite(pop) else "")
        handles.append(plt.Line2D([], [], color=color, lw=6, alpha=0.45, label=lab))
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"Formation scale factor $a_{1/2}$")
    ax.set_ylabel("count")
    handles += [
        plt.Line2D([], [], color="k", ls="--", lw=1.3, label="distribution median"),
        plt.Line2D([], [], color="k", ls="-", lw=1.8, label="population-curve"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left")


def plot_aperture_assembly(
    file_path: str | Path,
    mass_bin: float,
    h: float,
    half_width: float = 0.1,
    out_path: str | Path | None = None,
    rasterize_background: bool = True,
    dpi: int = 150,
) -> Path | None:
    """Make the 3x3 diagnostic panel for one aperture file + mass bin.

    Returns the saved figure path, or None if no halos fall in the bin.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        aperture = f.attrs["aperture"]
        a = f["time_axis/scale_factors"][:]
        masses = f["group/masses"][:]
        sel = select_mass_bin(masses, mass_bin, half_width, h)
        n_sel = int(sel.sum())
        if n_sel == 0:
            return None
        props = f["group/properties"]
        panel_data = {q: (props[q][:][sel] if q in props else None) for q, _, _ in _PANELS}
        a_half = f["group/scalars/a_half"][:][sel]
        a_half_present = f["group/scalars/a_half_present"][:][sel]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    axes = axes.ravel()
    for ax, (q, label, transform) in zip(axes[:8], _PANELS):
        values = panel_data[q]
        if values is None or not np.isfinite(values).any():
            _na_panel(ax, label)
        else:
            _evolution_panel(ax, a, values, label, transform, h,
                             rasterize=rasterize_background)
    _formation_panel(axes[8], a_half, a_half_present, a=a, m_traj=panel_data.get("M"))

    fig.suptitle(
        rf"Assembly histories — M{aperture},  "
        rf"$\log_{{10}}(M/{{\rm M_\odot}})={mass_bin:g}\pm{half_width:g}$,  "
        rf"$N={n_sel}$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = file_path.with_suffix(".pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Merger-rate diagnostics
# --------------------------------------------------------------------------- #

# imprint quantities to show as jump histograms (skipped if absent in the file)
_JUMP_QUANTITIES: tuple[tuple[str, str], ...] = (
    ("SpinBullock", r"$\Delta\lambda_{\rm Bullock}$"),
    ("ShapeS", r"$\Delta s$"),
    ("Beta", r"$\Delta\beta$"),
)


def _xi_panel(ax, xi_centers, dN_dxi, fit):
    good = np.isfinite(dN_dxi) & (dN_dxi > 0)
    x0 = fit.get("xi_fit_min", np.nan)
    fit_region = good & (xi_centers >= x0) if np.isfinite(x0) else good
    # data: filled points where fit (complete), open points below (incomplete)
    ax.loglog(xi_centers[fit_region], dN_dxi[fit_region], "o", color="C0", ms=4, label="data (fit)")
    incomplete = good & ~fit_region
    if incomplete.any():
        ax.loglog(xi_centers[incomplete], dN_dxi[incomplete], "o", mfc="none",
                  mec="grey", ms=4, label="incomplete (resolution)")
    if np.isfinite(fit.get("beta", np.nan)):
        lo = x0 if np.isfinite(x0) else xi_centers[good].min()
        xs = np.logspace(np.log10(lo), 0, 100)
        ys = fit["A"] * xs ** fit["beta"] * np.exp(fit["gamma"] / xs)
        ax.loglog(xs, ys, "-", color="C3", lw=1.8,
                  label=rf'$\beta={fit["beta"]:.2f}$, $\gamma={fit["gamma"]:.3f}$')
        if np.isfinite(x0):
            ax.axvline(x0, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"mass ratio $\xi=M_2/M_1$ (peak)")
    ax.set_ylabel(r"$dN/d\xi$ per descendant")
    ax.legend(fontsize=7)


def _rate_vs_a_panel(ax, a, rate):
    # major mergers are rare -> the per-halo median/IQR are ~0; the population
    # MEAN (rate per descendant per t_dyn averaged over the sample) is what matters.
    with np.errstate(all="ignore"):
        mean = np.nanmean(rate, axis=0)
        n_eff = np.sum(np.isfinite(rate), axis=0)
        sem = np.nanstd(rate, axis=0) / np.sqrt(np.maximum(n_eff, 1))
    ax.fill_between(a, np.clip(mean - sem, 0, None), mean + sem, color="C0", alpha=0.2,
                    label=r"$\pm$ s.e.m.")
    ax.semilogx(a, mean, color="C0", lw=2, label="population mean")
    ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"scale factor $a$")
    ax.set_ylabel(r"major mergers per $t_{\rm dyn}$ per descendant")
    ax.legend(fontsize=8)


def _nmajor_vs_mass_panel(ax, masses_log_msun, n_major):
    good = np.isfinite(masses_log_msun)
    ax.plot(masses_log_msun[good], n_major[good], ".", color="grey", alpha=0.3, ms=3)
    if good.sum() > 10:
        bins = np.linspace(np.nanmin(masses_log_msun[good]), np.nanmax(masses_log_msun[good]), 8)
        idx = np.digitize(masses_log_msun[good], bins)
        xc = 0.5 * (bins[:-1] + bins[1:])
        med = [np.median(n_major[good][idx == b]) if (idx == b).any() else np.nan
               for b in range(1, len(bins))]
        ax.plot(xc, med, "o-", color="C3", lw=1.8, label="median")
        ax.legend(fontsize=8)
    ax.set_xlabel(r"$\log_{10}(M_{0,\rm 200c}/{\rm M_\odot})$")
    ax.set_ylabel(r"$N_{\rm major}$ per descendant")


def _xi_compare_panel(ax, xi_peak, xi_pre):
    good = np.isfinite(xi_peak) & np.isfinite(xi_pre) & (xi_peak > 0) & (xi_pre > 0)
    ax.loglog(xi_pre[good], xi_peak[good], ".", color="C0", alpha=0.25, ms=3)
    lim = [1e-3, 1.2]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\xi$ (last pre-merger)")
    ax.set_ylabel(r"$\xi$ (peak / $t_{\rm max}$)")
    ax.set_title("stripping bias", fontsize=9)


def _jump_panel(ax, events, is_major, is_minor):
    plotted = False
    for q, lab in _JUMP_QUANTITIES:
        key = f"{q}_jump"
        if key not in events:
            continue
        vals = events[key][:]
        for mask, color, name in ((is_major, "C3", "major"), (is_minor, "C0", "minor")):
            d = vals[mask & np.isfinite(vals)]
            if d.size > 5:
                ax.hist(d, bins=30, histtype="step", density=True, color=color,
                        lw=1.5, label=f"{lab} ({name})")
                plotted = True
        break  # one quantity per panel (first available)
    if not plotted:
        _na_panel(ax, "property jump")
        return
    ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("property jump across merger")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)


def _alast_panel(ax, a_last_major):
    d = a_last_major[np.isfinite(a_last_major) & (a_last_major > 0)]
    if d.size:
        ax.hist(d, bins=np.logspace(np.log10(max(d.min(), 1e-2)), np.log10(d.max()), 25),
                color="C2", alpha=0.7)
        ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$a$ of last major merger")
    ax.set_ylabel("count")


def _bn_dNdxi_panel(ax, xc, dN_dxi, fit, label, color):
    good = np.isfinite(dN_dxi) & (dN_dxi > 0) & np.isfinite(xc)
    if not good.any():
        return
    ax.loglog(xc[good], dN_dxi[good], "o-", color=color, ms=4, lw=1.6, label=label)
    beta = fit.get("beta", np.nan)
    if np.isfinite(beta) and np.isfinite(fit.get("A", np.nan)):
        xr = np.logspace(np.log10(xc[good].min()), 0.0, 40)
        ax.loglog(xr, fit["A"] * xr ** beta, "--", color=color, lw=1.1, alpha=0.7,
                  label=rf"$\xi^{{{beta:.2f}}}$")
    ax.axvline(1.0 / 3.0, color="k", ls=":", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"mass ratio $\xi$")
    ax.set_ylabel(r"$dN/d\xi$ per descendant")


def _ahg_lgs_panel(ax, mass_centers, ahg, lgs):
    good = np.isfinite(mass_centers) & (np.isfinite(ahg) | np.isfinite(lgs))
    if not good.any():
        _na_panel(ax, "AHG / LGS")
        return
    ax.plot(mass_centers[good], ahg[good], "o-", color="C0", lw=1.6, label="AHG")
    ax.plot(mass_centers[good], lgs[good], "s--", color="C3", lw=1.4, label="LGS")
    ax.set_xlabel(r"$\log_{10}(M_0/{\rm M_\odot}\,h^{-1})$")
    ax.set_ylabel(r"per descendant, last $t_{\rm dyn}$")
    ax.legend(fontsize=8)


def _merger_z_panel(ax, z_fof, z_coal):
    bins = np.linspace(0, max(np.nanmax(z_fof) if z_fof.size else 1,
                              np.nanmax(z_coal) if z_coal.size else 1, 1), 30)
    if z_fof.size:
        ax.hist(z_fof[np.isfinite(z_fof)], bins=bins, histtype="step", color="C0",
                lw=1.6, density=True, label="FOF / infall")
    if z_coal.size:
        ax.hist(z_coal[np.isfinite(z_coal)], bins=bins, histtype="step", color="C3",
                lw=1.6, density=True, label="coalescence")
    ax.set_xlabel(r"merger redshift $z$")
    ax.set_ylabel("density")
    ax.set_title("timing: infall vs coalescence", fontsize=9)
    ax.legend(fontsize=8)


def plot_merger_diagnostics(
    file_path: str | Path, h: float, out_path: str | Path | None = None,
) -> Path | None:
    """6-panel merger diagnostic read directly from a (v3) merger-rate HDF5 file.

    Panels: FOF B/n dN/dxi + power-law fit; FOF vs coalescence dN/dxi overlay;
    AHG/LGS vs M0; xi_peak vs xi_premerge (stripping bias); B/n(M0) curves at z~0;
    merger-redshift distribution (infall vs coalescence timing).
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        events = {k: f[f"events/{k}"][:] for k in f["events"] if k != "table"}
        fof = _read_rate_group(f, "fof_merger_rate")
        coal = _read_rate_group(f, "coalescence_rate")
        ahg = (f["fof_merger_rate/ahg_lgs"] if "fof_merger_rate/ahg_lgs" in f else None)
        ahg_mc = ahg["mass_centers"][:] if ahg else np.array([])
        ahg_v = ahg["AHG"][:] if ahg else np.array([])
        lgs_v = ahg["LGS"][:] if ahg else np.array([])
        z_fof = f["infall/a"][:] if "infall/a" in f else np.array([])
        z_coal = events.get("z_merger", np.array([]))
    if z_fof.size:
        z_fof = 1.0 / z_fof - 1.0          # scale factor -> redshift

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes.ravel()
    if fof:
        _bn_dNdxi_panel(ax[0], fof["xi_centers"], fof["dN_dxi"], fof["fit"], "FOF B/n", "C0")
        ax[0].set_title("Fakhouri/Amoura FOF rate", fontsize=9); ax[0].legend(fontsize=8)
        _bn_dNdxi_panel(ax[1], fof["xi_centers"], fof["dN_dxi"], fof["fit"], "FOF", "C0")
        # B/n(M0) at z~0: line per mass bin
        bn = fof["bn"]; mc = fof["mass_centers"]; xc = fof["xi_centers"]
        j0 = int(np.argmin(np.abs(fof["z_of_snap"])))
        cmap = plt.get_cmap("viridis")
        nm = bn.shape[0]
        for mi in range(nm):
            y = bn[mi, :, j0]
            g = np.isfinite(y) & (y > 0)
            if g.sum() > 2:
                ax[4].loglog(xc[g], y[g], "-", color=cmap((mi + 0.5) / nm), lw=1.3,
                             label=rf"${mc[mi]:.1f}$")
        ax[4].set_xlabel(r"$\xi$"); ax[4].set_ylabel(r"$B/n$ at $z\!\approx\!0$")
        ax[4].legend(fontsize=6, title=r"$\log M_0$", title_fontsize=6, ncol=2)
    else:
        for i in (0, 1, 4):
            _na_panel(ax[i], "FOF rate")
    if coal:
        _bn_dNdxi_panel(ax[1], coal["xi_centers"], coal["dN_dxi"], coal["fit"],
                        "coalescence", "C3")
    ax[1].set_title("FOF vs coalescence", fontsize=9); ax[1].legend(fontsize=8)

    _ahg_lgs_panel(ax[2], ahg_mc, ahg_v, lgs_v)
    _xi_compare_panel(ax[3], events["xi_peak"], events["xi_premerge"])
    _merger_z_panel(ax[5], np.asarray(z_fof), np.asarray(z_coal))

    n_ev = int(events["xi_peak"].shape[0])
    fig.suptitle(f"Merger diagnostics — {file_path.name}  (N={n_ev} forest-wide events)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is None:
        out_path = file_path.with_suffix(".pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_last_merger_distribution(
    full_tree_path: str | Path,
    out_path: str | Path | None = None,
    cosmo=None,
    min_particles: int = 100,
) -> Path | None:
    """Distribution of the final-merger scale factor per tree (full_tree columns).

    Reads the per-node ``aLastMerger_{Coal,Infall}_{Major,Tenth,Any}`` columns at the
    **root** (tree-final, a=100/10^4) central halos and histograms ``a_last`` in log-a
    bins, one curve per mass-ratio threshold, for both timings. Returns None if the
    columns are absent (run ``compose_full_tree`` without ``--no_last_merger`` first).
    """
    from ..merger_trees.full_tree import FullTreeStore

    full_tree_path = Path(full_tree_path)
    st = FullTreeStore.open(full_tree_path, cosmo=cosmo)
    need = [f"aLastMerger_{tim}_{tag}" for tim in ("Coal", "Infall")
            for tag in ("Major", "Tenth", "Any")]
    if not all(st.has_column(c) for c in need):
        st.close()
        return None
    snap = st.column("SnapNum"); rows = np.arange(st.n_rows)
    final = snap.max()
    roots = np.flatnonzero((snap == final) & st.is_central(rows)
                           & (st.column("SubhaloLen") >= min_particles))
    cols = {c: st.column(c)[roots] for c in need}
    st.close()

    edges = np.logspace(np.log10(0.05), np.log10(100), 31)
    cen = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, timing in zip(axes, ("Coal", "Infall")):
        for tag, c in (("Major", "C3"), ("Tenth", "C1"), ("Any", "C0")):
            v = cols[f"aLastMerger_{timing}_{tag}"]; fin = np.isfinite(v)
            h, _ = np.histogram(v[fin], bins=edges)
            ax.step(cen, h, where="mid", color=c, lw=1.8,
                    label=f"{tag}: had={100*fin.mean():.0f}%, med a={np.nanmedian(v):.2f}")
        ax.set_xscale("log"); ax.axvline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("scale factor of last merger"); ax.set_ylabel("N tree-final halos / bin")
        ax.set_title(f"{timing} timing"); ax.legend(fontsize=8)
    fig.suptitle(f"Final-merger scale factor per tree — {full_tree_path.name} "
                 f"(a=100 finals; both participants >= {min_particles} part)")
    fig.tight_layout()
    if out_path is None:
        out_path = full_tree_path.with_name(full_tree_path.stem + "_last_merger_distribution.pdf")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_tree_vs_fulltree_mergers(
    orig_tree_path: str | Path,
    full_tree_path: str | Path,
    out_path: str | Path | None = None,
    cosmo=None,
) -> Path | None:
    """Merger counts vs scale factor: original GADGET tree vs composed full_tree.

    Runs forest-wide coalescence + infall detection on both trees and overlays the
    log-a histograms -- a consistency check that composition preserved the
    merger-relevant links/indexing (the curves should coincide exactly).
    """
    from ..merger_trees.full_tree import FullTreeStore
    from .mergers import detect_mergers_forest, detect_infalls

    edges = np.logspace(np.log10(0.02), np.log10(100), 38)
    cen = np.sqrt(edges[:-1] * edges[1:])

    def _counts(path):
        st = FullTreeStore.open(Path(path), cosmo=cosmo)
        grid = np.asarray(st.moments.snapshot_ids); a_grid = np.asarray(st.moments.scale_factors)
        ev = detect_mergers_forest(st)
        a_m = a_grid[np.searchsorted(grid, ev.snap_merger)]
        inf = detect_infalls(st, st.moments, mass_defs=("Virial",), backend=None,
                             min_host_particles=20, min_sec_particles=20, a_min=0.0)
        a_i = inf.a[~inf.is_splashout]
        st.close()
        return (np.histogram(a_m, bins=edges)[0], np.histogram(a_i, bins=edges)[0])

    hco, hio = _counts(orig_tree_path)
    hcf, hif = _counts(full_tree_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ho, hf, lab) in zip(axes, ((hco, hcf, "coalescence"), (hio, hif, "infall"))):
        ax.step(cen, ho, where="mid", lw=2, label="original trees.hdf5")
        ax.step(cen, hf, where="mid", lw=1, ls="--", label="full_tree.hdf5")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("scale factor a"); ax.set_ylabel("N mergers / bin"); ax.set_title(lab)
        ax.legend()
    ident = np.array_equal(hco, hcf) and np.array_equal(hio, hif)
    fig.suptitle(f"Merger counts vs a: original tree vs full_tree  (identical={ident})")
    fig.tight_layout()
    if out_path is None:
        out_path = Path(full_tree_path).with_name(Path(full_tree_path).stem + "_tree_vs_fulltree_mergers.pdf")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def _read_rate_group(f, name) -> dict | None:
    """Read a FoFMergerRate HDF5 group into a dict (or None if absent)."""
    if name not in f:
        return None
    g = f[name]
    out = {k: g[k][:] for k in ("xi_centers", "mass_centers", "dN_dxi", "bn",
                                "z_of_snap") if k in g}
    out["fit"] = {k: g.attrs.get(f"fit_{k}", np.nan) for k in ("A", "beta", "xi_fit_min")}
    return out


def plot_cumulative_rate_vs_a(
    file_path: str | Path,
    group: str = "fof_merger_rate",
    xi_mins=(1.0 / 3.0, 0.1, 0.01, 0.001),
    mass_range=None,
    out_path: str | Path | None = None,
) -> Path | None:
    """Cumulative merger rate ``dN(>ξ_min)/d(axis)`` per descendant vs scale factor.

    One curve per ``ξ_min`` threshold (Fakhouri+BK 2010 Fig. 3 / Dong+2022 Fig. 1
    right). Reads the (counts, n_desc, step, a) grid of ``group`` directly and sums
    ``counts/n_desc`` over ξ ≥ ξ_min (and the optional ``mass_range`` in log10
    Msun/h), then divides by the snapshot's Δ(axis). Left panel vs scale factor a,
    right panel vs (1+z) on a log axis for literature comparison.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        if group not in f:
            return None
        g = f[group]
        counts = g["counts"][:]; n_desc = g["n_desc"][:]
        xc = g["xi_centers"][:]; mc = g["mass_centers"][:]
        a = g["a_of_snap"][:]; z = g["z_of_snap"][:]; step = g["step_of_snap"][:]
        axis = g.attrs.get("rate_axis", "a")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_desc = counts / np.where(n_desc > 0, n_desc, np.nan)[:, None, :]
    mm = np.ones_like(mc, bool) if mass_range is None else (mc >= mass_range[0]) & (mc < mass_range[1])
    stepf = np.where(step > 0, step, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("plasma")
    for i, xm in enumerate(sorted(xi_mins, reverse=True)):
        cum = np.nansum(per_desc[mm][:, xc >= xm, :], axis=(0, 1))     # (ns,) per descendant
        rate = cum / stepf
        color = cmap((i + 0.5) / len(xi_mins))
        good = np.isfinite(rate) & (rate > 0)
        lab = rf"$\xi>{xm:.3g}$"
        axes[0].semilogy(a[good], rate[good], "o-", color=color, ms=3, lw=1.4, label=lab)
        gz = good & (z > -0.999)
        axes[1].loglog(1 + z[gz], rate[gz], "o-", color=color, ms=3, lw=1.4, label=lab)
    axes[0].axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[0].set_xlabel(r"scale factor $a$")
    axes[1].set_xlabel(r"$1+z$")
    ylab = rf"$dN(>\xi_{{\min}})/d{axis}$ per descendant"
    for ax in axes:
        ax.set_ylabel(ylab); ax.legend(fontsize=8)
    mr = "all masses" if mass_range is None else rf"${mass_range[0]:.1f}<\log M_0<{mass_range[1]:.1f}$"
    axes[0].set_title("vs scale factor", fontsize=10)
    axes[1].set_title("vs (1+z) [a<1 regime]", fontsize=10)
    fig.suptitle(f"Cumulative specific merger rate — {group} — {mr} — {file_path.name}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path is None:
        out_path = file_path.with_name(file_path.stem + "_cumulative_rate.pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Specific merger rate dN/dxi/dlogM0, resolved by mass + epoch (Dong et al. 2022)
# --------------------------------------------------------------------------- #

def plot_specific_rate_by_mass(
    file_path: str | Path,
    h: float,
    a_targets=(1.0, 10.0, 100.0),
    xi_complete: float = 0.1,  # unused (per-host-bin completeness used instead); kept for API
    out_path: str | Path | None = None,
) -> Path | None:
    """Panel-per-epoch plot of dN/dxi vs xi, resolved by HOST mass (Dong et al. 2022).

    Bins events by the main-progenitor peak (infall) particle count and draws each
    host-mass curve only above its host-dependent completeness limit
    ``xi_min ~ min_sec_particles / N_host`` (faded below). A ``xi^{-1.9}`` reference
    line marks the progenitor/halo-mass-function slope: well-resolved (massive) host
    bins should track it. Reads the merger-rate HDF5 directly.
    """
    from .merger_rates import specific_rate_by_mass

    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        ev = f["events"]
        min_sec = int(f["events"].attrs.get("min_sec_particles", 100)) \
            if "min_sec_particles" in f["events"].attrs else int(f.attrs.get("min_sec_particles", 100))
        cat = _CatalogView(
            xi_peak=ev["xi_peak"][:], a_merger=ev["a_merger"][:], status=ev["status"][:],
            M_main_peak=ev["M_main_peak"][:], desc_node=ev["desc_node"][:],
            main_peak_len=ev["main_peak_len"][:], sec_peak_len=ev["sec_peak_len"][:],
        )

    res = specific_rate_by_mass(cat, h=h, a_targets=a_targets, min_sec_particles=min_sec)
    xc = res["xi_centers"]
    hmass = res["host_mass_msun"]
    xicomp = res["xi_complete"]
    n_host = len(res["host_edges"]) - 1
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(a_targets), figsize=(5.2 * len(a_targets), 4.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, at in zip(axes, a_targets):
        rate = res["by_a"][at]["rate"]
        ndb = res["by_a"][at]["n_desc_bin"]
        n_in = res["by_a"][at]["n_in"]
        for hb in range(n_host):
            y = rate[hb]
            good = np.isfinite(y) & (y > 0)
            if not good.any():
                continue
            color = cmap((hb + 0.5) / n_host)
            lab = rf"${hmass[hb]:.1f}\!-\!{hmass[hb+1]:.1f}$ ($n_h\!=\!{ndb[hb]}$)"
            comp = good & (xc >= xicomp[hb])
            inc = good & (xc < xicomp[hb])
            ax.loglog(xc[comp], y[comp], "o-", color=color, ms=4, lw=1.5, label=lab)
            if inc.any():
                ax.loglog(xc[inc], y[inc], "o:", color=color, ms=3, lw=1.0, alpha=0.35)
            ax.axvline(xicomp[hb], color=color, ls=":", lw=0.8, alpha=0.4)
        # xi^-1.9 progenitor-MF reference (amplitude anchored to the panel)
        finite = np.concatenate([rate[hb][np.isfinite(rate[hb]) & (rate[hb] > 0)]
                                 for hb in range(n_host)]) if n_in else np.array([1.0])
        if finite.size:
            xr = np.logspace(np.log10(xc.min()), -0.4, 30)
            anchor = np.nanmedian(finite) * 0.3
            ax.loglog(xr, anchor * (xr / 0.1) ** -1.9, "k--", lw=1.1, alpha=0.6,
                      label=r"$\xi^{-1.9}$" if at == a_targets[0] else None)
        ax.axvline(1.0 / 3.0, color="k", ls="--", lw=0.6, alpha=0.3)
        ax.set_title(rf"$a={at:g}$  ($N_{{\rm merg}}={n_in}$)", fontsize=11)
        ax.set_xlabel(r"mass ratio $\xi=M_2/M_1$ (peak)")
    axes[0].set_ylabel(r"$dN/d\xi$  [per descendant, per unit $\xi$]")
    axes[-1].legend(fontsize=7, title=r"host $\log_{10}(M/{\rm M_\odot})$", title_fontsize=7)
    fig.suptitle(rf"Specific merger rate by HOST mass + epoch — {file_path.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path is None:
        out_path = file_path.with_name(file_path.stem + "_specific_rate.pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


class _CatalogView:
    """Minimal duck-typed view exposing the columns specific_rate_by_mass reads."""

    def __init__(self, xi_peak, a_merger, status, M_main_peak, desc_node,
                 main_peak_len, sec_peak_len):
        self.xi_peak = xi_peak
        self.a_merger = a_merger
        self.status = status
        self.M_main_peak = M_main_peak
        self.desc_node = desc_node
        self.main_peak_len = main_peak_len
        self.sec_peak_len = sec_peak_len


# --------------------------------------------------------------------------- #
# Dong et al. (2022) specific merger rate dN/dxi/dlogM + Eq.2 overlay
# --------------------------------------------------------------------------- #

def plot_dong_specific_rate(
    file_path: str | Path,
    mass_def: str = "Virial",
    host_mass_bins=(12.5, 13.5, 15.5),
    a_range=None,
    out_path: str | Path | None = None,
) -> Path | None:
    """Reconstruct Dong's dN/dxi/dlogM from the stored /infall + /host_growth and
    overlay the published Eq. 2 double Schechter. Curves are per host-mass bin,
    each drawn above its resolution completeness xi_min ~ min_sec/min_host."""
    from .merger_rates import dong_specific_rate, dong_eq2

    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        if "infall" not in f or "host_growth" not in f:
            return None
        inf = f["infall"]
        M_sat = inf[f"M_sat_{mass_def}"][:]
        M_host = inf[f"M_host_{mass_def}"][:]
        is_sp = inf["is_splashout"][:]
        a_inf = inf["a"][:]
        hg = f["host_growth"]
        growth_sum = hg[f"sum_dlogM_{mass_def}"][:]
        mass_edges = hg["mass_bin_edges"][:]
        growth_a = f["time_axis/scale_factors"][:]
        min_host = int(f.attrs.get("min_host_particles", 320))
        min_sec = int(f.attrs.get("infall_min_sec_particles", 30))

    res = dong_specific_rate(
        M_sat, M_host, is_sp, a_inf, growth_sum, mass_edges, growth_a,
        host_mass_bins=np.asarray(host_mass_bins, float), a_range=a_range,
    )
    xc = res["xi_centers"]; bins = res["host_mass_bins"]; nb = len(bins) - 1
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for hb in range(nb):
        y = res["rate"][hb]
        good = np.isfinite(y) & (y > 0)
        if not good.any():
            continue
        xi_complete = min_sec / 10.0 ** (bins[hb] - 10.0)        # ~min_sec/N_host_lo
        color = cmap((hb + 0.5) / nb)
        comp = good & (xc >= xi_complete)
        inc = good & (xc < xi_complete)
        ax.loglog(xc[comp], y[comp], "o-", color=color, ms=4, lw=1.5,
                  label=rf"${bins[hb]:.1f}\!-\!{bins[hb+1]:.1f}$")
        if inc.any():
            ax.loglog(xc[inc], y[inc], "o:", color=color, ms=3, lw=1.0, alpha=0.35)
        ax.axvline(xi_complete, color=color, ls=":", lw=0.7, alpha=0.4)
    xs = np.logspace(-3, 0, 200)
    ax.loglog(xs, dong_eq2(xs), "r-", lw=2.2, label="Dong+2022 Eq.2")
    ax.axvline(1.0 / 3.0, color="k", ls="--", lw=0.6, alpha=0.3)
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlabel(r"mass ratio $\xi=M_{\rm sat}/M_{\rm host}$ (virial, infall)")
    ax.set_ylabel(r"$dN_{\rm merge}/d\xi/d\log M_{\rm h}$")
    ttl = f"Dong specific merger rate ({mass_def}) vs Eq.2 — {file_path.name}"
    if a_range is not None:
        ttl += rf"  [$a\in[{a_range[0]:g},{a_range[1]:g})$]"
    ax.set_title(ttl, fontsize=11)
    ax.legend(fontsize=8, title=r"host $\log_{10}(M/{\rm M_\odot}h^{-1})$", title_fontsize=8)

    if out_path is None:
        out_path = file_path.with_name(file_path.stem + "_dong_specific_rate.pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
