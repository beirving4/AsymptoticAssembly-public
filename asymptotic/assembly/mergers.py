"""Merger detection + the flat per-event catalog (the N x k merger table).

A merger is a node with more than one direct progenitor: every node whose
``TreeMainProgenitor`` differs from a given progenitor is one secondary -> one
merger event. Detection is **forest-wide** -- :func:`detect_mergers_forest` scans
*every* node in the tree, not just the main branches of a selected z=0 descendant
set (that earlier, main-branch-conditioned detector systematically undercounted
off-main-branch mergers; see ``docs/merger_rate_estimator_fix_plan.md``).

This forest-wide subhalo-tree detector measures **coalescence-timing** mergers
(an HBT subhalo finally disrupts into its host, near the centre) -- the
Rodriguez-Gomez (2015) galaxy/subhalo merger rate. The Fakhouri-Ma / Amoura
**FOF** merger rate ``B/n(M,xi,z)`` is a *different*, earlier event (infall /
virial-crossing) and is built from the infall catalog (:func:`detect_infalls` +
``fof_merger_rate.py``), NOT from this detector.

The headline product is :class:`MergerCatalog`: one row per merger event, with a
fixed set of per-event features (the "k columns"). Rate statistics are computed
downstream from this table (see ``merger_rates.py``) -- this mirrors how the
merger-rate literature works (Fakhouri & Ma 2008; Fakhouri, Ma & Boylan-Kolchin
2010; Genel et al. 2009; Rodriguez-Gomez et al. 2015): store a flat event catalog,
then bin and integrate at analysis time.

Mass ratio
----------
``xi = M_secondary / M_primary``. Following Rodriguez-Gomez et al. (2015), the
primary ratio ``xi_peak`` uses each progenitor's **peak** ``SubhaloMass`` along its
own history (the "t_max" mass, before tidal stripping biases it low). The
last-pre-merger ratio ``xi_premerge`` is stored alongside for comparison. Peak
masses come from :func:`compute_peak_history`, a single DP over the whole tree.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from attrs import define, field

from .accretion import baseline_indices
from .cleaning import PointStatus
from ..merger_trees.topology import BranchTopology

# SubhaloMass is the bound mass on EVERY row, so each progenitor contributes its
# own mass to the ratio (a Group_* mass would only be valid central-central and
# routes satellites to the host -> spurious xi~1).
DEFAULT_MASS_COL = "SubhaloMass"
DEFAULT_MAJOR_RATIO = 1.0 / 3.0   # Rodriguez-Gomez 2015 / Fakhouri ~0.3
DEFAULT_MINOR_FLOOR = 0.01        # below this, xi is dominated by fragments/noise
DEFAULT_MIN_SEC_PARTICLES = 100   # secondary peak SubhaloLen floor (else FLOORED)

# Group mass definitions stored per event so any ratio/binning is reproducible
# downstream (no tree needed). Virial = Bryan-Norman Δ_vir (Dong et al. 2022 use this).
MASS_DEFS: tuple[str, ...] = ("Virial", "Crit200", "Mean200", "FoF", "Splashback")
# Dong-arm infall defaults (Dong+2022: host > 320 particles, secondary > ~30).
DEFAULT_MIN_HOST_PARTICLES = 320
DEFAULT_INFALL_MIN_SEC_PARTICLES = 30

# Per-event feature columns that always exist (imprint columns are appended).
KINEMATIC_COLUMNS: tuple[str, ...] = (
    "halo_idx", "desc_node", "desc_tree_id", "main_node", "sec_node",
    "snap_merger", "a_merger", "z_merger",
    "snap_main_peak", "a_main_peak", "snap_sec_peak", "a_sec_peak",
    "dt_to_desc_Gyr", "dt_since_peak_Gyr",
    "M0_Crit200", "M0_FoF", "M0_Splashback",
    "M_main_peak", "M_sec_peak", "M_main_premerge", "M_sec_premerge",
    "main_peak_len", "sec_peak_len",
    "xi_peak", "xi_premerge", "is_major", "status",
)


# --------------------------------------------------------------------------- #
# Peak-mass history (t_max)
# --------------------------------------------------------------------------- #

@define(slots=True)
class PeakHistory:
    """Per-row peak ``SubhaloMass`` over the main-progenitor history (t_max)."""

    peak_mass: np.ndarray   # (n_rows,) running max along the main-prog chain
    peak_snap: np.ndarray   # (n_rows,) SnapNum at which that max occurred
    peak_node: np.ndarray   # (n_rows,) absolute row at which that max occurred


def compute_peak_history(store, mass_col: str = DEFAULT_MASS_COL) -> PeakHistory:
    """Running max of ``mass_col`` along every row's main-progenitor history.

    DP over the whole tree in ascending snapshot order: a row's main progenitor
    lives at an earlier snapshot, so its peak is finalized first and
    ``peak[node] = max(mass[node], peak[main_prog[node]])``. O(n_rows).
    """
    snap = store.column("SnapNum")
    mass = store.column(mass_col).astype(float)
    tree_id = store.column("TreeID")
    mp_abs = store.abs_from_relative(tree_id, store.column("TreeMainProgenitor"))

    peak_mass = mass.copy()
    peak_snap = snap.copy()
    peak_node = np.arange(len(snap), dtype=np.int64)

    for s in np.unique(snap):                      # ascending; ~75 steps
        rows = np.flatnonzero(snap == s)
        mp = mp_abs[rows]
        valid = mp >= 0
        r, m = rows[valid], mp[valid]
        take_parent = peak_mass[m] >= mass[r]      # parent history dominates
        peak_mass[r] = np.where(take_parent, peak_mass[m], mass[r])
        peak_snap[r] = np.where(take_parent, peak_snap[m], snap[r])
        peak_node[r] = np.where(take_parent, peak_node[m], r)

    return PeakHistory(peak_mass, peak_snap, peak_node)


# --------------------------------------------------------------------------- #
# Merger detection (vectorized)
# --------------------------------------------------------------------------- #

@define(slots=True)
class RawEvents:
    """Bare (descendant, main-prog, secondary-prog) triples per merger event."""

    halo_idx: np.ndarray    # (N,) index into the descendant set
    desc_node: np.ndarray   # (N,) descendant absolute node id
    snap_merger: np.ndarray # (N,) snapshot at which the merger lands
    main_node: np.ndarray   # (N,) main-progenitor absolute node
    sec_node: np.ndarray    # (N,) secondary-progenitor absolute node

    @property
    def n_events(self) -> int:
        return len(self.snap_merger)


def detect_mergers_forest(store) -> RawEvents:
    """Forest-wide merger detection: every node with more than one progenitor.

    Scans **every** node in the tree (not just the main branches of a selected
    z=0 descendant set). A row ``r`` is a *secondary* progenitor -- i.e. one
    merger event landing on its descendant -- iff it has a valid descendant and
    is not that descendant's ``TreeMainProgenitor``. Among all progenitors of a
    descendant exactly one is the main progenitor, so this counts ``N_prog - 1``
    mergers per descendant (Fakhouri & Ma / Amoura recipe), forest-wide. Fully
    vectorized over all rows (no per-node Python, no topology).

    ``halo_idx`` re-indexes the *unique descendant nodes* that received a merger,
    so any per-descendant grouping downstream is over the merger's own descendant
    node (NOT a single z=0 set).
    """
    n = store.n_rows
    tid = store.column("TreeID")
    snap_col = store.column("SnapNum")
    desc_rel = store.column("TreeDescendant")
    main_rel = store.column("TreeMainProgenitor")

    rows = np.arange(n, dtype=np.int64)
    desc_abs = store.abs_from_relative(tid, desc_rel)        # each row's descendant
    valid = desc_abs >= 0
    # main progenitor of each row's descendant (relative index lives in desc's tree)
    main_of_desc = np.full(n, -1, dtype=np.int64)
    d = desc_abs[valid]
    main_of_desc[valid] = store.abs_from_relative(tid[d], main_rel[d])
    is_secondary = valid & (rows != main_of_desc)
    sel = np.flatnonzero(is_secondary)

    if sel.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return RawEvents(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())

    desc_node = desc_abs[sel]
    _, halo_idx = np.unique(desc_node, return_inverse=True)
    return RawEvents(
        halo_idx=halo_idx.astype(np.int64),
        desc_node=desc_node.astype(np.int64),
        snap_merger=snap_col[desc_node].astype(np.int64),
        main_node=main_of_desc[sel].astype(np.int64),
        sec_node=sel.astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# The N x k catalog
# --------------------------------------------------------------------------- #

@define(slots=True)
class MergerCatalog:
    """Flat per-event catalog -- the N x k merger table.

    Each attribute is a length-N array (one entry per merger event). ``imprints``
    holds the dynamically named property-imprint columns. :meth:`to_table` returns
    the literal ``(N, k)`` array + column names; :meth:`to_hdf5` writes both the
    per-column datasets and the stacked table.
    """

    # identity / topology
    halo_idx: np.ndarray
    desc_node: np.ndarray
    desc_tree_id: np.ndarray
    main_node: np.ndarray
    sec_node: np.ndarray
    # timing
    snap_merger: np.ndarray
    a_merger: np.ndarray
    z_merger: np.ndarray
    snap_main_peak: np.ndarray
    a_main_peak: np.ndarray
    snap_sec_peak: np.ndarray
    a_sec_peak: np.ndarray
    dt_to_desc_Gyr: np.ndarray
    dt_since_peak_Gyr: np.ndarray
    # descendant mass (Group, at desc snap)
    M0_Crit200: np.ndarray
    M0_FoF: np.ndarray
    M0_Splashback: np.ndarray
    # progenitor mass (SubhaloMass)
    M_main_peak: np.ndarray
    M_sec_peak: np.ndarray
    M_main_premerge: np.ndarray
    M_sec_premerge: np.ndarray
    # peak (infall) particle counts -> host-mass binning + per-bin xi completeness
    main_peak_len: np.ndarray
    sec_peak_len: np.ndarray
    # ratios + classification
    xi_peak: np.ndarray
    xi_premerge: np.ndarray
    is_major: np.ndarray
    status: np.ndarray
    # provenance
    major_ratio: float
    minor_floor: float
    imprint_aperture: str = "Crit200"           # primary (unnamespaced) aperture
    imprint_apertures: tuple = ()                # all apertures written (namespaced)
    imprints: dict[str, np.ndarray] = field(factory=dict)
    # multi-mass-definition host/progenitor masses (M0_<def>, M_main_<def>, M_sec_<def>)
    # so any ratio/binning in any mass definition is a local column op downstream.
    masses: dict[str, np.ndarray] = field(factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.snap_merger)

    @property
    def is_minor(self) -> np.ndarray:
        ok = self.status != PointStatus.INVALID
        return ok & (self.xi_peak >= self.minor_floor) & (self.xi_peak < self.major_ratio)

    def column_dict(self) -> dict[str, np.ndarray]:
        """Ordered name -> 1-D array for every numeric column (kinematics + masses + imprints)."""
        base = {name: getattr(self, name) for name in KINEMATIC_COLUMNS}
        base.update(self.masses)
        base.update(self.imprints)
        return base

    def to_table(self) -> tuple[np.ndarray, list[str]]:
        """The N x k float array + the column names (booleans/ints cast to float)."""
        cols = self.column_dict()
        names = list(cols)
        table = np.column_stack([np.asarray(cols[n], dtype=float) for n in names]) \
            if self.n_events else np.empty((0, len(names)), dtype=float)
        return table, names

    def to_hdf5(self, group) -> None:
        cols = self.column_dict()
        for name, arr in cols.items():
            group.create_dataset(name, data=arr)
        table, names = self.to_table()
        ds = group.create_dataset("table", data=table)
        ds.attrs["columns"] = np.array(names, dtype="S")
        group.attrs["n_events"] = self.n_events
        group.attrs["k_features"] = len(names)
        group.attrs["major_ratio"] = self.major_ratio
        group.attrs["minor_floor"] = self.minor_floor
        group.attrs["imprint_aperture"] = self.imprint_aperture
        if self.imprint_apertures:
            group.attrs["imprint_apertures"] = np.array(
                self.imprint_apertures, dtype="S")


def build_catalog(
    store,
    backend,
    events: RawEvents,
    peak: PeakHistory,
    moments,
    *,
    major_ratio: float = DEFAULT_MAJOR_RATIO,
    minor_floor: float = DEFAULT_MINOR_FLOOR,
    min_sec_particles: int = DEFAULT_MIN_SEC_PARTICLES,
    corrected_cells: np.ndarray | None = None,
    mass_col: str = DEFAULT_MASS_COL,
) -> MergerCatalog:
    """Assemble the kinematic + mass + ratio + status columns of the catalog.

    Topology-free: every column is read at the event's own descendant/progenitor
    nodes, so the same builder serves the forest-wide event set.
    """
    n = events.n_events
    grid = np.asarray(moments.snapshot_ids)
    a_grid = moments.scale_factors
    z_grid = moments.redshifts
    ages = moments.ages
    desc_col = len(grid) - 1  # descendants live at the final grid column

    submass = store.column(mass_col).astype(float)
    sublen = store.column("SubhaloLen")

    # --- timing ---
    merger_col = np.searchsorted(grid, events.snap_merger)
    peak_snap = peak.peak_snap[events.sec_node]
    peak_col = np.searchsorted(grid, peak_snap)
    main_peak_snap = peak.peak_snap[events.main_node]
    main_peak_col = np.searchsorted(grid, main_peak_snap)
    a_merger = a_grid[merger_col]
    z_merger = z_grid[merger_col]
    a_main_peak = a_grid[main_peak_col]
    a_sec_peak = a_grid[peak_col]
    dt_to_desc = ages[desc_col] - ages[merger_col]
    dt_since_peak = ages[merger_col] - ages[peak_col]

    # --- descendant masses (Group, read at the central descendant rows) ---
    def desc_mass(aperture: str) -> np.ndarray:
        col = backend.resolve_column("M", aperture)
        if col is None:
            return np.full(n, np.nan)
        return store.column(col).astype(float)[events.desc_node]

    M0_crit = desc_mass("Crit200")
    M0_fof = desc_mass("FoF")
    M0_sp = desc_mass("Splashback")

    # --- progenitor masses + ratios ---
    M_main_peak = peak.peak_mass[events.main_node]
    M_sec_peak = peak.peak_mass[events.sec_node]
    M_main_pre = submass[events.main_node]
    M_sec_pre = submass[events.sec_node]
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_peak = np.where(M_main_peak > 0, M_sec_peak / M_main_peak, np.nan)
        xi_pre = np.where(M_main_pre > 0, M_sec_pre / M_main_pre, np.nan)

    # peak (infall) particle counts of both progenitors
    main_peak_len = sublen[peak.peak_node[events.main_node]].astype(np.int64)
    sec_peak_len = sublen[peak.peak_node[events.sec_node]].astype(np.int64)

    # --- status ---
    status = np.full(n, PointStatus.SUCCESS, dtype=np.uint8)
    status[~np.isfinite(xi_peak)] = PointStatus.INVALID
    floored = (sec_peak_len < min_sec_particles) & (status != PointStatus.INVALID)
    status[floored] = PointStatus.FLOORED
    if corrected_cells is not None and n:
        on_corrected = corrected_cells[events.halo_idx, merger_col]
        status[on_corrected & (status == PointStatus.SUCCESS)] = PointStatus.POS_CORRECTED

    is_major = (xi_peak >= major_ratio) & (status != PointStatus.INVALID)

    # --- multi-mass-definition masses (own-halo at t_max for progenitors) --- #
    # M0_<def> at the central descendant row; M_main/M_sec_<def> at each
    # progenitor's SubhaloMass-peak (t_max) node, central-masked (Group_* is
    # broadcast to satellites, so a non-central peak node would read the host).
    main_pk = peak.peak_node[events.main_node]
    sec_pk = peak.peak_node[events.sec_node]
    cen_main = store.is_central(main_pk)
    cen_sec = store.is_central(sec_pk)
    explicit_M0 = {"Crit200", "FoF", "Splashback"}   # already stored as M0_* kinematic columns
    masses: dict[str, np.ndarray] = {}
    for d in MASS_DEFS:
        col = backend.resolve_column("M", d)
        if col is None:
            continue
        data = store.column(col).astype(float)
        if d not in explicit_M0:
            masses[f"M0_{d}"] = data[events.desc_node]
        masses[f"M_main_{d}"] = np.where(cen_main, data[main_pk], np.nan)
        masses[f"M_sec_{d}"] = np.where(cen_sec, data[sec_pk], np.nan)

    return MergerCatalog(
        halo_idx=events.halo_idx, desc_node=events.desc_node,
        desc_tree_id=store.column("TreeID")[events.desc_node],
        main_node=events.main_node, sec_node=events.sec_node,
        snap_merger=events.snap_merger.astype(np.int64),
        a_merger=a_merger, z_merger=z_merger,
        snap_main_peak=main_peak_snap.astype(np.int64), a_main_peak=a_main_peak,
        snap_sec_peak=peak_snap.astype(np.int64), a_sec_peak=a_sec_peak,
        dt_to_desc_Gyr=dt_to_desc, dt_since_peak_Gyr=dt_since_peak,
        M0_Crit200=M0_crit, M0_FoF=M0_fof, M0_Splashback=M0_sp,
        M_main_peak=M_main_peak, M_sec_peak=M_sec_peak,
        M_main_premerge=M_main_pre, M_sec_premerge=M_sec_pre,
        main_peak_len=main_peak_len, sec_peak_len=sec_peak_len,
        xi_peak=xi_peak, xi_premerge=xi_pre,
        is_major=is_major.astype(bool), status=status,
        major_ratio=float(major_ratio), minor_floor=float(minor_floor),
        masses=masses,
    )


def add_property_imprints(
    catalog: MergerCatalog,
    store,
    backend,
    events: RawEvents,
    peak: PeakHistory,
    moments,
    quantities: tuple[str, ...],
    apertures: str | Sequence[str],
    topo: BranchTopology | None = None,
    half_tdyn: float = 0.5,
    jump_max_snaps: int = 5,
) -> None:
    """Add per-event property-imprint columns to ``catalog.imprints`` in place.

    ``apertures`` may be a single aperture (str) or several. For each quantity
    ``Q`` available at an aperture we store the progenitor *contrast* columns
    ``<Q>_main_peak`` / ``<Q>_sec_peak`` -- ``Q`` at each progenitor's own
    peak-mass (t_max) FOF-central row (topology-free). With a single aperture the
    columns are unnamespaced (backward-compatible); with several they are written
    namespaced as ``<Q>_{main,sec}_peak_<Ap>`` for every aperture, plus an
    unnamespaced alias for the PRIMARY (``apertures[0]``) so existing readers keep
    working. The primary is recorded in the ``imprint_aperture`` attr; the full
    list in ``imprint_apertures``.

    Descendant *response* ``<Q>_jump`` = Q(a_merger + ~half_tdyn) - Q(a_merger -
    ~half_tdyn) is added when ``topo`` is the main-branch topology of the FINAL
    (z=0 / a=final) descendant set: each event's ``desc_node`` is looked up on the
    surviving branch that carries it (each node lies on at most one main branch),
    so the branch extends forward past the merger. The window is capped at
    ``jump_max_snaps`` grid steps each side -- the mean-density dynamical time runs
    away as a^{3/2} into the de Sitter future while age ~ ln a, so an uncapped
    +/-half_tdyn window degenerates into "final minus formation". ``jump_dlna``
    (actual ln a span) and ``jump_window_flag`` (0 full on-grid / 1 capped by
    jump_max_snaps / 3 descendant not on a surviving branch) record the window;
    events off a surviving branch get NaN jump + flag 3. ``topo=None`` skips jumps.
    """
    if isinstance(apertures, str):
        apertures = (apertures,)
    apertures = tuple(apertures)
    primary = apertures[0]
    multi = len(apertures) > 1
    catalog.imprint_aperture = primary
    catalog.imprint_apertures = apertures
    want_jump = topo is not None

    # (column tag, aperture) to compute once; single -> unnamespaced only.
    tag_ap = ([(f"_{ap}", ap) for ap in apertures] if multi
              else [("", primary)])

    def _alias_primary() -> None:
        """After computing namespaced columns, copy the primary to unnamespaced."""
        if not multi:
            return
        bases = ["main_peak", "sec_peak"] + (["jump"] if want_jump else [])
        for q in quantities:
            for base in bases:
                src = f"{q}_{base}_{primary}"
                if src in catalog.imprints:
                    catalog.imprints[f"{q}_{base}"] = catalog.imprints[src]

    if events.n_events == 0:
        for tag, ap in tag_ap:
            for q in quantities:
                if backend.available(q, ap):
                    catalog.imprints[f"{q}_main_peak{tag}"] = np.empty(0)
                    catalog.imprints[f"{q}_sec_peak{tag}"] = np.empty(0)
                    if want_jump:
                        catalog.imprints[f"{q}_jump{tag}"] = np.empty(0)
        if want_jump:
            catalog.imprints["jump_dlna"] = np.empty(0)
            catalog.imprints["jump_window_flag"] = np.empty(0, dtype=np.uint8)
        _alias_primary()
        return

    main_peak_central = store.central_row_of(peak.peak_node[events.main_node])
    sec_peak_central = store.central_row_of(peak.peak_node[events.sec_node])

    # --- progenitor-peak contrasts, per aperture ---
    for tag, ap in tag_ap:
        for q in quantities:
            col = backend.resolve_column(q, ap)
            if col is None:
                continue
            data = store.column(col).astype(float)
            catalog.imprints[f"{q}_main_peak{tag}"] = data[main_peak_central]
            catalog.imprints[f"{q}_sec_peak{tag}"] = data[sec_peak_central]

    if not want_jump:
        _alias_primary()
        return

    # --- descendant response (jump) on the surviving main-branch topology ---
    grid = topo.grid_snaps
    lna = np.log(np.asarray(moments.scale_factors, dtype=float))
    merger_col = np.searchsorted(grid, events.snap_merger)
    before_full = baseline_indices(moments.ages, moments.dynamical_times, half_tdyn)[merger_col]
    after_full = baseline_indices(moments.ages, moments.dynamical_times, -half_tdyn)[merger_col]
    col_before = np.maximum(before_full, merger_col - jump_max_snaps)
    col_after = np.minimum(after_full, merger_col + jump_max_snaps)
    capped = ((before_full < merger_col - jump_max_snaps)
              | (after_full > merger_col + jump_max_snaps))

    # each event's descendant node -> its surviving main-branch row (a node lies
    # on at most one main branch, so the map is single-valued; miss -> -1)
    order = np.argsort(topo.flat_node)
    fn_sorted = topo.flat_node[order]
    fb_sorted = topo.flat_halo_idx[order]
    pos = np.clip(np.searchsorted(fn_sorted, events.desc_node), 0, len(fn_sorted) - 1)
    on_branch = fn_sorted[pos] == events.desc_node
    branch_row = np.where(on_branch, fb_sorted[pos], -1)
    safe_row = np.where(on_branch, branch_row, 0)

    catalog.imprints["jump_dlna"] = np.where(
        on_branch, lna[col_after] - lna[col_before], np.nan)
    catalog.imprints["jump_window_flag"] = np.where(
        ~on_branch, 3, np.where(capped, 1, 0)).astype(np.uint8)

    for tag, ap in tag_ap:
        for q in quantities:
            if backend.resolve_column(q, ap) is None:
                continue
            grid_q = backend.gather(q, ap, topo)               # [n_surv, n_snaps]
            jump = grid_q[safe_row, col_after] - grid_q[safe_row, col_before]
            catalog.imprints[f"{q}_jump{tag}"] = np.where(on_branch, jump, np.nan)

    _alias_primary()


# --------------------------------------------------------------------------- #
# Dong et al. (2022) arm: infall events + host log-mass-growth table
# --------------------------------------------------------------------------- #
# In SUBFIND-HBT a satellite is tracked persistently, so an INFALL is a subhalo
# that was its own FOF central at the previous snapshot and becomes a satellite of
# a larger FOF (~ crossing the host R_vir); a SPLASHOUT is the reverse. The
# specific merger rate dN/dxi/dlogM (Dong 2022 Eq. 1) = (infall mergers of ratio
# xi) / (host logarithmic mass growth), with virial masses; it reproduces their
# universal double-Schechter Eq. 2. See Thesis Notes §8.

@define(slots=True)
class InfallCatalog:
    """Flat infall/splashout event catalog (the Dong arm).

    Masses are stored in every available definition (``M_sat_<def>`` /
    ``M_host_<def>``) so the ratio can be formed in any definition downstream.
    ``is_splashout`` separates the two event kinds (net merger = infall - splashout).
    """

    sat_node: np.ndarray        # (N,) the infalling subhalo node (now satellite)
    host_node: np.ndarray       # (N,) the FOF-central host node it fell into (post, =M0)
    host_prog_node: np.ndarray  # (N,) the host's main progenitor (pre-merger host A)
    prog_node: np.ndarray       # (N,) the satellite's last-central (infall) node
    snap: np.ndarray            # (N,)
    a: np.ndarray               # (N,) scale factor of the event
    sat_len: np.ndarray         # (N,) satellite particle count at infall (prog node)
    host_len: np.ndarray        # (N,) host particle count (post-merger, at host_node)
    is_splashout: np.ndarray    # (N,) bool
    # M_sat_<def> (infaller, last-central), M_host_<def> (post-merger host = M0),
    # M_host_prog_<def> (host's pre-merger progenitor = M_1 for the Fakhouri ratio).
    masses: dict[str, np.ndarray] = field(factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.snap)

    def to_hdf5(self, group) -> None:
        for name in ("sat_node", "host_node", "host_prog_node", "prog_node",
                     "snap", "a", "sat_len", "host_len", "is_splashout"):
            group.create_dataset(name, data=getattr(self, name))
        for name, arr in self.masses.items():
            group.create_dataset(name, data=arr)
        group.attrs["n_events"] = self.n_events
        group.attrs["n_infall"] = int((~self.is_splashout).sum())
        group.attrs["n_splashout"] = int(self.is_splashout.sum())


def detect_infalls(
    store,
    moments,
    mass_defs: tuple[str, ...] = MASS_DEFS,
    backend=None,
    min_host_particles: int = DEFAULT_MIN_HOST_PARTICLES,
    min_sec_particles: int = DEFAULT_INFALL_MIN_SEC_PARTICLES,
    a_min: float = 0.05,
) -> InfallCatalog:
    """Global infall/splashout detection via central<->satellite transitions.

    Vectorized over the whole tree. A node that was central at its main progenitor
    and is now a satellite is an *infall*; the reverse is a *splashout*. Masses are
    read in every ``mass_defs`` definition (``backend.resolve_column``): the
    satellite's own mass at its last-central node, the host's mass at the event.
    """
    n = store.n_rows
    snap = store.column("SnapNum")
    tid = store.column("TreeID")
    mp_abs = store.abs_from_relative(tid, store.column("TreeMainProgenitor"))
    rows = np.arange(n)
    is_cen = store.is_central(rows)
    host_cen = store.central_row_of(rows)
    sublen = store.column("SubhaloLen")

    grid = moments.snapshot_ids
    a_grid = moments.scale_factors
    a_of = a_grid[np.clip(np.searchsorted(grid, snap), 0, len(grid) - 1)]

    valid = mp_abs >= 0
    cen_mp = np.zeros(n, dtype=bool)
    cen_mp[valid] = is_cen[mp_abs[valid]]
    infall = valid & cen_mp & (~is_cen)
    splashout = valid & (~cen_mp) & is_cen

    sat_len = np.where(valid, sublen[np.where(valid, mp_abs, 0)], 0)
    host_len = sublen[host_cen]
    cut = (sat_len > min_sec_particles) & (host_len > min_host_particles) & (a_of >= a_min)
    keep = (infall | splashout) & cut
    idx = np.flatnonzero(keep)

    prog = mp_abs[idx]
    host = host_cen[idx]
    host_prog = mp_abs[host]                    # host's main progenitor (pre-merger host A)
    masses: dict[str, np.ndarray] = {}
    for d in mass_defs:
        col = backend.resolve_column("M", d) if backend is not None else f"Group_M_{d}"
        if col is None or not store.has_column(col):
            continue
        data = store.column(col).astype(float)
        masses[f"M_sat_{d}"] = data[prog]      # satellite's own mass at last-central node
        masses[f"M_host_{d}"] = data[host]     # post-merger host mass at the event (= M0)
        masses[f"M_host_prog_{d}"] = np.where(  # pre-merger host mass (M_1 for the ratio)
            host_prog >= 0, data[np.where(host_prog >= 0, host_prog, 0)], np.nan)

    return InfallCatalog(
        sat_node=idx.astype(np.int64), host_node=host.astype(np.int64),
        host_prog_node=host_prog.astype(np.int64),
        prog_node=prog.astype(np.int64), snap=snap[idx].astype(np.int64), a=a_of[idx],
        sat_len=sat_len[idx].astype(np.int64), host_len=host_len[idx].astype(np.int64),
        is_splashout=splashout[idx], masses=masses,
    )


def host_growth_histogram(
    store,
    moments,
    mass_bin_edges: np.ndarray,
    mass_defs: tuple[str, ...] = MASS_DEFS,
    backend=None,
    min_host_particles: int = DEFAULT_MIN_HOST_PARTICLES,
) -> dict:
    """Compact host log-mass-growth denominator for the Dong specific rate.

    For every central host that survives as a central to its descendant, sum the
    one-step ``Δlog10 M_host`` into a ``[mass_bin, snap]`` grid (and a count grid),
    per mass definition. This is the denominator of ``dN/dξ/dlogM`` (Dong Eq. 1);
    storing it pre-binned (rather than per-host) keeps the output tiny and lets the
    rate be reconstructed locally at any (coarser) host-mass binning. ``mass_bin_edges``
    are log10 M [Msun/h].
    """
    snap = store.column("SnapNum")
    tid = store.column("TreeID")
    desc_abs = store.abs_from_relative(tid, store.column("TreeDescendant"))
    rows = np.arange(store.n_rows)
    is_cen = store.is_central(rows)
    sublen = store.column("SubhaloLen")

    host_ok = is_cen & (desc_abs >= 0) & (sublen > min_host_particles)
    desc_cen = store.central_row_of(np.where(desc_abs >= 0, desc_abs, 0))
    host_ok &= is_cen[desc_cen]                       # host stays a central (clean self-growth)

    grid = moments.snapshot_ids
    snap_edges = np.append(grid - 0.5, grid[-1] + 0.5)

    out: dict = {"mass_bin_edges": mass_bin_edges, "snapshot_ids": grid}
    for d in mass_defs:
        col = backend.resolve_column("M", d) if backend is not None else f"Group_M_{d}"
        if col is None or not store.has_column(col):
            continue
        data = store.column(col).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logMh = np.log10(data) + 10.0          # Msun/h
            dlog = (np.log10(data[desc_cen]) - np.log10(data))  # base-10, one step
        ok = host_ok & np.isfinite(dlog) & (dlog > 0) & np.isfinite(logMh)
        sum_d, _, _ = np.histogram2d(logMh[ok], snap[ok], bins=[mass_bin_edges, snap_edges],
                                     weights=dlog[ok])
        cnt, _, _ = np.histogram2d(logMh[ok], snap[ok], bins=[mass_bin_edges, snap_edges])
        out[f"sum_dlogM_{d}"] = sum_d
        out[f"count_{d}"] = cnt
    return out


# --------------------------------------------------------------------------- #
# Per-node "scale factor of last merger" (forest-wide DP) -> full_tree columns
# --------------------------------------------------------------------------- #
# For EVERY node, the scale factor of the most recent merger along its own
# main-progenitor line. A single forward DP in ascending snapshot order (like
# compute_peak_history): a node inherits its parent's value unless it is itself a
# merger descendant, in which case it takes its own scale factor. Read at a root
# node (TreeDescendant<0, the a=100/10000 final halo) this is "when that
# descendant last merged"; read at any node it is that halo's last-merger epoch,
# and a - a_last is its time-since-last-merger relaxation clock. NaN = the line
# never had a qualifying merger. See Thesis Notes freezeout_merger_rate.md.

# (xi-threshold, suffix) for the stored columns. "Major" resets structure; "Any"
# is every resolved merger.
LAST_MERGER_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (1.0 / 3.0, "Major"), (0.1, "Tenth"), (0.0, "Any"),
)


def _propagate_last_event(store, a_of_node: np.ndarray, is_event: np.ndarray) -> np.ndarray:
    """Per-node scale factor of the most recent event along the main-prog line.

    Forward DP in ascending ``SnapNum``: event nodes keep their own ``a``;
    non-event nodes inherit their (already-finalized, earlier) main progenitor's
    value. NaN where no event occurs at or before the node along its branch.
    """
    snap = store.column("SnapNum")
    tid = store.column("TreeID")
    mp_abs = store.abs_from_relative(tid, store.column("TreeMainProgenitor"))
    a_last = np.where(is_event, a_of_node, np.nan).astype(float)
    for s in np.unique(snap):                       # ascending; ~75 steps
        rows = np.flatnonzero(snap == s)
        mp = mp_abs[rows]
        inherit = (mp >= 0) & ~is_event[rows]       # non-event -> take main prog's value
        r = rows[inherit]
        a_last[r] = a_last[mp[inherit]]
    return a_last


def compute_last_merger_fields(
    store,
    moments,
    *,
    backend=None,
    thresholds: tuple[tuple[float, str], ...] = LAST_MERGER_THRESHOLDS,
    mass_def: str = "Virial",
    min_particles: int = 100,
    a_min: float = 0.0,
) -> dict[str, np.ndarray]:
    """Per-node ``aLastMerger_{Coal,Infall}_{tag}`` arrays for every threshold.

    *Coalescence* timing: a node is a merger descendant when it has >1 progenitor
    (``detect_mergers_forest``); the ratio is the peak-``SubhaloMass`` ``xi_peak``.
    *Infall* timing: a node is a host that received an infall (``detect_infalls``);
    the ratio is ``M_sat/M_host_prog`` in ``mass_def``. Both are propagated along
    main-progenitor lines by :func:`_propagate_last_event`. Returns one float
    array per (timing, threshold), keyed ``aLastMerger_Coal_Major`` etc.

    ``min_particles`` is applied to **both** participants (the main/host and the
    secondary/infaller, at each one's peak/last-central SubhaloLen) so unresolved
    mergers -- where either side has < ``min_particles`` -- are excluded.
    """
    grid = np.asarray(moments.snapshot_ids)
    a_grid = np.asarray(moments.scale_factors, dtype=float)
    a_of_node = a_grid[np.searchsorted(grid, store.column("SnapNum"))]
    n = store.n_rows
    fields: dict[str, np.ndarray] = {}

    # --- coalescence (subhalo-tree >1-progenitor) --- #
    peak = compute_peak_history(store)
    ev = detect_mergers_forest(store)
    if ev.n_events:
        sublen = store.column("SubhaloLen")
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = peak.peak_mass[ev.sec_node] / peak.peak_mass[ev.main_node]
        main_len = sublen[peak.peak_node[ev.main_node]]   # both participants resolved
        sec_len = sublen[peak.peak_node[ev.sec_node]]
        base = np.isfinite(xi) & (xi > 0) \
            & (main_len >= min_particles) & (sec_len >= min_particles) \
            & (a_of_node[ev.desc_node] >= a_min)
    else:
        xi = np.empty(0)
        base = np.empty(0, bool)
    for tau, tag in thresholds:
        is_event = np.zeros(n, dtype=bool)
        if ev.n_events:
            sel = base & (xi >= tau)
            is_event[ev.desc_node[sel]] = True
        fields[f"aLastMerger_Coal_{tag}"] = _propagate_last_event(store, a_of_node, is_event)

    # --- infall (central->satellite / virial crossing) --- #
    # min_particles on both host and infaller (detect_infalls floors host_len + sat_len).
    infall = detect_infalls(
        store, moments, mass_defs=(mass_def,), backend=backend,
        min_host_particles=min_particles, min_sec_particles=min_particles,
        a_min=a_min,
    )
    if infall.n_events and f"M_host_prog_{mass_def}" in infall.masses:
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_inf = (np.asarray(infall.masses[f"M_sat_{mass_def}"], float)
                      / np.asarray(infall.masses[f"M_host_prog_{mass_def}"], float))
        base_inf = np.isfinite(xi_inf) & (xi_inf > 0) & (~infall.is_splashout)
    else:
        xi_inf = np.empty(0)
        base_inf = np.empty(0, bool)
    for tau, tag in thresholds:
        is_event = np.zeros(n, dtype=bool)
        if infall.n_events:
            sel = base_inf & (xi_inf >= tau)
            is_event[infall.host_node[sel]] = True
        fields[f"aLastMerger_Infall_{tag}"] = _propagate_last_event(store, a_of_node, is_event)

    return fields
