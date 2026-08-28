"""Cleaning, interpolation, and position-tracking validation/correction.

De-noising of tree-derived histories:
- ``clean_histories`` (vectorized over ``[n_halos, n_snaps]``) masks non-finite
  points and optionally enforces monotonic growth / light smoothing, returning a
  per-cell :class:`PointStatus`.
- ``loglog_interp`` reproduces the proven log-log interpolation from
  ``studies/mah/helpers.py``.
- ``position_corrections`` re-tracks each descendant by position+mass (tessera)
  and, where the tree main-progenitor link disagrees, SUBSTITUTES the
  position-matched halo's tree node so every gathered quantity is corrected
  consistently (flagging those cells ``POS_CORRECTED``).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

import numpy as np
from attrs import evolve
from scipy.interpolate import interp1d

from ..merger_trees.topology import BranchTopology


class PointStatus(IntEnum):
    SUCCESS = 0
    REDUCED_INTERVAL = 1        # < 1 t_dyn of history available for a rate
    FLOORED = 2                 # below the snapshot floor / unresolved
    INVALID = 3                 # NaN / non-finite
    POS_MISMATCH = 4            # tree link disagrees with position-match track
    POS_CORRECTED = 5           # value substituted from the position-matched halo
    FORMED_BEFORE_TRACKING = 6  # scalar: halo already > threshold at earliest snap


def loglog_interp(scale_factors: np.ndarray, values: np.ndarray) -> interp1d:
    """interp1d(log10 value, log10 a) on the finite, positive portion."""
    return interp1d(np.log10(values), np.log10(scale_factors))


def clean_histories(
    a: np.ndarray,
    values: np.ndarray,
    monotonic: bool = False,
    smooth: bool = False,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized clean of a ``[n_halos, n_snaps]`` property block.

    Returns ``(cleaned, status)``. Non-finite cells -> NaN + INVALID. With
    ``monotonic`` the running max (increasing-a order) is enforced per row over
    finite cells (NaN gaps preserved). With ``smooth`` a NaN-aware centred moving
    average is applied.
    """
    values = np.asarray(values, dtype=float)
    status = np.full(values.shape, PointStatus.SUCCESS, dtype=np.uint8)

    bad = ~np.isfinite(values)
    cleaned = values.copy()
    cleaned[bad] = np.nan
    status[bad] = PointStatus.INVALID

    if monotonic:
        filled = np.fmax.accumulate(np.where(bad, -np.inf, cleaned), axis=1)
        cleaned = filled
        cleaned[bad] = np.nan

    if smooth:
        cleaned = _moving_average_2d(cleaned, smooth_window)

    return cleaned, status


def _moving_average_2d(vals: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred moving average along axis 1 (cumsum-based, vectorized)."""
    n = vals.shape[1]
    half = window // 2
    finite = np.isfinite(vals).astype(float)
    v0 = np.where(finite > 0, vals, 0.0)
    zeros = np.zeros((vals.shape[0], 1))
    csum = np.concatenate([zeros, np.cumsum(v0, axis=1)], axis=1)
    ccnt = np.concatenate([zeros, np.cumsum(finite, axis=1)], axis=1)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    s = csum[:, hi] - csum[:, lo]
    c = ccnt[:, hi] - ccnt[:, lo]
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


def clean_history(
    a: np.ndarray, values: np.ndarray, monotonic: bool = False,
    smooth: bool = False, smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D convenience wrapper around :func:`clean_histories`."""
    cleaned, status = clean_histories(
        a, np.asarray(values)[None, :], monotonic=monotonic,
        smooth=smooth, smooth_window=smooth_window,
    )
    return cleaned[0], status[0]


def _apply_position_match(
    topo: BranchTopology,
    tree_group_nr: np.ndarray,
    central_row_for: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_match_distance: float | None,
    branch_id: np.ndarray,
    length: np.ndarray,
    snap_num: np.ndarray,
    group_nr: np.ndarray,
    match_distance: np.ndarray,
) -> tuple[BranchTopology, np.ndarray, np.ndarray]:
    """Substitute mismatched tree nodes from a flat position-match result.

    Shared by the live (tessera) and reuse (saved branch-catalog) paths. The
    node arrays are the per-branch sequences concatenated in ``branch_id`` order
    (``length[b]`` consecutive nodes per branch); ``branch_id[b]`` is the row of
    this run's ``topo`` (i.e. the descendant index). Fully vectorized.
    """
    n_halos, n_snaps = topo.n_halos, topo.n_snaps
    grid = np.asarray(topo.grid_snaps)

    node_grid = np.full((n_halos, n_snaps), -1, dtype=np.int64)
    node_grid[topo.flat_halo_idx, topo.flat_grid_col] = topo.flat_node
    pos_status = np.where(node_grid >= 0, PointStatus.SUCCESS, PointStatus.INVALID).astype(np.uint8)
    validated_group_nr = np.full((n_halos, n_snaps), np.nan)

    branch_id = np.asarray(branch_id, dtype=np.int64)
    length = np.asarray(length, dtype=np.int64)
    snap_flat = np.asarray(snap_num, dtype=np.int64)
    if branch_id.size and snap_flat.size:
        # node -> its branch's topo row (each branch owns length[b] consecutive nodes)
        i_of_node = np.repeat(branch_id, length)
        group_flat = np.asarray(group_nr, dtype=np.int64)
        dist_flat = np.asarray(match_distance, dtype=float)

        col = np.searchsorted(grid, snap_flat)
        col_c = np.clip(col, 0, n_snaps - 1)
        valid = (col < n_snaps) & (grid[col_c] == snap_flat)

        tg = np.full(snap_flat.shape, np.nan)
        tg[valid] = tree_group_nr[i_of_node[valid], col_c[valid]]
        valid &= np.isfinite(tg)

        far = ((dist_flat > max_match_distance) if max_match_distance is not None
               else np.zeros(snap_flat.shape, dtype=bool))
        mism = valid & ((group_flat != tg) | far)
        if mism.any():
            halo_idx = i_of_node[mism]
            col_idx = col_c[mism]
            matched_group = group_flat[mism]
            corrected_rows = central_row_for(grid[col_idx], matched_group)
            ok = corrected_rows >= 0
            hi, ci = halo_idx[ok], col_idx[ok]
            node_grid[hi, ci] = corrected_rows[ok]
            pos_status[hi, ci] = PointStatus.POS_CORRECTED
            validated_group_nr[hi, ci] = matched_group[ok]

    corrected_flat_node = node_grid[topo.flat_halo_idx, topo.flat_grid_col]
    corrected_topo = evolve(topo, flat_node=corrected_flat_node)
    return corrected_topo, pos_status, validated_group_nr


def position_corrections(
    catalog_dir,
    box_size: float,
    topo: BranchTopology,
    tree_group_nr: np.ndarray,
    desc_group_nr: np.ndarray,
    desc_snap: int,
    central_row_for: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_match_distance: float | None = None,
    n_threads: int = 0,
    snap_max: int | None = None,
) -> tuple[BranchTopology, np.ndarray, np.ndarray]:
    """Re-track by position+mass and substitute mismatched tree nodes.

    Returns ``(corrected_topo, pos_status, validated_group_nr)``:
    - ``corrected_topo`` is ``topo`` with ``flat_node`` swapped to the
      position-matched halo's central tree row wherever the tree link is suspect
      (so a subsequent gather corrects every quantity at once).
    - ``pos_status`` is ``[n_halos, n_snaps]`` uint8 (POS_CORRECTED on swapped cells).
    - ``validated_group_nr`` is ``[n_halos, n_snaps]`` float (the matched GroupNr on
      corrected cells, else NaN) — the validated halo identity.

    A cell is corrected only when the matched group differs from the tree branch's
    AND that group has a central tree node at that snapshot. Cells where the
    matched group has no tree node are left as-is (no reliable substitute).

    ``snap_max`` caps which catalog snapshots tessera preloads (effective cap is
    ``min(snap_max, desc_snap)`` since tracking is backward). ``None`` falls back
    to ``desc_snap`` — never load future-extended catalogs no branch reaches.
    """
    import tessera as ts

    reader = ts.io.HaloCatalogReader(str(catalog_dir))
    start_points = [
        ts.io.BranchStartPoint(snap_num=int(desc_snap), group_nr=int(g), branch_id=i)
        for i, g in enumerate(desc_group_nr)
    ]
    cfg = ts.io.PositionMatchConfig()
    # Tracking is backward from desc_snap, so snapshots above the descendant are
    # never consulted. Cap snap_max at desc_snap (or lower, if the caller passed a
    # tighter bound) so tessera's PRELOAD never loads future-extended catalogs —
    # bounds peak memory with identical match results.
    eff_snap_max = int(desc_snap) if snap_max is None else min(int(snap_max), int(desc_snap))
    result = ts.io.extract_position_branches_parallel(
        reader, float(box_size), start_points, config=cfg,
        strategy=ts.io.PositionTrackingStrategy.PRELOAD_CATALOGS, n_threads=n_threads,
        snap_max=eff_snap_max,
    )
    return _apply_position_match(
        topo, tree_group_nr, central_row_for, max_match_distance,
        branch_id=np.asarray(result.branch_id),
        length=np.asarray(result.length),
        snap_num=np.asarray(result.snap_num),
        group_nr=np.asarray(result.group_nr),
        match_distance=np.asarray(result.match_distance),
    )


def position_corrections_from_catalog(
    branch_catalog_path,
    topo: BranchTopology,
    tree_group_nr: np.ndarray,
    desc_group_nr: np.ndarray,
    central_row_for: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_match_distance: float | None = None,
) -> tuple[BranchTopology, np.ndarray, np.ndarray]:
    """Reuse a branch catalog's ``PositionTrackedBranches`` instead of re-running
    tessera, then apply the identical correction.

    The saved position-tracked branches (written by save_branch_catalog with
    ``--include_position_tracking``) are the SAME backward position match
    ``position_corrections`` would recompute. We remap them onto this run's
    descendants by FOF ``GroupNr`` (``root_group`` <-> ``desc_group_nr``) and
    apply the correction without reading any halo catalogs -- this is what lets
    histories be extracted when the raw catalogs are unavailable.

    Descendants absent from the saved table (e.g. beyond
    ``--bc_max_position_halos`` or selected differently) are left uncorrected;
    their count is reported via a warning. Raises ``KeyError`` if the file has no
    ``PositionTrackedBranches`` group.
    """
    import h5py

    branch_catalog_path = Path(branch_catalog_path)
    with h5py.File(branch_catalog_path, "r") as f:
        if "PositionTrackedBranches" not in f:
            raise KeyError(
                f"{branch_catalog_path.name} has no PositionTrackedBranches group "
                f"-- rebuild it with save_branch_catalog --include_position_tracking."
            )
        bi = f["PositionTrackedBranches"]["BranchIndex"]
        hd = f["PositionTrackedBranches"]["HaloData"]
        s_root_group = bi["root_group"][:].astype(np.int64)
        s_start = bi["start_offset"][:].astype(np.int64)
        s_len = bi["length"][:].astype(np.int64)
        s_snap = hd["snap_num"][:].astype(np.int64)
        s_group = hd["group_nr"][:].astype(np.int64)
        s_dist = hd["match_distance"][:].astype(float)

    # descendant FOF group -> saved branch index (first wins on the off chance of dups)
    saved_of_group: dict[int, int] = {}
    for b in range(s_root_group.size):
        saved_of_group.setdefault(int(s_root_group[b]), b)

    desc_group_nr = np.asarray(desc_group_nr, dtype=np.int64)
    out_bid: list[int] = []
    out_len: list[int] = []
    snap_parts, group_parts, dist_parts = [], [], []
    for i, g in enumerate(desc_group_nr):
        b = saved_of_group.get(int(g))
        if b is None:
            continue
        s0, L = int(s_start[b]), int(s_len[b])
        out_bid.append(i)
        out_len.append(L)
        snap_parts.append(s_snap[s0:s0 + L])
        group_parts.append(s_group[s0:s0 + L])
        dist_parts.append(s_dist[s0:s0 + L])

    n_missing = len(desc_group_nr) - len(out_bid)
    if n_missing:
        warnings.warn(
            f"position reuse: {n_missing}/{len(desc_group_nr)} descendants are not "
            f"in {branch_catalog_path.name}'s PositionTrackedBranches (left "
            f"uncorrected) -- likely beyond --bc_max_position_halos or a different "
            f"descendant selection.",
            stacklevel=2,
        )

    empty_i = np.empty(0, dtype=np.int64)
    return _apply_position_match(
        topo, tree_group_nr, central_row_for, max_match_distance,
        branch_id=np.asarray(out_bid, dtype=np.int64),
        length=np.asarray(out_len, dtype=np.int64),
        snap_num=np.concatenate(snap_parts) if snap_parts else empty_i,
        group_nr=np.concatenate(group_parts) if group_parts else empty_i,
        match_distance=np.concatenate(dist_parts) if dist_parts else np.empty(0, dtype=float),
    )
