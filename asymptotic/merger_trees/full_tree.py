"""tessera-backed merger-tree backend.

``FullTreeStore`` owns the HDF5 handle and the relative<->absolute index algebra;
``TesseraTreeBackend`` adds tessera topology (main-branch tracing) and the
canonical-name resolution that lets one code path read both the composed
``full_tree.hdf5`` (180 cols, 84 features) and GADGET-native ``trees.hdf5``
(33 cols). See ``docs/assembly_pipeline_requirements.md``.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

# GADGET-4 split-tree filename pattern: trees.<N>.hdf5
_SERIES_RE = re.compile(r"^trees\.(\d+)\.hdf5$")

from ..simulation.moments import MomentsInTime
from . import topology as topo_mod
from .backends import MergerTreeBackend, SUBHALO_FIELDS

#: raw tessera batch result (a dict of NumPy arrays from the pybind11 layer)
TesseraBatchResult: TypeAlias = Mapping[str, NDArray[Any]]
#: {snapshot: sorted unique GroupNr} for one descendant, ascending by snapshot
SnapshotOrder: TypeAlias = "OrderedDict[int, NDArray[Any]]"
#: the canonical saved-order structure: {descendant GroupNr: {"tree", "prog"}}
SavedOrderInfo: TypeAlias = dict[int, dict[str, SnapshotOrder]]

# Per-backend aperture aliases. GADGET-native trees.hdf5 names the virial
# overdensity "TopHat200"; the composed full_tree uses "Virial".
_APERTURE_ALIASES = {"Virial": ("Virial", "TopHat200")}

DEFAULT_DESC_SNAP = 74


def _series_in(directory: Path, stem: str) -> list[Path]:
    """Sorted ``<stem>.<N>.hdf5`` shard series in ``directory`` (empty if none)."""
    rx = re.compile(rf"^{re.escape(stem)}\.(\d+)\.hdf5$")
    return sorted(
        (p for p in directory.iterdir() if rx.match(p.name)),
        key=lambda p: int(rx.match(p.name).group(1)),
    )


def _discover_files(path: Path) -> list[Path]:
    """Resolve a path to the ordered list of tree files (mirrors tessera).

    Handles both the raw GADGET ``trees`` series and the composed
    ``full_tree`` series (batched compose writes ``full_tree.<N>.hdf5``):
    - an existing single file (``full_tree.hdf5`` / ``trees.hdf5``): itself;
    - a ``<stem>.<N>.hdf5`` shard file: its whole sibling series by stem;
    - a non-existent ``<stem>.hdf5`` (e.g. ``full_tree.hdf5`` for a batched
      sharded compose): the ``<stem>.<N>.hdf5`` series beside it;
    - a directory: the ``trees.<N>.hdf5`` series in it, else a lone ``trees.hdf5``.
    """
    path = Path(path)
    shard_member = re.match(r"^(.+)\.(\d+)\.hdf5$", path.name)

    if path.is_file():
        if shard_member is None:
            return [path]                   # single consolidated file
        # given one shard directly -> return its whole series by stem
        series = _series_in(path.parent, shard_member.group(1))
        return series or [path]

    if path.is_dir():
        series = _series_in(path, "trees")
        if series:
            return series
        solo = path / "trees.hdf5"
        if solo.is_file():
            return [solo]
        raise FileNotFoundError(f"No tree files found for {path}")

    # Non-existent <stem>.hdf5: look for the <stem>.<N>.hdf5 sharded series
    # beside it (the batched composed full_tree case), then fall back to the
    # raw trees series / lone trees.hdf5 in the same directory.
    directory = path.parent
    if not directory.exists():
        raise FileNotFoundError(path)
    if path.name.endswith(".hdf5"):
        stem = path.name[: -len(".hdf5")]
        series = _series_in(directory, stem)
        if series:
            return series
    series = _series_in(directory, "trees")
    if series:
        return series
    solo = directory / "trees.hdf5"
    if solo.is_file():
        return [solo]
    raise FileNotFoundError(f"No tree files found for {path}")


def _is_group_field(quantity: str) -> bool:
    """True when ``quantity`` is a FOF-group field (read at the central row)."""
    return quantity not in SUBHALO_FIELDS


class FullTreeStore:
    """Cached reader for a GADGET-4 merger tree (``TreeHalos`` group).

    Handles both a single consolidated file (``full_tree.hdf5`` / merged
    ``trees.hdf5``) and a GADGET-4 split series (``treedata/trees.<N>.hdf5``).
    For a split series the per-file ``TreeHalos`` columns are concatenated in
    file-index order; since GADGET-4 ``TreeTable.StartOffset`` is already a global
    offset, the resulting absolute row index matches tessera's ``tree_index``
    (tessera concatenates the same files in the same order) -- so topology rows
    from either tracer index the same gathered columns.
    """

    def __init__(self, path: Path, cosmo=None):
        self.path = Path(path).resolve()
        self._files = _discover_files(self.path)
        self._h5s = [h5py.File(f, "r") for f in self._files]
        self._h5 = self._h5s[0]   # file-0: shared schema + TreeTimes
        self._cache: dict[str, np.ndarray] = {}

        # per-file row counts -> global offsets (concatenation order == tessera's)
        self._file_nrows = [int(h["TreeHalos"]["SnapNum"].shape[0]) for h in self._h5s]
        self._n_rows = int(sum(self._file_nrows))

        # TreeTable: only root-bearing files carry it; StartOffset is already a
        # GLOBAL index in GADGET-4, so we concatenate without per-file shifting.
        # Two flavors of "no TreeTable" must both be skipped: raw GADGET shards
        # omit the group entirely, while composed full_tree.<N>.hdf5 shards always
        # create the group (deterministic layout) but leave it empty when their
        # source shard had Ntrees_ThisFile=0 -- so guard on the TreeID dataset.
        tree_ids_parts, start_parts = [], []
        for h in self._h5s:
            tt = h.get("TreeTable")
            if tt is not None and "TreeID" in tt:
                tree_ids_parts.append(tt["TreeID"][:])
                start_parts.append(tt["StartOffset"][:])
        tree_ids = np.concatenate(tree_ids_parts)
        start = np.concatenate(start_parts)
        # Lookup indexed by TreeID -> StartOffset (TreeIDs are dense 0..n-1).
        self._start_of_tree = np.zeros(int(tree_ids.max()) + 1, dtype=np.int64)
        self._start_of_tree[tree_ids] = start

        # cosmo only needed for GADGET-native trees (no baked Moment_*);
        # full_tree.hdf5 already carries the dynamical-time axis.
        self.moments = MomentsInTime.from_tree_times(self._h5["TreeTimes"], cosmo=cosmo)
        self._columns = set(self._h5["TreeHalos"].keys())
        # lazily built (SnapNum, GroupNr) -> central row index for position correction
        self._ckeys: np.ndarray | None = None
        self._crows: np.ndarray | None = None
        self._kmul: int = 0

    @classmethod
    def open(cls, path: str | Path, cosmo=None) -> "FullTreeStore":
        return cls(Path(path), cosmo=cosmo)

    @property
    def available_columns(self) -> set[str]:
        return self._columns

    def has_column(self, name: str) -> bool:
        return name in self._columns

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def box_size(self) -> float | None:
        """Box size in comoving Mpc/h from GADGET ``/Parameters/BoxSize``, or None.

        Lets the backend report box_size without instantiating tessera's C++
        MergerTree -- needed for batched sharded full_tree.<N>.hdf5 series, which
        tessera can't open via the nominal full_tree.hdf5 path. ``BoxSize`` is in
        the sim's length unit; ``UnitLength_in_cm`` (if present) converts it to
        Mpc/h (identity when the unit is already Mpc).
        """
        if "Parameters" not in self._h5:
            return None
        p = self._h5["Parameters"].attrs
        if "BoxSize" not in p:
            return None
        box = float(p["BoxSize"])
        if "UnitLength_in_cm" in p:
            box *= float(p["UnitLength_in_cm"]) / 3.085677581491367e24  # -> Mpc/h
        return box

    def column(self, name: str) -> np.ndarray:
        """Bulk-read one ``TreeHalos`` column once and cache it.

        For a split series the per-file slices are concatenated in file-index
        order (= the global row order tessera uses).
        """
        if name not in self._cache:
            if len(self._h5s) == 1:
                self._cache[name] = self._h5s[0]["TreeHalos"][name][:]
            else:
                self._cache[name] = np.concatenate(
                    [h["TreeHalos"][name][:] for h in self._h5s]
                )
        return self._cache[name]

    def abs_from_relative(self, tree_id: np.ndarray, rel: np.ndarray) -> np.ndarray:
        """Absolute row from per-tree relative index (-1 stays -1)."""
        rel = np.asarray(rel)
        out = self._start_of_tree[np.asarray(tree_id)] + rel
        return np.where(rel < 0, -1, out)

    def central_row_of(self, rows: np.ndarray) -> np.ndarray:
        """Absolute FOF-central row for each row (where ``Group_*`` values live)."""
        rows = np.asarray(rows)
        tid = self.column("TreeID")[rows]
        first = self.column("TreeFirstHaloInFOFgroup")[rows]
        return self.abs_from_relative(tid, first)

    def is_central(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        return self.central_row_of(rows) == rows

    def central_row_for(self, snaps: np.ndarray, group_nrs: np.ndarray) -> np.ndarray:
        """Map (SnapNum, GroupNr) -> central tree row (or -1 if absent).

        Used by position correction to resolve a position-matched group at a
        snapshot back to its tree node. Builds a sorted key index once.
        """
        if self._ckeys is None:
            snap_all = self.column("SnapNum").astype(np.int64)
            gnr_all = self.column("GroupNr").astype(np.int64)
            rows = np.arange(self.n_rows)
            central = self.is_central(rows)
            self._kmul = int(gnr_all.max()) + 1
            keys = snap_all[central] * self._kmul + gnr_all[central]
            order = np.argsort(keys, kind="stable")
            self._ckeys = keys[order]
            self._crows = rows[central][order]

        q = np.asarray(snaps, dtype=np.int64) * self._kmul + np.asarray(group_nrs, dtype=np.int64)
        pos = np.clip(np.searchsorted(self._ckeys, q), 0, len(self._ckeys) - 1)
        found = self._ckeys[pos] == q
        return np.where(found, self._crows[pos], -1)

    def close(self) -> None:
        for h in self._h5s:
            h.close()
        self._cache.clear()


def _normalize_desc_rows(desc_nodes: ArrayLike, n_rows: int) -> NDArray[np.int64]:
    """Validate and normalize selected rows for a tessera batch call.

    Shared by both batch entry points: integer dtype, one-dimensional, in
    ``[0, n_rows)``, returned C-contiguous ``int64`` exactly as the native
    APIs require.
    """
    desc = np.asarray(desc_nodes)
    if desc.dtype.kind not in "iu":
        raise TypeError(f"desc_nodes must be integer rows, got dtype {desc.dtype}")
    # ndim first: ascontiguousarray would promote a 0-d scalar to 1-D
    if desc.ndim != 1:
        raise ValueError("desc_nodes must be one-dimensional")
    desc = np.ascontiguousarray(desc, dtype=np.int64)
    if desc.size and (int(desc.min()) < 0 or int(desc.max()) >= n_rows):
        raise ValueError(
            f"desc_nodes contain rows outside the valid TreeHalos range [0, {n_rows})"
        )
    return desc


def _native_array(
    result: TesseraBatchResult, key: str, dtype: type[np.generic]
) -> NDArray[Any]:
    """One field of a native batch result, held to the frozen contract.

    The record promises tessera's exact layout and dtypes, so a field that
    does not already satisfy it is a defect to report — never something to
    silently cast, copy, or reshape into compliance.
    """
    if key not in result:
        raise RuntimeError(
            f"tessera progenitor-tree result is missing the {key!r} field"
        )
    arr = result[key]
    if (
        not isinstance(arr, np.ndarray)
        or arr.dtype != np.dtype(dtype)
        or arr.ndim != 1
        or not arr.flags.c_contiguous
    ):
        got = (
            f"dtype={arr.dtype}, ndim={arr.ndim}, "
            f"c_contiguous={bool(arr.flags.c_contiguous)}"
            if isinstance(arr, np.ndarray) else f"type={type(arr).__name__}"
        )
        raise RuntimeError(
            f"tessera progenitor-tree result field {key!r} violates the "
            f"native contract (expected a 1-D C-contiguous "
            f"{np.dtype(dtype)} array; got {got})"
        )
    return arr


def _progenitor_topology_from_batch(
    result: TesseraBatchResult,
    desc_nodes: NDArray[np.int64],
) -> topo_mod.ProgenitorTreeTopology:
    """Wrap tessera's all-progenitor batch result in a topology record.

    tessera's layout is already canonical here (deterministic DFS preorder,
    input order preserved, exclusive-prefix offsets), so this validates
    dtypes, layout, and the branch bookkeeping, then adopts the native
    arrays directly — no reordering, casting, or copying of the potentially
    enormous flat arrays.
    """
    start_row = _native_array(result, "start_row", np.int64)
    offsets = _native_array(result, "offset", np.int64)
    lengths = _native_array(result, "length", np.int64)
    flat_node = _native_array(result, "flat_tree_index", np.int64)
    flat_snap = _native_array(result, "flat_snap_num", np.int32)
    n = len(desc_nodes)
    total = int(lengths.sum()) if n else 0

    expected_offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(lengths[:-1], out=expected_offsets[1:])
    if (
        len(offsets) != n
        or len(lengths) != n
        or (n and int(lengths.min()) < 1)
        or len(flat_node) != len(flat_snap)
        or len(flat_node) != total
        or not np.array_equal(offsets, expected_offsets)
        or not np.array_equal(start_row, desc_nodes)
    ):
        raise RuntimeError(
            "tessera progenitor-tree tracer returned an inconsistent layout "
            "(offsets/lengths/flat sizes or echoed start rows do not match)"
        )
    # every tree starts at its requested descendant row
    if n and not np.array_equal(flat_node[offsets], desc_nodes):
        raise RuntimeError(
            "tessera progenitor-tree results do not start at their requested "
            "descendant rows"
        )

    return topo_mod.ProgenitorTreeTopology(
        desc_node=desc_nodes,
        tree_start=offsets,
        tree_length=lengths,
        flat_node=flat_node,
        flat_snap=flat_snap,
    )


def _snapshot_group_order(
    rows: NDArray[np.int64],
    snaps: NDArray[Any],
    group_nr: NDArray[Any],
) -> SnapshotOrder:
    """``{snapshot: sorted unique GroupNr}`` for one descendant's node set.

    Bulk NumPy only: one gather of the pre-read ``GroupNr`` column at the
    node rows, one stable sort by snapshot, then one ``np.unique`` per
    snapshot. The loop is over snapshots, never over nodes.
    """
    order: SnapshotOrder = OrderedDict()
    if rows.size == 0:
        return order
    sort_idx = np.argsort(snaps, kind="stable")
    snaps_sorted = snaps[sort_idx]
    groups_sorted = group_nr[rows[sort_idx]]
    uniq_snaps, starts = np.unique(snaps_sorted, return_index=True)
    bounds = np.append(starts, snaps_sorted.size)
    for k, snap in enumerate(uniq_snaps):
        order[int(snap)] = np.unique(groups_sorted[bounds[k] : bounds[k + 1]])
    return order


def build_saved_order_info(
    backend: "TesseraTreeBackend",
    desc_nodes: ArrayLike,
    a_min: float = 0.05,
    parallel: bool = True,
    n_threads: int = 0,
) -> SavedOrderInfo:
    """Build the canonical saved-order mapping for the selected descendants.

    Two native tessera calls with identical selection/``a_min``/threading:
    the all-progenitor tracer supplies the ``"tree"`` half and the accepted
    main-branch tracer the ``"prog"`` half. ``GroupNr`` is read once in bulk
    (the store caches it) and gathered at the returned rows; there is no
    Python per-node traversal anywhere.

    Returns ``{descendant GroupNr: {"tree": …, "prog": …}}`` — the structure
    ``asymptotic.assembly.trees.save_merger_tree_info`` writes and
    ``load_merger_tree_info`` reads back. Snapshot keys ascend and each
    snapshot's group ids are sorted and unique.

    Because the artifact is keyed by descendant ``GroupNr``, two selected
    descendants sharing one ``GroupNr`` (including the same row requested
    twice) cannot both be represented and raise ``ValueError`` rather than
    silently overwriting each other.
    """
    tree_topo = backend.progenitor_trees(
        desc_nodes, a_min=a_min, parallel=parallel, n_threads=n_threads
    )
    branch_topo = backend.main_branches(
        desc_nodes, a_min=a_min, parallel=parallel, n_threads=n_threads
    )
    group_nr = backend.store.column("GroupNr")  # one bulk (cached) column read
    desc_groups = group_nr[tree_topo.desc_node]

    uniq, counts = np.unique(desc_groups, return_counts=True)
    if np.any(counts > 1):
        clashing = [int(g) for g in uniq[counts > 1]]
        raise ValueError(
            "selected descendants share GroupNr "
            f"{clashing}: a saved-order artifact is keyed by descendant "
            "GroupNr and cannot represent them separately; select one row "
            "per descendant group"
        )

    info: SavedOrderInfo = {}
    for i in range(tree_topo.n_desc):
        lo = int(branch_topo.branch_start[i])
        hi = lo + int(branch_topo.branch_length[i])
        info[int(desc_groups[i])] = {
            "tree": _snapshot_group_order(
                tree_topo.nodes_of(i), tree_topo.snaps_of(i), group_nr
            ),
            "prog": _snapshot_group_order(
                branch_topo.flat_node[lo:hi], branch_topo.flat_snap[lo:hi], group_nr
            ),
        }
    return info


def _topology_from_batch(
    result: TesseraBatchResult,
    desc_nodes: np.ndarray,
    desc_subhalo_nr: np.ndarray,
    desc_tree_id: np.ndarray,
    grid_snaps: np.ndarray,
) -> topo_mod.BranchTopology:
    """Convert tessera's batch-tracer result into a ``BranchTopology``.

    tessera emits each branch descendant-first (starting row first,
    descending scale factor); ``BranchTopology`` stores each branch ascending
    by snapshot. Each branch segment is reversed with a per-branch C-level
    slice copy into preallocated int64 arrays — no per-node Python work and
    no total-size temporaries beyond the two output arrays. Offsets,
    lengths, array sizes, and branch boundaries are validated before
    returning.
    """
    offsets = np.asarray(result["offset"], dtype=np.int64)
    lengths = np.asarray(result["length"], dtype=np.int64)
    src_nodes = np.asarray(result["flat_tree_index"], dtype=np.int64)
    src_snaps = np.asarray(result["flat_snap_num"])
    n = len(desc_nodes)
    total = int(lengths.sum()) if n else 0

    expected_offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        np.cumsum(lengths[:-1], out=expected_offsets[1:])
    if (
        len(offsets) != n
        or len(lengths) != n
        or (n and int(lengths.min()) < 1)
        or len(src_nodes) != len(src_snaps)
        or len(src_nodes) != total
        or not np.array_equal(offsets, expected_offsets)
        or not np.array_equal(
            np.asarray(result["start_row"], dtype=np.int64), desc_nodes
        )
    ):
        raise RuntimeError(
            "tessera batch tracer returned an inconsistent branch layout "
            "(offsets/lengths/flat sizes or echoed start rows do not match)"
        )

    flat_node = np.empty(total, dtype=np.int64)
    flat_snap = np.empty(total, dtype=np.int64)
    for i in range(n):
        lo = offsets[i]
        hi = lo + lengths[i]
        flat_node[lo:hi] = src_nodes[lo:hi][::-1]
        flat_snap[lo:hi] = src_snaps[lo:hi][::-1]

    # after reversal every branch must end at its requested descendant row
    if n and not np.array_equal(flat_node[offsets + lengths - 1], desc_nodes):
        raise RuntimeError(
            "tessera batch tracer branches do not terminate at their "
            "requested descendant rows"
        )

    return topo_mod.BranchTopology(
        desc_node=desc_nodes,
        desc_subhalo_nr=np.asarray(desc_subhalo_nr, dtype=np.int64),
        desc_tree_id=np.asarray(desc_tree_id, dtype=np.int64),
        branch_start=offsets,
        branch_length=lengths,
        flat_node=flat_node,
        flat_snap=flat_snap,
        grid_snaps=np.asarray(grid_snaps, dtype=np.int64),
    )


class TesseraTreeBackend(MergerTreeBackend):
    name = "tessera"

    def __init__(self, store: FullTreeStore):
        try:
            import tessera as ts
        except ModuleNotFoundError as err:
            store.close()
            if err.name == "tessera":
                # only true top-level absence gets the actionable message
                raise ImportError(
                    "The tessera merger-tree backend requires the project's "
                    "supported tessera build (C++ tree traversal), which is "
                    "not installed in this environment. Install it from the "
                    "tessera source tree; the unrelated 'tessera' "
                    "distribution on PyPI is NOT the required package."
                ) from err
            raise  # a module missing INSIDE tessera: propagate unchanged
        except ImportError:
            # an import failure inside tessera (e.g. its native extension):
            # propagate unchanged rather than misreporting it as absence
            store.close()
            raise

        io_mod = getattr(ts, "io", None)
        if (
            io_mod is None
            or not hasattr(io_mod, "MergerTree")
            or not hasattr(io_mod, "extract_branches_from_rows")
        ):
            store.close()
            raise ImportError(
                "The importable 'tessera' module is not a compatible tessera "
                "build: it lacks the io.MergerTree / "
                "io.extract_branches_from_rows API. Rebuild/reinstall "
                "tessera from its current source tree (an older or unrelated "
                "'tessera' package cannot be used)."
            )

        self.store = store
        self._ts = ts
        # tessera's C++ MergerTree is built lazily. Its file discovery
        # resolves the same single-file and sharded <stem>.<N>.hdf5 layouts
        # as FullTreeStore, in the same numeric order, so batch topology rows
        # and store columns index one dataset.
        self._tree_obj = None
        self._tracker_obj = None

    @property
    def _tree(self):
        if self._tree_obj is None:
            self._tree_obj = self._ts.io.MergerTree(str(self.store.path))
        return self._tree_obj

    @property
    def _tracker(self):
        if self._tracker_obj is None:
            self._tracker_obj = self._ts.io.HaloTracker(self._tree)
        return self._tracker_obj

    @classmethod
    def open(cls, path: str | Path, cosmo=None) -> "TesseraTreeBackend":
        return cls(FullTreeStore.open(path, cosmo=cosmo))

    @property
    def moments(self) -> MomentsInTime:
        return self.store.moments

    @property
    def box_size(self) -> float:
        # Prefer the Header BoxSize (no tessera needed -> works on sharded trees);
        # fall back to tessera's MergerTree only if the header lacks it.
        bs = self.store.box_size
        return float(bs) if bs is not None else float(self._tree.box_size())

    # --- canonical name resolution ---------------------------------------- #

    def resolve_column(self, quantity: str, aperture: str) -> str | None:
        if quantity in SUBHALO_FIELDS:
            return quantity if self.store.has_column(quantity) else None
        # FOF mass has no Group_M_FoF column; it is the verbatim GroupMass.
        if quantity == "M" and aperture == "FoF" and self.store.has_column("GroupMass"):
            return "GroupMass"
        for ap in _APERTURE_ALIASES.get(aperture, (aperture,)):
            col = f"Group_{quantity}_{ap}"
            if self.store.has_column(col):
                return col
        return None

    # --- descendant selection + topology ---------------------------------- #

    def select_descendant_nodes(
        self, snap_num: int = DEFAULT_DESC_SNAP, min_len: int = 1000
    ) -> np.ndarray:
        snap = self.store.column("SnapNum")
        sublen = self.store.column("SubhaloLen")
        rows = np.arange(self.store.n_rows)
        mask = (snap == snap_num) & (sublen >= min_len) & self.store.is_central(rows)
        return rows[mask]

    def main_branches(
        self,
        desc_nodes: np.ndarray,
        a_min: float = 0.05,
        parallel: bool = True,
        n_threads: int = 0,
    ) -> topo_mod.BranchTopology:
        """Per-descendant main-progenitor topology (tessera C++ batch tracer).

        The authoritative production traversal is tessera's
        ``extract_branches_from_rows``: one GIL-released native call tracing
        every requested descendant (any absolute row, duplicates allowed,
        input order preserved) in parallel with OpenMP. ``parallel=True``
        passes ``n_threads`` through (0 keeps tessera's automatic thread
        count); ``parallel=False`` runs the same C++ batch tracer with
        exactly one thread. There is no Python traversal in production and no
        silent fallback — without a usable tessera the backend cannot be
        constructed. The vectorized NumPy walk survives only as the private
        test oracle ``_main_branches_numpy_oracle``.
        """
        desc = _normalize_desc_rows(desc_nodes, self.store.n_rows)

        # descendant metadata from the selected store rows, exactly as before
        # (rows validated above: negative indices cannot wrap)
        subhalo_nr = self.store.column("SubhaloNr")[desc]
        tree_id = self.store.column("TreeID")[desc].astype(np.int64)

        result = self._ts.io.extract_branches_from_rows(
            self._tree,
            desc,
            a_min=a_min,
            n_threads=n_threads if parallel else 1,
        )
        return _topology_from_batch(
            result, desc, subhalo_nr, tree_id,
            np.asarray(self.moments.snapshot_ids, dtype=np.int64),
        )

    def progenitor_trees(
        self,
        desc_nodes: ArrayLike,
        a_min: float = 0.05,
        parallel: bool = True,
        n_threads: int = 0,
    ) -> topo_mod.ProgenitorTreeTopology:
        """Per-descendant all-progenitor topology (tessera C++ batch tracer).

        One GIL-released native ``extract_progenitor_trees_from_rows`` call
        returns every progenitor reachable from each selected row — the
        ``TreeFirstProgenitor`` plus ``TreeNextProgenitor`` sibling-chain
        traversal, not just the main line. Any absolute row may be selected,
        duplicates are allowed, input order is preserved, and ``a_min``
        prunes a progenitor together with its subtree (equality inclusive,
        the starting row always kept). ``parallel=True`` passes ``n_threads``
        through (0 keeps tessera's automatic thread count);
        ``parallel=False`` runs the same C++ tracer with exactly one thread.
        There is no Python traversal and no fallback of any kind.

        The capability is checked here rather than in the constructor so a
        main-branch-only workflow still works against an older tessera build.
        """
        extract = getattr(self._ts.io, "extract_progenitor_trees_from_rows", None)
        if extract is None:
            raise ImportError(
                "The installed tessera build lacks the "
                "io.extract_progenitor_trees_from_rows API required for "
                "all-progenitor (full-tree) extraction. Rebuild/reinstall "
                "tessera from its current source tree; main-branch "
                "workflows continue to work with the older build."
            )
        desc = _normalize_desc_rows(desc_nodes, self.store.n_rows)
        result = extract(
            self._tree,
            desc,
            a_min=a_min,
            n_threads=n_threads if parallel else 1,
        )
        return _progenitor_topology_from_batch(result, desc)

    def _main_branches_halotracker_oracle(
        self,
        desc_nodes: np.ndarray,
        a_min: float = 0.05,
    ) -> topo_mod.BranchTopology:
        """Private integration oracle: per-descendant tessera ``trace_main_branch``.

        One Python<->C++ crossing per descendant; never called by production
        code. Kept only so tests can cross-check the batch tracer against the
        long-validated serial path. Assumes every descendant sits at the same
        (maximum) reference snapshot, as the serial path always did.
        """
        desc_nodes = np.ascontiguousarray(np.asarray(desc_nodes), dtype=np.int64)
        subhalo_nr = self.store.column("SubhaloNr")[desc_nodes]
        tree_id = self.store.column("TreeID")[desc_nodes].astype(np.int64)
        snap_at = self.store.column("SnapNum")[desc_nodes].astype(np.int64)
        desc_snap = int(snap_at.max()) if len(snap_at) else DEFAULT_DESC_SNAP

        def trace(sub_nr: int) -> tuple[np.ndarray, np.ndarray]:
            branch = self._tracker.trace_main_branch(
                desc_snap, sub_nr, backward=True, forward=False, a_min=a_min
            )
            nodes = np.fromiter((h.tree_index for h in branch), dtype=np.int64, count=len(branch))
            snaps = np.fromiter((h.snap_num for h in branch), dtype=np.int64, count=len(branch))
            return nodes, snaps

        return topo_mod.build_main_branches(
            trace=trace,
            desc_node=desc_nodes,
            desc_subhalo_nr=subhalo_nr,
            desc_tree_id=tree_id,
            grid_snaps=self.moments.snapshot_ids,
        )

    def _main_branches_numpy_oracle(
        self,
        desc_nodes: np.ndarray,
        a_min: float = 0.05,
    ) -> topo_mod.BranchTopology:
        """Private deterministic test oracle: vectorized ``TreeMainProgenitor`` walk.

        Never called by production code — the authoritative traversal is
        tessera's C++ batch tracer. Retained purely so tests can verify the
        batch path against an independent implementation built on store
        columns only: starting at the descendant rows, repeatedly advance
        every still-active branch by one ``TreeMainProgenitor`` hop (relative
        -> absolute via the global ``TreeTable``), recording each visited node
        and stopping a branch when the link is ``-1`` or the progenitor's
        scale factor falls below ``a_min`` (~``n_snaps`` vectorized
        iterations).
        """
        desc_nodes = np.ascontiguousarray(np.asarray(desc_nodes), dtype=np.int64)
        subhalo_nr = self.store.column("SubhaloNr")[desc_nodes]
        tree_id = self.store.column("TreeID")[desc_nodes].astype(np.int64)
        n = len(desc_nodes)
        main_prog = self.store.column("TreeMainProgenitor")
        tid_col = self.store.column("TreeID")
        snap_col = self.store.column("SnapNum")
        a_grid = np.asarray(self.moments.scale_factors, dtype=float)

        cur = np.asarray(desc_nodes, dtype=np.int64).copy()
        active = np.ones(n, dtype=bool)
        halo_parts, node_parts, snap_parts = [], [], []

        while active.any():
            ai = np.flatnonzero(active)
            c = cur[ai]
            halo_parts.append(ai)
            node_parts.append(c)
            snap_parts.append(snap_col[c])
            # advance to the main progenitor (relative -> absolute via TreeTable)
            nxt = self.store.abs_from_relative(tid_col[c], main_prog[c])
            safe = np.where(nxt >= 0, nxt, 0)
            a_next = np.where(nxt >= 0, a_grid[snap_col[safe]], -np.inf)
            cont = (nxt >= 0) & (a_next >= a_min)
            cur[ai] = np.where(cont, nxt, c)
            active[ai] = cont

        halo = np.concatenate(halo_parts) if halo_parts else np.empty(0, np.int64)
        node = np.concatenate(node_parts) if node_parts else np.empty(0, np.int64)
        snap = np.concatenate(snap_parts) if snap_parts else np.empty(0, np.int64)

        # group by descendant, snap ascending (matches build_main_branches order)
        order = np.lexsort((snap, halo))
        halo, node, snap = halo[order], node[order], snap[order]
        lengths = np.bincount(halo, minlength=n).astype(np.int64)
        starts = np.zeros(n, dtype=np.int64)
        if n:
            starts[1:] = np.cumsum(lengths)[:-1]

        return topo_mod.BranchTopology(
            desc_node=np.asarray(desc_nodes, dtype=np.int64),
            desc_subhalo_nr=np.asarray(subhalo_nr, dtype=np.int64),
            desc_tree_id=np.asarray(tree_id, dtype=np.int64),
            branch_start=starts,
            branch_length=lengths,
            flat_node=node.astype(np.int64),
            flat_snap=snap.astype(np.int64),
            grid_snaps=np.asarray(self.moments.snapshot_ids, dtype=np.int64),
        )

    # --- property gather --------------------------------------------------- #

    def _gather_column(self, col: str, topo: topo_mod.BranchTopology, use_central: bool) -> np.ndarray:
        data = self.store.column(col)
        rows = self.store.central_row_of(topo.flat_node) if use_central else topo.flat_node
        if data.ndim == 1:
            return topo.scatter(data[rows].astype(float))
        # vector field [N,k] -> [n_halos, n_snaps, k]
        k = data.shape[1]
        out = np.full((topo.n_halos, topo.n_snaps, k), np.nan, dtype=float)
        out[topo.flat_halo_idx, topo.flat_grid_col] = data[rows]
        return out

    def gather(self, quantity: str, aperture: str, topo: topo_mod.BranchTopology) -> np.ndarray:
        col = self.resolve_column(quantity, aperture)
        if col is None:
            raise KeyError(f"{quantity}@{aperture} not available in {self.store.path.name}")
        return self._gather_column(col, topo, use_central=_is_group_field(quantity))

    def gather_positions(self, topo: topo_mod.BranchTopology) -> np.ndarray:
        # Prefer the threaded FOF GroupPos (true group centre, finite on every
        # row); fall back to per-subhalo SubhaloPos on trees that lack it.
        col = "GroupPos" if self.store.has_column("GroupPos") else "SubhaloPos"
        return self._gather_column(col, topo, use_central=False)

    def close(self) -> None:
        self.store.close()
