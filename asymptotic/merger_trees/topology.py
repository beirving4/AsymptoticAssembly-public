"""Merger-tree topology records.

``BranchTopology`` holds per-descendant *main-progenitor* node lists: a
backend supplies a ``trace`` callable returning the absolute node ids +
snapshot numbers along one halo's main branch, and this module flattens them
into a compact ragged structure (start/length offsets) plus the machinery to
scatter gathered columns onto the global snapshot grid.

``ProgenitorTreeTopology`` holds the *all-progenitor* counterpart — every
progenitor reachable from each selected descendant — in tessera's native
flattened layout. The two are deliberately separate records: a full tree has
many nodes per (descendant, snapshot), which is exactly the invariant
``BranchTopology.scatter`` relies on, so the full-tree record has no
``scatter``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from attrs import define
from numpy.typing import NDArray
from tqdm.auto import tqdm

# trace(subhalo_nr) -> (nodes, snaps), each 1-D, in ANY order (we sort).
TraceFn = Callable[[int], tuple[np.ndarray, np.ndarray]]


@define(slots=True)
class BranchTopology:
    """Flattened main-progenitor branches for a set of descendants.

    ``flat_node`` holds absolute node ids for every (halo, snapshot) along each
    branch, concatenated; ``branch_start``/``branch_length`` index into it. Each
    branch is sorted by snapshot ascending (earliest progenitor first).
    """

    desc_node: np.ndarray          # (n_halos,) descendant node id (e.g. snap-74 row)
    desc_subhalo_nr: np.ndarray    # (n_halos,)
    desc_tree_id: np.ndarray       # (n_halos,)
    branch_start: np.ndarray       # (n_halos,) offset into flat_*
    branch_length: np.ndarray      # (n_halos,)
    flat_node: np.ndarray          # (sum length,) absolute node ids
    flat_snap: np.ndarray          # (sum length,) snapshot ids
    grid_snaps: np.ndarray         # (n_snaps,) global snapshot axis (sorted)

    @property
    def n_halos(self) -> int:
        return len(self.desc_node)

    @property
    def n_snaps(self) -> int:
        return len(self.grid_snaps)

    @property
    def flat_halo_idx(self) -> np.ndarray:
        """(sum length,) halo index for each flat entry (for vectorized scatter)."""
        return np.repeat(np.arange(self.n_halos), self.branch_length)

    @property
    def flat_grid_col(self) -> np.ndarray:
        """(sum length,) column on the global grid for each flat entry."""
        return np.searchsorted(self.grid_snaps, self.flat_snap)

    def scatter(self, flat_values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Place per-node ``flat_values`` into a ``[n_halos, n_snaps]`` grid."""
        out = np.full((self.n_halos, self.n_snaps), fill, dtype=float)
        out[self.flat_halo_idx, self.flat_grid_col] = flat_values
        return out


@define(slots=True)
class ProgenitorTreeTopology:
    """Flattened all-progenitor trees for a set of selected descendants.

    Holds tessera's native batch layout verbatim: tree ``i`` occupies
    ``flat_node[tree_start[i] : tree_start[i] + tree_length[i]]`` (and the
    same slice of ``flat_snap``), in tessera's deterministic depth-first
    preorder — the selected descendant first, then each node's
    ``TreeFirstProgenitor`` subtree before its ``TreeNextProgenitor``
    siblings. Descendant order (including duplicates and caller-chosen
    ordering) is preserved, and ``flat_snap`` keeps the native ``int32``
    dtype.

    Unlike :class:`BranchTopology` this record has no ``scatter``: a full
    tree holds many nodes per (descendant, snapshot), so a
    ``[n_halos, n_snaps]`` grid cannot represent it.
    """

    desc_node: NDArray[np.int64]    # (n_desc,) requested descendant rows, echoed
    tree_start: NDArray[np.int64]   # (n_desc,) offset into flat_* per descendant
    tree_length: NDArray[np.int64]  # (n_desc,) nodes per descendant
    flat_node: NDArray[np.int64]    # (sum length,) absolute rows, DFS preorder
    flat_snap: NDArray[np.int32]    # (sum length,) snapshot numbers (native dtype)

    @property
    def n_desc(self) -> int:
        return len(self.desc_node)

    @property
    def n_nodes(self) -> int:
        return len(self.flat_node)

    def nodes_of(self, i: int) -> NDArray[np.int64]:
        """Absolute rows of descendant ``i``'s tree (a view, never a copy)."""
        lo = int(self.tree_start[i])
        return self.flat_node[lo : lo + int(self.tree_length[i])]

    def snaps_of(self, i: int) -> NDArray[np.int32]:
        """Snapshot numbers of descendant ``i``'s tree (a view, never a copy)."""
        lo = int(self.tree_start[i])
        return self.flat_snap[lo : lo + int(self.tree_length[i])]


def build_main_branches(
    trace: TraceFn,
    desc_node: np.ndarray,
    desc_subhalo_nr: np.ndarray,
    desc_tree_id: np.ndarray,
    grid_snaps: np.ndarray,
    progress: bool = True,
) -> BranchTopology:
    """Trace each descendant's main branch and flatten.

    ``trace(subhalo_nr)`` returns ``(nodes, snaps)`` for the main-progenitor
    branch; ordering is not trusted (sorted by snapshot here).
    """
    nodes_list: list[np.ndarray] = []
    snaps_list: list[np.ndarray] = []
    lengths = np.empty(len(desc_node), dtype=np.int64)

    iterator = enumerate(desc_subhalo_nr)
    if progress:
        iterator = tqdm(iterator, total=len(desc_node), desc="Tracing main branches")

    for i, subhalo_nr in iterator:
        nodes, snaps = trace(int(subhalo_nr))
        order = np.argsort(snaps, kind="stable")
        nodes_list.append(np.asarray(nodes)[order])
        snaps_list.append(np.asarray(snaps)[order])
        lengths[i] = len(nodes)

    flat_node = (
        np.concatenate(nodes_list).astype(np.int64) if len(nodes_list) else np.empty(0, np.int64)
    )
    flat_snap = (
        np.concatenate(snaps_list).astype(np.int64) if len(snaps_list) else np.empty(0, np.int64)
    )
    starts = np.zeros(len(desc_node), dtype=np.int64)
    if len(lengths):
        starts[1:] = np.cumsum(lengths)[:-1]

    return BranchTopology(
        desc_node=np.asarray(desc_node, dtype=np.int64),
        desc_subhalo_nr=np.asarray(desc_subhalo_nr, dtype=np.int64),
        desc_tree_id=np.asarray(desc_tree_id, dtype=np.int64),
        branch_start=starts,
        branch_length=lengths,
        flat_node=flat_node,
        flat_snap=flat_snap,
        grid_snaps=np.asarray(grid_snaps, dtype=np.int64),
    )
