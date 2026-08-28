"""Merger-tree backend interface.

The assembly pipeline works in *canonical* (quantity, aperture) names and never
hard-codes a file's column names. Concrete backends translate canonical names to
their own columns and supply branch topology + a property gather. Tessera is the
sole production backend (reads both ``full_tree.hdf5`` and GADGET-native
``trees.hdf5``); the public ytree backend was retired before release, and
``arbor*`` inputs are rejected explicitly.

See ``docs/assembly_pipeline_requirements.md`` for the full spec.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..simulation.moments import MomentsInTime

# --- Canonical vocabulary -------------------------------------------------- #

# The 7 primary apertures shared by the shape/energetics features.
CANONICAL_APERTURES: tuple[str, ...] = (
    "Bound", "Crit200", "Crit500", "FoF", "Mean200", "Splashback", "Virial",
)

# The 12 shape/energetics/spin/anisotropy quantities (full_tree only).
SHAPE_ENERGETICS_QUANTITIES: tuple[str, ...] = (
    "Beta", "BetaCyl", "Energy", "KinToPot",
    "ShapeQ", "ShapeS", "Triaxiality",
    "ReducedShapeQ", "ReducedShapeS", "ReducedTriaxiality",
    "SpinBullock", "SpinPeebles",
)

# Per-aperture structural quantities (mass/radius/velocity + derived).
BASE_QUANTITIES: tuple[str, ...] = (
    "M", "R", "V", "Concentration", "CrossingTime", "RLagrangian",
)

# Aperture-independent subhalo fields, read at the branch row directly.
SUBHALO_FIELDS: tuple[str, ...] = (
    "SubhaloPos", "SubhaloVel", "SubhaloSpin",
    "SubhaloMass", "SubhaloVmax", "SubhaloVmaxRad", "SubhaloLen",
)

DEFAULT_QUANTITIES: tuple[str, ...] = (
    "M", "R", "V", "Concentration", *SHAPE_ENERGETICS_QUANTITIES,
)


def canonical_name(quantity: str, aperture: str) -> str:
    """Logical name used as the dataset key in output files, e.g. ``ShapeS``.

    Per-aperture files key by quantity alone (the aperture is the file), so this
    is just the quantity; kept as a function so callers don't assume the rule.
    """
    return quantity


class MergerTreeBackend(ABC):
    """Common surface over a merger-tree source.

    Topology is returned as absolute node ids (tree-storage rows for tessera)
    and a gather pulls arbitrary canonical columns at those nodes. ``use_central``
    routes ``Group_*`` reads through the FOF-central row (tessera) where the
    composed values live; subhalo fields are read at the node directly.
    """

    #: human-readable backend tag stored in output provenance
    name: str = "base"

    @property
    @abstractmethod
    def moments(self) -> MomentsInTime:
        """The global snapshot time axis."""

    @property
    @abstractmethod
    def box_size(self) -> float:
        ...

    @abstractmethod
    def resolve_column(self, quantity: str, aperture: str) -> str | None:
        """Backend column name for a canonical (quantity, aperture), or None."""

    def available(self, quantity: str, aperture: str) -> bool:
        return self.resolve_column(quantity, aperture) is not None

    @abstractmethod
    def select_descendant_nodes(self, snap_num: int, min_len: int) -> np.ndarray:
        """Central halos at ``snap_num`` with ``SubhaloLen >= min_len`` (node ids)."""

    @abstractmethod
    def main_branches(self, desc_nodes, a_min: float, parallel: bool = True, n_threads: int = 0):
        """Build per-descendant main-progenitor topology (see topology.BranchTopology).

        ``parallel``/``n_threads`` request a parallel batch tracer where the backend
        supports one (tessera); backends without one accept and ignore them.
        """

    @abstractmethod
    def gather(self, quantity: str, aperture: str, topo) -> np.ndarray:
        """NaN-padded ``[n_halos, n_snaps]`` scalar array on the global grid.

        The backend routes ``Group_*`` quantities through the FOF-central row
        automatically; subhalo fields are read at the branch node.
        """

    @abstractmethod
    def gather_positions(self, topo) -> np.ndarray:
        """``[n_halos, n_snaps, 3]`` comoving branch positions (NaN-padded)."""

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> "MergerTreeBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_merger_tree(
    path: str | Path, backend: str = "auto", cosmo=None
) -> MergerTreeBackend:
    """Factory: pick a backend by file type (or force it).

    Valid choices are ``auto`` and ``tessera``. ``auto`` routes
    ``*full_tree.hdf5`` / ``*trees.hdf5`` (single-file or ``.<N>.hdf5``
    sharded, or a directory holding a shard series) to tessera; ytree
    ``arbor*`` inputs are rejected explicitly — the public ytree backend was
    retired before release. ``cosmo`` is optional and only used to synthesize
    the age / dynamical-time axis when the tree lacks a baked ``Moment_*``
    axis (GADGET-native ``trees.hdf5``); ignored for ``full_tree.hdf5``,
    which already carries it.
    """
    path = Path(path)
    choice = backend.lower()

    if choice == "ytree":
        raise ValueError(
            "The public ytree backend was retired before release; the Tessera "
            "C++ backend is the authoritative merger-tree backend (choices: "
            "auto|tessera). The historical implementation survives only as an "
            "excluded migration example under postprocessing/legacy_ytree/."
        )

    if choice == "auto":
        choice = _detect_backend(path)

    if choice == "tessera":
        from .full_tree import TesseraTreeBackend
        return TesseraTreeBackend.open(path, cosmo=cosmo)

    raise ValueError(f"Unknown backend {backend!r} (auto|tessera)")


def _detect_backend(path: Path) -> str:
    name = path.name.lower()
    # Arbor rejection is explicit and precedes the generic .h5 fallback so a
    # ytree arbor is never misclassified as a Tessera/GADGET tree.
    if "arbor" in name or (path.is_dir() and any(path.glob("arbor*"))):
        raise ValueError(
            f"{path} looks like a ytree arbor; the public ytree backend was "
            "retired before release and arbors are not supported. Production "
            "traversal uses the Tessera C++ backend on GADGET-4 "
            "full_tree.hdf5 / trees.hdf5 outputs."
        )
    if name.endswith(".hdf5") or name.endswith(".h5"):
        # full_tree.hdf5 / trees.hdf5 / trees.<N>.hdf5 are all GADGET-4 -> tessera
        return "tessera"
    # a directory holding a GADGET-4 split series (e.g. treedata/trees.<N>.hdf5)
    if path.is_dir() and any(path.glob("trees.*.hdf5")):
        return "tessera"
    raise ValueError(
        f"Cannot auto-detect backend for {path!r}; pass backend='tessera' explicitly"
    )
