"""Runtime state derived from a composed run specification.

`RunStore` and `RunContext` are the concrete records that satisfy the public
`SimulationStore` and `AnalysisContext` protocols structurally — no
inheritance, no registration.

`RunStore`'s path defaults are pure derivations from the configured root:
nothing is discovered, checked for existence, or created. `RunContext`
construction adds no path discovery, no private lookup, and no directory
creation either — but it does build a `Cosmology`, and that helper reads the
parameter file the configuration points at. Configuration chooses the
location; this module never guesses one.

The scientific helpers are imported inside `build_analysis_context` so that
importing the configuration boundary stays free of the numerical stack until
after the thread environment is set.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import define, field

from .registry import validate_run_spec
from .specs import AnalysisRunSpec, StorageSpec, WorkflowSpecT

if TYPE_CHECKING:
    from ..cosmo.model import Cosmology


@define(frozen=True, slots=True, kw_only=True)
class RunStore:
    """Where a simulation's products live — the four members public analysis
    functions reach through ``ctx.io``, all resolved and non-optional."""

    sim_data_dir: Path = field()
    full_catalog_dir: Path = field()
    power_spectra_dir: Path = field()
    use_tree: bool = field(default=False)


def build_run_store(storage: StorageSpec) -> RunStore:
    """Resolve a storage spec into the concrete store.

    Unset locations take their one documented pure derivation from
    ``sim_data_dir``; explicit values win. Paths are never resolved, so a
    relative root stays relative.
    """
    root = storage.sim_data_dir
    if root is None:
        raise ValueError(
            "storage.sim_data_dir is required to build a run store: it is the "
            "root the unset locations derive from")

    return RunStore(
        sim_data_dir=root,
        full_catalog_dir=(
            root / "full_catalogs" if storage.full_catalog_dir is None
            else storage.full_catalog_dir),
        power_spectra_dir=(
            root / "powerspecs" if storage.power_spectra_dir is None
            else storage.power_spectra_dir),
        use_tree=storage.use_tree,
    )


@define(frozen=True, slots=True, kw_only=True)
class RunContext:
    """The analysis state public functions consume.

    The two order paths are optional inputs — a workflow that never reads them
    must not be made to invent one — but the public accessors are plain
    ``Path``, so an absent path fails only where it is actually used.
    """

    sim_cosmo: Cosmology = field()
    link_fraction: float = field()
    min_resolved_halo_mass: float = field()
    io: RunStore = field()
    _tree_order_path: Path | None = field(default=None)
    _prog_order_path: Path | None = field(default=None)

    @property
    def tree_order_path(self) -> Path:
        return _require_path(self._tree_order_path, "tree_order_path")

    @property
    def prog_order_path(self) -> Path:
        return _require_path(self._prog_order_path, "prog_order_path")


def _require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(
            f"this run has no {name}: set storage.{name} for a workflow that "
            "reads saved merger-tree order artifacts")
    return value


def build_analysis_context(spec: AnalysisRunSpec[WorkflowSpecT]) -> RunContext:
    """Derive the runtime context from a validated run specification."""
    validate_run_spec(spec)

    storage = _require_group(spec.storage, "storage")
    cosmology = _require_group(spec.cosmology, "cosmology")
    resolution = _require_group(spec.resolution, "resolution")

    # deferred so the configuration boundary imports nothing numerical
    from ..bounded.fof import get_loss_cfg_linking_length
    from ..particles.masses import get_min_resolved_mass
    from ..utils.analysis import get_cosmology_model

    store = build_run_store(storage)
    return RunContext(
        sim_cosmo=get_cosmology_model(
            sim_data_dir=store.sim_data_dir,
            cosmo_name=cosmology.name,
            param_path=cosmology.param_path,
        ),
        link_fraction=(
            get_loss_cfg_linking_length(seed_num=spec.simulation.seed_num)
            if resolution.link_fraction is None else resolution.link_fraction),
        min_resolved_halo_mass=get_min_resolved_mass(
            cosmo_name=cosmology.name,
            box_size=spec.simulation.box_size,
            min_num_resolved_particles=resolution.min_resolved_particles,
            num_particles_in_sim=spec.simulation.num_particles,
        ),
        io=store,
        tree_order_path=storage.tree_order_path,
        prog_order_path=storage.prog_order_path,
    )


def _require_group[GroupT](group: GroupT | None, name: str) -> GroupT:
    if group is None:
        raise ValueError(
            f"building a runtime context requires the {name!r} group; add it "
            "to the composed configuration")
    return group
