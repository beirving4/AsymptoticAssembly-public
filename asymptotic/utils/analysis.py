from __future__ import annotations

# Above the warning filter so they stay genuine top-of-file imports (B1d2).
from collections.abc import Iterator
from typing import Protocol

import warnings
warnings.filterwarnings("ignore")

import pdb
import time
import numpy as np
import multiprocessing as mp


from pathlib import Path
from attrs import define
from tqdm.auto import tqdm
from datetime import timedelta
from argparse import Namespace
from collections.abc import Collection

from ..cosmo.config import CosmoType
from ..cosmo.model import Cosmology
from ..simulation.moments import Moment, MomentsInTime
from ..bounded.fof import (
    FOFGroup, 
    FOFGroups, 
    FOFEvoData, 
    FOFGroupEvoData,
    get_loss_cfg_linking_length
)
from ..particles.masses import get_min_resolved_mass
from ..particles.collection import BoxParticleCollection
from ..assembly.trees import (
    load_prog_order_info,
    load_tree_order_info
)



MIN_NUM_RESOLVED_PARTICLES = 1000

# Module-level holder for the FOFEvoData instance
_FOF = None  # type: FOFEvoData

def _init_worker(fof_group_evo: FOFEvoData) -> None:
    """
    Run once per worker process. Stores fof_group_evo in a global,
    so that subsequent tasks only send small evo_path dicts.
    """
    global _FOF
    _FOF = fof_group_evo

def _extract(evo_path: dict[int, int]) -> FOFGroupEvoData:
    """
    Runs in the worker; pulls the global _FOF and extracts one branch.
    """
    return _FOF.get_main_progenitor_branch(evo_path)

class SimulationStore(Protocol):
    """Where a simulation's products live, as the public analysis functions
    read them.

    Structural and read-only: any object exposing these four members satisfies
    it, which is why the functions below never need a concrete configuration
    class. Deliberately *not* ``runtime_checkable`` — no ``isinstance`` check
    exists or is wanted — and deliberately without the order-path builders,
    which are an implementation detail of whoever derives the two order paths.
    """

    @property
    def sim_data_dir(self) -> Path: ...

    @property
    def full_catalog_dir(self) -> Path: ...

    @property
    def power_spectra_dir(self) -> Path: ...

    @property
    def use_tree(self) -> bool: ...


class AnalysisContext(Protocol):
    """The analysis state the public functions in this module consume.

    Read-only by construction: the members are properties, so an
    implementation may narrow ``io`` to a concrete store type. Nothing here is
    validated or checked at runtime.
    """

    @property
    def sim_cosmo(self) -> Cosmology: ...

    @property
    def link_fraction(self) -> float: ...

    @property
    def min_resolved_halo_mass(self) -> float: ...

    @property
    def tree_order_path(self) -> Path: ...

    @property
    def prog_order_path(self) -> Path: ...

    @property
    def io(self) -> SimulationStore: ...


@define(slots=True)
class AnalysisConfig:
    box_size: int
    num_particles: int
    seed_num: int
    min_resolved_halo_mass: float
    link_fraction: float
    cosmo_name: str
    sim_cosmo: Cosmology
    io: IOConfig
    
    @classmethod
    def from_analysis_args(
            cls, 
            analysis_args: Namespace, 
            min_num_resolved_particles: int = MIN_NUM_RESOLVED_PARTICLES
        ) -> AnalysisConfig:

        # lazy: directory_finder encodes private cluster layouts and is
        # excluded from the public wheel
        from .directory_finder import IOConfig, get_param_path

        io = IOConfig.from_analysis_args(analysis_args)
        io.param_path = get_param_path(
            sim_data_dir=io.sim_data_dir, 
            cosmo_name=io.cosmo_name, 
            box_size=analysis_args.box_size
        )

        print(f"{io.cosmo_name = }")
        print(f"{io.param_path = }")
        print(f"{io.param_path.exists() = }\n")

        sim_cosmo = get_cosmology_model(
            sim_data_dir=io.sim_data_dir, 
            cosmo_name=io.cosmo_name, 
            param_path=io.param_path
        )

        min_resolved_halo_mass = get_min_resolved_mass(
            cosmo_name=io.cosmo_name, 
            box_size=analysis_args.box_size, 
            min_num_resolved_particles=min_num_resolved_particles,
            num_particles_in_sim=analysis_args.num_particles
        )
        
        return cls(
            box_size=analysis_args.box_size,
            num_particles=analysis_args.num_particles,
            seed_num=(
                analysis_args.seed_num if io.cosmo_name != "primary" else -1
            ),
            min_resolved_halo_mass=min_resolved_halo_mass,
            link_fraction=get_loss_cfg_linking_length(seed_num=analysis_args.seed_num),
            cosmo_name=io.cosmo_name,
            sim_cosmo=sim_cosmo,
            io=io
        )

    @property
    def tree_order_path(self) -> Path:
        return self.io.get_tree_order_path(
            box_size=self.box_size, 
            num_particles=self.num_particles,
            seed_num=self.seed_num
        )

    @property
    def prog_order_path(self) -> Path:
        return self.io.get_prog_order_path(
            box_size=self.box_size, 
            num_particles=self.num_particles,
            seed_num=self.seed_num
        )
    
def get_mdefs_to_eval(mdefs: str | Collection[str], allow_fof: bool = False) -> list[str]:
    if isinstance(mdefs, str) and mdefs == "fof" and not allow_fof:
        raise ValueError(
            "Cannot evaluate just FOF mass definition as a mass definition."
        )
    elif isinstance(mdefs, str):
        return [mdefs] 
    else:
        return [mdef for mdef in mdefs if ((mdef != "fof") and not allow_fof)]

def display_time(elapsed_time: float) -> str:
    if elapsed_time < 1.0:
        return f"{elapsed_time:.6f} seconds"
    td = timedelta(seconds=elapsed_time)
    return str(td)


def get_cosmology_model(
        sim_data_dir: Path, cosmo_name: str, param_path: Path | None
    ) -> Cosmology:    

    if cosmo_name in {
        CosmoType.PRIMARY.value, 
        CosmoType.TOY_MODEL_A.value, 
        CosmoType.PLANCK.value
    }:
        if param_path is None:
            raise ValueError(
                f"cosmology {cosmo_name!r} reads a parameter file; "
                "param_path is required"
            )

        return Cosmology.from_directory(
            sim_dir=sim_data_dir,
            name=cosmo_name,
            param_path=param_path,
            from_music=False
        ) 
    
    else:

        return Cosmology.from_directory(sim_data_dir)

def get_file_iterators(
        is_batched: bool,
        sim_data_dir: Path,
    ) -> tuple[int, Iterator[Path], str | None]:

    if is_batched:
        num_files = sum(1 for _ in sim_data_dir.glob("groups_*"))
        file_iterator = sim_data_dir.glob("groups_*")
        fof_data_dir_name = "groups"
    else:
        num_files = sum(1 for _ in sim_data_dir.glob("fof_subhalo_tab_*.hdf5"))
        file_iterator = sim_data_dir.glob("fof_subhalo_tab_*.hdf5")
        fof_data_dir_name = None

    return num_files, file_iterator, fof_data_dir_name

def get_fof_groups(
        cfg: AnalysisContext,
        args: Namespace,
        snapshot_id: int,
        min_count_above_max_bin_hist: int = 0,
        use_full_catalog: bool | None = None,
    ) -> FOFGroups:
    """Load FOFGroups for one snapshot.

    ``use_full_catalog`` defaults to whatever ``args.use_full_catalog`` is
    (``False`` if absent), so callers that just forward their own
    argparse namespace get the flag for free without changes. Passing
    ``True`` (or setting ``--use_full_catalog`` on the CLI) routes through
    the Phase 1 full_catalog reader and auto-populates the solo-derived
    mass_defs slots, removing the need for a separate
    ``add_solo_boundaries`` call downstream.
    """
    if use_full_catalog is None:
        use_full_catalog = bool(getattr(args, "use_full_catalog", False))
    return FOFGroups.from_catalog(
        catalog_dir=cfg.io.sim_data_dir,
        snapshot_id=snapshot_id,
        cosmo=cfg.sim_cosmo,
        b=cfg.link_fraction,
        min_resolved_mass=cfg.min_resolved_halo_mass,
        min_count_above_max_bin_hist=min_count_above_max_bin_hist,
        region_scaling=args.query_scaling,
        scaling_def=args.scaling_def,
        is_batched=args.is_batched,
        in_comoving=True,
        use_full_catalog=use_full_catalog,
        full_catalog_dir=cfg.io.full_catalog_dir,
    )

def get_fof_group_evo_data(
        cfg: AnalysisContext,
        args: Namespace,
        min_count_above_max_bin_hist: int = 0,
        snapshots_to_skip: Collection[int] | None = None,
        use_full_catalog: bool | None = None,
    ) -> FOFEvoData:
    """Multi-snap FOFEvoData loader.

    ``use_full_catalog`` defaults to ``bool(args.use_full_catalog)`` (or
    ``False`` when the namespace lacks the attribute), so any script
    that exposes ``--use_full_catalog`` on its argparse gets the flag
    routed for free. Setting it to ``True`` makes each per-snap
    FOFGroups come back with solo-derived mass_defs slots already
    populated (sp/bound/ta/inf/hs/scale/<4s/max/marg_bound), which is
    what scripts like save_branch_catalog need to filter halos by
    e.g. ``--mdef bound`` or ``--mdef sp``.
    """
    if use_full_catalog is None:
        use_full_catalog = bool(getattr(args, "use_full_catalog", False))
    return FOFEvoData.from_directory(
        sim_dir=cfg.io.sim_data_dir,
        cosmo=cfg.sim_cosmo,
        b=cfg.link_fraction,
        region_scaling=args.query_scaling,
        in_comoving=True,
        min_resolved_mass=cfg.min_resolved_halo_mass,
        min_count_above_max_bin_hist = min_count_above_max_bin_hist,
        snapshots_to_skip = snapshots_to_skip,
        is_batched=args.is_batched,
        use_full_catalog=use_full_catalog,
        full_catalog_dir=cfg.io.full_catalog_dir,
    )

def stack_with_variable_length_arrays(
        arrays: Collection[np.ndarray], 
        num_features: int
    ) -> np.ndarray:

    k = len(arrays)
    N_max = max(array.shape[1] for array in arrays)

    data = np.zeros((k, num_features, N_max))
    mask = np.ones((k, num_features, N_max), dtype=bool)

    for i, array in enumerate(arrays):
        data[i, :, :array.shape[1]] = array
        mask[i, :, :array.shape[1]] = False

    return np.ma.masked_array(data, mask=mask)

def get_box_particles(
        data_dir: Path, 
        moment: Moment, 
        use_tree: bool = False, 
        is_batched: bool = False,
        compute_kd_tree: bool = True,
        initialize_sub_boxes: bool = True,
        num_splits: int = 1,
        use_initial_conditions: bool = False
    ) -> BoxParticleCollection:

    if use_tree:

        kd_tree_dir = Path(data_dir, "kd_trees/particles")
        kd_tree_path = Path(
            kd_tree_dir, f"snapshot_{moment.snapshot_id:03d}_tree.npz"
        )

        if not kd_tree_path.exists(): 
            raise ValueError("Need to correct path for the kd tree!")

        print(f"Loading Box Particles from {kd_tree_path}")
        
        box_particles = BoxParticleCollection.from_kd_tree_file(kd_tree_path, moment)
    
    else: 

        print(f"Loading Box Particles from {data_dir}")

        box_particles = BoxParticleCollection.from_snapshot_file(
            sim_dir=data_dir,
            moment=moment,
            is_batched=is_batched,
            compute_kd_tree=compute_kd_tree,
            for_initial_condition=use_initial_conditions
        )

    if initialize_sub_boxes:
        box_particles.initialize_empty_sub_boxes(num_splits=num_splits)

    return box_particles
    
def get_moments(cfg: AnalysisContext, args: Namespace) -> MomentsInTime:
    moments = MomentsInTime.from_power_spec_data(
        power_spec_dir=cfg.io.power_spectra_dir,
        sim_cosmo=cfg.sim_cosmo 
    )  
    moments.add_times(cfg.sim_cosmo)
    return moments

def get_snapshot_ids_from_spectra(cfg: AnalysisContext, args: Namespace) -> np.ndarray:
    return get_moments(cfg, args).snapshot_ids

def get_snapshot_ids_from_halos(cfg: AnalysisContext, args: Namespace) -> np.ndarray:
    fof_group_evo_data = get_fof_group_evo_data(cfg, args)
    return fof_group_evo_data.moments.snapshot_ids

def get_snapshots_to_eval(
        snapshot_ids: Collection[int] | np.ndarray, 
        prog_snaps_to_eval: int | Collection[int] | np.ndarray,
        desc_snap_id: int
    ) -> Collection[int]:

    if prog_snaps_to_eval == -1:
        snaps_to_eval = [
            desc_snap_id, 
            *[
                snap_id for snap_id in snapshot_ids
                if snap_id < desc_snap_id
            ]
        ]
    elif isinstance(prog_snaps_to_eval, int):
        snaps_to_eval = [desc_snap_id, prog_snaps_to_eval]
    else:
        snaps_to_eval = [desc_snap_id, *prog_snaps_to_eval]

    return sorted(snaps_to_eval, reverse=True)

def get_snapshot_timeline_ids(
        snapshot_ids: Collection[int],
        max_tree_snap_id: int,
        prog_snaps_to_eval: int | Collection[int] | np.ndarray,
    ) -> dict[str, int | Collection[int]]:

    desc_snap_id = min(max(snapshot_ids), max_tree_snap_id)
    
    return {
        "desc_snap_id": desc_snap_id,
        "snaps_to_eval" : get_snapshots_to_eval(
            snapshot_ids=snapshot_ids, 
            prog_snaps_to_eval=prog_snaps_to_eval, 
            desc_snap_id=desc_snap_id
        )
    }

def _cohort_sample_seed(args: Namespace) -> int:
    """Deterministic, non-negative seed for the descendant-cohort subsample.

    Derived from seed_num and the target mass bin so reruns -- and the
    symmetric vs fixed-target composition modes -- draw the SAME halos from an
    over-full mass bin. Coerced non-negative because np.random.default_rng
    rejects negative seeds and primary-cosmo runs carry seed_num = -1.
    """
    base = getattr(args, "seed_num", 0)
    base = 0 if (base is None or base < 0) else int(base)
    mass_bin = getattr(args, "mass_bin", 0.0) or 0.0
    return (base * 1_000_003 + int(round(float(mass_bin) * 100))) % (2 ** 32)


def get_desc_sample_from_args(
        desc_halos: FOFGroups,
        args: Namespace,
        min_resolved_halo_mass: float,
        sampling_mdef: str | None = None
    ) -> FOFGroups:
    return get_desc_sample(
        all_desc_halos=desc_halos,
        mdef=(
            sampling_mdef if (sampling_mdef is not None) else args.mdef
        ),
        target_sample_mass=args.mass_bin,
        lower_bound_mass=args.lower_sample_limit,
        upper_bound_mass=args.upper_sample_limit,
        min_resolved_mass=min_resolved_halo_mass,
        max_sample_size=args.max_sample_size,
        seed=_cohort_sample_seed(args)
    )

def get_desc_sample(
        all_desc_halos: FOFGroups,
        mdef: str,
        target_sample_mass: float,
        lower_bound_mass: float,
        upper_bound_mass: float,
        min_resolved_mass: float,
        max_sample_size: int,
        seed: int | None = None
    ) -> FOFGroups:

    resolved_halos = all_desc_halos.get_resolved_groups(mdef, min_resolved_mass)

    if len(resolved_halos) == 0:
        raise ValueError("No resolved halos found")

    target_sample = resolved_halos.get_subset_in_mass_range(
        mass_def_key=mdef,
        min_mass=10.0**(target_sample_mass - lower_bound_mass),
        max_mass=10.0**(target_sample_mass + upper_bound_mass)
    )

    if len(target_sample) > max_sample_size:
        # Seeded so an over-full mass bin yields the SAME cohort on every run
        # (and across composition modes). Without this the unseeded draw made
        # symmetric vs fixed-target runs land on different halos, and a bad
        # draw could whiff the early-snap progenitor walk wholesale.
        return target_sample.sample(max_sample_size, seed=seed)
    return target_sample

# def get_mpbs_from_order_as_dict(
#         order: dict[int, dict[int, int]], 
#         fof_group_evo: FOFEvoData,
#     ) -> dict[int, FOFGroupEvoData]:

#     return {
#         group_id: fof_group_evo.get_main_progenitor_branch(evo_path)
#         for group_id, evo_path in tqdm(
#             order.items(), desc="Extracting MPBs from Order as Dict"
#         )
#     }

# def get_mpbs_from_order_as_list(
#         order: dict[int, dict[int, int]],
#         fof_group_evo: FOFEvoData,
#     ) -> list[FOFGroupEvoData]:

#     return [
#         fof_group_evo.get_main_progenitor_branch(evo_path)
#         for evo_path in tqdm(
#             order.values(), desc="Extracting MPBs from Order as List"
#         )
#     ]


def get_mpbs_from_order_as_dict(
        order: dict[int, dict[int, int]], 
        fof_group_evo: FOFEvoData,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> dict[int, FOFGroupEvoData]:

    items = list(order.items())
    keys, paths = zip(*items)
    num_workers = mp.cpu_count() if n_jobs == -1 else n_jobs
    chunksize = max(1, len(paths) // (num_workers * 4))

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(fof_group_evo,)
    ) as pool:
        mpb_iter = pool.imap(_extract, paths, chunksize=chunksize)
        results = list(tqdm(mpb_iter, total=len(paths), desc="Extracting MPBs (dict)"))

    return dict(zip(keys, results))


def get_mpbs_from_order_as_list(
        order: dict[int, dict[int, int]],
        fof_group_evo: FOFEvoData,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> list[FOFGroupEvoData]:

    paths = list(order.values())
    num_workers = mp.cpu_count() if n_jobs == -1 else n_jobs
    # choose a sensible chunksize to amortize IPC
    chunksize = max(1, len(paths) // (num_workers * 4))

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(fof_group_evo,)
    ) as pool:
        # imap yields results as they come in
        mpb_iter = pool.imap(_extract, paths, chunksize=chunksize)
        return list(tqdm(mpb_iter, total=len(paths), desc="Extracting MPBs (list)"))


def get_main_progenitor_branches_from_order(
        order: dict[int, dict[int, int]], 
        fof_group_evo: FOFEvoData,
        group_ids: Collection[int],
        return_as_dict: bool = False,
    ) -> dict[int, FOFGroupEvoData] | list[FOFGroupEvoData]:

    if target_orders := {
        group_id: order[group_id]
        for group_id in group_ids
        if group_id in order
    }:
        return (
            get_mpbs_from_order_as_dict(
                order=target_orders, 
                fof_group_evo=fof_group_evo,
            )
            if return_as_dict else 
            get_mpbs_from_order_as_list(
                order=target_orders, 
                fof_group_evo=fof_group_evo,
            )
        )
    else:
        raise ValueError("No descendants found in order dict")


def is_valid_mpb(mbp: FOFGroupEvoData, target_snap_ids: Collection[int]) -> bool:
    return all(
        snap_idx in mbp.moments.snapshot_ids for snap_idx in target_snap_ids
    )

def _object_array(items: list) -> np.ndarray:
    """1-D object array holding each item atomically. A plain
    ``np.array(list, dtype=object)`` recurses into sized iterables like
    FOFGroupEvoData and unpacks them into rows."""
    arr = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        arr[idx] = item
    return arr


def filter_valid_mpbs(
        mpbs: Collection[FOFGroupEvoData], target_snap_ids: Collection[int]
    ) -> np.ndarray:
    """Branches containing every requested snapshot, as a 1-D object array
    of the branch objects themselves."""
    return _object_array(
        [mpb for mpb in mpbs if is_valid_mpb(mpb, target_snap_ids)]
    )


def get_valid_mpbs_from_order(
        order: dict[int, dict[int, int]],
        fof_group_evo: FOFEvoData,
        desc_group_ids: Collection[int],
        snaps_to_eval: Collection[int],
        return_as_dict: bool = False
    ) -> FOFGroupEvoData | np.ndarray | dict[int, FOFGroupEvoData]:
    """Extract branches for ``desc_group_ids`` and keep those containing
    every snapshot in ``snaps_to_eval``.

    Returns ``{group_id: branch}`` in requested-ID order (dict mode), the
    single surviving branch itself (list mode, one survivor), or a 1-D
    object ndarray of the surviving branches (list mode, several
    survivors). IDs absent from ``order`` and branches missing a requested
    snapshot are omitted and never relabel another branch. Raises
    ``ValueError`` when nothing survives.
    """
    mpbs = get_main_progenitor_branches_from_order(
        order=order,
        fof_group_evo=fof_group_evo,
        group_ids=desc_group_ids,
        return_as_dict=return_as_dict
    )

    if return_as_dict:
        # filter as (group_id, branch) pairs so keys can never shift
        valid = {
            group_id: mpb
            for group_id, mpb in mpbs.items()
            if is_valid_mpb(mpb, snaps_to_eval)
        }
        if not valid:
            raise ValueError("No valid main progenitor branches found")
        return valid

    valid_mpbs = filter_valid_mpbs(mpbs, target_snap_ids=snaps_to_eval)
    if len(valid_mpbs) == 0:
        raise ValueError("No valid main progenitor branches found")
    return valid_mpbs[0] if len(valid_mpbs) == 1 else valid_mpbs

def get_mpb_for_density_maps(
        cfg: AnalysisContext,
        args: Namespace,
        fof_group_evo_data: FOFEvoData,
    ) -> FOFGroupEvoData:

    order = get_tree_order(cfg, fof_group_evo_data, sampling_mdef=args.mdef)

    if (args.halo_id != -1) and (args.halo_id not in order):
        raise ValueError(f"{args.halo_id} not found in order")

    if args.halo_id == -1 and args.target_halo_mass == 0.0:
        raise ValueError("Need to provide either halo_id or target_halo_mass")

    if args.halo_id != -1:
        return fof_group_evo_data.get_main_progenitor_branch(order[args.halo_id])

    target_desc_halos = fof_group_evo_data.descendants
    sampling_masses = target_desc_halos.mass_defs[args.mdef].mass
    target_loc = np.argmax(np.abs(sampling_masses - args.target_halo_mass))
    target_desc_id = target_desc_halos.ids[target_loc]

    if target_desc_id not in order:
        raise ValueError(f"{target_desc_id} not found in order")
    else:
        return fof_group_evo_data.get_main_progenitor_branch(order[target_desc_id])
    


def get_tree_order(
        cfg: AnalysisContext,
        fof_group_evo_data: FOFEvoData | None = None,  # kept for caller compatibility
        sampling_mdef: str = "200m",
    ) -> dict[int, dict[int, int]]:

    order_path = cfg.tree_order_path if cfg.io.use_tree else cfg.prog_order_path

    if not order_path.exists():
        raise FileNotFoundError(
            f"Saved merger-tree order artifact not found: {order_path}. "
            "Automatic ytree/arbor synthesis was retired (A3b2); this "
            "compatibility loader now requires a precomputed order artifact. "
            "Build one from the trees with "
            "asymptotic.merger_trees.full_tree.build_saved_order_info "
            "(Tessera C++ all-progenitor traversal) and write it with "
            "asymptotic.assembly.trees.save_merger_tree_info; for production "
            "main-branch analysis use asymptotic.merger_trees directly."
        )

    return (
        load_tree_order_info(order_path)
        if cfg.io.use_tree else
        load_prog_order_info(order_path)
    )

def get_main_prog_info(
        tree_order: dict[int, dict[int, int]],
        fof_group_evo_data: FOFEvoData,
        target_desc_halos: FOFGroups,
        snaps_to_eval: Collection[int],
        run_validation: bool = True,
        return_as_dict: bool = False
    ) -> tuple[
        FOFGroupEvoData | list[FOFGroupEvoData] | np.ndarray | dict[int, FOFGroupEvoData],
        Collection[int],
    ]:
    """Returns ``(main_progs, target_desc_group_ids)``. With validation,
    ``main_progs`` follows the ``get_valid_mpbs_from_order`` contract: the
    single surviving branch itself, a 1-D object ndarray of branches, or
    ``{group_id: branch}``; without validation it is the raw list/dict from
    extraction. The ID collection matches the surviving branches."""
    if run_validation:

        main_progs = get_valid_mpbs_from_order(
            order=tree_order,
            fof_group_evo=fof_group_evo_data,
            desc_group_ids=target_desc_halos.ids,
            snaps_to_eval=snaps_to_eval,
            return_as_dict=return_as_dict
        )

        desc_snap_id = target_desc_halos.moment.snapshot_id

        if return_as_dict:
            target_desc_group_ids = [
                mpb[desc_snap_id].id_number for mpb in main_progs.values()
            ]
        elif isinstance(main_progs, FOFGroupEvoData):
            # single survivor: the branch itself (iterating it would walk
            # its per-snapshot halos, not a branch collection)
            target_desc_group_ids = [main_progs[desc_snap_id].id_number]
        else:
            target_desc_group_ids = [
                mpb[desc_snap_id].id_number for mpb in main_progs
            ]

    else:

        main_progs = get_main_progenitor_branches_from_order(
            order=tree_order,
            fof_group_evo=fof_group_evo_data,
            group_ids=target_desc_halos.ids,
            return_as_dict=return_as_dict,
        )

        target_desc_group_ids = target_desc_halos.ids

    return main_progs, target_desc_group_ids

def get_single_mpb(
        cfg: AnalysisContext,
        fof_group_evo_data: FOFEvoData,
        sampling_mdef: str = "200m"
    ) -> FOFGroupEvoData:

    order = get_tree_order(cfg, fof_group_evo_data, sampling_mdef)
        
    desc_halos = fof_group_evo_data[max(fof_group_evo_data.data)]
    desc_mass_loc = np.argmax(desc_halos.mass_defs[sampling_mdef].mass)

    heaviest_desc_id = desc_halos.ids[desc_mass_loc]
    evo_path = order[heaviest_desc_id]

    return fof_group_evo_data.get_main_progenitor_branch(evo_path)

def get_halo_timelines_info(
        cfg: AnalysisContext,
        args: Namespace,
        fof_group_evo_data: FOFEvoData,
        run_validation: bool = True,
        return_as_dict: bool = False,
        sampling_mdef: str = "200m"
    ) -> dict:

    order = get_tree_order(cfg, fof_group_evo_data, sampling_mdef)

    timeline_ids = get_snapshot_timeline_ids(
        snapshot_ids=fof_group_evo_data.moments.snapshot_ids,
        max_tree_snap_id=max(order.keys()),
        prog_snaps_to_eval=args.prog_snaps_to_eval
    )
        
    desc_snap_id = timeline_ids["desc_snap_id"]
    snaps_to_eval = timeline_ids["snaps_to_eval"]

    print(f"{desc_snap_id = }")
    print(f"Evaluating {snaps_to_eval} Snapshots:\n")

    target_desc_halos = get_desc_sample_from_args(
        desc_halos=fof_group_evo_data[desc_snap_id],
        args=args,
        min_resolved_halo_mass=cfg.min_resolved_halo_mass,
        sampling_mdef=sampling_mdef
    )

    organize_tree_start = time.time()

    main_progs, target_desc_group_ids = get_main_prog_info(
        tree_order=order,
        fof_group_evo_data=fof_group_evo_data,
        target_desc_halos=target_desc_halos,
        snaps_to_eval=snaps_to_eval,
        run_validation=run_validation,
        return_as_dict=return_as_dict,
    )

    organize_tree_finish = time.time()
    organize_tree_time = organize_tree_finish - organize_tree_start

    print(f"Organized Tree in {display_time(organize_tree_time)}\n")

    return {
        "snapshots_to_eval": snaps_to_eval,
        "desc_snap_id": desc_snap_id,
        "target_desc_group_ids": target_desc_group_ids,
        "main_progs": main_progs,
    }
