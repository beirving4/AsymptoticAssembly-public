from __future__ import annotations

import h5py
import numpy as np, pdb

from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Iterable
from attrs import define, field
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from collections import OrderedDict



from ..cosmo.model import Cosmology
from ..simulation.evo import EvolutionData
from ..mass_def.data import MassDefinitions
from ..simulation.moments import Moment, MomentsInTime
from ..bounded.fof import FOFGroups, FOFEvoData, FOFGroupEvoData

from ..utils.get_data import get_fof_data, get_fof_subfind_data # add get tree data


#: canonical filename suffixes of the split saved-order pair
_TREE_SUFFIX = "_tree_order.hdf5"
_PROG_SUFFIX = "_prog_order.hdf5"


def _object_array(items: Sequence[Any]) -> np.ndarray:
    """1-D object array holding each item atomically.

    ``np.array(list, dtype=object)`` recurses into sized/iterable objects —
    ``GroupTree`` and ``FOFGroupEvoData`` both inherit ``EvolutionData``'s
    ``__len__``/``__iter__`` — and unpacks them into rows whenever their
    shapes line up. Preallocating and assigning per element stores the
    objects themselves, preserving identity for any collection size.
    """
    arr = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        arr[idx] = item
    return arr


def save_merger_tree_info(save_base: Path, tree_info: dict) -> None:
    """Write ``tree_info`` as the canonical split saved-order pair.

    Creates ``<save_base>_tree_order.hdf5`` and ``<save_base>_prog_order.hdf5``.
    Each file holds one HDF5 group per descendant (named ``str(group_id)``)
    containing a ``tree`` (resp. ``prog``) subgroup with one dataset per
    snapshot (named ``str(snap_id)``). ``load_merger_tree_info`` reads the
    pair back into the same ``{group_id: {"tree": OrderedDict, "prog":
    OrderedDict}}`` structure accepted here and by
    ``GroupTreeData.from_tree_order_data``.
    """
    tree_path = Path(save_base.parent, f"{save_base.name}{_TREE_SUFFIX}")
    prog_path = Path(save_base.parent, f"{save_base.name}{_PROG_SUFFIX}")

    print(f"Saving Merger Tree Info to {tree_path}")

    with h5py.File(tree_path, 'w') as hf:
        for group_id, group_data in tqdm(tree_info.items(), desc="Saving Merger Tree Info"):
            # Create a group for each group_id
            group_id_str = str(group_id)
            group_grp = hf.create_group(group_id_str)

            # Create subgroups for "tree" and "prog"
            tree_grp = group_grp.create_group("tree")
            for snap_id, array in group_data["tree"].items():
                tree_grp.create_dataset(str(snap_id), data=array)

    print(f"Saving Progenitor Line Info to {prog_path}")

    with h5py.File(prog_path, 'w') as hf:
            for group_id, group_data in tqdm(tree_info.items(), desc="Saving Progenitor Line Info"):
                group_id_str = str(group_id)
                group_grp = hf.create_group(group_id_str)

                prog_grp = group_grp.create_group("prog")
                for snap_id, array in group_data["prog"].items():
                    prog_grp.create_dataset(str(snap_id), data=array)


def _load_order_kind(load_path: Path, kind: str, desc: str) -> dict:
    """Read one kind ("tree"/"prog") from a saved-order file: numerically
    sorted ``{group_id: OrderedDict[snap_id, np.ndarray]}``, stored dtypes
    preserved."""
    data = {}
    with h5py.File(load_path, 'r') as hf:
        for group_id_str in tqdm(sorted(hf.keys(), key=int), desc=desc):
            group_grp = hf[group_id_str]
            if kind not in group_grp:
                raise ValueError(
                    f"{load_path}: group {group_id_str!r} has no {kind!r} "
                    f"subgroup; not a valid saved {kind}-order artifact"
                )
            order = OrderedDict()
            kind_grp = group_grp[kind]
            for snap_id_str in sorted(kind_grp.keys(), key=int):
                order[int(snap_id_str)] = kind_grp[snap_id_str][...]
            data[int(group_id_str)] = order
    return data


def load_tree_order_info(load_path: Path) -> dict:
    return _load_order_kind(load_path, "tree", "Loading Merger Tree Info")


def load_prog_order_info(load_path: Path) -> dict:
    return _load_order_kind(load_path, "prog", "Loading Progenitor Line Info")


def _is_combined_order_file(path: Path) -> bool:
    """Historical combined layout: 'tree' AND 'prog' under every group."""
    with h5py.File(path, 'r') as hf:
        groups = list(hf.keys())
        return bool(groups) and all(
            "tree" in hf[g] and "prog" in hf[g] for g in groups
        )


def load_merger_tree_info(
        tree_order_path: Path,
        prog_order_path: Path | None = None,
    ) -> dict:
    """Load a saved order artifact into the canonical group-first structure
    ``{group_id: {"tree": OrderedDict[snap_id, np.ndarray], "prog": ...}}``
    — the shape ``save_merger_tree_info`` writes and
    ``GroupTreeData.from_tree_order_data`` consumes.

    Accepts the canonical split pair (the ``*_prog_order.hdf5`` companion is
    inferred from a ``*_tree_order.hdf5`` name unless ``prog_order_path`` is
    given explicitly) or a historical combined file carrying both subgroups
    under every group. Group and snapshot IDs are ints in ascending
    numerical order; stored array dtypes and values are preserved.
    """
    tree_order_path = Path(tree_order_path)

    if not tree_order_path.exists():
        raise FileNotFoundError(
            f"saved order artifact not found: {tree_order_path}"
        )

    if prog_order_path is None:
        if _is_combined_order_file(tree_order_path):
            prog_order_path = tree_order_path
        elif tree_order_path.name.endswith(_TREE_SUFFIX):
            prog_order_path = tree_order_path.with_name(
                tree_order_path.name[: -len(_TREE_SUFFIX)] + _PROG_SUFFIX
            )
        else:
            raise ValueError(
                f"{tree_order_path} is neither a combined order file nor a "
                f"standard *{_TREE_SUFFIX} artifact; pass prog_order_path "
                "explicitly"
            )
    else:
        prog_order_path = Path(prog_order_path)

    if not prog_order_path.exists():
        raise FileNotFoundError(
            f"saved order artifact not found: {prog_order_path}"
        )

    tree_data = load_tree_order_info(tree_order_path)
    prog_data = load_prog_order_info(prog_order_path)

    if set(tree_data) != set(prog_data):
        raise ValueError(
            f"descendant sets differ between {tree_order_path} and "
            f"{prog_order_path}: missing from prog side: "
            f"{sorted(set(tree_data) - set(prog_data))}; missing from tree "
            f"side: {sorted(set(prog_data) - set(tree_data))}"
        )

    return {
        group_id: {"tree": tree_data[group_id], "prog": prog_data[group_id]}
        for group_id in sorted(tree_data)
    }


def get_tree_info_from_prog_halo(
        full_tree_info: dict, 
        prog_group_id: int, 
        prog_snap_id: int
    ) -> dict:

    return next(
        (
            tree_info
            for tree_info in full_tree_info.values()
            if (
                (prog_snap_id in tree_info) and
                prog_group_id in tree_info[prog_snap_id]
            )
        ),
        {},
    )

def get_tree_info_subset_from_prog_halos(
        full_tree_info: dict, 
        prog_group_ids: Collection[int], 
        prog_snap_id: int
    ) -> dict:

    results = {}

    for prog_group_id in tqdm(prog_group_ids, desc="Extracting Tree Subset"):
        if tree_info := get_tree_info_from_prog_halo(
            full_tree_info=full_tree_info,
            prog_group_id=prog_group_id,
            prog_snap_id=prog_snap_id,
        ):
            results[prog_group_id] = tree_info

    return results

def get_prog_info_from_prog_halo(
        full_prog_info: dict, 
        prog_group_id: int, 
        prog_snap_id: int
    ) -> dict:

    return next(
        (
            prog_info
            for prog_info in full_prog_info.values()
            if (
                (prog_snap_id in prog_info) and
                prog_group_id in prog_info[prog_snap_id]
            )
        ),
        {},
    )

def get_prog_info_subset_from_prog_halos(
        full_prog_info: dict, 
        prog_group_ids: Collection[int], 
        prog_snap_id: int
    ) -> dict:

    results = {}

    for prog_group_id in tqdm(prog_group_ids, desc="Extracting Progenitor Subset"):
        if prog_info := get_prog_info_from_prog_halo(
            full_prog_info=full_prog_info,
            prog_group_id=prog_group_id,
            prog_snap_id=prog_snap_id,
        ):
            results[prog_group_id] = prog_info

    return results

def get_prog_info_subset_from_tree_halos(
        full_tree_info: dict, 
        full_prog_info: dict,
        group_ids: Collection[int], 
        prog_snap_id: int
    ) -> dict:

    tree_results = get_tree_info_subset_from_prog_halos(
        full_tree_info=full_tree_info,
        prog_group_ids=group_ids,
        prog_snap_id=prog_snap_id,
    )

    return {
        group_id : prog_info
        for group_id, prog_info in full_prog_info.items()
        if group_id in tree_results
    }

    # for group_id in tqdm(group_ids, desc="Extracting Progenitor Subset"):
    #     if tree_info := get_tree_info_from_prog_halo(
    #         full_tree_info=full_tree_info,
    #         prog_group_id=group_id,
    #         prog_snap_id=prog_snap_id,
    #     ):
    #         prog_info = tree_info
    #         results[prog_group_id] = prog_info

    # return results


@define(slots=True)
class MergerTree(ABC):
    pass

@define(slots=True)
class MergerTreeData(ABC):
    pass # Collection of merger trees... 

# @define(slots=True)
# class GroupTree(MergerTree):
#     pass


@define(slots=True)
class GroupTree(EvolutionData):
    id_number: int
    moments: MomentsInTime
    data: OrderedDict[int, FOFGroups]

    main_prog_branch_ids: np.ndarray

    def __repr__(self) -> str:
        return super().__repr__()

    @classmethod
    def from_tree_order(
            cls,
            desc_group_id: int,
            tree_order: OrderedDict[int, np.ndarray],
            prog_order: OrderedDict[int, np.ndarray],
            fof_group_evo: FOFEvoData,
        ) -> GroupTree:
        
        data = OrderedDict()
        moments = MomentsInTime()

        for snap_id, group_ids in tree_order.items():

            try: 
                data[snap_id] = fof_group_evo[snap_id].get_subset(group_ids)
                moments.add_moment(fof_group_evo[snap_id].moment)
            except (KeyError, IndexError) as e:
                continue 

        # pdb.set_trace()

        return cls(
            id_number=desc_group_id,
            moments=moments,
            data=OrderedDict(sorted(data.items())),
            main_prog_branch_ids=np.vstack([
                (snap_id, int(group_ids[0]))
                for snap_id, group_ids in prog_order.items()
            ])
        )
    
    @classmethod
    def from_fof_group_evo(
            cls, 
            fof_group_evo: FOFEvoData, 
            desc_group_id: int,
        ) -> GroupTree:
    
        ...
    
    @property
    def main_progenitor_branch(self) -> FOFGroupEvoData:
        
        moments = MomentsInTime()
        data = OrderedDict() 

        for snap_id, group_id in self.main_prog_branch_ids:

            if ((groups := self.data.get(snap_id)) is None): continue 

            try: 
                data[snap_id] = groups[group_id]
                moments.add_moment(groups.moment)
            except IndexError:
                continue 

        return FOFGroupEvoData(
            moments=moments,
            data=OrderedDict(sorted(data.items()))
        )
    
    @property
    def tree_order(self) -> OrderedDict[int, np.ndarray]:
        return OrderedDict([
            (snap_id, np.asarray([group.id_number for group in groups]))
            for snap_id, groups in self.data.items()
        ])

    

@define(slots=True)
class GroupTreeData:
    id_numbers: np.ndarray
    moments: MomentsInTime
    data: np.ndarray

    def __getitem__(self, idx: int) -> GroupTree:
        return self.data[idx]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterable[GroupTree]:
        yield from self.data


    @classmethod
    def from_tree_order_data(
            cls, 
            fof_group_evo: FOFEvoData,
            full_order_info: dict[int, OrderedDict[int, np.ndarray]],
        ) -> GroupTreeData:
        
        data, desc_group_ids = [], []

        for group_id, order in tqdm(full_order_info.items(), desc="Generating Group Trees"):
            try:
                data.append(
                    GroupTree.from_tree_order(
                        desc_group_id=group_id, 
                        fof_group_evo=fof_group_evo,
                        tree_order=order["tree"],
                        prog_order=order["prog"]
                    )
                )
                desc_group_ids.append(group_id)
            except IndexError:
                continue

        return GroupTreeData(
            id_numbers=np.asarray(desc_group_ids),
            moments=fof_group_evo.moments,
            data=_object_array(data)
        )

    @classmethod
    def from_saved_tree_order(
            cls,
            fof_group_evo: FOFEvoData,
            tree_order_path: Path,
            prog_order_path: Path | None = None,
        ) -> GroupTreeData:
        """Build from a saved order artifact: the canonical split pair
        (companion inferred unless given) or a historical combined file."""
        tree_order_info = load_merger_tree_info(
            tree_order_path, prog_order_path=prog_order_path
        )

        return cls.from_tree_order_data(fof_group_evo, tree_order_info)
    

    
    @property
    def main_progenitor_branches(self) -> np.ndarray:
        return _object_array(
            [group_tree.main_progenitor_branch for group_tree in self]
        )
    

    # def get_group_id_evos(
    #         self, group_ids: Collection[int], snapshot_ids: Collection[int]
    #     ) -> dict[int, np.ndarray | Collection[int]]:
        
    #     evo_results = defaultdict(list)

    #     ... 



def save_tree(hdf_file: h5py.File, tree: GroupTree) -> None:

    tree_group = hdf_file.create_group(f"tree_{tree.id_number}")

    tree_order = tree.tree_order

    for snap_idx, group_ids in tree_order.items():
        tree_group.create_dataset(f"snap_{snap_idx}", data=group_ids)


def save_group_tree_data(
        save_filepath, group_tree_data: GroupTreeData | GroupTree
    ) -> None:

    with h5py.File(save_filepath, "w") as hdf_file:
        if isinstance(group_tree_data, GroupTree):
            save_tree(hdf_file, group_tree_data)
        else:
            for tree in tqdm(group_tree_data.data, desc="Saving trees"):
                save_tree(hdf_file, tree)
        





