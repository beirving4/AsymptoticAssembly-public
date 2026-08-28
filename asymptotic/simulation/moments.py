from __future__ import annotations

import h5py
import pickle, pdb
import numbers
import numpy as np

from pathlib import Path
from typing import Iterable
from attrs import define, field
from collections.abc import Collection

from ..cosmo.model import Cosmology
from ..utils.get_data import (
    get_snapshot_mapping, get_snapshot_mapping_from_power_spectra
)

@define(slots=True, order=True)
class Moment:
    snapshot_id: int
    scale_factor: float
    redshift: float = field(default=np.nan)
    age: float = field(default=np.nan)
    lookback_time: float = field(default=np.nan)
    hubble_epoch: float = field(default=np.nan)
    dynamical_time: float = field(default=np.nan)

    def __attrs_post_init__(self) -> None:
        if np.isnan(self.redshift):
            self.redshift = 1.0 / self.scale_factor - 1.0


    def __eq__(self, value: Moment) -> bool:
        return (
            self.scale_factor == value.scale_factor
            and self.redshift == value.redshift
            and self.age == value.age
            and self.lookback_time == value.lookback_time
            and self.hubble_epoch == value.hubble_epoch
            and self.dynamical_time == value.dynamical_time
        )
    
    def add_times(self, cosmo: Cosmology) -> None:
        self.age = float(cosmo.age(self.redshift))
        self.lookback_time = float(cosmo.lookback_time(self.redshift))
        self.hubble_epoch = float(cosmo.hubble_epoch(self.redshift))
        self.dynamical_time = float(cosmo.get_dynamical_time(self.redshift))

    def get_age_wrt_present(self, present_day_age: float) -> float:
        return self.age / present_day_age

    @classmethod
    def null_initialize(cls) -> Moment:
        return cls(
            snapshot_id=-1,
            scale_factor=np.nan,
            redshift=np.nan,
            age=np.nan,
            lookback_time=np.nan,
            hubble_epoch=np.nan,
            dynamical_time=np.nan
        )
    
    @classmethod
    def for_initial_condition(cls, redshift: float) -> Moment:
        scale_factor = 1.0 / (1.0 + redshift)
        return cls(
            snapshot_id=-1,
            scale_factor=scale_factor,
            redshift=redshift,
            age=np.nan,
            lookback_time=np.nan,
            hubble_epoch=np.nan,
            dynamical_time=np.nan
        )


@define(slots=True)
class MomentsInTime:
    snapshot_ids: np.ndarray = field(default=np.empty(0))
    scale_factors: np.ndarray = field(default=np.empty(0))
    redshifts: np.ndarray = field(default=np.empty(0))

    # Ages are in Gyr
    ages: np.ndarray = field(default=np.empty(0))
    lookback_times: np.ndarray = field(default=np.empty(0))
    dynamical_times: np.ndarray = field(default=np.empty(0))

    hubble_epochs: np.ndarray = field(default=np.empty(0))

    initial: Moment = field(default=Moment.null_initialize())

    def __attrs_post_init__(self) -> None:
        if (len(self.redshifts) == 0) and (len(self.scale_factors) > 0):
            self.redshifts = 1.0 / self.scale_factors - 1.0

        
        self.snapshot_ids = self.snapshot_ids.astype(int)

    def __getitem__(self, snapshot_id: int) -> Moment:
        return self.get_snapshot_instance(snapshot_id)
    
    def __len__(self) -> int:
        return len(self.snapshot_ids)
    

    def __repr__(self) -> str:
        return (
            f"MomentsInTime("
            f"n_snapshots={len(self)}, "
            f"snap_ids={self.snapshot_ids.min():.0f}-{self.snapshot_ids.max():.0f}, "
            f"a={self.scale_factors.min():.3f}-{self.scale_factors.max():.3f})"
        )
    
    @property
    def as_dict(self) -> dict[int, Moment]:
        return {
            int(snapshot_id): self[snapshot_id]
            for snapshot_id in self.snapshot_ids
        }
    
    def get_by_index(self, index: int) -> Moment:
        if index >= len(self.snapshot_ids) or index < 0:
            raise IndexError("Index out of bounds")
        return Moment(
            snapshot_id=safe_1D_indexing(self.snapshot_ids, index),
            scale_factor=safe_1D_indexing(self.scale_factors, index),
            redshift=safe_1D_indexing(self.redshifts, index),
            age=safe_1D_indexing(self.ages, index),
            lookback_time=safe_1D_indexing(self.lookback_times, index),
            hubble_epoch=safe_1D_indexing(self.hubble_epochs, index),
            dynamical_time=safe_1D_indexing(self.dynamical_times, index)
        )
    
    @property
    def snap_to_scale_factor(self) -> dict[int, float]:
        if len(self.scale_factors) == 0:
            return {}
        return {
            int(sid): a 
            for sid, a in zip(self.snapshot_ids, self.scale_factors)
        }
    
    @property
    def snap_to_redshift(self) -> dict[int, float]:
        if len(self.redshifts) == 0:
            return {}
        return {
            int(sid): z 
            for sid, z in zip(self.snapshot_ids, self.redshifts)
        }
    
    @property
    def snap_to_age(self) -> dict[int, float]:
        if len(self.ages) == 0:
            return {}
        return {
            int(sid): t_age 
            for sid, t_age in zip(self.snapshot_ids, self.ages)
        }
    
    @property
    def snap_to_lookback_time(self) -> dict[int, float]:
        if len(self.lookback_times) == 0:
            return {}
        return {
            int(sid): t_look 
            for sid, t_look in zip(self.snapshot_ids, self.lookback_times)
        }
    
    @property
    def snap_to_hubble_epoch(self) -> dict[int, float]:
        if len(self.hubble_epochs) == 0:
            return {}
        return {
            int(sid): epoch 
            for sid, epoch in zip(self.snapshot_ids, self.hubble_epochs)
        }
    
    @property
    def present_day_age(self) -> float:
        dist_from_present = np.abs(np.round(self.scale_factors - 1.0, 3))
        present_idx = np.argmin(dist_from_present)
        return self.ages[present_idx]

    @property
    def ages_wrt_present(self) -> np.ndarray:
        return (
            np.empty(0)
            if len(self.ages) == 0
            else (self.ages / self.present_day_age)
        )

    @property
    def inital_age_wrt_present(self) -> float:
        return self.initial.get_age_wrt_present(self.present_day_age)
    
    @property
    def full_scale_factors(self) -> np.ndarray:
        if len(self.scale_factors) == 0:
            return np.empty(0)
        return np.concatenate(([self.initial.scale_factor], self.scale_factors))
    
    @property
    def full_redshifts(self) -> np.ndarray:
        if len(self.redshifts) == 0:
            return np.empty(0)
        return np.concatenate(([self.initial.redshift], self.redshifts))
    
    @property
    def full_ages(self) -> np.ndarray:
        if len(self.ages) == 0:
            return np.empty(0)
        return np.concatenate(([self.initial.age], self.ages))

    @property
    def full_lookback_times(self) -> np.ndarray:
        if len(self.lookback_times) == 0:
            return np.empty(0)
        return np.concatenate(([self.initial.lookback_time], self.lookback_times))
    
    @property
    def full_hubble_epochs(self) -> np.ndarray:
        if len(self.hubble_epochs) == 0:
            return np.empty(0)
        return np.concatenate(([self.initial.hubble_epoch], self.hubble_epochs))
    
    @property
    def ages_into_future_wrt_present(self) -> np.ndarray:
        if len(self.ages) == 0:
            return np.empty(0)
        return (self.ages / self.present_day_age) - 1.0

    @property
    def full_ages_into_future_wrt_present(self) -> np.ndarray:
        if len(self.ages) == 0:
            return np.empty(0)
        initial_wrt_present = self.initial.get_age_wrt_present(self.present_day_age)
        return np.concatenate((
            [initial_wrt_present - 1.0],
            (self.ages / self.present_day_age) - 1.0
        ))

    @property
    def full_ages_wrt_present(self) -> np.ndarray:
        if len(self.ages_wrt_present) == 0:
            return np.empty(0)
        return np.concatenate((
            [self.inital_age_wrt_present],
            self.ages_wrt_present
        ))
    
    def add_times(self, cosmo: Cosmology) -> None:

        self.ages = np.asarray(cosmo.age(self.redshifts))
        self.lookback_times = np.asarray(cosmo.lookback_time(self.redshifts))
        self.dynamical_times = np.asarray(cosmo.get_dynamical_time(self.redshifts))
        self.hubble_epochs = np.asarray(cosmo.hubble_epoch(self.redshifts))
    
    def _append_optional_attr(
            self, attr_name: str, value_to_be_appended: float,
        ) -> None:

        attr_array = getattr(self, attr_name)
        if (attr_array.size > 0) and np.isfinite(value_to_be_appended):
            setattr(
                self, attr_name, np.append(attr_array, value_to_be_appended)
            )
        elif (attr_array.size == 0) and np.isfinite(value_to_be_appended):
            setattr(
                self, attr_name, np.array([value_to_be_appended])
            )
        else:
            return
    
    def _append_moment_values(self, moment: Moment) -> None:
        self.snapshot_ids = np.append(self.snapshot_ids, moment.snapshot_id)
        self.scale_factors = np.append(self.scale_factors, moment.scale_factor)
        self.redshifts = np.append(self.redshifts, moment.redshift)

        self._append_optional_attr("ages", moment.age)
        self._append_optional_attr("lookback_times", moment.lookback_time)
        self._append_optional_attr("hubble_epochs", moment.hubble_epoch)
        self._append_optional_attr("dynamical_times", moment.dynamical_time)

    
    def _sort_by_snapshot_ids(self) -> None:
        sort_index = np.argsort(self.snapshot_ids)
        self.snapshot_ids = self.snapshot_ids[sort_index]
        self.scale_factors = self.scale_factors[sort_index]
        self.redshifts = self.redshifts[sort_index]

        if self.ages.size > 0:
            self.ages = self.ages[sort_index]
        if self.lookback_times.size > 0:
            self.lookback_times = self.lookback_times[sort_index]
        if self.hubble_epochs.size > 0:
            self.hubble_epochs = self.hubble_epochs[sort_index]
        if self.dynamical_times.size > 0:
            self.dynamical_times = self.dynamical_times[sort_index]

    def add_moment(self, moment: Moment) -> None:
        if moment.snapshot_id in self.snapshot_ids:
            return  # Already have this moment
        
        self._append_moment_values(moment)
        self._sort_by_snapshot_ids()

    def get_subset(self, idxs: int | np.int_ | Collection[int | np.int_]) -> Moment | MomentsInTime:

        if isinstance(idxs, (int, np.int_)):
            return self[idxs]
        else:
            return MomentsInTime(
                snapshot_ids=self.snapshot_ids[idxs],
                scale_factors=self.scale_factors[idxs],
                redshifts=self.redshifts[idxs],
                ages=self.ages[idxs],
                lookback_times=self.lookback_times[idxs],
                hubble_epochs=self.hubble_epochs[idxs],
                dynamical_times=self.dynamical_times[idxs],
            )
    
    def get_subset_by_attribute(
            self, key_attr: str,
            attr_value: float | int | Iterable[float | int],
        ) -> MomentsInTime:

        if isinstance(attr_value, Collection) and len(attr_value) == 0:
            raise ValueError("attr_value must be a non-empty collection")
        
        values = self.map_by_attribute(key_attr=key_attr, attr_value=attr_value)
        idxs = np.where(np.isin(self.snapshot_ids, values))[0]

        # Always subset by the POSITIONAL index array (returns a MomentsInTime),
        # even for a single match. The old `get_subset(idxs[0])` special case passed
        # a positional index to get_subset(int), which reinterprets an int as a
        # snapshot_id (via __getitem__ -> get_snapshot_instance) and raised whenever
        # that index was not itself a valid snapshot id. Every caller uses the result
        # as `moments=` for an Evo object, i.e. expects a MomentsInTime.
        return self.get_subset(idxs)


    def map_by_attribute(
            self, key_attr: str, 
            attr_value: float | int | Collection[float | int], 
            return_attr: str = "snapshot_id"
        ) -> float | int | np.ndarray:

        if isinstance(attr_value, Collection):
            return np.array([
                self.map_by_attribute(key_attr, value, return_attr) 
                for value in attr_value
                if isinstance(value, (float, np.float64, int, np.int_))
            ])
        index = np.argmin(np.abs(getattr(self, f"{key_attr}s") - attr_value))
        return getattr(self.get_by_index(index), return_attr)
    
        
    
    def get_snapshot_instance(self, snapshot_id: int) -> Moment: 
        index = np.where(self.snapshot_ids == snapshot_id)[0][0]
        return self.get_by_index(index)

    @classmethod
    def from_data(
            cls, sim_dir: Path,  sim_cosmo: Cosmology, 
            snapshot_dir_name: str | None = None
        ) -> MomentsInTime:
        
        mapping = get_snapshot_mapping(sim_dir, snapshot_dir_name)

        return cls(
            snapshot_ids=mapping["snapshot_ids"],
            scale_factors=mapping["scale_factors"],
            redshifts=mapping["redshifts"],
            ages=sim_cosmo.age(mapping["redshifts"]),
            lookback_times=sim_cosmo.lookback_time(mapping["redshifts"]),
            hubble_epochs=sim_cosmo.hubble_epoch(mapping["redshifts"]),
            dynamical_times=sim_cosmo.get_dynamical_time(mapping["redshifts"]),
        ) 

    @classmethod
    def from_power_spec_data(
            cls, 
            power_spec_dir: Path,  
            sim_cosmo: Cosmology, 
        ) -> MomentsInTime:
        
        mapping = get_snapshot_mapping_from_power_spectra(power_spec_dir)

        # pdb.set_trace()

        if 'initial' in mapping:
            initial_redshift = mapping['initial'][0]
            initial_moment = Moment.for_initial_condition(initial_redshift)
            initial_moment.add_times(sim_cosmo)
        else:
            initial_moment = Moment.null_initialize()

        return cls(
            snapshot_ids=mapping["snapshot_ids"],
            scale_factors=mapping["scale_factors"],
            redshifts=mapping["redshifts"],
            ages=sim_cosmo.age(mapping["redshifts"]),
            lookback_times=sim_cosmo.lookback_time(mapping["redshifts"]),
            hubble_epochs=sim_cosmo.hubble_epoch(mapping["redshifts"]),
            dynamical_times=sim_cosmo.get_dynamical_time(mapping["redshifts"]),
            initial=initial_moment
        ) 
    
    

    def to_hdf5(self, moments_h5py_group: h5py.Group) -> None:
        """Write this MomentsInTime instance to an HDF5 group."""
        def _write(name: str, arr: np.ndarray) -> None:
            # Ensure we always write something, even if optional arrays are empty.
            data = np.asarray(arr)
            if name in moments_h5py_group:
                del moments_h5py_group[name]
            moments_h5py_group.create_dataset(name, data=data)

        # Required
        _write("snapshot_ids", np.asarray(self.snapshot_ids, dtype=int))
        _write("scale_factors", np.asarray(self.scale_factors, dtype=float))

        # Optional / derived
        _write("redshifts", np.asarray(self.redshifts, dtype=float))
        _write("ages", np.asarray(self.ages, dtype=float))
        _write("lookback_times", np.asarray(self.lookback_times, dtype=float))
        _write("hubble_epochs", np.asarray(self.hubble_epochs, dtype=float))
        _write("dynamical_times", np.asarray(self.dynamical_times, dtype=float))

    def save(self, directory: Path) -> None:
        """Save moments metadata to an HDF5 file for portability.

        Writes to: <directory>/sim_data.h5 under group "moments_in_time".
        """
        filepath = Path(directory, "sim_data.h5")
        with h5py.File(filepath, "a") as sim_h5_file:
            if "moments_in_time" in sim_h5_file:
                del sim_h5_file["moments_in_time"]
            grp = sim_h5_file.create_group("moments_in_time")
            grp.attrs["class"] = self.__class__.__name__
            grp.attrs["schema_version"] = 1
            self.to_hdf5(grp)

    def save_pickle(self, directory: Path) -> None:
        """Backwards-compatible pickle save (legacy)."""
        filepath = Path(directory, "snapshot_mapping.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, directory: Path) -> MomentsInTime:
        filepath = Path(directory, "snapshot_mapping.pkl")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_hdf5(cls, moments_h5py_group: h5py.Group) -> MomentsInTime:
        def _read(name: str, default: np.ndarray) -> np.ndarray:
            return (
                np.asarray(moments_h5py_group[name][()]) 
                if (name in moments_h5py_group) else 
                default
            )

        snapshot_ids = _read("snapshot_ids", np.empty(0, dtype=int))
        scale_factors = _read("scale_factors", np.empty(0, dtype=float))

        # redshifts can be derived in __attrs_post_init__ if missing
        redshifts = _read("redshifts", np.empty(0, dtype=float))

        ages = _read("ages", np.empty(0, dtype=float))
        lookback_times = _read("lookback_times", np.empty(0, dtype=float))
        dynamical_times = _read("dynamical_times", np.empty(0, dtype=float))

        # Support both new and legacy dataset names
        if "hubble_epochs" in moments_h5py_group:
            hubble_epochs = moments_h5py_group["hubble_epochs"][()]
        elif "hubble_times" in moments_h5py_group:
            hubble_epochs = moments_h5py_group["hubble_times"][()]
        else:
            hubble_epochs = np.empty(0, dtype=float)

        return cls(
            snapshot_ids=snapshot_ids,
            scale_factors=scale_factors,
            redshifts=redshifts,
            ages=ages,
            lookback_times=lookback_times,
            hubble_epochs=hubble_epochs,
            dynamical_times=dynamical_times,
        )

    @classmethod
    def from_tree_times(
        cls, tree_times_group: h5py.Group, cosmo: Cosmology | None = None
    ) -> MomentsInTime:
        """Build a MomentsInTime from a GADGET-4 / composed-tree ``TreeTimes`` group.

        Maps the ``Moment_*`` arrays (each length n_snapshots) onto the global
        time axis. Used by the tessera-backed assembly pipeline so the 75-snap
        grid stored in ``full_tree.hdf5`` / ``trees.hdf5`` is reused verbatim
        rather than recomputed from the cosmology.

        ``full_tree.hdf5`` carries the rich ``Moment_*`` axis directly. The
        GADGET-native ``trees.hdf5`` only has ``Time`` (scale factor) + ``Redshift``;
        for it, pass ``cosmo`` to compute ``age`` / ``lookback_time`` /
        ``hubble_epoch`` / ``dynamical_time`` from the redshifts (mirrors what
        ``compose_full_tree`` bakes in). Without ``cosmo`` those fields are NaN.
        """
        def _read(name: str, default: np.ndarray | None = None) -> np.ndarray | None:
            return np.asarray(tree_times_group[name][()]) if name in tree_times_group else default

        scale_factors = _read("Moment_scale_factor", _read("Time"))
        if scale_factors is None:
            raise KeyError("TreeTimes has neither Moment_scale_factor nor Time")
        n = len(scale_factors)
        nan = np.full(n, np.nan)

        snapshot_ids = _read("Moment_snapshot_id")
        if snapshot_ids is None:
            snapshot_ids = np.arange(n, dtype=int)

        redshifts = _read("Moment_redshift", _read("Redshift"))
        if redshifts is None:
            redshifts = 1.0 / scale_factors - 1.0

        # Each cosmology-derived field: use the baked Moment_* if present,
        # else compute from the cosmology if supplied, else NaN.
        def _timed(moment_key: str, cosmo_fn) -> np.ndarray:
            baked = _read(moment_key)
            if baked is not None:
                return baked
            return np.asarray(cosmo_fn(redshifts)) if cosmo is not None else nan

        return cls(
            snapshot_ids=snapshot_ids,
            scale_factors=scale_factors,
            redshifts=redshifts,
            ages=_timed("Moment_age", lambda z: cosmo.age(z)),
            lookback_times=_timed("Moment_lookback_time", lambda z: cosmo.lookback_time(z)),
            hubble_epochs=_timed("Moment_hubble_epoch", lambda z: cosmo.hubble_epoch(z)),
            dynamical_times=_timed("Moment_dynamical_time", lambda z: cosmo.get_dynamical_time(z)),
        )

    @classmethod
    def load(cls, saved_data_dir: Path) -> MomentsInTime:
        filepath = Path(saved_data_dir, "sim_data.h5")
        with h5py.File(filepath, "r") as sim_h5_file:
            if "moments_in_time" not in sim_h5_file:
                raise KeyError(
                    f"Group 'moments_in_time' not found in {filepath}. "
                    "Did you save MomentsInTime with MomentsInTime.save(...)?"
                )
            return cls.from_hdf5(sim_h5_file["moments_in_time"])
        
    
    @classmethod
    def joint(cls, moments: Collection[MomentsInTime]) -> MomentsInTime:
        """Alternative constructor that returns the union of multiple MomentsInTime.
        """
        return get_joint_moments(moments)

def safe_1D_indexing(array1D: np.ndarray, index: int) -> numbers.Number | np.ndarray:
    if array1D.ndim == 0 or array1D.size == 0:
        return array1D

    if array1D.ndim > 1:
        raise ValueError("Array must be 1D")   
    
    if index >= len(array1D) or index < 0:
        raise IndexError("Index out of bounds")
    
    return array1D[index]



# Need a funciton for a unified sample moments ... 
# make a join moments function that gets the unique snapshot instances 
# to create a singular instance. Only join them if they share the same 
# seed/cosmology in the sample. (Samples like the sandbox should return 
# multiple cosmologies since there are primary and toy_model_b sims that
# comprise it)
def get_joint_moments(moments: Collection[MomentsInTime]) -> MomentsInTime:
    """Return a MomentsInTime that is the union of a collection of inputs.

    Behavior:
    - If the collection is empty or all inputs are empty, returns an empty MomentsInTime.
    - **Fast path:** If any input already contains *all* snapshot_ids present in the
      whole collection, return that instance directly.
    - Otherwise, build the union over snapshot_ids and merge per-snapshot metadata.
      For each snapshot_id, prefer the *first available finite* value seen across
      the inputs; fill missing redshift from scale factor when possible.
    """
    # Normalize and drop non-MomentsInTime inputs
    ms: list[MomentsInTime] = [m for m in moments if isinstance(m, MomentsInTime)]
    if not ms:
        return MomentsInTime()

    # Drop empty inputs
    non_empty = [m for m in ms if getattr(m, "snapshot_ids", np.empty(0)).size > 0]
    if not non_empty:
        return MomentsInTime()

    # Compute sorted union of snapshot ids using numpy for speed
    union_ids: np.ndarray | None = None
    for m in non_empty:
        arr = m.snapshot_ids.astype(int, copy=False)
        union_ids = arr if union_ids is None else np.union1d(union_ids, arr)

    assert union_ids is not None  # for type checkers

    # Fast path: if any instance already covers the full union, return it
    for m in non_empty:
        a = np.sort(m.snapshot_ids.astype(int, copy=False))
        if a.size == union_ids.size and np.array_equal(a, union_ids):
            return m

    # Build index map from snapshot id to position in the union
    pos = {sid: i for i, sid in enumerate(union_ids.tolist())}
    L = union_ids.size

    # Initialize accumulators with NaN
    scales = np.full(L, np.nan, dtype=float)
    reds   = np.full(L, np.nan, dtype=float)
    ages   = np.full(L, np.nan, dtype=float)
    looks  = np.full(L, np.nan, dtype=float)
    hubs   = np.full(L, np.nan, dtype=float)
    dyns   = np.full(L, np.nan, dtype=float)

    def _merge(acc: np.ndarray, src_arr: np.ndarray, idx: np.ndarray) -> None:
        """Merge finite values from src_arr into acc at positions idx where acc is NaN."""
        if src_arr.size == 0:
            return
        src = np.asarray(src_arr, dtype=float)
        mask = np.isnan(acc[idx]) & np.isfinite(src)
        if mask.any():
            acc[idx[mask]] = src[mask]

    # Fill per attribute, preferring first finite value encountered
    for m in non_empty:
        sids = m.snapshot_ids.astype(int, copy=False)
        idx = np.fromiter((pos[int(s)] for s in sids), dtype=int, count=sids.size)

        # Scale factors & redshifts
        _merge(scales, m.scale_factors if m.scale_factors.size else np.full(sids.size, np.nan), idx)
        _merge(reds,   m.redshifts     if m.redshifts.size     else np.full(sids.size, np.nan), idx)

        # Optional arrays
        _merge(ages,  m.ages           if m.ages.size           else np.full(sids.size, np.nan), idx)
        _merge(looks, m.lookback_times if m.lookback_times.size else np.full(sids.size, np.nan), idx)
        _merge(hubs,  m.hubble_epochs  if m.hubble_epochs.size  else np.full(sids.size, np.nan), idx)
        _merge(dyns,  m.dynamical_times if m.dynamical_times.size else np.full(sids.size, np.nan), idx)

    # Fill in any missing redshift from available scale factor
    need_reds = np.isnan(reds) & np.isfinite(scales)
    if need_reds.any():
        reds[need_reds] = 1.0 / scales[need_reds] - 1.0

    # Optional arrays: if no finite values, keep them empty to match class semantics
    def _empty_if_all_nan(arr: np.ndarray) -> np.ndarray:
        return np.empty(0) if (arr.size and not np.isfinite(arr).any()) else arr

    ages  = _empty_if_all_nan(ages)
    looks = _empty_if_all_nan(looks)
    hubs  = _empty_if_all_nan(hubs)

    return MomentsInTime(
        snapshot_ids=union_ids.astype(int, copy=False),
        scale_factors=scales,
        redshifts=reds,
        ages=ages,
        lookback_times=looks,
        hubble_epochs=hubs,
        dynamical_times=dyns,
    )


def get_correct_snapshot_mapping(moments: MomentsInTime, wrt: str) -> dict[int, float]:

    match wrt:
        case "scale_factor":
            return moments.snap_to_scale_factor
        case "redshift":
            return moments.snap_to_redshift
        case "age":
            return moments.snap_to_age
        case "lookback_time":
            return moments.snap_to_lookback_time
        case "hubble_epoch":
            return moments.snap_to_hubble_epoch
        case "dynamical_time":
            return moments.snap_to_dynamical_time
        case _:
            raise ValueError(f"Unknown mapping type: {wrt}")