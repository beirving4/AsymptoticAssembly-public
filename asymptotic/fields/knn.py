from __future__ import annotations

import re
import h5py
import numpy as np, pdb

from pathlib import Path
from tqdm.auto import tqdm
from typing import Generator
from collections.abc import Iterator
from collections import OrderedDict
from attrs import define, field
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from colossus.halo import profile_composite
from colossus.lss.peaks import nonLinearMass, lagrangianR


from ..utils.jackknife import (
    jackknife_resampling,
    jackknife_estimate,
    jackknife_error,
    get_jackknife_subbox_samples
)

from ..cosmo.model import Cosmology
from ..simulation.evo import EvolutionData
from ..particles.masses import get_particle_mass
from ..simulation.moments import Moment, MomentsInTime

type NNDistOutput = dict[str, dict[int, float | np.ndarray]] | tuple[dict[str, dict[int, float | np.ndarray]], dict[int, interp1d]]

@define(slots=True)
class NNDistributionStats:
    median: float
    pct5: float
    pct25: float
    pct75: float
    pct95: float

    in_comoving: bool = field(default=False)

    @classmethod
    def null_initialize(cls) -> NNDistributionStats:
        return cls(
            median=np.nan,
            pct5=np.nan,
            pct25=np.nan,
            pct75=np.nan,
            pct95=np.nan
        )
    
    @property
    def is_null(self) -> bool:
        return np.logical_and.reduce([
            self.median is np.nan,
            self.pct5 is np.nan,
            self.pct25 is np.nan,
            self.pct75 is np.nan,
            self.pct95 is np.nan
        ])
    
    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "median": self.median,
            "pct5": self.pct5,
            "pct25": self.pct25,
            "pct75": self.pct75,
            "pct95": self.pct95
        }


@define(slots=True)
class kNNDistributionStats: 
    median: dict[int, float]
    pct5: dict[int, float]
    pct25: dict[int, float]
    pct75: dict[int, float]
    pct95: dict[int, float]

    in_comoving: bool = field(default=False)

    def __getitem__(self, k: int) -> NNDistributionStats:
        return NNDistributionStats(
            median=self.median[k],
            pct5=self.pct5[k],
            pct25=self.pct25[k],
            pct75=self.pct75[k],
            pct95=self.pct95[k]
        )

    def __iter__(self) -> Generator[tuple[int, NNDistributionStats], None, None]:
        for k in self.median.keys():
            yield k, self[k]

    @classmethod
    def null_initialize(cls, max_num_neighbors: int) -> kNNDistributionStats:
        return cls(
            median={k: np.nan for k in range(1, max_num_neighbors + 1)},
            pct5={k: np.nan for k in range(1, max_num_neighbors + 1)},
            pct25={k: np.nan for k in range(1, max_num_neighbors + 1)},
            pct75={k: np.nan for k in range(1, max_num_neighbors + 1)},
            pct95={k: np.nan for k in range(1, max_num_neighbors + 1)}
        )

    @classmethod
    def from_data(
            cls,
            knn_data: dict[str, dict[int, float | np.ndarray]],
            in_comoving: bool = False
        ) -> kNNDistributionStats:
        
        return cls(
            median=knn_data["medians"],
            pct5=knn_data["pct5s"],
            pct25=knn_data["pct25s"],
            pct75=knn_data["pct75s"],
            pct95=knn_data["pct95s"],
            in_comoving=in_comoving
        )
    
    @property
    def is_null(self) -> bool:
        return all(stats.is_null for _, stats in self)
    
    @property
    def as_dict(self) -> dict[int, dict[str, float]]:
        return {k: stats.as_dict for k, stats in self}

    def display(self, k: int) -> None:
        stats = self[k]
        print(f"k={k}:")
        print(f"  Median: {stats.median}")
        print(f"  5th Percentile: {stats.pct5}")
        print(f"  25th Percentile: {stats.pct25}")
        print(f"  75th Percentile: {stats.pct75}")
        print(f"  95th Percentile: {stats.pct95}")

@define(slots=True)
class NNDistribution:
    k: int
    radii: np.ndarray
    cdf: np.ndarray
    peaked_cdf: np.ndarray
    cic_prob: np.ndarray
    stats: NNDistributionStats 

    in_comoving: bool = field(default=False)
    for_data_randoms: bool = field(default=False)

    @classmethod
    def null_initialize(cls) -> NNDistribution:
        return cls(
            k=-1,
            radii=np.empty(0),
            cdf=np.empty(0),
            peaked_cdf=np.empty(0),
            cic_prob=np.empty(0),
            stats=NNDistributionStats.null_initialize(),
        )
    
    @property
    def is_null(self) -> bool:
        return self.k == -1

    @property
    def pdf(self) -> np.ndarray:
        """
        Probability density function (PDF) as the derivative of the CDF with respect to radius.
        Returns an empty array if radii or cdf are empty or mismatched.
        """
        if self.radii is None or self.cdf is None:
            return np.array([])
        if len(self.radii) == 0 or len(self.cdf) == 0:
            return np.array([])
        if len(self.radii) != len(self.cdf):
            return np.array([])
        return np.gradient(self.cdf, self.radii)
    
    @property
    def as_dict(self) -> dict[str, float | np.ndarray]:
        return {
            "k": self.k,
            "radii": self.radii,
            "cdf": self.cdf,
            "peaked_cdf": self.peaked_cdf,
            "cic_prob": self.cic_prob,
            "stats": self.stats.as_dict
        }
    
@define(slots=True)
class NNDistributionEvo(EvolutionData):
    data: OrderedDict[int, NNDistribution]

    def __repr__(self) -> str:
        return super().__repr__()

@define(slots=True)
class NNDistributionData:
    estimate: NNDistribution
    errors: NNDistribution

    in_comoving: bool = field(default=False)
    for_data_randoms: bool = field(default=False)

    @property
    def as_dict(self) -> dict[str, float | np.ndarray]:
        return {
            "estimate": self.estimate.as_dict,
            "errors": self.errors.as_dict
        }

    def display_stats(self) -> None:
        try:
            display_stats_w_errors(self)
        except ValueError:
            display_stats(self)


@define(slots=True)
class NNDistributionEvoData(EvolutionData):
    data: OrderedDict[int, NNDistributionData]

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def estimate_evo(self) -> NNDistributionEvo:
        return NNDistributionEvo(
            moments=self.moments,
            data=OrderedDict(
                (k, v.estimate) for k, v in self.data.items()
            )
        )
    
    @property
    def errors_evo(self) -> NNDistributionEvo:
        return NNDistributionEvo(
            moments=self.moments,
            data=OrderedDict(
                (k, v.errors) for k, v in self.data.items()
            )
        )


@define(slots=True)
class kNNDistribution:
    radii: dict[int, np.ndarray]
    cdfs: dict[int, np.ndarray]
    peaked_cdfs: dict[int, np.ndarray]
    cic_probs: dict[int, np.ndarray]
    stats: kNNDistributionStats

    in_comoving: bool = field(default=False)
    for_data_randoms: bool = field(default=False)

    def __getitem__(self, k: int) -> NNDistribution:
        return NNDistribution(
            k=k,
            radii=self.radii[k],
            cdf=self.cdfs[k],
            peaked_cdf=self.peaked_cdfs[k],
            cic_prob=self.cic_probs[k-1],
            stats=self.stats[k],
            in_comoving=self.in_comoving,
            for_data_randoms=self.for_data_randoms
        )
    

    def __repr__(self) -> str:
        return (
            f"kNNDistribution(max_num_neighbors={self.max_num_neighbors}, "
            f"in_comoving={self.in_comoving})"
        )
    
    @classmethod
    def null_initialize(cls, max_num_neighbors: int) -> kNNDistribution:
        return cls(
            radii={k: np.empty(0) for k in range(1, max_num_neighbors + 1)},
            cdfs={k: np.empty(0) for k in range(1, max_num_neighbors + 1)},
            peaked_cdfs={k: np.empty(0) for k in range(1, max_num_neighbors + 1)},
            cic_probs={k: np.empty(0) for k in range(max_num_neighbors)},
            stats=kNNDistributionStats.null_initialize(max_num_neighbors),
        )

    @property
    def is_null(self) -> bool:
        return self.max_num_neighbors == 0 or self.stats.is_null
    
    @classmethod
    def from_data(
            cls, 
            knn_data: dict[str, dict[int, float | np.ndarray]],
            in_comoving: bool = False,
            for_data_randoms: bool = False
        ) -> kNNDistribution:

        return cls(
            radii=knn_data["interped_radii"],
            cdfs=knn_data["cdfs"],
            peaked_cdfs=knn_data["peaked_cdfs"],
            cic_probs=knn_data["cic_probs"],
            stats=kNNDistributionStats.from_data(knn_data),
            in_comoving=in_comoving,
            for_data_randoms=for_data_randoms
        )

    
    @property
    def vpf(self) -> np.ndarray:
        if 0 not in self.cic_probs:
            raise ValueError("CIC probabilities not computed for k=0 (VPF)")
        return self.cic_probs[0]

    @property
    def max_num_neighbors(self) -> int:
        return max(self.radii.keys())

    def convert_to_comoving(self, scale_factor: float) -> None:
        for k, r in self.radii.items():
            self.radii[k] = r / scale_factor
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        for k, r in self.radii.items():
            self.radii[k] = r * scale_factor
        self.in_comoving = False

    def display_distribution_stats(self, k: int) -> None:
        if k not in self.radii:
            raise ValueError(f"No data for k={k}")
        self.stats[k].display() 

    @property
    def as_dict(self) -> dict[str, dict[int, np.ndarray] | dict[int, float]]:
        return {
            "interped_radii": self.radii,
            "cdfs": self.cdfs,
            "peaked_cdfs": self.peaked_cdfs,
            "cic_probs": self.cic_probs,
            "medians": self.stats.median,
            "pct5s":  self.stats.pct5,
            "pct25s": self.stats.pct25,
            "pct75s": self.stats.pct75,
            "pct95s": self.stats.pct95,
        }
    

@define(slots=True)
class kNNDistributionEvo(EvolutionData): 
    data: OrderedDict[int, kNNDistribution]

    def get_nn_distribution_evo(self, k: int) -> NNDistributionEvo:
        return NNDistributionEvo(
            moments=self.moments,
            data=OrderedDict(
                (snap_idx, v[k]) for snap_idx, v in self.data.items()
                if v[k] is not None
            )
        )

@define(slots=True)
class kNNDistributionData:
    estimate: kNNDistribution
    errors: kNNDistribution

    in_comoving: bool = field(default=False)
    for_data_randoms: bool = field(default=False)

    def __getitem__(self, k: int) -> NNDistributionData:
        return NNDistributionData(
            estimate=self.estimate[k],
            errors=self.errors[k],
        )

    def __repr__(self) -> str:
        return (
            f"kNNDistributionData(max_num_neighbors={self.max_num_neighbors}, "
            f"for_dr={self.for_data_randoms}, "
            f"in_comoving={self.in_comoving}, has_errors={self.has_errors})"
        )

    def is_k_available(self, k: int) -> bool:
        return (k in self.estimate.radii) and (k in self.errors.radii)
    
    @classmethod
    def from_data(
            cls, 
            comoving_positions: np.ndarray,
            comoving_box_size: float,
            num_query_points: int, 
            max_num_neighbors: int,
            num_interp_points: int = 1000,
            sub_box_info: dict[int, list[tuple[float, float]]] | None = None,
            kd_tree: cKDTree | None = None,
            use_vectorized: bool = True, 
            run_jackknife: bool = True,
            in_comoving: bool = True,
            use_population_jk: bool = False,
            use_fast_population_jk: bool = False,
            use_data_randoms: bool = True,
        ) -> kNNDistributionData: 

        if run_jackknife and sub_box_info is None:
            raise ValueError("sub_box_info must be provided if run_jackknife is True")
        
        if kd_tree is None:
            kd_tree = get_cKDTRee(comoving_positions, comoving_box_size)

        if run_jackknife:

            knn_data = get_jackknife_nn_distributions(
                comoving_positions=comoving_positions,
                sub_box_info=sub_box_info,
                comoving_box_size=comoving_box_size,
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                kd_tree=kd_tree,
                num_interp_points=num_interp_points,
                use_vectorized=use_vectorized,
                use_population_jk=use_population_jk,
                use_fast_population_jk=use_fast_population_jk,
                use_data_randoms=use_data_randoms
            )

            estimate = kNNDistribution.from_data(knn_data["estimate"])
            errors = kNNDistribution.from_data(knn_data["errors"])

        else:

            knn_data = get_nn_distributions(
                kd_tree=kd_tree,
                box_size=comoving_box_size,
                num_query_points=num_query_points,
                max_num_neighbors=max_num_neighbors,
                num_interp_points=num_interp_points,
                use_vectorized=use_vectorized,
                use_data_randoms=use_data_randoms
            )

            estimate = kNNDistribution.from_data(knn_data)
            errors = kNNDistribution.null_initialize(max_num_neighbors)

        return cls(estimate=estimate, errors=errors, in_comoving=in_comoving)


    def convert_to_comoving(self, scale_factor: float) -> None:
        self.estimate.convert_to_comoving(scale_factor)
        self.errors.convert_to_comoving(scale_factor)
        self.in_comoving = True

    def convert_to_physical(self, scale_factor: float) -> None:
        self.estimate.convert_to_physical(scale_factor)
        self.errors.convert_to_physical(scale_factor)
        self.in_comoving = False

    @property
    def max_num_neighbors(self) -> int:
        return self.estimate.max_num_neighbors
    
    @property
    def has_errors(self) -> bool:
        return not self.errors.is_null
    
    def get_two_point_radial_bounds(
            self, 
            target_pcdf_value: float,
            for_data_randoms: bool = True
        ) -> tuple[float, float]:
        
        try:
            data = self[2 if for_data_randoms else 1].estimate
        except KeyError as e:
            raise KeyError("k=2 data not available for radial bounds computation") from e
        
        r_min, r_max = get_two_point_radial_bounds(
            radii=data.radii,
            peaked_cdf=data.peaked_cdf,
            target_pcdf_value=target_pcdf_value,
        )

        lower_bound = np.nanmax([r_min, data.radii[0]])
        upper_bound = np.nanmin([r_max, data.radii[-1]])

        return lower_bound, upper_bound

    def get_estimated_two_point_mass_range(
            self,
            target_pcdf_value: float,
            cosmo: Cosmology,
            use_uniform_sphere: bool = False,
            for_data_randoms: bool = True
        ) -> tuple[float, float]:

        lower_bound, upper_bound = self.get_two_point_radial_bounds(
            target_pcdf_value=target_pcdf_value,
            for_data_randoms=for_data_randoms
        )

        return get_estimated_two_point_mass_range(
            r_min=lower_bound,
            r_max=upper_bound,
            cosmo=cosmo,
            use_uniform_sphere=use_uniform_sphere
        )

    def get_estimated_two_point_particle_range(
            self,
            target_pcdf_value: float,
            cosmo: Cosmology,
            sim_particle_count: int,
            sim_box_size: int,
            use_uniform_sphere: bool = False
        ) -> tuple[int, int]:

        min_mass, max_mass = self.get_estimated_two_point_mass_range(
            target_pcdf_value=target_pcdf_value,
            cosmo=cosmo,
            use_uniform_sphere=use_uniform_sphere
        )

        particle_mass = get_particle_mass(
            cosmo_name=cosmo.name,
            box_size=sim_box_size,
            num_particles=sim_particle_count
        )

        return (int(min_mass / particle_mass), int(max_mass / particle_mass))
    
    @property
    def as_dict(self) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
        return {
            "estimate": self.estimate.as_dict,
            "errors":  self.errors.as_dict,
        }

    def save(self, filepath: Path) -> None:
        save_knn_distribution_data(filepath, self.as_dict)


    @classmethod
    def null_initialize(cls, max_num_neighbors: int) -> kNNDistributionData:
        return cls(
            estimate=kNNDistribution.null_initialize(max_num_neighbors),
            errors=kNNDistribution.null_initialize(max_num_neighbors)
        )

    @classmethod
    def load(cls, filepath: Path) -> kNNDistributionData:
        knn_data_dict = load_knn_distribution_data(filepath)

        # Extract inferred pair-type flag (default True -> DR for legacy files)
        for_dr = bool(knn_data_dict.pop("_for_data_randoms", True))

        estimate = kNNDistribution.from_data(
            knn_data_dict["estimate"],
            for_data_randoms=for_dr
        )
        errors   = kNNDistribution.from_data(
            knn_data_dict["errors"],
            for_data_randoms=for_dr
        )

        return cls(estimate=estimate, errors=errors, for_data_randoms=for_dr)
    

@define(slots=True)
class kNNDistributionEvoData(EvolutionData):
    data: OrderedDict[int, kNNDistributionData]

    def __repr__(self) -> str:
        return super().__repr__()


    @property
    def for_data_randoms(self) -> bool:
        # Assume all snapshots have the same flag
        first_key = next(iter(self.data))
        return self.data[first_key].for_data_randoms

    def get_nn_estimate_evo(self, k: int) -> kNNDistributionEvo:
        return kNNDistributionEvo(
            moments=self.moments,
            data=OrderedDict(
                (snap_idx, v[k].estimate) for snap_idx, v in self.data.items()
                if v[k] is not None
            )
        )  

    def get_nn_errors_evo(self, k: int) -> kNNDistributionEvo:
        return kNNDistributionEvo(
            moments=self.moments,
            data=OrderedDict(
                (snap_idx, v[k].errors) for snap_idx, v in self.data.items()
                if v[k] is not None
            )
        ) 
    
    @classmethod
    def load(
            cls, 
            directory: Path, 
            moments: MomentsInTime,
            sim_cosmo: Cosmology,
            for_data_data: bool = True,
            for_jackknifed_file: bool = False
        ) -> kNNDistributionEvoData:

        if not directory.is_dir():
            raise ValueError(f"Directory {directory} does not exist")

        data = {}
        new_moments = MomentsInTime()

        # Accept both: ..._snap005_k2.hdf5 and ..._snap005_k2_jackknifed.hdf5
        pat = re.compile(r"_snap(\d+)_k(\d+)(?:_jackknifed)?\.hdf5$")

        for filepath in directory.glob("*.hdf5"):
            name = filepath.name

            # If the flag is True, only load the `_jackknifed` files;
            # if False, explicitly skip them.
            if (
                for_jackknifed_file
                and (not name.endswith("_jackknifed.hdf5"))
                or (not for_jackknifed_file)
                and (name.endswith("_jackknifed.hdf5"))
            ):
                continue

            # If the flag is True, only load the files with `_dd_knns_` in the name;
            # if False, explicitly skip them.
            if (
                (for_data_data and ("_dr_knns_" in name))
                or (not for_data_data and ("_dd_knns_" in name))
            ):
                continue

            m = pat.search(name)
            if not m:
                # Name doesn't match expected pattern; skip quietly
                continue

            # group(1) is the numeric snapshot id from `_snap###`
            snapshot_id = int(m[1])

            try:
                new_moments.add_moment(moments[snapshot_id])
            except IndexError:
                print(f"Warning: Snapshot ID {snapshot_id} not found in provided moments.")
                continue

            data[snapshot_id] = kNNDistributionData.load(filepath)

        new_moments.add_times(sim_cosmo)

        return cls(moments=new_moments, data=OrderedDict(sorted(data.items())))

    
@define(slots=True)
class JointNNDistribution:
    k: int
    radii_a: np.ndarray
    radii_b: np.ndarray
    cdf: np.ndarray
    peaked_cdf: np.ndarray
    cic_prob: np.ndarray
    ... 


def wrap_to_box(x: np.ndarray, L: float, atol: float = 1e-12) -> np.ndarray:
    """
    Normalize coordinates to [0, L) with numerical safety.
    Works for 1D or ND arrays (last axis are coords).
    """
    # 1) modulo wrap (handles negatives and >= L)
    y = np.mod(x, L)

    # 2) values numerically equal to L -> 0 (open interval)
    # use <= to catch tiny overshoots like L + 1e-16 after mod due to fp error
    y = np.where(y >= L - atol, 0.0, y)
    return y.astype(np.float64, copy=False)

def get_cKDTRee(comoving_positions: np.ndarray, comoving_box_size: float) -> cKDTree:
    comoving_positions = wrap_to_box(comoving_positions, comoving_box_size)
    return cKDTree(comoving_positions)


def pick_highest_k_per_snapshot(
        knn_dir: Path, for_jackknifed_file: bool = False
    ) -> dict[int, Path]:
    """
    Return {snapshot_id: filepath_with_largest_k} for files like
    ..._snap010_k2.hdf5 in `knn_dir`.
    """
    pat = re.compile(r"_snap(\d+)_k(\d+)(?:_jackknifed)?\.hdf5$")
    best: dict[int, tuple[int, Path]] = {}

    for fp in knn_dir.glob("*.hdf5"):
        name = fp.name

        # Filter by suffix according to the flag
        if (
            for_jackknifed_file
            and (not name.endswith("_jackknifed.hdf5"))
            or (not for_jackknifed_file)
            and (name.endswith("_jackknifed.hdf5"))
        ):
            continue

        m = pat.search(name)
        if not m:
            continue
        snap, kval = int(m[1]), int(m[2])
        cur = best.get(snap)
        if (cur is None) or (kval > cur[0]):
            best[snap] = (kval, fp)

    # flatten to {snap: path}
    return {snap: path for snap, (_, path) in best.items()}



def get_vectorized_queried_radii(
        kd_tree: cKDTree,
        query_points: np.ndarray,
        max_num_neighbors: int,
    ) -> np.ndarray:
    
    distances, _ = kd_tree.query(
        x=query_points,
        k=max_num_neighbors,
        workers=-1,
    )
    return np.vstack(distances)

def get_queried_radii(
        kd_tree: cKDTree,
        query_points: np.ndarray,
        max_num_neighbors: int,
    ) -> np.ndarray:
    
    queried_radii = []
    for query_point in tqdm(query_points, desc="Querying KDTree"):
        distances, _ = kd_tree.query(
            x=query_point,
            k=max_num_neighbors,
            workers=-1,
        )
        queried_radii.append(distances)
    return np.vstack(queried_radii)

def _sorted_kth_radii(query_radii: np.ndarray, k: int) -> np.ndarray:
    """Return the sorted radii for the k-th nearest neighbor (1-indexed k)."""
    if k < 1 or k > query_radii.shape[1]:
        raise ValueError("k is out of bounds for query_radii")
    return np.sort(query_radii[:, k - 1])

def _make_rgrid(sorted_radii: np.ndarray, num_interp_points: int) -> np.ndarray:
    """Build a log-spaced radius grid spanning the sorted radii range."""
    rmin = float(sorted_radii[0])
    rmax = float(sorted_radii[-1])
    if rmin <= 0 or not np.isfinite(rmin) or not np.isfinite(rmax):
        rmin = np.finfo(float).tiny
    return np.logspace(np.log10(rmin), np.log10(rmax), num=num_interp_points)

def _build_cdf_interp(sorted_radii: np.ndarray) -> interp1d:
    """Empirical CDF interpolator for the provided sorted radii."""
    n = sorted_radii.size
    raw_cdf = np.arange(1, n + 1, dtype=float) / float(n)
    return interp1d(
        x=sorted_radii, y=raw_cdf, kind="linear",
        bounds_error=False, fill_value=(0.0, 1.0), assume_sorted=True,
    )

def _evaluate_distributions(rgrid: np.ndarray, cdf_interp: interp1d) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate CDF and peaked-CDF on a provided radius grid."""
    cdf = np.asarray(cdf_interp(rgrid), dtype=float)
    peaked = np.where(cdf <= 0.5, cdf, 1.0 - cdf)
    return cdf, peaked

def _compute_summary_from_cdf(rgrid: np.ndarray, cdf: np.ndarray) -> dict[str, float]:
    """Compute quantile radii from a CDF sampled on rgrid (uses inversion)."""
    r = np.asarray(rgrid, dtype=float)
    c = np.asarray(cdf, dtype=float)
    c = np.clip(c, 0.0, 1.0)
    keep = np.r_[True, np.diff(c) > 0]   # drop flats so interp x increases
    r_m = r[keep]; c_m = c[keep]
    
    if r_m.size < 2:
        return {
            "pct5s": np.nan, 
            "pct25s": np.nan, 
            "medians": np.nan, 
            "pct75s": np.nan, 
            "pct95s": np.nan
        }
    
    inv = interp1d(
        c_m, r_m, 
        kind="linear", 
        bounds_error=False, 
        fill_value=(np.nan, np.nan), 
        assume_sorted=True
    )
    probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    vals = inv(probs)

    return {
        "pct5s": float(vals[0]), 
        "pct25s": float(vals[1]), 
        "medians": float(vals[2]),
        "pct75s": float(vals[3]), 
        "pct95s": float(vals[4]),
    }

def _compute_cic_for_index(
        m: int, rgrid: np.ndarray, cdf_interps: dict[int, interp1d]
    ) -> np.ndarray:
    """Compute CIC probability for index m (0-based: m=0 is VPF) on a given r-grid."""
    rgrid = np.asarray(rgrid, dtype=float)
    if m == 0:
        if 1 not in cdf_interps:
            return np.full_like(rgrid, np.nan, dtype=float)
        return 1.0 - np.asarray(cdf_interps[1](rgrid), dtype=float)
    k = m
    if (k not in cdf_interps) or ((k + 1) not in cdf_interps):
        return np.full_like(rgrid, np.nan, dtype=float)
    cdf_k  = np.asarray(cdf_interps[k](rgrid), dtype=float)
    cdf_k1 = np.asarray(cdf_interps[k + 1](rgrid), dtype=float)
    return cdf_k - cdf_k1


def initialize_nn_dist_dict() -> dict[str, dict[int, float | np.ndarray]]:
    return {
        "interped_radii": {},
        "cdfs": {},
        "peaked_cdfs": {},
        "medians": {},
        "pct5s": {},
        "pct25s": {},
        "pct75s": {},
        "pct95s": {},
        "cic_probs": {},
    }

def get_dr_nn_query_points(box_size: float, num_query_points: int) -> np.ndarray:
    return np.random.uniform(
        low=0.0, 
        high=box_size, 
        size=(num_query_points, 3)
    )

def get_dd_nn_query_points(
        data_positions: np.ndarray,
        num_query_points: int,
    ) -> np.ndarray:
    assert data_positions.shape[0] >= num_query_points, (
        f"Not enough data points to sample from "
        f"({data_positions.shape[0]} < {num_query_points})" 
    )
    if num_query_points == data_positions.shape[0]:
        return data_positions.copy()
    
    sample_idx = np.random.choice(
        data_positions.shape[0], 
        size=num_query_points, 
        replace=False
    )
    return data_positions[sample_idx]

def get_query_points(
        data_positions: np.ndarray | None,
        box_size: float,
        num_query_points: int,
        use_data_randoms: bool = True,
    ) -> np.ndarray:
    if use_data_randoms and (data_positions is not None):
        return get_dd_nn_query_points(
            data_positions=data_positions, 
            num_query_points=num_query_points
        )
    else:
        return get_dr_nn_query_points(
            box_size=box_size, 
            num_query_points=num_query_points
        )

def get_nn_distributions(
        kd_tree: cKDTree,
        box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        query_points: np.ndarray | None = None,
        return_interps: bool = False,
        use_data_randoms: bool = True,
    ) -> NNDistOutput:
    """
    Compute k-NN CDF/PCDF/CIC distributions for k=1..max_num_neighbors.
    Returns nn_dist_dict, and optionally the in-memory CDF interpolators when return_interps=True.
    """
    if query_points is None:
        query_points = get_query_points(
            data_positions=kd_tree.data if use_data_randoms else None,
            box_size=box_size,
            num_query_points=num_query_points,
            use_data_randoms=use_data_randoms,
        )
    assert query_points.shape == (num_query_points, 3), "Invalid query_points shape"

    if use_vectorized:
        query_radii = get_vectorized_queried_radii(
            kd_tree=kd_tree, 
            query_points=query_points, 
            max_num_neighbors=max_num_neighbors
        )
    else:
        query_radii = get_queried_radii(
            kd_tree=kd_tree, 
            query_points=query_points, 
            max_num_neighbors=max_num_neighbors
        )

    nn_dist_dict = initialize_nn_dist_dict()
    cdf_interps: dict[int, interp1d] = {}

    # Per-k CDFs/interps and summaries
    for k in range(1, max_num_neighbors + 1):
        sorted_k = _sorted_kth_radii(query_radii, k)
        rgrid_k  = _make_rgrid(sorted_k, num_interp_points)
        cdf_k_i  = _build_cdf_interp(sorted_k)
        cdf_interps[k] = cdf_k_i

        nn_dist_dict['interped_radii'][k] = rgrid_k
        cdf_k, pcdf_k = _evaluate_distributions(rgrid_k, cdf_k_i)
        nn_dist_dict['cdfs'][k] = cdf_k
        nn_dist_dict['peaked_cdfs'][k] = pcdf_k

        stats = _compute_summary_from_cdf(rgrid_k, cdf_k)
        nn_dist_dict['medians'][k] = stats['medians']
        nn_dist_dict['pct5s'][k]   = stats['pct5s']
        nn_dist_dict['pct25s'][k]  = stats['pct25s']
        nn_dist_dict['pct75s'][k]  = stats['pct75s']
        nn_dist_dict['pct95s'][k]  = stats['pct95s']

    # CIC on aligned grids
    for m in range(max_num_neighbors):
        rgrid = nn_dist_dict['interped_radii'][m + 1]
        nn_dist_dict['cic_probs'][m] = _compute_cic_for_index(m, rgrid, cdf_interps)

    return (nn_dist_dict, cdf_interps) if return_interps else nn_dist_dict


def _init_jk_output(
        full_nn: dict[str, dict[int, float | np.ndarray]], 
        max_k: int
    ) -> tuple[dict[str, dict[int, float | np.ndarray]], dict[str, dict[int, float | np.ndarray]]]:
    """Initialize separate estimate and error dicts using full-sample r-grids."""
    est = initialize_nn_dist_dict()
    err = initialize_nn_dist_dict()

    # copy r-grids from the full run so all folds align
    for k in range(1, max_k + 1):
        est["interped_radii"][k] = full_nn["interped_radii"][k]
        err["interped_radii"][k] = full_nn["interped_radii"][k]
    return est, err


def _init_fold_containers(max_k: int) -> tuple[dict, ...]:
    """Allocate per-fold containers for curves and scalar stats."""
    cdf_folds  = {k: [] for k in range(1, max_k + 1)}
    pcdf_folds = {k: [] for k in range(1, max_k + 1)}
    cic_folds  = {m: [] for m in range(max_k)}

    scalar_folds = {
        "medians": {k: [] for k in range(1, max_k + 1)},
        "pct5s":   {k: [] for k in range(1, max_k + 1)},
        "pct25s":  {k: [] for k in range(1, max_k + 1)},
        "pct75s":  {k: [] for k in range(1, max_k + 1)},
        "pct95s":  {k: [] for k in range(1, max_k + 1)},
    }
    return cdf_folds, pcdf_folds, cic_folds, scalar_folds


def _init_jk_args(
        full_nn: dict[str, dict[int, float | np.ndarray]], 
        max_k: int
    ) -> tuple[dict, ...]:

    est, err = _init_jk_output(full_nn, max_k)
    cdf_folds, pcdf_folds, cic_folds, scalar_folds = _init_fold_containers(max_k)
    
    return est, err, cdf_folds, pcdf_folds, cic_folds, scalar_folds


def _accumulate_fold_results(
        interps: dict[int, interp1d],
        out_rgrids: dict[int, np.ndarray],
        max_k: int,
        cdf_folds: dict[int, list[np.ndarray]],
        pcdf_folds: dict[int, list[np.ndarray]],
        cic_folds: dict[int, list[np.ndarray]],
        scalar_folds: dict[str, dict[int, list[np.ndarray]]],
    ) -> None:
    """Evaluate CDF/PCDF/CIC on shared grids for a single fold and append to containers."""
    # per-k curves and per-fold scalar stats
    for k in range(1, max_k + 1):
        rgrid = out_rgrids[k]
        cdf_k = np.clip(np.asarray(interps[k](rgrid), dtype=float), 0.0, 1.0)
        pcdf_k = np.where(cdf_k <= 0.5, cdf_k, 1.0 - cdf_k)
        cdf_folds[k].append(cdf_k)
        pcdf_folds[k].append(pcdf_k)

        stats_k = _compute_summary_from_cdf(rgrid, cdf_k)
        scalar_folds["medians"][k].append(np.array([stats_k["medians"]], dtype=float))
        scalar_folds["pct5s"][k].append(np.array([stats_k["pct5s"]], dtype=float))
        scalar_folds["pct25s"][k].append(np.array([stats_k["pct25s"]], dtype=float))
        scalar_folds["pct75s"][k].append(np.array([stats_k["pct75s"]], dtype=float))
        scalar_folds["pct95s"][k].append(np.array([stats_k["pct95s"]], dtype=float))

    # per-m CIC on r-grid of k=m+1
    for m in range(max_k):
        rgrid_m = out_rgrids[m + 1]
        if m == 0:
            cic_m = 1.0 - np.asarray(interps[1](rgrid_m), dtype=float)
        else:
            cdf_m   = np.asarray(interps[m](rgrid_m), dtype=float)
            cdf_mp1 = np.asarray(interps[m + 1](rgrid_m), dtype=float)
            cic_m = cdf_m - cdf_mp1
        cic_folds[m].append(cic_m)


def _finalize_curve_estimates(
        est: dict[str, dict[int, float | np.ndarray]],
        err: dict[str, dict[int, float | np.ndarray]],
        max_k: int,
        cdf_folds: dict[int, list[np.ndarray]],
        pcdf_folds: dict[int, list[np.ndarray]],
        cic_folds: dict[int, list[np.ndarray]],
    ) -> None:
    """Compute jackknife estimates & errors for CDF/PCDF/CIC curves and write into est/err."""
    for k in range(1, max_k + 1):
        est["cdfs"][k]      = jackknife_estimate(cdf_folds[k])
        err["cdfs"][k]      = jackknife_error(cdf_folds[k])
        est["peaked_cdfs"][k] = jackknife_estimate(pcdf_folds[k])
        err["peaked_cdfs"][k] = jackknife_error(pcdf_folds[k])
    for m in range(max_k):
        est["cic_probs"][m] = jackknife_estimate(cic_folds[m])
        err["cic_probs"][m] = jackknife_error(cic_folds[m])


def _finalize_stats_from_cdf(
        est: dict[str, dict[int, float | np.ndarray]],
        err: dict[str, dict[int, float | np.ndarray]],
        max_k: int,
        scalar_folds: dict[str, dict[int, list[np.ndarray]]],
    ) -> None:
    """Compute stats from the JK-estimated CDF and JK errors from per-fold scalar stats."""
    for k in range(1, max_k + 1):
        rgrid = est["interped_radii"][k]
        stats_from_est = _compute_summary_from_cdf(rgrid, est["cdfs"][k])
        est["medians"][k] = stats_from_est["medians"]
        est["pct5s"][k]   = stats_from_est["pct5s"]
        est["pct25s"][k]  = stats_from_est["pct25s"]
        est["pct75s"][k]  = stats_from_est["pct75s"]
        est["pct95s"][k]  = stats_from_est["pct95s"]

        err["medians"][k] = float(jackknife_error(scalar_folds["medians"][k])[0])
        err["pct5s"][k]   = float(jackknife_error(scalar_folds["pct5s"][k])[0])
        err["pct25s"][k]  = float(jackknife_error(scalar_folds["pct25s"][k])[0])
        err["pct75s"][k]  = float(jackknife_error(scalar_folds["pct75s"][k])[0])
        err["pct95s"][k]  = float(jackknife_error(scalar_folds["pct95s"][k])[0])

def _get_cdf_interps_only(
        kd_tree: cKDTree,
        query_points: np.ndarray,
        max_num_neighbors: int,
        use_vectorized: bool = True,
    ) -> dict[int, interp1d]:
    """
    Return only the empirical CDF interpolators for k=1..max_num_neighbors,
    avoiding r-grid construction and curve evaluation.
    """
    if use_vectorized:
        query_radii = get_vectorized_queried_radii(
            kd_tree=kd_tree,
            query_points=query_points,
            max_num_neighbors=max_num_neighbors,
        )
    else:
        query_radii = get_queried_radii(
            kd_tree=kd_tree,
            query_points=query_points,
            max_num_neighbors=max_num_neighbors,
        )

    interps: dict[int, interp1d] = {}
    for k in range(1, max_num_neighbors + 1):
        sorted_k = _sorted_kth_radii(query_radii, k)
        interps[k] = _build_cdf_interp(sorted_k)
    return interps



def get_query_jackknife_nn_distributions(
        comoving_positions: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],  # {1: [(x_i,x_f),(y_i,y_f),(z_i,z_f)], ...}
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        kd_tree: cKDTree | None = None,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        use_data_randoms: bool = True,
    ) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """
    Compute jackknife estimates and errors for k-NN distributions using leave-one-subbox-out resamples (query-based JK).
    Returns a dict with keys "estimate" and "errors" containing the respective output dicts.
    """

    if kd_tree is None:
        kd_tree = cKDTree(comoving_positions, boxsize=comoving_box_size)

    # Master set of random query points across the full volume
    query_points = get_query_points(
        data_positions=comoving_positions,
        box_size=comoving_box_size,
        num_query_points=num_query_points,
        use_data_randoms=use_data_randoms,
    )
    # query_points = np.random.uniform(low=0.0, high=comoving_box_size, size=(num_query_points, 3))

    # Partition query points into sub-box samples
    query_samples = get_jackknife_subbox_samples(
        sub_box_info=sub_box_info,
        coordinates=query_points,
        return_as_mask=False,
    )

    # --- Full sample only to define the shared r-grids per k ---
    full_nn = get_nn_distributions(
        kd_tree=kd_tree,
        box_size=comoving_box_size,
        num_query_points=num_query_points,
        max_num_neighbors=max_num_neighbors,
        num_interp_points=num_interp_points,
        use_vectorized=use_vectorized,
        query_points=query_points,
        return_interps=False,   # no need for interps here
    )

    # Initialize output and per-fold containers
    (
        est, err, cdf_folds, pcdf_folds, cic_folds, scalar_folds
    ) = _init_jk_args(full_nn, max_num_neighbors)

    # Evaluate each JK fold on the shared grids (build only the interps per fold)
    for qp_resample in query_samples:
        interps = _get_cdf_interps_only(
            kd_tree=kd_tree,
            query_points=qp_resample,
            max_num_neighbors=max_num_neighbors,
            use_vectorized=use_vectorized,
        )
        _accumulate_fold_results(
            interps=interps,
            out_rgrids=est["interped_radii"],
            max_k=max_num_neighbors,
            cdf_folds=cdf_folds,
            pcdf_folds=pcdf_folds,
            cic_folds=cic_folds,
            scalar_folds=scalar_folds,
        )

    # Finalize curves and stats
    _finalize_curve_estimates(
        est=est, err=err, max_k=max_num_neighbors,
        cdf_folds=cdf_folds, pcdf_folds=pcdf_folds, cic_folds=cic_folds
    )
    _finalize_stats_from_cdf(
        est=est, err=err, max_k=max_num_neighbors, scalar_folds=scalar_folds
    )

    return {"estimate": est, "errors": err}


def get_population_jackknife_nn_distributions(
        comoving_positions: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        kd_tree: cKDTree | None = None,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        use_data_randoms: bool = True,
    ) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """
    Compute jackknife estimates and errors for k-NN distributions using leave-one-subbox-out resamples (population-based JK).
    Returns a dict with keys "estimate" and "errors" containing the respective output dicts.
    """
    # Build KDTree for full sample (for defining shared r-grids)
    if kd_tree is None:
        kd_tree_full = cKDTree(comoving_positions, boxsize=comoving_box_size)
    else:
        kd_tree_full = kd_tree

    # Master set of random query points across the full volume (shared for all folds)
    query_points = get_query_points(
        data_positions=comoving_positions,
        box_size=comoving_box_size,
        num_query_points=num_query_points,
        use_data_randoms=use_data_randoms,
    )

    # --- Full sample only to define the shared r-grids per k ---
    full_nn = get_nn_distributions(
        kd_tree=kd_tree_full,
        box_size=comoving_box_size,
        num_query_points=num_query_points,
        max_num_neighbors=max_num_neighbors,
        num_interp_points=num_interp_points,
        use_vectorized=use_vectorized,
        query_points=query_points,
        return_interps=False,
    )

    # Initialize output and per-fold containers
    est, err = _init_jk_output(full_nn, max_num_neighbors)
    cdf_folds, pcdf_folds, cic_folds, scalar_folds = _init_fold_containers(max_num_neighbors)

    # For each sub-box, remove the population inside that sub-box and build a KDTree
    for sub_box_bounds in sub_box_info.values():
        # sub_box_bounds: [(x0,x1),(y0,y1),(z0,z1)]
        x0, x1 = sub_box_bounds[0]
        y0, y1 = sub_box_bounds[1]
        z0, z1 = sub_box_bounds[2]
        pos = comoving_positions
        keep_mask = (
            (pos[:, 0] < x0) | (pos[:, 0] > x1) |
            (pos[:, 1] < y0) | (pos[:, 1] > y1) |
            (pos[:, 2] < z0) | (pos[:, 2] > z1)
        )
        # Points outside the sub-box are kept
        kd_tree_fold = cKDTree(comoving_positions[keep_mask], boxsize=comoving_box_size)
        interps = _get_cdf_interps_only(
            kd_tree=kd_tree_fold,
            query_points=query_points,
            max_num_neighbors=max_num_neighbors,
            use_vectorized=use_vectorized,
        )
        _accumulate_fold_results(
            interps=interps,
            out_rgrids=est["interped_radii"],
            max_k=max_num_neighbors,
            cdf_folds=cdf_folds,
            pcdf_folds=pcdf_folds,
            cic_folds=cic_folds,
            scalar_folds=scalar_folds,
        )

    # Finalize curves and stats
    _finalize_curve_estimates(
        est=est, err=err, max_k=max_num_neighbors,
        cdf_folds=cdf_folds, pcdf_folds=pcdf_folds, cic_folds=cic_folds
    )
    _finalize_stats_from_cdf(
        est=est, err=err, max_k=max_num_neighbors, scalar_folds=scalar_folds
    )

    return {"estimate": est, "errors": err}

def get_fast_population_jackknife_nn_distributions(
        comoving_positions: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        kd_tree: cKDTree | None = None,
        use_vectorized: bool = True,
        use_data_randoms: bool = True,
    ) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """
    Fast approximation to population jackknife:
    - Builds a single KDTree on the full population.
    - Uses a shared set of query points.
    - For each fold, skips query points inside the omitted sub-box,
      approximating the effect of removing that region's population.
    - Cheaper than full population JK, but underestimates variance slightly.
    """

    if kd_tree is None:
        kd_tree = cKDTree(comoving_positions, boxsize=comoving_box_size)

    # Shared query points across all folds
    query_points = get_query_points(
        data_positions=comoving_positions,
        box_size=comoving_box_size,
        num_query_points=num_query_points,
        use_data_randoms=use_data_randoms,
    )

    fold_results = {}

    # Compute once: cdf interpolators from full population for shared r-grid
    master_interps, r_grid = _get_cdf_interps_only(
        kd_tree,
        query_points,
        max_num_neighbors=max_num_neighbors,
        use_vectorized=use_vectorized,
    )

    def get_keep_mask(bounds: list[tuple[float, float]]) -> np.ndarray:
        (x0, x1), (y0, y1), (z0, z1) = bounds
        in_box_mask = (
            (query_points[:, 0] >= x0) & (query_points[:, 0] < x1) &
            (query_points[:, 1] >= y0) & (query_points[:, 1] < y1) &
            (query_points[:, 2] >= z0) & (query_points[:, 2] < z1)
        )
        return ~in_box_mask

    for bounds in sub_box_info.values():
        # Determine which query points are outside the excluded sub-box
        keep_mask = get_keep_mask(bounds)
        fold_interps = {k: v[keep_mask] for k, v in master_interps.items()}
        
        _accumulate_fold_results(fold_results, fold_interps)

    curve_estimates = _finalize_curve_estimates(fold_results)
    return _finalize_stats_from_cdf(curve_estimates, r_grid)


def get_jackknife_nn_distributions(
        comoving_positions: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        kd_tree: cKDTree | None = None,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        use_population_jk: bool = False,
        use_fast_population_jk: bool = False,
        use_data_randoms: bool = True,
    ) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """
    Main jackknife API. Three modes:
    - Default (population_jk=False, fast_population_jk=False): query-based JK.
    - population_jk=True: population-based JK (rebuilds a KDTree per fold).
    - fast_population_jk=True: approximate population JK, skips query points
      in omitted sub-boxes using a single KDTree.
    """

    if use_fast_population_jk:
        return get_fast_population_jackknife_nn_distributions(
            comoving_positions,
            sub_box_info,
            comoving_box_size,
            num_query_points,
            max_num_neighbors,
            kd_tree=kd_tree,
            use_vectorized=use_vectorized,
            use_data_randoms=use_data_randoms
        )
    elif use_population_jk:
        return get_population_jackknife_nn_distributions(
            comoving_positions,
            sub_box_info,
            comoving_box_size,
            num_query_points,
            max_num_neighbors,
            kd_tree=kd_tree,
            num_interp_points=num_interp_points,
            use_vectorized=use_vectorized,
            use_data_randoms=use_data_randoms
        )
    else:
        return get_query_jackknife_nn_distributions(
            comoving_positions,
            sub_box_info,
            comoving_box_size,
            num_query_points,
            max_num_neighbors,
            kd_tree=kd_tree,
            num_interp_points=num_interp_points,
            use_vectorized=use_vectorized,
            use_data_randoms=use_data_randoms
        )


def _validate_target(pc: np.ndarray, target: float) -> bool:
    """Return True if target is attainable by the PCDF (between its min and peak)."""
    if not np.isfinite(target):
        return False
    peak_val = np.nanmax(pc)
    pc_min = np.nanmin(pc)
    return (pc_min <= target <= peak_val)

def _peak_span_and_value(pc: np.ndarray) -> tuple[int, int, float]:
    """
    Return (i_left, i_right, peak_val) for the plateau of maximal PCDF.
    If the peak is a single point, i_left == i_right.
    """
    peak_val = np.nanmax(pc)
    peak_mask = (pc == peak_val)
    idx = np.where(peak_mask)[0]
    return int(idx[0]), int(idx[-1]), float(peak_val)

def _build_inverse_on_side(
        r: np.ndarray, pc: np.ndarray, start: int, stop: int, *, increasing: bool
    ) -> interp1d:
    """
    Build an interpolator mapping PCDF -> radius on one monotone side.
    Slice is inclusive of 'stop'. If 'increasing' is False, reverse arrays so
    the PCDF passed to interp1d is increasing. Plateaus are dropped.
    """
    # inclusive slice
    r_side = r[start:stop + 1]
    pc_side = pc[start:stop + 1]

    if not increasing:
        r_side = r_side[::-1]
        pc_side = pc_side[::-1]

    # Drop flats so x for interp1d is strictly increasing
    keep = np.r_[True, np.diff(pc_side) > 0]
    r_side = r_side[keep]
    pc_side = pc_side[keep]

    return interp1d(
        pc_side, r_side, kind="linear",
        bounds_error=False, fill_value=np.nan, assume_sorted=True
    )

def get_two_point_radial_bounds(
        radii: np.ndarray, 
        peaked_cdf: np.ndarray, 
        target_pcdf_value: float
    ) -> tuple[float, float]:
    """
    Return (r_low, r_high) such that PCDF(r_low) = PCDF(r_high) = pctile_bound.
    Assumes PCDF is unimodal: rises to a single peak then falls.
    - radii: sorted ascending
    - peaked_cdf: values aligned with radii
    - target_pcdf_value: a PCDF value in [0, 0.5] (typical), but any value up to the peak works.
    Returns (nan, nan) if the target is unattainable.
    """
    r = np.asarray(radii, dtype=float)
    pc = np.asarray(peaked_cdf, dtype=float)

    if r.ndim != 1 or pc.ndim != 1 or r.size != pc.size or r.size == 0:
        raise ValueError("Inputs must be 1D arrays of equal nonzero length.")

    if not _validate_target(pc, target_pcdf_value):
        return float('nan'), float('nan')

    i_peak_l, i_peak_r, peak_val = _peak_span_and_value(pc)

    # If the target is the peak, return the plateau radii
    if np.isclose(target_pcdf_value, peak_val):
        return float(r[i_peak_l]), float(r[i_peak_r])

    # Left side: strictly increasing up to the left edge of the peak
    f_left = _build_inverse_on_side(r, pc, start=0, stop=i_peak_l, increasing=True)
    # Right side: strictly decreasing after the right edge of the peak
    f_right = _build_inverse_on_side(r, pc, start=i_peak_r, stop=len(r)-1, increasing=False)

    r_low = float(f_left(target_pcdf_value))
    r_high = float(f_right(target_pcdf_value))

    return r_low, r_high

def get_joint_nn_distribution(
        comoving_positions_a: np.ndarray,
        comoving_positions_b: np.ndarray,
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        query_points: np.ndarray | None = None,
        return_interps: bool = False,
    ) -> NNDistOutput:
    """
    Compute joint k-NN distributions for two catalogs A and B over a *common* set of
    query points in a periodic box. For each k, the joint CDF over radius r is
        P( r ≥ max(r_k^A, r_k^B) ),
    i.e., the probability that a sphere of radius r centered on a random point
    contains at least k neighbors from A *and* at least k neighbors from B.

    Parameters
    ----------
    comoving_positions_a, comoving_positions_b : (N,3) float arrays
        Particle positions in comoving units within [0, L)^3.
    comoving_box_size : float
        Periodic box size L (same for both catalogs).
    num_query_points : int
        Number of random query points for the empirical distribution.
    max_num_neighbors : int
        Maximum k for the kNN statistics.
    num_interp_points : int, optional
        Number of radii points in the log-spaced evaluation grid per k.
    use_vectorized : bool, optional
        If True, use a single vectorized KDTree query for all points.
    query_points : (num_query_points,3) array, optional
        Pre-specified query points. If None, drawn uniformly in the box.
    return_interps : bool, optional
        If True, also return the per-k CDF interpolators (mapping r → CDF).

    Returns
    -------
    nn_dist_dict : dict
        Dictionary with the same structure as returned by `get_nn_distributions`:
        - interped_radii[k]: r-grid for that k
        - cdfs[k]: joint CDF at those r
        - peaked_cdfs[k]: min(CDF, 1-CDF)
        - cic_probs[m]: left empty for now (filled later when we formalize joint-CIC)
        - medians/pct5s/pct25s/pct75s/pct95s : scalar summary radii
    (optional) cdf_interps : dict[int, interp1d]
        If `return_interps=True`, per-k CDF(r) interpolators.
    """

    # Build KDTrees with periodic boundaries for each catalog
    tree_a = cKDTree(comoving_positions_a, boxsize=comoving_box_size)
    tree_b = cKDTree(comoving_positions_b, boxsize=comoving_box_size)

    # Common query points
    if query_points is None:
        query_points = np.random.uniform(
            low=0.0, high=comoving_box_size, size=(num_query_points, 3)
        )
    assert query_points.shape == (num_query_points, 3), "Invalid query_points shape"

    # Query distances to the k nearest neighbors for both catalogs
    if use_vectorized:
        d_a, _ = tree_a.query(x=query_points, k=max_num_neighbors, workers=-1)
        d_b, _ = tree_b.query(x=query_points, k=max_num_neighbors, workers=-1)
    else:
        dists_a = []
        dists_b = []
        for qp in query_points:
            da, _ = tree_a.query(x=qp, k=max_num_neighbors, workers=-1)
            db, _ = tree_b.query(x=qp, k=max_num_neighbors, workers=-1)
            dists_a.append(da)
            dists_b.append(db)
        d_a = np.vstack(dists_a)
        d_b = np.vstack(dists_b)

    # Ensure (num_query_points, max_k)
    if d_a.ndim == 1:
        d_a = d_a[:, None]
    if d_b.ndim == 1:
        d_b = d_b[:, None]

    nn_dist_dict = initialize_nn_dist_dict()
    cdf_interps: dict[int, interp1d] = {}

    # For each k, construct the order-statistic for the joint event: max(r_k^A, r_k^B)
    for k in range(1, max_num_neighbors + 1):
        rka = d_a[:, k - 1]
        rkb = d_b[:, k - 1]
        r_joint = np.maximum(rka, rkb)  # sphere must enclose both k-th neighbors

        # Sort and build log-spaced r-grid
        r_joint_sorted = np.sort(r_joint)
        rgrid = _make_rgrid(r_joint_sorted, num_interp_points)

        # Empirical CDF for r_joint
        cdf_i = _build_cdf_interp(r_joint_sorted)
        cdf_interps[k] = cdf_i

        nn_dist_dict["interped_radii"][k] = rgrid
        cdf_k, pcdf_k = _evaluate_distributions(rgrid, cdf_i)
        nn_dist_dict["cdfs"][k] = cdf_k
        nn_dist_dict["peaked_cdfs"][k] = pcdf_k

        # Scalar summaries from CDF
        stats = _compute_summary_from_cdf(rgrid, cdf_k)
        nn_dist_dict["medians"][k] = stats["medians"]
        nn_dist_dict["pct5s"][k]   = stats["pct5s"]
        nn_dist_dict["pct25s"][k]  = stats["pct25s"]
        nn_dist_dict["pct75s"][k]  = stats["pct75s"]
        nn_dist_dict["pct95s"][k]  = stats["pct95s"]

    # NOTE: We intentionally leave `cic_probs` empty for the joint case here.
    # A consistent joint-CIC definition (e.g., over the max-radius process or a
    # 2D occupancy grid) will be added in a later pass.

    return (nn_dist_dict, cdf_interps) if return_interps else nn_dist_dict


# =========================
# Joint kNN cross-correlation ([-1, 1]) and transition scale helpers
# =========================

def _maybe_downsample_to_match_number_density(
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        box_size: float,
        *,
        rng: np.random.Generator | None = None,
        tol: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Randomly downsample the denser catalog so that both have the same \bar n.
    Returns possibly-downsampled (A, B).
    """
    if rng is None:
        rng = np.random.default_rng()
    vol = box_size**3
    n_a = pos_a.shape[0] / vol
    n_b = pos_b.shape[0] / vol
    if n_a <= 0 or n_b <= 0:
        return pos_a, pos_b
    if abs(n_a - n_b) / max(n_a, n_b) <= tol:
        return pos_a, pos_b
    if n_a > n_b:
        keep = rng.choice(pos_a.shape[0], size=int(round(n_b * vol)), replace=False)
        return pos_a[keep], pos_b
    else:
        keep = rng.choice(pos_b.shape[0], size=int(round(n_a * vol)), replace=False)
        return pos_a, pos_b[keep]


def _common_rgrid_for_k(
        r_a: np.ndarray, r_b: np.ndarray, r_ab: np.ndarray, num_interp_points: int
    ) -> np.ndarray:
    rmin = float(max(min(r_a), min(r_b), min(r_ab)))
    rmax = float(min(max(r_a), max(r_b), max(r_ab)))
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmin >= rmax:
        # fallback: use merged range
        rmin = float(min(r_a[0], r_b[0], r_ab[0]))
        rmax = float(max(r_a[-1], r_b[-1], r_ab[-1]))
        rmin = max(rmin, np.finfo(float).tiny)
    return np.logspace(np.log10(rmin), np.log10(rmax), num=num_interp_points)


def _safe_corr_from_probs(p_ab: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Compute rho = (p_ab - p_a p_b)/sqrt(p_a(1-p_a)p_b(1-p_b)) with safe masking."""
    num = p_ab - (p_a * p_b)
    den = np.sqrt(np.clip(p_a * (1.0 - p_a) * p_b * (1.0 - p_b), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):  # avoid warnings near 0/1
        rho = num / den
    # Clip to [-1,1] and set ill-defined points to nan
    rho = np.clip(rho, -1.0, 1.0)
    bad = (den <= 0) | ~np.isfinite(rho)
    rho[bad] = np.nan
    return rho


def get_joint_knn_correlations(
        comoving_positions_a: np.ndarray,
        comoving_positions_b: np.ndarray,
        comoving_box_size: float,
        num_query_points: int,
        max_num_neighbors: int,
        num_interp_points: int = 1000,
        use_vectorized: bool = True,
        query_points: np.ndarray | None = None,
        match_number_density: bool = True,
        seed: int | None = None,
        return_components: bool = False,
    ) -> dict[str, dict[int, np.ndarray]]:
    """
    Compute the scale-dependent **kNN cross-correlation coefficient** (in [-1,1])
    between two catalogs, together with the independence ratio R.

    Returns a dict with per-k arrays on a shared r-grid:
      {
        'radii': {k: r_k},
        'rho':   {k: rho_k(r)},
        'R':     {k: R_k(r)},
        # optionally the raw probabilities
        'pA': {k: p_A(r)}, 'pB': {k: p_B(r)}, 'pAB': {k: p_AB(r)}
      }
    """
    rng = np.random.default_rng(seed)

    # Optionally downsample to match \bar n
    if match_number_density:
        comoving_positions_a, comoving_positions_b = _maybe_downsample_to_match_number_density(
            comoving_positions_a, comoving_positions_b, comoving_box_size, rng=rng
        )

    # Common query points
    if query_points is None:
        query_points = rng.uniform(0.0, comoving_box_size, size=(num_query_points, 3))

    # Single-catalog CDFs (interps only) for A and B on their native supports
    tree_a = cKDTree(comoving_positions_a, boxsize=comoving_box_size)
    tree_b = cKDTree(comoving_positions_b, boxsize=comoving_box_size)

    nn_a, interps_a = get_nn_distributions(
        kd_tree=tree_a, box_size=comoving_box_size, num_query_points=num_query_points,
        max_num_neighbors=max_num_neighbors, num_interp_points=num_interp_points,
        use_vectorized=use_vectorized, query_points=query_points, return_interps=True
    )
    nn_b, interps_b = get_nn_distributions(
        kd_tree=tree_b, box_size=comoving_box_size, num_query_points=num_query_points,
        max_num_neighbors=max_num_neighbors, num_interp_points=num_interp_points,
        use_vectorized=use_vectorized, query_points=query_points, return_interps=True
    )

    # Joint CDF interps via max(r_k^A, r_k^B)
    nn_ab, interps_ab = get_joint_nn_distribution(
        comoving_positions_a, comoving_positions_b,
        comoving_box_size=comoving_box_size, num_query_points=num_query_points,
        max_num_neighbors=max_num_neighbors, num_interp_points=num_interp_points,
        use_vectorized=use_vectorized, query_points=query_points, return_interps=True,
    )

    out: dict[str, dict[int, np.ndarray]] = {key: {} for key in ("radii", "rho", "R")}
    if return_components:
        out |= {"pA": {}, "pB": {}, "pAB": {}}

    for k in range(1, max_num_neighbors + 1):
        rgrid = _common_rgrid_for_k(
            nn_a["interped_radii"][k], nn_b["interped_radii"][k], nn_ab["interped_radii"][k],
            num_interp_points
        )
        pA  = np.asarray(interps_a[k](rgrid), dtype=float)
        pB  = np.asarray(interps_b[k](rgrid), dtype=float)
        pAB = np.asarray(interps_ab[k](rgrid), dtype=float)

        # Ratio to independence and correlation coefficient
        with np.errstate(invalid="ignore", divide="ignore"):
            R = pAB / np.clip(pA * pB, np.finfo(float).tiny, None)
        R[~np.isfinite(R)] = np.nan
        rho = _safe_corr_from_probs(pAB, pA, pB)

        out["radii"][k] = rgrid
        out["R"][k]     = R
        out["rho"][k]   = rho
        if return_components:
            out["pA"][k], out["pB"][k], out["pAB"][k] = pA, pB, pAB

    return out


def find_knn_transition_scale(
        corr_out: dict[str, dict[int, np.ndarray]],
        *,
        ks: list[int] | range | None = None,
        epsilon: float = 0.05,
        p_window: tuple[float, float] = (0.01, 0.99),
        prefer_stat: str = "R",
    ) -> float:
    """
    Given the output of `get_joint_knn_correlations`, pick a transition radius:
        r_trans = sup{ r : min_{k in K} S_k(r) >= 1+epsilon }, where S is R by default.
    We also require that probabilities are well-posed (avoid p ~ 0 or 1) by checking that
    rho is finite; alternatively you can use p-windowing if components were returned.

    Parameters
    ----------
    corr_out : dict returned by get_joint_knn_correlations
    ks : list or range of k to enforce jointly; default uses all available k
    epsilon : plateau tolerance for R >= 1 + epsilon
    p_window : (lo, hi) used implicitly via finite rho mask; set tighter if needed
    prefer_stat : 'R' or 'rho' to use for thresholding (for rho, we look for high positive values)

    Returns
    -------
    r_trans : float (nan if no radius satisfies the criterion)
    """
    if ks is None:
        ks = list(corr_out["radii"].keys())

    # Build a common r grid across the selected ks by intersecting supports
    # (we discretize by union and then mask per-k)
    r_union = np.unique(np.concatenate([corr_out["radii"][k] for k in ks]))

    # Interpolate the chosen statistic onto r_union per k
    vals = []
    finite_masks = []
    for k in ks:
        r_k  = corr_out["radii"][k]
        sk   = corr_out[prefer_stat][k]
        # Linear in log-r for robustness
        f = interp1d(np.log(r_k), sk, kind="linear", bounds_error=False, fill_value=np.nan)
        v = f(np.log(r_union))
        vals.append(v)
        finite_masks.append(np.isfinite(v))

    vals = np.vstack(vals)  # (len(ks), len(r_union))
    finite = np.all(np.vstack(finite_masks), axis=0)

    if prefer_stat == "R":
        good = (vals >= (1.0 + epsilon)) & finite
    else:  # prefer_stat == 'rho'
        # heuristic: require rho >= 0.5 as "high" correlation plateau
        good = (vals >= 0.5) & finite

    if not np.any(np.all(good, axis=0)):
        return float("nan")

    return float(r_union[np.where(np.all(good, axis=0))[0].max()])



def display_stats(nn_dist: NNDistributionData) -> None: 
    print(f"k={nn_dist.estimate.k}")
    print(f"Median radius: {nn_dist.estimate.median:.3f}")
    print(f"5th percentile: {nn_dist.estimate.pct5:.3f}")
    print(f"25th percentile: {nn_dist.estimate.pct25:.3f}")
    print(f"75th percentile: {nn_dist.estimate.pct75:.3f}")
    print(f"95th percentile: {nn_dist.estimate.pct95:.3f}")


def display_stats_w_errors(nn_dist: NNDistributionData) -> None:
    if nn_dist.errors.is_null: 
        raise ValueError("Cannot display stats for null NNDistribution")
    print(f"k={nn_dist.estimate.k}")
    print(f"Median radius: {nn_dist.estimate.median:.3f} +/ {nn_dist.errors.median:.3f}")
    print(f"5th percentile: {nn_dist.estimate.pct5:.3f} +/ {nn_dist.errors.pct5:.3f}")
    print(f"25th percentile: {nn_dist.estimate.pct25:.3f} +/ {nn_dist.errors.pct25:.3f}")
    print(f"75th percentile: {nn_dist.estimate.pct75:.3f} +/ {nn_dist.errors.pct75:.3f}")
    print(f"95th percentile: {nn_dist.estimate.pct95:.3f} +/ {nn_dist.errors.pct95:.3f}")


def _write_branch(h5group: h5py.Group, branch: dict[str, dict[int, float | np.ndarray]]) -> None:
    """
    Write one branch (estimate/errors) to HDF5 under h5group.
    Expects keys: interped_radii, cdfs, peaked_cdfs, cic_probs, medians, pct5s, pct25s, pct75s, pct95s.
    """
    # array-valued per-k fields
    for key in ["interped_radii", "cdfs", "peaked_cdfs", "cic_probs"]:
        subgrp = h5group.require_group(key)
        for k, arr in branch.get(key, {}).items():
            dset_name = f"k{int(k):03d}"
            data = np.asarray(arr)
            if dset_name in subgrp:
                del subgrp[dset_name]
            subgrp.create_dataset(dset_name, data=data)

    # scalar per-k fields
    for stat_key in ["medians", "pct5s", "pct25s", "pct75s", "pct95s"]:
        subgrp = h5group.require_group(stat_key)
        for k, val in branch.get(stat_key, {}).items():
            dset_name = f"k{int(k):03d}"
            val_arr = np.asarray(val)
            data = val_arr.reshape(()) if np.size(val_arr) == 1 else val_arr
            if dset_name in subgrp:
                del subgrp[dset_name]
            subgrp.create_dataset(dset_name, data=data)


def _read_branch(h5group: h5py.Group) -> dict[str, dict[int, float | np.ndarray]]:

    branch = initialize_nn_dist_dict()
    def _read_numeric_group(name: str, is_scalar: bool) -> None:
        if name not in h5group:
            return
        grp = h5group[name]
        for dname, dset in grp.items():
            try:
                k = int(dname[1:])
            except Exception:
                continue
            arr = dset[()]
            branch[name][k] = float(arr) if is_scalar else np.array(arr)

    for name in ["interped_radii", "cdfs", "peaked_cdfs", "cic_probs"]:
        _read_numeric_group(name, is_scalar=False)
    for name in ["medians", "pct5s", "pct25s", "pct75s", "pct95s"]:
        _read_numeric_group(name, is_scalar=True)

    return branch


def save_knn_distribution_data(
        filepath: Path, 
        knn_data_dict: dict[str, dict[str, dict[int, float | np.ndarray]]]
    ) -> None:
    """
    Save kNNDistributionData to a HDF5 file.
    Accepts dicts produced by:
        - get_jackknife_nn_distributions() -> {"estimate": {...}, "errors": {...}}
        - Or, emulate non-JK by passing {"estimate": get_nn_distributions(...), "errors": initialize_nn_dist_dict()}
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "w") as h5:
        for branch_name in ["estimate", "errors"]:
            grp = h5.require_group(branch_name)
            branch = knn_data_dict.get(branch_name, {})
            _write_branch(grp, branch)



def _load_knn_distribution_data_v2(h5: h5py.File) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """Read the newer v2 format with top-level 'estimate' and 'errors' groups."""
    out: dict[str, dict[str, dict[int, float | np.ndarray]]] = {
        "estimate": (
            _read_branch(h5["estimate"])
            if "estimate" in h5
            else initialize_nn_dist_dict()
        )
    }
    if "errors" in h5:
        out["errors"] = _read_branch(h5["errors"])  # type: ignore[arg-type]
    else:
        out["errors"] = initialize_nn_dist_dict()
    return out


def _read_legacy_branch(h5group: h5py.Group | h5py.File) -> dict[str, dict[int, float | np.ndarray]]:
    """Read legacy (v1) layout with fields at the root and datasets named '1', '2', ..."""
    branch = initialize_nn_dist_dict()

    def _read_numeric_group(name: str, is_scalar: bool) -> None:
        if name not in h5group:
            return
        grp = h5group[name]
        for dname, dset in grp.items():
            # Accept either 'k###' or plain integer names
            try:
                k = int(dname[1:]) if dname.startswith("k") else int(dname)
            except Exception:
                continue
            arr = dset[()]
            branch[name][k] = float(arr) if is_scalar else np.array(arr)

    for name in ["interped_radii", "cdfs", "peaked_cdfs", "cic_probs"]:
        _read_numeric_group(name, is_scalar=False)
    for name in ["medians", "pct5s", "pct25s", "pct75s", "pct95s"]:
        _read_numeric_group(name, is_scalar=True)

    return branch


def _load_knn_distribution_data_v1(h5: h5py.File) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """Read the older v1 format (no top-level 'estimate'/'errors')."""
    estimate_branch = _read_legacy_branch(h5)
    return {"estimate": estimate_branch, "errors": initialize_nn_dist_dict()}

def _infer_for_data_randoms_from_filename(filepath: Path) -> bool:
    """
    Return True if file corresponds to data–random (dr) pairs, False for data–data (dd).
    Backward compatible: if neither token is present, assume 'dr'.
    Expected tokens in filename: '_dd_knns_' or '_dr_knns_'.
    """
    name = filepath.name
    if re.search(r"_dd_knns_", name):
        return False
    return True if re.search(r"_dr_knns_", name) else True

def load_knn_distribution_data(filepath: Path) -> dict[str, dict[str, dict[int, float | np.ndarray]]]:
    """
    Load kNNDistributionData from HDF5 in either of two layouts:
      - v2 (new): top-level groups 'estimate' and 'errors' (each contains per-k data)
      - v1 (legacy): per-k data groups at the root (treated as 'estimate'; 'errors' empty)

    Additionally infers pair type from filename tokens:
      - contains '_dd_knns_'  -> data–data (for_data_randoms = False)
      - contains '_dr_knns_'  -> data–random (for_data_randoms = True)
      - neither present       -> default to data–random (True) for backward compatibility

    The inferred boolean is attached under the private key '_for_data_randoms' in the return dict.
    """
    for_dr = _infer_for_data_randoms_from_filename(filepath)
    try:
        with h5py.File(filepath, "r") as h5:
            keys = set(h5.keys())
            if ("estimate" in keys) or ("errors" in keys):
                out = _load_knn_distribution_data_v2(h5)
            else:
                legacy_fields = {
                    "interped_radii", "cdfs", "peaked_cdfs", "cic_probs",
                    "medians", "pct5s", "pct25s", "pct75s", "pct95s"
                }
                if legacy_fields & keys:
                    out = _load_knn_distribution_data_v1(h5)
                else:
                    out = {
                        "estimate": initialize_nn_dist_dict(),
                        "errors":   initialize_nn_dist_dict()
                    }
        # Attach inferred pair-type flag for the caller
        out["_for_data_randoms"] = for_dr
        return out
    except OSError as e:
        raise FileNotFoundError(f"Could not find file at {filepath}") from e

def _coerce_knn(v: kNNDistribution | None) -> kNNDistribution:
    return v if isinstance(v, kNNDistribution) else kNNDistribution.null_initialize(1)

def _coerce_knn_data(v: kNNDistributionData | None) -> kNNDistributionData:
    return v if isinstance(v, kNNDistributionData) else kNNDistributionData.null_initialize(1)


def get_estimated_two_point_mass_range(
        r_min: float, 
        r_max: float, 
        cosmo: Cosmology,
        use_uniform_sphere: bool = True
    ) -> tuple[float, float]:

    if use_uniform_sphere:
        """Return estimated Lagrangian mass range (min, max) in physical units for given radii."""
        m_min = float(cosmo.get_lagrangian_mass(r_min, a=1.0))
        m_max = float(cosmo.get_lagrangian_mass(r_max, a=1.0))
    else:
        """Return estimated enclosed mass range (min, max) in physical units for given radii."""
        
        # For nonlinear mass scale: nu=1.0
        R200m = lagrangianR(nonLinearMass(0.0))


        # Standard Values from Diemer & Kravtsov 2023
        prof = profile_composite.compositeProfile(
            inner_name="diemer23b",
            outer_names=["infalling"],
            z=0.0,
            rhos=10000.0 *cosmo.matter_density(z=0.0),  # in physical units
            rs=0.2,
            rt=(1.9 - 0.18) * R200m,
            alpha=0.18,
            beta=3.0,
            eta=0.1,
            pl_delta_1=10.0**1.2,
            pl_s=0.0,
            pl_zeta=0.5,
            pl_delta_max=10.0**1.2,
            R200m=R200m,
        )
        m_min = float(prof.enclosedMass(r_min))
        m_max = float(prof.enclosedMass(r_max))

    return m_min, m_max

