from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np

from pathlib import Path
from collections import defaultdict

from ..particles.bounded import (
    concatenate_milestone_evo,
    concatenate_so_values_evo,
    concatenate_prog_radial_distances,
    vectorize_bounded_counts,
    vectorize_query_radii_evo,
)

snap_str_to_int = lambda snap_str: int(snap_str.split("_")[1])

def save_prog_radial_distances(
        hdf_file: h5py.File, 
        prog_radial_distances: dict[int, dict[str, np.ndarray]]
    ) -> None:

    key = lambda idx, scaling: f"prog_radial_distances/snapshot_{idx:03d}/{scaling}"

    for snap_idx, radial_distances in prog_radial_distances.items():
        for scaling_key, radial_values in radial_distances.items():
            hdf_file[key(snap_idx, scaling_key)] = radial_values


def load_prog_radial_distances(hdf_file: h5py.File) -> dict[int, dict[str, np.ndarray]]:

    key = lambda snap_str, scaling: f"prog_radial_distances/{snap_str}/{scaling}"

    return {
        snap_str_to_int(snap_str) : {
            scaling_key: np.asarray(hdf_file[key(snap_str, scaling_key)][:])
            for scaling_key in ("enclosed_steepest", "splashback", "200m")
        }
        for snap_str in hdf_file["prog_radial_distances"].keys()
    }

def save_radial_boundary_evo(
        hdf_file: h5py.File,
        radial_boundary_evo: dict[str, dict[int, float]]
    ) -> None:

    key = lambda idx, scaling: f"radial_boundary_evo/snapshot_{idx:03d}/{scaling}"

    for scaling_key, boundary_evo in radial_boundary_evo.items():
        for snap_idx, boundary_value in boundary_evo.items():
            hdf_file[key(snap_idx, scaling_key)] = boundary_value


def load_radial_boundary_evo(hdf_file: h5py.File) -> dict[str, dict[int, float]]:

    key = lambda snap_str, scaling: f"radial_boundary_evo/{snap_str}/{scaling}"

    return {
        scaling_key: {
            snap_str_to_int(snap_str): float(hdf_file[key(snap_str, scaling_key)][()])
            for snap_str in hdf_file["radial_boundary_evo"].keys()
        }
        for scaling_key in ("enclosed_steepest", "splashback", "200m")
    }

def save_distributions(
        hdf_file: h5py.File,
        distributions: dict[str, dict[tuple[int, int], dict]]
    ) -> None:

    def key(snap_pair: tuple[int, int], scaling: str, data_key: str) -> str:
        return (
            f"distributions/{snap_pair[0]:03d}_{snap_pair[1]:03d}"
            f"/{scaling}/{data_key}"
        )

    for scaling_key, snap_pair_data in distributions.items():
        for snap_pair, data in snap_pair_data.items():
            for data_key, data_values in data.items():
                hdf_file[key(snap_pair, scaling_key, data_key)] = data_values


def load_distributions(hdf_file: h5py.File) -> dict[str, dict[tuple[int, int], dict]]:

    def key(snap_str: str, scaling: str, data_key: str) -> str:
        return f"distributions/{snap_str}/{scaling}/{data_key}"
    
    snap_str_to_tuple = lambda snap_str: tuple(map(int, snap_str.split("_")))


    return {
        scaling_key : {
            snap_str_to_tuple(snap_str) : {
                data_key: np.asarray(hdf_file[key(snap_str, scaling_key, data_key)][:])
                for data_key in ("counts", "xedges", "yedges")
            }
            for snap_str in hdf_file["distributions"].keys()
        }
        for scaling_key in ("enclosed_steepest", "splashback", "200m")
    }


def save_bounded_stats(
        hdf_file: h5py.File,
        bounded_stats: dict[int, dict[str, dict[str, float | np.ndarray]]]
    ) -> None:

    key = lambda idx, scaling, data_key: f"bounded_stats/snapshot_{idx:03d}/{scaling}/{data_key}"

    for snap_idx, scaling_data in bounded_stats.items():
        for scaling_key, data in scaling_data.items():
            for data_key, data_value in data.items():
                hdf_file[key(snap_idx, scaling_key, data_key)] = data_value


def load_bounded_stats(hdf_file: h5py.File) -> dict[int, dict[str, dict[str, float | np.ndarray]]]:

    key = lambda snap_str, scaling, data_key: f"bounded_stats/{snap_str}/{scaling}/{data_key}"

    # stats = defaultdict(dict)
    stats = defaultdict(lambda: defaultdict(dict))

    for snap_str in hdf_file["bounded_stats"].keys():
        snap_idx = snap_str_to_int(snap_str)
        for scaling_key in ("enclosed_steepest", "splashback", "200m"):
            stats[snap_idx][scaling_key]["halo_pct"] = np.asarray(
                hdf_file[key(snap_str, scaling_key, "halo_pct")][:]
            )
            stats[snap_idx][scaling_key]["total_bound_pcts"] = float(
                hdf_file[key(snap_str, scaling_key, "total_bound_pcts")][()]
            )

    return stats

def save_query_radii_evo(
        hdf_file: h5py.File,
        query_radii_evo: dict[int, dict[str, list]]
    ) -> None:

    key = lambda idx, scaling: f"query_radii_evo/snapshot_{idx:03d}/{scaling}"

    for snap_idx, scaling_data in query_radii_evo.items():
        for scaling_key, data in scaling_data.items():
            hdf_file[key(snap_idx, scaling_key)] = data


def load_query_radii_evo(hdf_file: h5py.File) -> dict[int, dict[str, np.ndarray]]:

    key = lambda snap_str, scaling: f"query_radii_evo/{snap_str}/{scaling}"

    return {
        snap_str_to_int(snap_str): {
            scaling_key: np.asarray(hdf_file[key(snap_str, scaling_key)][:])
            for scaling_key in ("enclosed_steepest", "splashback", "200m")
        }
        for snap_str in hdf_file["query_radii_evo"].keys()
    }

def save_bounded_counts(
        hdf_file: h5py.File,
        bounded_counts: dict[int, dict[str, list]]
    ) -> None:

    key = lambda idx, scaling: f"bounded_counts/snapshot_{idx:03d}/{scaling}"

    for snap_idx, scaling_data in bounded_counts.items():
        for scaling_key, data in scaling_data.items():
            hdf_file[key(snap_idx, scaling_key)] = data


def load_bounded_counts(hdf_file: h5py.File) -> dict[int, dict[str, np.ndarray]]:

    key = lambda snap_str, scaling: f"bounded_counts/{snap_str}/{scaling}"

    return {
        snap_str_to_int(snap_str): {
            scaling_key: np.asarray(hdf_file[key(snap_str, scaling_key)][:])
            for scaling_key in ("enclosed_steepest", "splashback", "200m")
        }
        for snap_str in hdf_file["bounded_counts"].keys()
    }

def save_milestones_evo(    
        hdf_file: h5py.File,
        milestones_evo: dict[int, dict[str, np.ndarray | list]]
    ) -> None:

    key = lambda idx, milestone: f"milestones_evo/snapshot_{idx:03d}/{milestone}"

    for snap_idx, milestone_data in milestones_evo.items():
        for milestone_key, data in milestone_data.items():
            hdf_file[key(snap_idx, milestone_key)] = data


def load_milestones_evo(hdf_file: h5py.File) -> dict[int, dict[str, np.ndarray]]:

    key = lambda snap_str, milestone: f"milestones_evo/{snap_str}/{milestone}"

    return {
        snap_str_to_int(snap_str): {
            milestone_key: np.asarray(hdf_file[key(snap_str, milestone_key)][:])
            for milestone_key in (
                'scale_radius', 'scale_mass', 'enclosed_scale_rho', 
                "M4rs", "enclosed_4rs_rho", "four_scale",
                'splashback', 'last_bound', 'splashback_mass', 
                'last_bound_mass', 'v_splash', 'v_last', 'v_scale',
                'v_4rs', 'enclosed_splashback_rho', 'enclosed_last_bound_rho',
                'enclosed_max_rho', 'v_max', 'r_max', 'M_max',

            )
        }
        for snap_str in hdf_file["milestones_evo"].keys()
    }

def save_so_values_evo(
        hdf_file: h5py.File,
        so_values_evo: dict[int, dict[str, dict[str, np.ndarray | list]]]
    ) -> None:

    def key(snap_idx: int, value_key: str, mdef_key: str) -> str:
        return f"so_values_evo/snapshot_{snap_idx:03d}/{value_key}/{mdef_key}"
    
    for snap_idx, value_data in so_values_evo.items():
        for value_key, mdef_data in value_data.items():
            for mdef_key, mdef_values in mdef_data.items():
                hdf_file[key(snap_idx, value_key, mdef_key)] = mdef_values


def load_so_values_evo(hdf_file: h5py.File) -> dict[int, dict[str, dict[str, np.ndarray]]]:

    def key(snap_str: str, value_key: str, mdef_key: str) -> str:
        return f"so_values_evo/{snap_str}/{value_key}/{mdef_key}"

    return {
        snap_str_to_int(snap_str): {
            value_key: {
                mdef_key: np.asarray(hdf_file[key(snap_str, value_key, mdef_key)][:])
                for mdef_key in ('200c', '200m', '500c', 'vir')
            }
            for value_key in ('rho', 'M', 'R', 'V')
        }
        for snap_str in hdf_file["so_values_evo"].keys()
    }


def save_bounded_hdf5(
        save_filepath: Path,
        radial_boundary_evo: dict[str, dict[int, float]],
        distributions: dict[str, dict[tuple[int, int], dict]],
        bounded_counts: dict[int, dict[str, list]],
        bounded_stats: dict[int, dict[str, dict[str, float | np.ndarray]]],
        query_radii_evo: dict[int, dict[str, list]],
        milestones_evo: dict[int, dict[str, np.ndarray]],
        so_values_evo: dict[int, dict[str, dict[str, np.ndarray]]],
    ) -> None: 

    with h5py.File(save_filepath, "w") as hdf_file:
        save_radial_boundary_evo(hdf_file, radial_boundary_evo)
        save_distributions(hdf_file, distributions)
        save_bounded_counts(hdf_file, bounded_counts)
        save_bounded_stats(hdf_file, bounded_stats)
        save_query_radii_evo(hdf_file, query_radii_evo)
        save_milestones_evo(hdf_file, milestones_evo)
        save_so_values_evo(hdf_file, so_values_evo)

def save_bounded_chkpt_hdf5(
        chkpt_filepath: Path,
        prog_radial_distances: dict[int, dict[str, np.ndarray | list]],
        bounded_counts: dict[int, dict[str, list]],
        query_radii_evo: dict[int, dict[str, list]],
        milestones_evo: dict[int, dict[str, np.ndarray | list]],
        so_values_evo: dict[int, dict[str, dict[str, np.ndarray | list]]],
    ) -> None: 

    chkt_pt_prog_radial_distances = concatenate_prog_radial_distances(prog_radial_distances)
    chkt_pt_bounded_counts = vectorize_bounded_counts(bounded_counts)
    chkt_pt_query_radii_evo = vectorize_query_radii_evo(query_radii_evo)
    chkt_pt_milestones_evo = concatenate_milestone_evo(milestones_evo)
    chkt_pt_so_values_evo = concatenate_so_values_evo(so_values_evo)

    with h5py.File(chkpt_filepath, "w") as hdf_file:
        save_prog_radial_distances(hdf_file, chkt_pt_prog_radial_distances)
        save_bounded_counts(hdf_file, chkt_pt_bounded_counts)
        save_query_radii_evo(hdf_file, chkt_pt_query_radii_evo)
        save_milestones_evo(hdf_file, chkt_pt_milestones_evo)
        save_so_values_evo(hdf_file, chkt_pt_so_values_evo)


def load_bounded_hdf5(save_filepath: Path) -> dict:

    bounded_data = {}

    with h5py.File(save_filepath, "r") as hdf_file:
        bounded_data["radial_boundary_evo"] = load_radial_boundary_evo(hdf_file)
        bounded_data["distributions"] = load_distributions(hdf_file)
        bounded_data["bounded_counts"] = load_bounded_counts(hdf_file)
        bounded_data["bounded_stats"] = load_bounded_stats(hdf_file)
        bounded_data["query_radii_evo"] = load_query_radii_evo(hdf_file)
        bounded_data["milestones_evo"] = load_milestones_evo(hdf_file)
        bounded_data["so_values_evo"] = load_so_values_evo(hdf_file)


    return bounded_data

def load_bounded_chkpt_hdf5(chkpt_filepath: Path) -> dict:

    bounded_data = {}

    with h5py.File(chkpt_filepath, "r") as hdf_file:
        bounded_data["prog_radial_distances"] = load_prog_radial_distances(hdf_file)
        bounded_data["bounded_counts"] = load_bounded_counts(hdf_file)
        bounded_data["query_radii_evo"] = load_query_radii_evo(hdf_file)
        bounded_data["milestones_evo"] = load_milestones_evo(hdf_file)
        bounded_data["so_values_evo"] = load_so_values_evo(hdf_file)


    return bounded_data


def verify_loaded_bounded_hdf5(
        save_filepath: Path,
        radial_boundary_evo: dict[str, dict[int, float]],
        distributions: dict[str, dict[tuple[int, int], dict]],
        bounded_counts: dict[int, dict[str, list]],
        bounded_stats: dict[int, dict[str, dict[str, float | np.ndarray]]],
        query_radii_evo: dict[int, dict[str, list]],
        milestones_evo: dict[int, dict[str, np.ndarray]],
        so_values_evo: dict[int, dict[str, dict[str, np.ndarray]]],
    ) -> None:

    loaded_data = load_bounded_hdf5(save_filepath)

    assert radial_boundary_evo.keys() == loaded_data["radial_boundary_evo"].keys()
    assert distributions.keys() == loaded_data["distributions"].keys()
    assert bounded_stats.keys() == loaded_data["bounded_stats"].keys()
    assert query_radii_evo.keys() == loaded_data["query_radii_evo"].keys()
    assert bounded_counts.keys() == loaded_data["bounded_counts"].keys()
    assert milestones_evo.keys() == loaded_data["milestones_evo"].keys()
    assert so_values_evo.keys() == loaded_data["so_values_evo"].keys()

    for scaling_key, boundary_evo in radial_boundary_evo.items():
        assert scaling_key in loaded_data["radial_boundary_evo"]
        assert boundary_evo.keys() == loaded_data["radial_boundary_evo"][scaling_key].keys()
        for snap_idx, boundary_value in boundary_evo.items():
            assert np.isclose(boundary_value, loaded_data["radial_boundary_evo"][scaling_key][snap_idx])

    for scaling_key, snap_pair_data in distributions.items():
        assert scaling_key in loaded_data["distributions"]
        assert snap_pair_data.keys() == loaded_data["distributions"][scaling_key].keys()
        for snap_pair, data in snap_pair_data.items():
            assert data.keys() == loaded_data["distributions"][scaling_key][snap_pair].keys()
            for data_key, data_values in data.items():
                assert np.allclose(data_values, loaded_data["distributions"][scaling_key][snap_pair][data_key])

    for snap_idx, scaling_data in bounded_stats.items():
        assert snap_idx in loaded_data["bounded_stats"]
        assert scaling_data.keys() == loaded_data["bounded_stats"][snap_idx].keys()
        for scaling_key, stat_data in scaling_data.items():
            assert stat_data.keys() == loaded_data["bounded_stats"][snap_idx][scaling_key].keys()
            for stat_key, stat_value in stat_data.items():
                if isinstance(stat_value, np.ndarray):
                    assert np.allclose(stat_value, loaded_data["bounded_stats"][snap_idx][scaling_key][stat_key])
                else:
                    assert np.isclose(stat_value, loaded_data["bounded_stats"][snap_idx][scaling_key][stat_key])

    for snap_idx, scaling_data in query_radii_evo.items():
        assert snap_idx in loaded_data["query_radii_evo"]
        assert scaling_data.keys() == loaded_data["query_radii_evo"][snap_idx].keys()
        for scaling_key, radial_values in scaling_data.items():
            assert np.allclose(radial_values, loaded_data["query_radii_evo"][snap_idx][scaling_key])


    for snap_idx, scaling_data in bounded_counts.items():
        assert snap_idx in loaded_data["bounded_counts"]
        assert scaling_data.keys() == loaded_data["bounded_counts"][snap_idx].keys()
        for scaling_key, counts in scaling_data.items():
            assert np.allclose(counts, loaded_data["bounded_counts"][snap_idx][scaling_key])

    for snap_idx, milestones_data in milestones_evo.items():
        assert snap_idx in loaded_data["milestones_evo"]
        assert milestones_data.keys() == loaded_data["milestones_evo"][snap_idx].keys()
        for milestone_key, milestones in milestones_data.items():
            assert np.allclose(
                    milestones, 
                    loaded_data["milestones_evo"][snap_idx][milestone_key],
                    equal_nan=True
                )

    for snap_idx, so_values in so_values_evo.items():
        assert snap_idx in loaded_data["so_values_evo"]
        assert so_values.keys() == loaded_data["so_values_evo"][snap_idx].keys()
        for so_name, so_data in so_values.items():
            assert so_data.keys() == loaded_data["so_values_evo"][snap_idx][so_name].keys()
            for mdef, mdef_values in so_data.items():
                assert np.allclose(mdef_values, loaded_data["so_values_evo"][snap_idx][so_name][mdef])


def verify_loaded_bounded_chkpt_hdf5(
        chkpt_filepath: Path,
        prog_radial_distances: dict[int, dict[str, np.ndarray]],
        bounded_counts: dict[int, dict[str, list]],
        query_radii_evo: dict[int, dict[str, list]],
        milestones_evo: dict[int, dict[str, np.ndarray]],
        so_values_evo: dict[int, dict[str, dict[str, np.ndarray]]],
    ) -> None:

    loaded_data = load_bounded_chkpt_hdf5(chkpt_filepath)

    assert prog_radial_distances.keys() == loaded_data["prog_radial_distances"].keys()
    assert query_radii_evo.keys() == loaded_data["query_radii_evo"].keys()
    assert bounded_counts.keys() == loaded_data["bounded_counts"].keys()
    assert milestones_evo.keys() == loaded_data["milestones_evo"].keys()
    assert so_values_evo.keys() == loaded_data["so_values_evo"].keys()

    for snap_idx, radial_distances in prog_radial_distances.items():
        assert snap_idx in loaded_data["prog_radial_distances"].keys()
        assert radial_distances.keys() == loaded_data["prog_radial_distances"][snap_idx].keys()
        for scaling_key, radial_values in radial_distances.items():
            assert np.allclose(
                np.concatenate(radial_values), 
                loaded_data["prog_radial_distances"][snap_idx][scaling_key]
            )


    for snap_idx, scaling_data in query_radii_evo.items():
        assert snap_idx in loaded_data["query_radii_evo"]
        assert scaling_data.keys() == loaded_data["query_radii_evo"][snap_idx].keys()
        for scaling_key, radial_values in scaling_data.items():
            assert np.allclose(
                np.asarray(radial_values), 
                loaded_data["query_radii_evo"][snap_idx][scaling_key]
            )


    for snap_idx, scaling_data in bounded_counts.items():
        assert snap_idx in loaded_data["bounded_counts"]
        assert scaling_data.keys() == loaded_data["bounded_counts"][snap_idx].keys()
        for scaling_key, counts in scaling_data.items():
            assert np.allclose(
                np.asarray(counts), 
                loaded_data["bounded_counts"][snap_idx][scaling_key]
            )

    for snap_idx, milestones_data in milestones_evo.items():
        assert snap_idx in loaded_data["milestones_evo"]
        key_set_diff = (
            set(milestones_data.keys()) - set(loaded_data["milestones_evo"][snap_idx].keys())
        )
        assert key_set_diff == {"slope", "enclosed_slope"}
        for milestone_key, milestones in milestones_data.items():
            if milestone_key in key_set_diff: continue
            assert np.allclose(
                    np.stack(milestones), 
                    loaded_data["milestones_evo"][snap_idx][milestone_key],
                    equal_nan=True
                )

    for snap_idx, so_values in so_values_evo.items():
        assert snap_idx in loaded_data["so_values_evo"]
        assert so_values.keys() == loaded_data["so_values_evo"][snap_idx].keys()
        for so_name, so_data in so_values.items():
            assert so_data.keys() == loaded_data["so_values_evo"][snap_idx][so_name].keys()
            for mdef, mdef_values in so_data.items():
                assert np.allclose(
                    np.stack(mdef_values), 
                    loaded_data["so_values_evo"][snap_idx][so_name][mdef]
                )
