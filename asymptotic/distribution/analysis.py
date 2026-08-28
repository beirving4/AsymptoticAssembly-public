import h5py
import numpy as np

from pathlib import Path
from collections import defaultdict

def get_solo_profiles(solo_path: Path) -> dict:
    solo_profiles = {}
    
    with h5py.File(solo_path, 'r') as h5_file:
        # Handle snapshots (e.g., 'snapshot_009')
        for snapshot_str in h5_file.keys():
            snapshot_id = int(snapshot_str.split('_')[-1])
            solo_profiles[snapshot_id] = {}
            
            # Handle mass definitions (e.g., 'M200m')
            for mass_def_str in h5_file[snapshot_str].keys():
                solo_profiles[snapshot_id][mass_def_str] = {}
                mass_def_group = h5_file[snapshot_str][mass_def_str]
                
                # Handle mass bins (e.g., 'M12')
                for mass_bin_str in mass_def_group.keys():
                    mass_bin = int(mass_bin_str.split('M')[-1])
                    solo_profiles[snapshot_id][mass_def_str][mass_bin] = {}
                    mass_bin_group = mass_def_group[mass_bin_str]
                    
                    # Handle all profile types for this mass bin
                    for profile_type in mass_bin_group.keys():
                        profile_data = mass_bin_group[profile_type]
                        
                        # Handle both array datasets and scalar datasets
                        if profile_data.shape == ():  # scalar value
                            solo_profiles[snapshot_id][mass_def_str][mass_bin][profile_type] = \
                                profile_data[()]
                        else:  # array value
                            solo_profiles[snapshot_id][mass_def_str][mass_bin][profile_type] = \
                                profile_data[:]
    
    return solo_profiles


def load_target_profiles(target_path: Path) -> dict:
    target_profiles = {}

    with h5py.File(target_path, 'r') as h5_file:
        for snapshot_str in h5_file.keys():
            snapshot_id = int(snapshot_str.split('_')[-1])
            target_profiles[snapshot_id] = {}

            for mass_def_str in h5_file[snapshot_str].keys():
                target_profiles[snapshot_id][mass_def_str] = {}
                mass_def_group = h5_file[snapshot_str][mass_def_str]

                for profile_str in mass_def_group.keys():
                    target_profiles[snapshot_id][mass_def_str][profile_str] = {}
                    profile_group = mass_def_group[profile_str]

                    if isinstance(profile_group, h5py.Group):
                        for mass_bin_str in profile_group.keys():
                            mass_bin_group = profile_group[mass_bin_str]

                            if isinstance(mass_bin_group, h5py.Group):
                                target_profiles[snapshot_id][mass_def_str][profile_str][mass_bin_str] = {}
                                for sub_key in mass_bin_group.keys():
                                    target_profiles[snapshot_id][mass_def_str][profile_str][mass_bin_str][sub_key] = \
                                        mass_bin_group[sub_key][()]
                            else:
                                # Handle both array datasets and scalar datasets
                                target_profiles[snapshot_id][mass_def_str][
                                    profile_str
                                ][mass_bin_str] = (
                                    mass_bin_group[()]
                                    if mass_bin_group.shape == ()
                                    else mass_bin_group[:]
                                )
                    elif profile_group.shape == ():  # scalar value
                        target_profiles[snapshot_id][mass_def_str][profile_str] = profile_group[()]
                    else:  # array value
                        target_profiles[snapshot_id][mass_def_str][profile_str] = profile_group[:]

    return target_profiles


def stack_solo_profiles(data: dict) -> dict:
    """
    Stack arrays from nested dictionaries while handling special cases.
    
    Parameters
    ----------
    data : dict
        Nested dictionary with structure data[snap_idx][mass_def][mass_bin][measurement]
        where each value is a numpy array
    
    Returns
    -------
    dict
        Stacked dictionary with structure data[snap_idx][mass_def][measurement]
        where values are stacked arrays across mass bins
    """
    result = {}

    for snap_idx, value in data.items():
        result[snap_idx] = {}

        for mass_def in value:
            result[snap_idx][mass_def] = {}

            # Get all measurements for this mass_def
            mass_bins = data[snap_idx][mass_def]
            all_measurements = set().union(*(bin_data.keys() for bin_data in mass_bins.values()))

            for measurement in all_measurements:
                # Collect valid arrays for this measurement across mass bins
                arrays_to_stack = []

                for mass_bin in mass_bins:
                    if measurement not in mass_bins[mass_bin]:
                        continue

                    arr = mass_bins[mass_bin][measurement]

                    # Skip arrays with 0 in the last dimension
                    if arr.shape[-1] == 0:
                        continue

                    arrays_to_stack.append(arr)

                if not arrays_to_stack:
                    continue

                # Handle stacking based on array dimensions
                try:
                    # For 1D arrays of shape (N,), use concatenate
                    if all(arr.ndim == 1 for arr in arrays_to_stack):
                        stacked = np.concatenate(arrays_to_stack)
                    else:
                        # For higher dimensional arrays, use vstack
                        stacked = np.vstack(arrays_to_stack)

                    result[snap_idx][mass_def][measurement] = stacked

                except ValueError as e:
                    print(f"Warning: Could not stack arrays for {snap_idx}/{mass_def}/{measurement}: {e}")
    return result


def get_fully_stacked(stacked_profiles: dict) -> dict:

    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fully = defaultdict(lambda: defaultdict(dict))

    for profile_data in stacked_profiles.values():
        for snap_idx, snap_data in profile_data.items():
            for mass_def, mass_def_data in snap_data.items():
                for measurement, measurement_data in mass_def_data.items():
                    tmp[snap_idx][mass_def][measurement].append(measurement_data)

    for snap_idx, snap_data in tmp.items():
        for mass_def, mass_def_data in snap_data.items():
            for measurement, measurement_data in mass_def_data.items():
                if measurement == "anisotropies": 
                    for i, data in enumerate(measurement_data):
                        if tuple(data.shape[1:]) != (3, 99): 
                            measurement_data.pop(i)

                if measurement == "shapes": 
                    for i, data in enumerate(measurement_data):
                        if tuple(data.shape[1:]) != (12, 99): 
                            measurement_data.pop(i)

                fully[snap_idx][mass_def][measurement] = np.concatenate(measurement_data)

    return fully

def get_round_one_mass_bins(masses: np.ndarray) -> np.ndarray:
    if masses.ndim == 2:
        return np.unique(np.round(masses[:, -1], 1)) 
    return np.unique(np.round(masses, 1))

def get_round_one_mass_bin_evo(staked_data: dict) -> dict:
    mass_bin_evo = defaultdict(dict)
    for snap_idx, snap_data in staked_data.items():
        for mass_def, mass_def_data in snap_data.items():
            if not mass_def_data: continue
            mass_bin_evo[snap_idx][mass_def] = get_round_one_mass_bins(
                masses=mass_def_data['masses']
            )

    return mass_bin_evo

def get_mass_bin_mask(
        masses: np.ndarray, lower_mass_limit: float, upper_mass_limit: float
    ) -> np.ndarray:
    return np.logical_and(masses >= lower_mass_limit, masses < upper_mass_limit)

def get_mass_bin_profiles(
        profiles: np.ndarray,
        masses: np.ndarray,
        target_mass_bin: float,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf, 
        use_round_one: bool = False
    ) -> np.ndarray:

    if use_round_one:
        lower_bound, upper_bound = 0.05, 0.05

    lower_mass = target_mass_bin - lower_bound
    upper_mass = target_mass_bin + upper_bound

    return profiles[get_mass_bin_mask(masses, lower_mass, upper_mass)]

def get_mass_bin_median_profiles(
        profiles: np.ndarray,
        masses: np.ndarray,
        target_mass_bin: float,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf, 
        use_round_one: bool = False
    ) -> np.ndarray:

    mass_bin_profiles = get_mass_bin_profiles(
        profiles, 
        masses, 
        target_mass_bin, 
        lower_bound, 
        upper_bound, 
        use_round_one
    )

    return np.nanmedian(mass_bin_profiles, axis=0)

def get_mass_bin_central_tendency(
        profiles: np.ndarray,
        masses: np.ndarray,
        target_mass_bin: float,
        lower_bound: float = -np.inf,   
        upper_bound: float = np.inf,
        use_round_one: bool = False
    ) -> dict[str, np.ndarray]:

    mass_bin_profiles = get_mass_bin_profiles(
        profiles, 
        masses, 
        target_mass_bin, 
        lower_bound, 
        upper_bound, 
        use_round_one
    )

    pct25 = np.nanpercentile(mass_bin_profiles, 25, axis=0)
    pct75 = np.nanpercentile(mass_bin_profiles, 75, axis=0)
    log_std = np.nanstd(np.log10(mass_bin_profiles), axis=0)

    return {
        "mean" : np.nanmean(mass_bin_profiles, axis=0),
        "median": np.nanmedian(mass_bin_profiles, axis=0),
        "pct25": pct25,
        "pct75": pct75,
        "std" : 10.0**log_std,
        "iqr" : pct75 - pct25,
        "error" : 0.5 * (pct75 - pct25),
    }

def get_target_profiles(
        profiles: np.ndarray,
        group_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> np.ndarray:

    return profiles[np.isin(group_ids, target_ids)]

def get_target_median_profiles(
        profiles: np.ndarray,
        group_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> np.ndarray:

    target_profiles = get_target_profiles(profiles, group_ids, target_ids)
    return np.nanmedian(target_profiles, axis=0)

def get_target_central_tendency(
        profiles: np.ndarray,
        group_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:

    target_profiles = get_target_profiles(profiles, group_ids, target_ids)
    log_std = np.nanstd(np.log10(target_profiles), axis=0)
    pct25 = np.nanpercentile(target_profiles, 25, axis=0)
    pct75 = np.nanpercentile(target_profiles, 75, axis=0)

    return {
        "mean" : np.nanmean(target_profiles, axis=0),
        "median": np.nanmedian(target_profiles, axis=0),
        "pct25": pct25,
        "pct75": pct75,
        "std" : 10.0**log_std,
        "iqr" : pct75 - pct25,
        "error" : 0.5 * (pct75 - pct25),
    }

def get_round_one_profiles(fully_stacked_profiles: dict) -> dict:

    round_one_bin_evo = get_round_one_mass_bin_evo(fully_stacked_profiles)

    result = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    skip_measurements = {'group_ids'}

    for snap_idx, snap_data in fully_stacked_profiles.items():
        for mass_def, mass_def_data in snap_data.items():
            if not mass_def_data:
                continue
                
            # Get masses for this snapshot/mass_def
            masses = mass_def_data['masses']
            masses = masses[:, -1] if masses.ndim == 2 else masses
            mass_bins = round_one_bin_evo[snap_idx][mass_def]

            for measurement in mass_def_data:
                if measurement in skip_measurements:
                    continue
                    
                profiles = mass_def_data[measurement]

                # Calculate statistics for each mass bin
                for mass_bin in mass_bins:
                    
                    try:
                        result[snap_idx][mass_def][mass_bin][measurement] = get_mass_bin_central_tendency(
                            profiles=profiles, 
                            masses=masses, 
                            target_mass_bin=mass_bin, 
                            use_round_one=True
                        )
                    except IndexError as error: 
                        print(f"Error: {error} for {snap_idx}/{mass_def}/{mass_bin}/{measurement}")
                        continue


    return result

def save_round_one_profiles(filepath: Path, round_one_profiles: dict) -> None:

    # save round_one_profiles to hdf5 
    with h5py.File(filepath, "w") as f:
        for snap_idx, snap_data in round_one_profiles.items():
            snap_group = f.create_group(f"snap_{snap_idx}")

            for mass_def, mass_def_data in snap_data.items():
                mass_def_group = snap_group.create_group(mass_def)

                for mass_bin, mass_bin_data in mass_def_data.items():
                    mass_bin_group = mass_def_group.create_group(str(mass_bin))

                    for measurement, measurement_data in mass_bin_data.items():

                        if isinstance(measurement_data, dict):
                            measurement_group = mass_bin_group.create_group(measurement)
                            for sub_measurement, sub_measurement_data in measurement_data.items():
                                measurement_group.create_dataset(sub_measurement, data=sub_measurement_data)
                        else:
                            mass_bin_group.create_dataset(measurement, data=measurement_data)

def load_round_one_profiles(filepath: Path) -> dict:

    loaded_round_one_profiles = {}

    with h5py.File(filepath, "r") as f:
        for snap_idx, snap_data in f.items():
            loaded_round_one_profiles[int(snap_idx[5:])] = {}

            for mass_def, mass_def_data in snap_data.items():
                loaded_round_one_profiles[int(snap_idx[5:])][mass_def] = {}

                for mass_bin, mass_bin_data in mass_def_data.items():
                    loaded_round_one_profiles[int(snap_idx[5:])][mass_def][float(mass_bin)] = {}

                    for measurement, measurement_data in mass_bin_data.items():
                        if isinstance(measurement_data, h5py.Dataset):
                            loaded_round_one_profiles[int(snap_idx[5:])][mass_def][float(mass_bin)][measurement] = measurement_data[()]
                        else:
                            loaded_round_one_profiles[int(snap_idx[5:])][mass_def][float(mass_bin)][measurement] = {}
                            for sub_measurement, sub_measurement_data in measurement_data.items():
                                loaded_round_one_profiles[int(snap_idx[5:])][mass_def][float(mass_bin)][measurement][sub_measurement] = sub_measurement_data[()]
                        # loaded_round_one_profiles[int(snap_idx[5:])][mass_def][float(mass_bin)][measurement] = measurement_data[()]

    return loaded_round_one_profiles

def save_milestone_values(
        filepath: Path, 
        milestone_values: dict[str, dict[int, dict[str, np.ndarray]]]
    ) -> None:
    
    with h5py.File(filepath, "w") as h5_file:
        for sim_name, sim_values in milestone_values.items():
            for snap_idx, snap_values in sim_values.items():
                for var_name, var_values in snap_values.items():
                    h5_file.create_dataset(
                        f"{sim_name}/{snap_idx}/{var_name}", 
                        data=var_values,
                    )   

def load_milestone_values(filepath: Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    
    milestone_values = defaultdict(lambda: defaultdict(dict))
    
    with h5py.File(filepath, "r") as h5_file:
        for sim_name in h5_file.keys():
            for snap_idx in h5_file[sim_name].keys():
                for var_name in h5_file[sim_name][snap_idx].keys():
                    milestone_values[sim_name][int(snap_idx)][var_name] = h5_file[sim_name][snap_idx][var_name][:]

    return milestone_values