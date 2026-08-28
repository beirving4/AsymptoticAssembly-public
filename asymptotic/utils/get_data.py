from __future__ import annotations

import numbers

import h5py, re, pdb
import numpy as np
import configparser

from typing import Any

from pathlib import Path
from tqdm.auto import tqdm
from collections.abc import Collection
from scipy.interpolate import interp1d
from collections import OrderedDict, defaultdict

from .physical_catalog import get_physical_catalog_path

type PhysicalCatalog = dict[str, dict[int, dict[str, np.ndarray]]]
type ConcatenatedPhysicalCatalog = dict[str, dict[str, np.ndarray]]

TRANSFER_FN_NAMES = {"eisenstein": "eisenstein98", "camb": "camb", "class": "class"}

convert_mass = lambda mass: 10.0 + np.log10(mass)

def get_snapshot_mapping(
        data_dir: Path, 
        snap_dir_name: str | None = None
    ) -> dict[str, np.ndarray]:
    if snap_dir_name is None:
        snap_name_template = "snapshot_*.hdf5"
    else:
        snap_name_template = f"{snap_dir_name}*/snapshot_*.hdf5"

    snapshot_ids, scale_factors, redshifts = set(), set(), set() 
    
    for snapshot in data_dir.glob(snap_name_template):
        if snap_dir_name is None:
            snapshot_ids.add(int(snapshot.stem.split("_")[1]))
        else:
            snapshot_ids.add(int(snapshot.stem.split(".")[0].split('_')[1]))
        with h5py.File(snapshot, "r") as f:
            scale_factors.add(f["Header"].attrs["Time"])
            redshifts.add(f["Header"].attrs["Redshift"])

    return {
        "snapshot_ids" : np.array(sorted(snapshot_ids)), 
        "scale_factors" : np.array(sorted(scale_factors)), 
        "redshifts" : np.array(sorted(redshifts, reverse=True))
    }

def get_transfer_fn_name(option: int) -> str:
    if option in {0, 1}:
        return TRANSFER_FN_NAMES["eisenstein"]
    elif option in {2, 3}:
        return TRANSFER_FN_NAMES["class"]
    else:
        raise ValueError(f"Transfer function {option} not supported")


def read_music_ic_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)
    return config


def read_from_music(filepath: Path) -> dict:
    print("Reading Cosmology from MUSIC parameter file...")
    config = read_music_ic_config(filepath)
    cosmo_config = config["cosmology"]
    cosmology_cfg = {}
    for parameter, value in cosmo_config.items():
        value = value.strip()
        if parameter == "transfer":
            cosmology_cfg[parameter] = TRANSFER_FN_NAMES[value]
        else:
            cosmology_cfg[parameter] = float(value) if "." in value else int(value)
    return cosmology_cfg


def read_gadget4_simulation_config(filepath: Path) -> dict:
    config = {}
    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        # Remove comments and leading/trailing whitespace
        line = re.sub(r"%.*", "", line).strip()
        if not line:
            continue  # Skip empty lines

        # Split each line into key and value
        parts = line.split()
        key, value = parts[0], parts[1:]

        # Check if value is numeric and convert it
        if all(part.replace(".", "", 1).isdigit() for part in value):
            value = [int(part) if part.isdigit() else float(part) for part in value]
            if len(value) == 1:
                value = value[0]

        # Store key-value pair in the dictionary
        config[key] = value

    return config


def read_from_gadget(filepath: Path) -> dict:
    print("Reading Cosmology from Gadget4 parameter file...")
    config = read_gadget4_simulation_config(filepath)
    return {
        "H0": float(config["HubbleParam"] * config["Hubble"]),
        "Omega_m": float(config["Omega0"]),
        "Omega_b": float(config["OmegaBaryon"]),
        "Omega_L": float(config["OmegaLambda"]),
        "sigma_8": float(config["Sigma8"]),
        "nspec": float(config["PrimordialIndex"]),
        "w0": -1.0,
        "wa": 0.0,
        "transfer": get_transfer_fn_name(int(config["PowerSpectrumType"])),
    }


def extract_data(h5_file: h5py.File | h5py.Group, base_key=None) -> dict:
    data = {}
    for key in h5_file.keys():
        full_key = f"{base_key}/{key}" if base_key else key
        if isinstance(h5_file[key], h5py.Dataset):
            data[full_key] = h5_file[key][()]
        elif isinstance(h5_file[key], h5py.Group):
            data |= extract_data(h5_file[key], full_key)
    return data

def _sort_glob_by_file_number(files):
    """Sort glob results by file number (the second-to-last segment).

    This ensures consistent ordering when loading batched HDF5 files,
    regardless of filesystem glob ordering.
    """
    return sorted(files, key=lambda f: int(f.name.split('.')[-2]))


def is_array(value: np.ndarray) -> bool:
    return isinstance(value, np.ndarray) and value.ndim > 0

def both_arrays(value_1: np.ndarray, value_2: np.ndarray) -> bool:
    return is_array(value_1) and is_array(value_2)

def is_partial(value: str) -> bool:
    return value.endswith("ThisFile")

def is_value(value: numbers.Number) -> bool:
    return isinstance(value, numbers.Number)

def is_partial_value(data: dict, key: str) -> bool:
    return is_partial(key) and is_value(data[key])

def both_partial(data1: dict, data2: dict, key: str) -> bool:
    return (
        is_partial_value(data1, key) 
        and 
        is_partial_value(data2, key)
    )

def run_joint_concatenation(array1: np.ndarray, array2: np.ndarray) -> bool:
    return (array1.ndim == 1) and (array2.ndim == 1)

def copy_action(joint: dict, key: str, value: Any) -> None:
    joint[key] = value

def sum_action(joint: dict, key: str, x: numbers.Number, y: numbers.Number) -> None:
    joint[key] = x + y

def concatenate_action(joint: dict, key: str, x: np.ndarray, y: np.ndarray) -> None:
    joint[key] = (
        np.concatenate((x, y)) 
        if run_joint_concatenation(x, y) else 
        np.vstack((x, y))
    )

def join_action(joint: dict, key: str, subset1: dict, subset2: dict) -> None:
    
    value1, value2 = subset1[key], subset2[key]

    if both_arrays(value1, value2):
        if key == "MassTable":
            copy_action(joint, key, value1)
        else:
            concatenate_action(joint, key, value1, value2)
    elif both_partial(subset1, subset2, key):
        sum_action(joint, key, value1, value2)
    else:
        copy_action(joint, key, value1)

# check if this is wrong for snapshots... 
def join_data_subsets(data_subset_1: dict, data_subset_2: dict) -> dict:
    """Merge two GADGET-4 shard dicts into one combined dict.

    Keys present in both are joined via ``join_action`` (concatenate
    arrays, sum ``*_ThisFile`` counts, copy scalars). Keys present in
    only one side are passed through — previously they were dropped,
    which caused any field absent from one shard (e.g. Group/* in a
    shard where Ngroups_ThisFile=0) to disappear from the final merged
    dict, creating silent data loss across the concatenated catalog.
    """
    if not data_subset_1:
        return data_subset_2
    if not data_subset_2:
        return data_subset_1

    joint: dict = {}
    all_keys = data_subset_1.keys() | data_subset_2.keys()
    for key in all_keys:
        if key in data_subset_1 and key in data_subset_2:
            join_action(joint, key, data_subset_1, data_subset_2)
        elif key in data_subset_1:
            joint[key] = data_subset_1[key]
        else:
            joint[key] = data_subset_2[key]

    return joint

def to_physical_units(data: dict, is_in_comoving_units: bool, data_type: str) -> None:
    scaling_factor = data["Time"] if is_in_comoving_units else 1.0
    # Softening-specific scale factor: the primary sim convention holds
    # the softening fixed at its comoving length for a >= 1 (so the
    # physical softening does NOT grow unbounded into the future).
    # Cosmological lengths (BoxSize, positions, R_200m, etc.) continue
    # to scale linearly with a.
    softening_scaling_factor = (min(scaling_factor, 1.0)
                                if is_in_comoving_units else 1.0)

    # GADGET-4 velocity unit conventions (comoving runs). These are SEPARATE
    # from the linear comoving->physical position scaling above:
    #   - Snapshot PartType*/Velocities carry a_scaling = 0.5, so the physical
    #     peculiar velocity is v_pec = sqrt(a) * u_stored.
    #   - Group/Subhalo catalog velocities are stored as a * v_pec, so the
    #     peculiar velocity is v_cat / a (verified against the member-particle
    #     bulk: |GroupVel| = a^1.5 * |mean raw particle v|).
    # Both reduce to the identity at a = 1, so a <= 1 outputs are unchanged.
    # Skipping these inflates late-time radial-velocity dispersions because the
    # raw particle and halo velocities sit on mismatched a-scalings and the
    # bulk-velocity subtraction no longer cancels.
    velocity_scaling_factor = np.sqrt(scaling_factor)
    catalog_velocity_scaling_factor = 1.0 / scaling_factor

    def apply_factor(name: str) -> None:
        if name in data:
            data[name] = scaling_factor * data[name]

    def scale_field(name: str, factor: float) -> None:
        if name in data:
            data[name] = factor * data[name]

    # Update FOF Group Data
    if data_type == "fof":
        apply_factor("BoxSize")
        apply_factor("Group/GroupPos")
        apply_factor("Group/Group_R_Crit200")
        apply_factor("Group/Group_R_Mean200")
        apply_factor("Group/Group_R_TopHat200")
        apply_factor("Group/Group_R_Crit500")
        apply_factor("Subhalo/SubhaloCM")
        apply_factor("Subhalo/SubhaloPos")
        apply_factor("Subhalo/SubhaloHalfmassRad")
        apply_factor("Subhalo/SubhaloHalfmassRadType")
        scale_field("Group/GroupVel", catalog_velocity_scaling_factor)
        scale_field("Subhalo/SubhaloVel", catalog_velocity_scaling_factor)


    # Update Snapshot Data
    if data_type == "snap":
        apply_factor("BoxSize")
        apply_factor("PartType1/Coordinates")
        scale_field("PartType1/Velocities", velocity_scaling_factor)
        data["Softening"] = softening_scaling_factor * data["SofteningComovingClass0"]
        data["MaxSoftening"] = softening_scaling_factor * data["SofteningMaxPhysClass0"]


def get_valid_subhalo_ids_and_locs(
        group_ids: np.ndarray,
        subhalo_grnr: np.ndarray,
        n_subhalos: int,
        valid_mask: np.ndarray
    ) -> dict[str, np.ndarray]:
    # Create an array of subhalo IDs of length n_subhalos.
    subhalo_ids = np.arange(n_subhalos)
    
    # Determine the valid group IDs based on the valid_mask.
    valid_group_ids = group_ids[valid_mask]
    
    # Compute a boolean mask for the subhalos that are in one of the valid groups.
    computed_valid_locs = np.isin(subhalo_grnr, valid_group_ids)
    
    # Ensure that valid_subhalo_locs always has length n_subhalos.
    if subhalo_grnr.size == n_subhalos:
        valid_subhalo_locs = computed_valid_locs
    else:
        # Initialize a full-length boolean mask as all False.
        valid_subhalo_locs = np.zeros(n_subhalos, dtype=bool)
        # Copy the computed values into the beginning of this full-length array.
        num_to_fill = min(subhalo_grnr.size, n_subhalos)
        valid_subhalo_locs[:num_to_fill] = computed_valid_locs[:num_to_fill]
    
    # Use the full-length mask to select the valid subhalo IDs.
    valid_subhalo_ids = subhalo_ids[valid_subhalo_locs]
    
    return {
        "group_ids": valid_group_ids,
        "subhalo_locs": valid_subhalo_locs,
        "subhalo_ids": valid_subhalo_ids
    }



def clean_fof_groups(fof_data: dict) -> dict:

    subhalo_grnr = np.asarray(fof_data["Subhalo/SubhaloGroupNr"])
    group_first_subhalo_ids = np.asarray(fof_data["Group/GroupFirstSub"])

    fof_masses = np.asarray(fof_data["Group/GroupMass"])
    masses500c = np.asarray(fof_data["Group/Group_M_Crit500"])
    radii500c = np.asarray(fof_data["Group/Group_R_Crit500"])
    masses200c = np.asarray(fof_data["Group/Group_M_Crit200"])
    radii200c = np.asarray(fof_data["Group/Group_R_Crit200"])
    masses200m = np.asarray(fof_data["Group/Group_M_Mean200"])
    radii200m = np.asarray(fof_data["Group/Group_R_Mean200"])
    # GADGET-4 calls the virial-overdensity SO "TopHat200"; the full_catalog
    # spec renames it to "Virial" (docs/full_catalog_spec.md §v0.3). Accept
    # whichever name is present so the same clean_fof_groups serves both
    # sources without translation.
    vir_m_key = (
        "Group/Group_M_TopHat200" if "Group/Group_M_TopHat200" in fof_data
        else "Group/Group_M_Virial"
    )
    vir_r_key = (
        "Group/Group_R_TopHat200" if "Group/Group_R_TopHat200" in fof_data
        else "Group/Group_R_Virial"
    )
    massesVir = np.asarray(fof_data[vir_m_key])
    radiiVir = np.asarray(fof_data[vir_r_key])
    group_particle_counts = np.asarray(fof_data["Group/GroupLen"])
    group_positions = np.asarray(fof_data["Group/GroupPos"])
    group_velocities = np.asarray(fof_data["Group/GroupVel"])

    # Use original group IDs if available (batched data that was sorted),
    # otherwise use sequential IDs (single file or unsorted data)
    group_ids = fof_data.get('_original_group_ids', np.arange(fof_masses.size))

    valid_group_mask = np.logical_and.reduce([
        (radii200m > 0.0),
        (masses200m > 0.0),
        (radii200c > 0.0),
        (masses200c > 0.0),
        (radii500c > 0.0),
        (masses500c > 0.0),
        (fof_masses > 0.0),
        (massesVir > 0.0),
        (radiiVir > 0.0),
    ])

    valid_ids_and_locs = get_valid_subhalo_ids_and_locs(
        group_ids=group_ids,
        subhalo_grnr=subhalo_grnr,
        n_subhalos=fof_data["Nsubhalos_Total"],
        valid_mask=valid_group_mask
    )


    cleaned = {
        "box_size" : fof_data["BoxSize"],
        "scale_factor": fof_data["Time"],
        "redshift": fof_data["Redshift"],
        "valid_subhalo_ids": valid_ids_and_locs["subhalo_ids"],
        "valid_subhalo_locs": valid_ids_and_locs["subhalo_locs"],
        "group_ids": valid_ids_and_locs["group_ids"],
        "group_first_subhalo_ids": group_first_subhalo_ids[valid_group_mask],
        "masses": convert_mass(fof_masses[valid_group_mask]),
        "masses200c": convert_mass(masses200c[valid_group_mask]),
        "radii200c": radii200c[valid_group_mask],
        "masses200m": convert_mass(masses200m[valid_group_mask]),
        "radii200m": radii200m[valid_group_mask],
        "masses500c": convert_mass(masses500c[valid_group_mask]),
        "radii500c": radii500c[valid_group_mask],
        "massesVirial": convert_mass(massesVir[valid_group_mask]),
        "radiiVirial": radiiVir[valid_group_mask],
        "particle_counts": group_particle_counts[valid_group_mask],
        "positions": group_positions[valid_group_mask],
        "velocities": group_velocities[valid_group_mask],
    }

    # ---- Pass-through for solo-derived mdef fields (full_catalog source) ----
    # These keys only exist when use_full_catalog=True upstream; the GADGET-4
    # path never produces them, so the loop is a no-op for legacy callers.
    # Output keys follow the existing flat naming convention ("masses200m",
    # "radii200c"): "masses_<descriptor.lower()>", "radii_<descriptor.lower()>",
    # "velocities_<descriptor.lower()>",
    # "radii_lagrangian_<descriptor.lower()>".
    _SOLO_DESCRIPTORS = (
        "Splashback", "Bound", "Lambda", "HydrostaticEq",
        "Turnaround", "Infall", "Busha05", "FourScale", "Scale", "Vmax",
    )
    for descriptor in _SOLO_DESCRIPTORS:
        d_lower = descriptor.lower()
        for q, prefix in (
            ("M",            f"masses_{d_lower}"),
            ("R",            f"radii_{d_lower}"),
            ("V",            f"velocities_{d_lower}"),
            ("RLagrangian",  f"radii_lagrangian_{d_lower}"),
        ):
            src_key = f"Group/Group_{q}_{descriptor}"
            if src_key not in fof_data:
                continue
            arr = np.asarray(fof_data[src_key])[valid_group_mask]
            cleaned[prefix] = convert_mass(arr) if q == "M" else arr

    return cleaned

def from_single_fof_file(
        fof_filepath: Path, return_scale_factor_if_empty: bool = False
    ) -> dict:
    """Read one fof_subhalo_tab shard.

    A GADGET-4 batched FOF shard can legitimately have
    ``Ngroups_ThisFile=0`` while still carrying Subhalo entries — e.g.
    when a single large FOF group spans many shards, and one of those
    shards holds only the group's satellite subhalos. Returning ``{}``
    in that case (previous behavior) caused ``join_data_subsets`` to
    silently skip the shard entirely, losing all of its subhalos and
    shifting every subsequent shard's subhalo_nr → array-position
    mapping. Downstream SubhaloMapping then returned wrong group_ids
    for any subhalo_nr after the first dropped shard.

    Fix: only return empty when the shard has NEITHER Group NOR
    Subhalo data. Otherwise return whatever is present so
    join_data_subsets concatenates the subhalos correctly.
    """

    try:
        with h5py.File(fof_filepath, "r") as f:
            has_group = "Group" in f.keys()
            has_subhalo = "Subhalo" in f.keys()
            if not (has_group or has_subhalo):
                if return_scale_factor_if_empty:
                    return {"scale_factor": f["Header"].attrs["Time"]}
                return {}
            return extract_data(f) | dict(f["Header"].attrs.items())
    except OSError as ose:
        raise OSError(f"Could not open {fof_filepath}") from ose


def from_split_fof_files(
        directory: Path,
        snap_idx: int,
        return_scale_factor_if_empty: bool = False
    ) -> dict:
    fof_dir = Path(directory, f"groups_{snap_idx:03d}")
    fof_data = {}
    fof_files = _sort_glob_by_file_number(fof_dir.glob(f"fof_subhalo_tab_{snap_idx:03d}.*.hdf5"))
    for fof_filepath in fof_files:
        if subfile_fof_data := from_single_fof_file(fof_filepath):
            fof_data = join_data_subsets(fof_data, subfile_fof_data)

    if fof_data or not return_scale_factor_if_empty:
        return fof_data
    # Just grab the scale factor from any file
    with h5py.File(fof_filepath, "r") as f:
        return {"scale_factor": f["Header"].attrs["Time"]}


def sort_batched_subfind_data(fof_subfind_data: dict) -> np.ndarray:
    """Sort subfind data by SubhaloGroupNr and return the sort order.

    Uses stable sort to preserve original order within groups when data
    is already sorted (which is the case for properly concatenated files).
    This maintains the SubhaloNr at position i = i relationship.
    """
    order = np.argsort(fof_subfind_data['Subhalo/SubhaloGroupNr'], kind='stable')
    for key, value in fof_subfind_data.items():
        if key.startswith("Subhalo/") and isinstance(value, np.ndarray):
            fof_subfind_data[key] = fof_subfind_data[key][order]
    return order

def sort_batched_fof_data(fof_subfind_data: dict) -> None:
    """Sort FOF group data by GroupFirstSub.

    Uses stable sort for consistency with subfind sorting.
    Also stores the original group IDs before sorting, which is needed
    for correct subhalo filtering in clean_fof_groups.
    """
    n_groups = len(fof_subfind_data['Group/GroupFirstSub'])
    # Store original group IDs (0, 1, 2, ...) BEFORE sorting
    # These are needed because SubhaloGroupNr references original IDs
    fof_subfind_data['_original_group_ids'] = np.arange(n_groups)

    order = np.argsort(fof_subfind_data['Group/GroupFirstSub'], kind='stable')
    for key in fof_subfind_data:
        if key.startswith("Group/") and isinstance(fof_subfind_data[key], np.ndarray):
            fof_subfind_data[key] = fof_subfind_data[key][order]
    # Also sort the original group IDs so they stay aligned with the sorted data
    fof_subfind_data['_original_group_ids'] = fof_subfind_data['_original_group_ids'][order]


def sort_batched_fof_subfind_data(fof_subfind_data: dict) -> np.ndarray | None:
    """Sort FOF and subfind data. Returns the subfind sort order."""
    sort_batched_fof_data(fof_subfind_data)
    subfind_sort_order = sort_batched_subfind_data(fof_subfind_data)
    # Store the sort order in the data dict for use by prog/desc loading
    fof_subfind_data['_subfind_sort_order'] = subfind_sort_order
    return subfind_sort_order


def get_fof_data(
        directory: Path, 
        snap_idx: int, 
        is_batched: bool = False,
        in_comoving: bool = True,
        return_scale_factor_if_empty: bool = False, 
        display_if_empty: bool = False
    ) -> dict:

    if is_batched:

        if (
            fof_data := from_split_fof_files(
                directory=directory,  
                snap_idx=snap_idx,
                return_scale_factor_if_empty=return_scale_factor_if_empty
            )
        ):

            sort_batched_fof_subfind_data(fof_data)

    else:

        fof_data = from_single_fof_file(
            fof_filepath=Path(directory, f"fof_subhalo_tab_{snap_idx:03d}.hdf5"), 
            return_scale_factor_if_empty=return_scale_factor_if_empty
        )

    try:
        to_physical_units(fof_data, in_comoving, "fof")
    except KeyError:
        if return_scale_factor_if_empty and display_if_empty:
            print(f"No group information on {snap_idx:03d}, but returning scale factor!")
        if display_if_empty:
            print(f"No group information on {snap_idx:03d}!")
        

    return fof_data

# Need a single and split version to call on in this function like others ... 
def get_subfind_data(
        raw_fof_data: dict, 
        valid_subhalo_mask: np.ndarray,
    ) -> dict:
    data = {} 

    num_raw_subhalos = raw_fof_data['Nsubhalos_ThisFile']

    if valid_subhalo_mask.size != num_raw_subhalos:
        valid_subhalo_mask = valid_subhalo_mask[:num_raw_subhalos]


    for key, value in raw_fof_data.items(): # Add the mismatch clause here too! 
        if key.startswith("Subhalo/"):
            new_key = key.replace("Subhalo/", "")
            if "Mass" in new_key: value = convert_mass(value)
            data[new_key] = value[valid_subhalo_mask]

    return data

def convert_snap_data_to_physical_units(snap_data: dict) -> None:
    is_in_comoving_units = bool(snap_data["ComovingIntegrationOn"])
    scaling_factor = snap_data["Time"] if is_in_comoving_units else 1.0
    # Softening is held fixed at its comoving value for a >= 1 — see
    # to_physical_units above for the rationale.
    softening_scaling_factor = (min(scaling_factor, 1.0)
                                if is_in_comoving_units else 1.0)
    # GADGET-4 snapshot velocities carry a_scaling = 0.5: the physical peculiar
    # velocity is v_pec = sqrt(a) * u_stored (NOT the linear position factor).
    # Identity at a = 1; see to_physical_units for the full rationale and the
    # matching GroupVel = a * v_pec (catalog) convention.
    velocity_scaling_factor = np.sqrt(scaling_factor)

    # Update Snapshot Data
    snap_data["BoxSize"] *= scaling_factor
    snap_data["PartType1/Coordinates"] *= scaling_factor
    snap_data["PartType1/Velocities"] = (
        velocity_scaling_factor * snap_data["PartType1/Velocities"]
    )
    snap_data["Softening"] = softening_scaling_factor * snap_data["SofteningComovingClass0"]
    snap_data["MaxSoftening"] = softening_scaling_factor * snap_data["SofteningMaxPhysClass0"]


def from_single_snap_file(snap_filepath: Path) -> dict:
    with h5py.File(snap_filepath, "r") as f:
        header_data = {name: np.array(data) for name, data in f["Header"].attrs.items()}
        parameter_data = {
            name: np.array(data) for name, data in f["Parameters"].attrs.items()
        }
        snap_data = extract_data(f)
        return header_data | parameter_data | snap_data
    
def from_split_snap_files(directory: Path, snap_idx: int) -> dict:
    snap_dir = Path(directory, f"snapdir_{snap_idx:03d}")
    snap_data = {}
    snap_files = _sort_glob_by_file_number(snap_dir.glob(f"snapshot_{snap_idx:03d}.*.hdf5"))
    for snap_filepath in snap_files:
        snap_data = join_data_subsets(snap_data, from_single_snap_file(snap_filepath))

    for key in snap_data:
        if key != "MassTable" and not key.startswith('PartType'):
            snap_data[key] = np.unique(snap_data[key])[0]

    for key in snap_data:
        if key == "MassTable": continue
        if (not key.startswith('PartType')):
            snap_data[key] = np.unique(snap_data[key])[0]
    
    return snap_data

def from_split_init_snap_files(directory: Path) -> dict:
    snap_dir = Path(directory, "snapdir_000")
    snap_data = {}
    snap_files = _sort_glob_by_file_number(snap_dir.glob("snapshot_ics_000.*.hdf5"))
    for snap_filepath in snap_files:
        snap_data = join_data_subsets(snap_data, from_single_snap_file(snap_filepath))

    for key in snap_data:
        if key != "MassTable" and not key.startswith('PartType'):
            snap_data[key] = np.unique(snap_data[key])[0]

    for key in snap_data:
        if key == "MassTable": continue
        if (not key.startswith('PartType')):
            snap_data[key] = np.unique(snap_data[key])[0]
    
    return snap_data
def get_snap_data(
        directory: Path, snap_idx: int, 
        is_batched: bool = False, 
        loaded_kd_tree: bool = False
    ) -> dict:

    snapshot_data = (
        from_split_snap_files(directory, snap_idx) 
        if is_batched else 
        from_single_snap_file(Path(directory, f"snapshot_{snap_idx:03d}.hdf5"))
    )

    convert_snap_data_to_physical_units(snapshot_data)

    if not loaded_kd_tree:
        return snapshot_data
    
    tree_path = Path(directory, f"kd_tree_{snap_idx:03d}.npz")
    tree_data = np.load(tree_path, allow_pickle=True)
    return snapshot_data | {"kd_tree": tree_data["kd_tree"].item()}


def get_initial_snap_data(
        directory: Path,
        is_batched: bool = False, 
        loaded_kd_tree: bool = False
    ) -> dict:

    snapshot_data = (
        from_split_init_snap_files(directory) 
        if is_batched else 
        from_single_snap_file(Path(directory, "snapshot_ics_000.hdf5"))
    )

    convert_snap_data_to_physical_units(snapshot_data)

    if not loaded_kd_tree:
        return snapshot_data
    
    tree_path = Path(directory, "kd_tree_ics.npz")
    tree_data = np.load(tree_path, allow_pickle=True)
    return snapshot_data | {"kd_tree": tree_data["kd_tree"].item()}


def from_single_prog_file(progenitor_filepath: Path) -> dict:
    if not progenitor_filepath.exists(): 
        return {}
    with h5py.File(progenitor_filepath, "r") as h5_file:
        return extract_data(h5_file)

def from_split_prog_files(directory: Path, snap_idx: int) -> dict:
    prog_dir = Path(directory, f"groups_{snap_idx:03d}")
    prog_data = {}
    prog_files = _sort_glob_by_file_number(prog_dir.glob(f"subhalo_prog_{snap_idx:03d}.*.hdf5"))
    for prog_filepath in prog_files:
        prog_data = join_data_subsets(prog_data, from_single_prog_file(prog_filepath))
    return prog_data

def sort_batched_prog_data(prog_data: dict, sort_order: np.ndarray | None = None) -> None:
    """Sort prog data by the given order, or by SubhaloNr if no order given.

    Uses stable sort for consistency when falling back to SubhaloNr sorting.
    """
    if sort_order is None:
        sort_order = np.argsort(prog_data['Subhalo/SubhaloNr'], kind='stable')
    for key in prog_data:
        prog_data[key] = prog_data[key][sort_order]

def get_prog_data(
        directory: Path,
        snap_idx: int,
        valid_subhalo_mask: np.ndarray,
        is_batched: bool = False,
        sort_order: np.ndarray | None = None
    ) -> dict:

    if is_batched:
        if (prog_data := from_split_prog_files(directory, snap_idx)):
            # Use the same sort order as subfind data (sorted by SubhaloGroupNr)
            # to ensure mask alignment
            sort_batched_prog_data(prog_data, sort_order=sort_order)
    else:
        filepath = Path(directory, f"subhalo_prog_{snap_idx:03d}.hdf5")
        prog_data = from_single_prog_file(filepath)

    # Handle size mismatch between prog data and mask (can happen if prog files
    # have fewer subhalos than fof_subhalo files)
    first_key = next((k for k in prog_data if isinstance(prog_data[k], np.ndarray)), None)
    if first_key is not None:
        num_prog_subhalos = len(prog_data[first_key])
        if valid_subhalo_mask.size != num_prog_subhalos:
            # Truncate mask to match prog data size
            valid_subhalo_mask = valid_subhalo_mask[:num_prog_subhalos]

    for key, value in prog_data.items():
        if isinstance(value, np.ndarray):
            prog_data[key] = value[valid_subhalo_mask]

    return prog_data


def from_single_desc_file(descendant_filepath: Path) -> dict:
    if not descendant_filepath.exists(): return {}
    with h5py.File(descendant_filepath, "r") as h5_file:
        return extract_data(h5_file)

def from_split_desc_files(directory: Path, snap_idx: int) -> dict:
    desc_dir = Path(directory, f"groups_{snap_idx:03d}")
    desc_data = {}
    desc_files = _sort_glob_by_file_number(desc_dir.glob(f"subhalo_desc_{snap_idx:03d}.*.hdf5"))
    for desc_filepath in desc_files:
        desc_data = join_data_subsets(desc_data, from_single_desc_file(desc_filepath))
    return desc_data

def sort_batched_desc_data(desc_data: dict, sort_order: np.ndarray | None = None) -> None:
    """Sort desc data by the given order, or by SubhaloNr if no order given.

    Uses stable sort for consistency when falling back to SubhaloNr sorting.
    """
    if sort_order is None:
        sort_order = np.argsort(desc_data['Subhalo/SubhaloNr'], kind='stable')
    for key in desc_data:
        desc_data[key] = desc_data[key][sort_order]

def get_desc_data(
        directory: Path,
        snap_idx: int,
        valid_subhalo_mask: np.ndarray,
        is_batched: bool = False,
        sort_order: np.ndarray | None = None
    ) -> dict:

    if is_batched:
        if (desc_data := from_split_desc_files(directory, snap_idx)):
            # Use the same sort order as subfind data (sorted by SubhaloGroupNr)
            # to ensure mask alignment
            sort_batched_desc_data(desc_data, sort_order=sort_order)
    else:
        filepath = Path(directory, f"subhalo_desc_{snap_idx:03d}.hdf5")
        desc_data = from_single_desc_file(filepath)

    # Handle size mismatch between desc data and mask (can happen if desc files
    # have fewer subhalos than fof_subhalo files)
    first_key = next((k for k in desc_data if isinstance(desc_data[k], np.ndarray)), None)
    if first_key is not None:
        num_desc_subhalos = len(desc_data[first_key])
        if valid_subhalo_mask.size != num_desc_subhalos:
            # Truncate mask to match desc data size
            valid_subhalo_mask = valid_subhalo_mask[:num_desc_subhalos]

    for key, value in desc_data.items():
        if isinstance(value, np.ndarray):
            desc_data[key] = value[valid_subhalo_mask]

    return desc_data


def get_subhalo_data(
        directory: Path,
        snap_idx: int,
        prev_loaded_fof_data: dict[str, np.ndarray],
        valid_subhalo_mask: np.ndarray,
        is_batched: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:

    # Get the sort order used for subfind data (if batched)
    # This ensures prog/desc data is sorted the same way as subfind
    subfind_sort_order = prev_loaded_fof_data.get('_subfind_sort_order')

    subfind_data = get_subfind_data(
        raw_fof_data=prev_loaded_fof_data,
        valid_subhalo_mask=valid_subhalo_mask,
    )

    prog_data = get_prog_data(
        directory=directory,
        snap_idx=snap_idx,
        valid_subhalo_mask=valid_subhalo_mask,
        is_batched=is_batched,
        sort_order=subfind_sort_order
    )

    desc_data = get_desc_data(
        directory=directory,
        snap_idx=snap_idx,
        valid_subhalo_mask=valid_subhalo_mask,
        is_batched=is_batched,
        sort_order=subfind_sort_order
    )

    return {"subfind" : subfind_data, "prog" : prog_data, "desc" : desc_data}


# ---- Full-catalog reader (compose_full_catalog.py output) ----

# Solo-milestone descriptors that compose_full_catalog.py writes into /Group.
# Kept in sync with SOLO_TO_DESCRIPTOR in compose_full_catalog.py.
_FULL_CATALOG_SOLO_DESCRIPTORS = (
    "Splashback", "Bound", "Lambda", "HydrostaticEq",
    "Turnaround", "Infall", "Busha05", "FourScale", "Scale", "Vmax",
)


def _full_catalog_dir(catalog_dir: Path) -> Path:
    """Convention: full_catalogs lives as a child of catalog_dir.

    For L=256 with catalog_dir=gadget4/output/, this is
    gadget4/output/full_catalogs/ — alongside the raw fof_subhalo_tab
    files. For L=32 with catalog_dir=halo_catalogs/, this is
    halo_catalogs/full_catalogs/ — alongside the groups_<snap>/ shard
    subdirectories. Matches compose_full_catalog.py's default
    --output_dir.
    """
    return Path(catalog_dir) / "full_catalogs"


def _find_full_catalog_shards(full_cat_dir: Path, snap_idx: int) -> list[Path]:
    """Return numerically-sorted shard list (matches compose's writer convention)."""
    def _shard_index(path: Path) -> int:
        try:
            return int(path.stem.rsplit(".", 1)[-1])
        except ValueError:
            return -1
    if not full_cat_dir.exists():
        return []
    single = full_cat_dir / f"full_catalog_{snap_idx:03d}.hdf5"
    if single.exists():
        return [single]
    return sorted(
        full_cat_dir.glob(f"full_catalog_{snap_idx:03d}.*.hdf5"),
        key=_shard_index,
    )


def _load_full_catalog_as_fof_data(
        catalog_dir: Path,
        snap_idx: int,
        display_if_empty: bool = True,
        in_comoving: bool = True,
        full_catalog_dir: Path | None = None,
    ) -> dict:
    """Read a full_catalog_<snap>.<shard>.hdf5 set and return a dict shaped
    like get_fof_data's output, plus solo-derived Group_<Q>_<Descriptor>
    fields.

    Output dict keys:
        - "Group/<field>"   for every dataset under /Group/ (including the
          GADGET-4 verbatim fields like GroupMass, Group_M_Mean200, and the
          new solo descriptors like Group_M_Splashback). The Virial mass/
          radius are exposed under their full_catalog names
          (Group_M_Virial / Group_R_Virial); clean_fof_groups accepts either
          GADGET-4 (Group_M_TopHat200) or full_catalog (Group_M_Virial)
          naming.
        - "Subhalo/<field>" for every dataset under /Subhalo/ (verbatim
          GADGET-4 fields + treelink/desc/prog passthrough).
        - Top-level header attrs ("BoxSize", "Time", "Redshift",
          "Nsubhalos_Total", "Ngroups_Total", "Cosmo_*", "Moment_*", …),
          matching the pattern from from_single_fof_file's
          `extract_data(f) | dict(f["Header"].attrs.items())`.

    Concatenates per-field arrays across shards in numerical shard order.
    Returns {} when no shards are found (parity with get_fof_data).
    """
    full_cat_dir = (
        Path(full_catalog_dir)
        if full_catalog_dir is not None else
        _full_catalog_dir(catalog_dir)
    )
    shards = _find_full_catalog_shards(full_cat_dir, snap_idx)
    if not shards:
        if display_if_empty:
            print(
                f"No full_catalog files for snap {snap_idx} under "
                f"{full_cat_dir} — was compose_full_catalog.py run for this snap?"
            )
        return {}

    accumulated: dict[str, np.ndarray] = {}
    header_attrs: dict | None = None

    def _open_with_retry(path: Path, attempts: int = 4, delay_s: float = 2.0):
        """h5py.File open with retry on transient TimeoutError.

        macOS Spotlight indexing of freshly-written HDF5 files
        intermittently produces errno 60 ('Operation timed out') when
        we re-read full_catalogs right after a compose sweep. The file
        is fine — a short backoff lets the indexer release the read
        block. We retry a small number of times before re-raising.
        """
        import time as _time
        last_err: BaseException | None = None
        for k in range(attempts):
            try:
                return h5py.File(path, "r")
            except (TimeoutError, OSError) as err:
                if "time" not in str(err).lower() and "60" not in str(err):
                    raise
                last_err = err
                if k < attempts - 1:
                    _time.sleep(delay_s * (k + 1))
        if last_err is not None:
            raise last_err
        raise RuntimeError("_open_with_retry: unreachable")

    for shard in shards:
        with _open_with_retry(shard) as f:
            if header_attrs is None:
                # Cache header attrs from the first shard. ThisFile counts
                # are shard-local; the *_Total fields are sim-wide and
                # identical across shards.
                header_attrs = dict(f["Header"].attrs.items())
            for grp_name in ("Group", "Subhalo"):
                if grp_name not in f:
                    continue
                for key in f[grp_name].keys():
                    full_key = f"{grp_name}/{key}"
                    arr = f[grp_name][key][...]
                    if full_key in accumulated:
                        accumulated[full_key] = np.concatenate(
                            [accumulated[full_key], arr], axis=0,
                        )
                    else:
                        accumulated[full_key] = arr

    result: dict = {**accumulated, **(header_attrs or {})}

    # The cached Header *_Total / *_ThisFile counts come from shard 0 only and
    # are NOT reliably sim-wide for these batched catalogs (each shard can store
    # its own shard-local total). Downstream clean_fof_groups + get_subfind_data
    # size their subhalo masks from Nsubhalos_Total / Nsubhalos_ThisFile and
    # MUST match the concatenated arrays, so recompute the counts from the actual
    # concatenated array lengths rather than trusting the header.
    _sub_keys = [k for k in accumulated if k.startswith("Subhalo/")]
    _grp_keys = [k for k in accumulated if k.startswith("Group/")]
    if _sub_keys:
        n_sub = int(len(accumulated[_sub_keys[0]]))
        result["Nsubhalos_Total"] = n_sub
        result["Nsubhalos_ThisFile"] = n_sub
    if _grp_keys:
        n_grp = int(len(accumulated[_grp_keys[0]]))
        result["Ngroups_Total"] = n_grp
        result["Ngroups_ThisFile"] = n_grp

    # Empty-snap short-circuit. compose_full_catalog always creates
    # /Group and /Subhalo (even when n_groups == n_subhalos == 0) so the
    # on-disk layout stays consistent, but downstream clean_fof_groups
    # then blows up on missing required fields like
    # Subhalo/SubhaloGroupNr (which GADGET-4 omits entirely on empty
    # snaps). Match the GADGET path's behavior: when both totals are
    # zero, return {} so the get_fof_subfind_data `if fof_data:` guard
    # skips the clean / subhalo pass.
    n_groups_total    = int(result.get("Ngroups_Total", 0))
    n_subhalos_total  = int(result.get("Nsubhalos_Total", 0))
    if n_groups_total == 0 and n_subhalos_total == 0:
        return {}

    # Mirror sort_batched_fof_subfind_data when we crossed shard boundaries
    # (matches what get_fof_data does on the batched path). For single-shard
    # full_catalogs this is a no-op except for adding _original_group_ids.
    if len(shards) > 1:
        sort_batched_fof_subfind_data(result)

    # Match the comoving→physical conversion get_fof_data performs on
    # raw fof_subhalo_tab data. full_catalog stores radii / positions in
    # comoving Mpc/h relative to a=1 (verbatim GADGET-4 native fields
    # are already comoving, and solo-derived R_<descriptor> are divided
    # by scale_factor in compose_full_catalog._scatter_solo_to_global).
    # Without this multiplication by `Time` (= a) at snap-read time,
    # downstream code that expects physical units (phusis particle
    # extraction, R_200m-radius-cuts, etc.) sees radii undershooting by
    # a factor of `a` and rejects every halo at late snaps.
    try:
        to_physical_units(result, in_comoving, "fof")
    except KeyError:
        # Empty-snap branch above already returned {}, so we should
        # only reach here with a well-formed dict — but be tolerant.
        pass

    return result


def get_fof_subfind_data(
        directory: Path, snap_idx: int,
        is_batched: bool = False,
        in_comoving: bool = True,
        display_if_empty: bool = True,
        use_full_catalog: bool = False,
        full_catalog_dir: Path | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
    """Load FOF + subhalo data for one snapshot.

    When use_full_catalog=True, reads the per-snap full_catalog HDF5
    (compose_full_catalog.py output) from `directory / "full_catalogs/"`
    instead of the GADGET-4 fof_subhalo_tab files. Pass
    `full_catalog_dir` to read them from somewhere else entirely — the
    Planck sims keep their full_catalogs under the user's own Oak tree
    because the GADGET-4 catalogs live in another user's space (see
    IOConfig.full_catalog_dir).

    The downstream cleaning, masking, and subhalo data path are
    unchanged — clean_fof_groups handles either naming, and
    get_subhalo_data still reads subhalo_prog / subhalo_desc from
    `directory` (full_catalog doesn't replace those sidecars).
    """
    if use_full_catalog:
        fof_data = _load_full_catalog_as_fof_data(
            catalog_dir=directory,
            snap_idx=snap_idx,
            display_if_empty=display_if_empty,
            in_comoving=in_comoving,
            full_catalog_dir=full_catalog_dir,
        )
    else:
        fof_data = get_fof_data(
            directory, snap_idx,
            is_batched=is_batched,
            in_comoving=in_comoving,
            display_if_empty=display_if_empty,
        )

    if fof_data:
        cleaned_fof_data = clean_fof_groups(fof_data)

        cleaned_subhalo_data = get_subhalo_data(
            directory=directory,
            snap_idx=snap_idx,
            prev_loaded_fof_data=fof_data,
            valid_subhalo_mask=cleaned_fof_data["valid_subhalo_locs"],
            is_batched=is_batched,
        )

        return {"fof": cleaned_fof_data, **cleaned_subhalo_data}

    else:
        return {}
    

def load_subfind_data(
        directory: Path, 
        snap_idx: int, 
        is_batched: bool = False,
        in_comoving: bool = True
    ) -> dict[str, np.ndarray]:

    return get_fof_subfind_data(
        directory=directory, 
        snap_idx=snap_idx, 
        is_batched=is_batched,
        in_comoving=in_comoving
    )["subfind"]

def get_catalog_snap_ids(directory: Path, is_batched: bool = False) -> list[int]:
    
    if is_batched:
        pattern = "groups_*/fof_subhalo_tab_*.*.hdf5"
        return sorted(
            {
                int(filepath.stem.split("_")[-1].split(".")[0]) 
                for filepath in directory.glob(pattern)
            }
        )
    else:
        pattern = "fof_subhalo_tab_*.hdf5"
        return sorted(
            {
                int(filepath.stem.split("_")[-1]) 
                for filepath in directory.glob(pattern)
            }
        )
    
def get_phys_catalog_snap_ids(directory: Path, seed_idx: int = -1) -> list[int]:

    phys_cat_dir = Path(directory, "physical_catalogs")
    pattern = (
        f"L*_N*_box{seed_idx}_phys_catalog_*.hdf5"
        if seed_idx != -1 else
        "L*_N*_primary_phys_catalog_*.hdf5"
    )

    return sorted(
        {
            int(filepath.stem.split("_")[-1]) 
            for filepath in phys_cat_dir.glob(pattern)
        }
    )

def get_all_fof_subfind_catalogs(
        directory: Path,
        is_batched: bool = False,
        in_comoving: bool = True,
        display_if_empty: bool = True,
        snapshots_to_skip: Collection[int] | None = None,
        use_full_catalog: bool = False,
        full_catalog_dir: Path | None = None,
    ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """Multi-snap variant of :func:`get_fof_subfind_data`.

    When ``use_full_catalog=True``, snap discovery still scans the GADGET-4
    catalog directory (so the snap set is canonical), but each snap's
    payload is loaded via the Phase 1 full_catalog reader. Snaps lacking
    a full_catalog file return ``{}`` from the loader and are dropped
    from the returned dict — matching the existing behavior for missing
    GADGET-4 snaps.
    """
    catalog_snap_ids = get_catalog_snap_ids(directory, is_batched=is_batched)

    if snapshots_to_skip is not None:
        catalog_snap_ids = [
            snap_id
            for snap_id in catalog_snap_ids
            if snap_id not in snapshots_to_skip
        ]

    desc = "Loading Full Catalog Data" if use_full_catalog else "Loading Gadget Catalog Data"
    return {
        snap_id: get_fof_subfind_data(
            directory=directory,
            snap_idx=snap_id,
            is_batched=is_batched,
            in_comoving=in_comoving,
            display_if_empty=display_if_empty,
            use_full_catalog=use_full_catalog,
            full_catalog_dir=full_catalog_dir,
        )
        for snap_id in tqdm(catalog_snap_ids, desc=desc)
    }
    

def get_sim_data(directory: Path, snap_idx: int, is_batched: bool = False) -> dict:
    snap_data = get_snap_data(directory, snap_idx, is_batched=is_batched)
    return {
        "snapshot": snap_data,
        **get_fof_subfind_data(
            directory, snap_idx, in_comoving=bool(snap_data["ComovingIntegrationOn"])
        ),
    }


def get_particle_data(directory: Path) -> dict:
    num_snapshots = len(list(directory.glob("snapshot_*.hdf5")))
    return {
        snap_id: get_snap_data(directory, snap_id)
        for snap_id in tqdm(range(num_snapshots), desc="Loading Particle Data")
    }


def extract_global_power_spectrum_params(lines: list) -> dict[str, float | int | None]:
    if all(len(line.split(' ')) == 1 for line in lines[-3:]):
        effective_num_tracers = float(lines.pop().split()[0])
        total_count = int(np.ceil(np.cbrt(float(lines.pop().split()[0])))**3)
        particle_mass = float(lines.pop().split()[0])
    else: 
        effective_num_tracers, total_count, particle_mass = None, None, None

    return {
        "effective_num_tracers": effective_num_tracers, 
        "total_count": total_count, 
        "particle_mass": particle_mass
    }

def load_gadget_linear_ps(filename: Path, Dgrowth: float) -> dict[str, float | np.ndarray]:
    ps_data = np.loadtxt(filename, skiprows=1)
    k, delta_sq = ps_data[:, 0], (ps_data[:, 1] / Dgrowth**2)
    return {
        "growth_factor" : Dgrowth,
        "wavenumbers" : k, 
        "normalized" : delta_sq,
        "amplitudes" : delta_sq * (2 * np.pi**2) / k**3
    }

def load_music_linear_ps(filename: Path, Dgrowth: float) -> dict[str, float | np.ndarray]:
    ps_data = np.loadtxt(filename, skiprows=2)

    # The power spectrum definition is smaller than CAMB by a factor 8 pi^3.
    k, Pk = ps_data[:, 0], (ps_data[:, -1] / Dgrowth**2) * (2 * np.pi)**3

    return {
        "growth_factor" : Dgrowth,
        "wavenumbers" : k, 
        "normalized" : Pk * (4 * np.pi * k**3), # Due (2 pi)^-3 factor in MUSIC
        "amplitudes" : Pk
    }

def load_linear_ps(
        input_ps_filename: Path, Dgrowth: float, use_music: bool = False
    ) -> dict[str, float |np.ndarray]:

    return (
        load_music_linear_ps(input_ps_filename, Dgrowth)
        if use_music else 
        load_gadget_linear_ps(input_ps_filename, Dgrowth)
    )

NUM_FOLDED_POWER_SPECTRA_PIECES = 3
FOLDING_FACTOR = 16
TARGET_BIN_COUNT = 50   # will add this to the config file later
MIN_MODE_COUNT = 8      #  # rebin to get at least this number of modes per bin (will add this to the config file later)

def get_powerspec_paths(power_spec_dir: Path) -> OrderedDict[int, Path]:
    snap_num = lambda filename: int(filename.stem.split("_")[1])
    return OrderedDict({
        snap_num(filepath) : filepath
        for filepath in sorted(power_spec_dir.glob("powerspec_*.txt"))
    })

def get_for_colossus_powerspec_paths(power_spec_dir: Path) -> OrderedDict[int, Path]:
    snap_num = lambda filename: int(filename.stem.split("_")[-1])
    return OrderedDict({
        snap_num(filepath) : filepath
        for filepath in sorted(power_spec_dir.glob("for_colossus_*.txt"))
    })



def get_power_spec(
        filepath: Path, 
        input_ps_path: Path, 
        use_music_init_ps: bool = False,
    ) -> dict[str, float | dict[str, np.ndarray]]:
    Kall, Dall = [0], [0]

    FoldFac         = FOLDING_FACTOR
    MinModeCount    = MIN_MODE_COUNT
    TargetBinNummer = TARGET_BIN_COUNT 

    with open(filepath, "r") as file:

        shot_wavenumbers, shot_noise = [], []
        piece_k, piece_Delta2 = [], []

        for piece in range(NUM_FOLDED_POWER_SPECTRA_PIECES):

            Time = float(file.readline())
            Bins = int(file.readline())
            BoxSize = float(file.readline())
            Ngrid = float(file.readline())
            Dgrowth = float(file.readline())

            da = [
                [float(num) for num in file.readline().strip().split()]
                for _ in range(Bins)
            ]
            da = np.asarray(da)

            K         = da[:,0]
            Delta2    = da[:,1]
            ModePow   = da[:,2]
            ModeCount = da[:,3]
            Shot      = da[:,4]

            # There's the input power spectrum plot in the reference code, but 
            # we'll skip that for now
            if (piece == 0):

                linear_power_spec = load_linear_ps(
                    input_ps_path, 
                    Dgrowth=Dgrowth,
                    use_music=use_music_init_ps
                )
                k_lin = linear_power_spec["wavenumbers"]
                D2_lin = linear_power_spec["normalized"]

            # The shot noise is plotted here in the reference code, but we'll skip that for now
            shot_wavenumbers.extend(K)
            shot_noise.extend(Shot)

            Delta2 -= Shot
            SumPower = Delta2 * ModeCount

            kmin = 2*np.pi/BoxSize * float(FoldFac)**piece
            kmax = 2*np.pi/BoxSize * Ngrid/2.0/4  * float(FoldFac)**piece

            if (piece > 0):
                kmin = 2*np.pi/BoxSize * Ngrid/2.0/4 * float(FoldFac)**(piece-1.0)

            MinDlogK = (np.log10(np.max(K)) - np.log10(np.min(K))) / TargetBinNummer

            istart = 0
            ind = np.array(istart)
            k_list = [0]
            Delta2_list = [0]
            Delta2lin_list = [0]
            count_list = [0]

            while (istart < Bins):
                count = np.sum(ModeCount[ind])
                deltak = (np.log10(np.max(K[ind])) - np.log10(np.min(K[ind])))

                istart += 1
                if deltak >= MinDlogK or (np.max(K[ind]) < 2.0 * kmin and piece == 0):
                    kk = np.exp(np.sum(np.log(K[ind]) * ModeCount[ind]) / np.sum(ModeCount[ind]))
                    d2 = np.sum(Delta2[ind] * ModeCount[ind]) / np.sum(ModeCount[ind])
                    Delta2lin = np.exp(np.interp(np.log(K[ind]), np.log(k_lin), np.log(D2_lin)))
                    d2lin = np.sum(Delta2lin * ModeCount[ind]) / np.sum(ModeCount[ind])

                    if d2 > 0:
                        k_list = np.append(k_list, kk)
                        Delta2_list = np.append(Delta2_list, d2)
                        Delta2lin_list = np.append(Delta2lin_list, d2lin)
                        count_list = np.append(count_list, np.sum(ModeCount[ind]))
                    ind = np.array(istart)

                else:
                    ind = np.append(ind,istart)


            # Convert lists to numpy arrays and slice off the initial zero
            k_list = np.array(k_list[1:])
            Delta2_list = np.array(Delta2_list[1:])
            Count_list = np.array(count_list[1:])
            Delta2lin_list = np.array(Delta2lin_list[1:])

            # Select the measurements within the range [kmin, kmax] and adopt them for our total measurement
            ind = np.where((k_list >= kmin) & (k_list <= kmax))

            Kall = np.concatenate((Kall, k_list[ind]))
            Dall = np.concatenate((Dall, Delta2_list[ind]))

            # get rid off the trailing zeros
            Kall, Dall = Kall[1:], Dall[1:]

            Sall = np.exp(np.interp(np.log(Kall), np.log(K), np.log(Shot)))

            ind = np.where(Dall > Sall/2)[0]

            piece_k.extend(Kall[ind])
            piece_Delta2.extend(Dall[ind])

        sorted_idxs = np.argsort(piece_k)
        piece_k = np.asarray(piece_k)[sorted_idxs]
        piece_Delta2 = np.asarray(piece_Delta2)[sorted_idxs]

        piece_k = np.asarray(piece_k)
        piece_Delta2 = np.asarray(piece_Delta2)
        shot_wavenumbers = np.asarray(shot_wavenumbers)
        shot_noise = np.asarray(shot_noise)

        output = {
            "scale_factor" : Time,
            "linear" : linear_power_spec,
            "shot_noise" : {
                "wavenumbers" : shot_wavenumbers,
                "normalized" : shot_noise,
                "amplitudes" : shot_noise * (2 * np.pi**2) / shot_wavenumbers**3
            }
        }

        nonlinear = get_nonlinear_power_spec(
            shot_wavenumbers=shot_wavenumbers,
            shot_noise=shot_noise,
            pieced_wavenumbers=piece_k,
            pieced_normalized=piece_Delta2
        )

        if all((value.size == 0) for value in nonlinear.values()):
            print(f"snap_id = {int(filepath.stem.split('_')[1])}")
            print("The minimum wavenumber is not the same as the shot noise wavenumber")

        return output | {"nonlinear" : nonlinear}


def get_nonlinear_power_spec(
        shot_wavenumbers: np.ndarray, 
        shot_noise: np.ndarray, 
        pieced_wavenumbers: np.ndarray, 
        pieced_normalized: np.ndarray
    ) -> dict[str, np.ndarray]:

    if np.all(
        (np.log10(pieced_wavenumbers) > 0.0) & (np.log10(pieced_normalized) > 0.0)
    ):
        return {
            "wavenumbers" : np.empty(0), 
            "normalized" : np.empty(0),
            "amplitudes" : np.empty(0)
        }
    
    shot_noise_interp = interp1d(
        np.log(shot_wavenumbers), np.log(shot_noise), kind="linear"
    )
    interp_shot_noise = np.exp(shot_noise_interp(np.log(pieced_wavenumbers)))

    above_shot_noise_mask = (pieced_normalized > interp_shot_noise)

    if np.all(above_shot_noise_mask):
        mask = np.ones(len(pieced_wavenumbers), dtype=bool)
    else:
        chop_index = np.argmin(above_shot_noise_mask)
        mask = np.arange(len(pieced_wavenumbers)) <= chop_index

    wavenumbers = np.asarray(pieced_wavenumbers)[mask]
    normalized = np.asarray(pieced_normalized)[mask]
    amplitudes = normalized * (2 * np.pi**2) / wavenumbers**3

    return {
        "wavenumbers": wavenumbers,
        "normalized": normalized,
        "amplitudes": amplitudes,
    }

        
def get_power_spec_evo(
        power_spec_dir : Path, 
        use_music_init_ps: bool = False,
    ) -> dict[int, dict[str, float | dict[str, np.ndarray]]]:

    snap_num = lambda filename: int(filename.stem.split("_")[1])
    init_ps_name = "input_powerspec.txt" if use_music_init_ps else "inputspec_snapshot.txt"
    input_ps_path = Path(power_spec_dir, init_ps_name)

    return {
        snap_num(filepath) : get_power_spec(
            filepath, input_ps_path, use_music_init_ps=use_music_init_ps
        )
        for filepath in power_spec_dir.glob("powerspec_*.txt")
    }

def get_snapshot_mapping_from_power_spectra(power_spec_dir : Path) -> dict[str, np.ndarray]:

    snapshot_ids, scale_factors, redshifts = set(), set(), set()

    for filepath in power_spec_dir.glob("powerspec_*.txt"):
        snap_idx = int(filepath.stem.split("_")[1])
        with open(filepath, "r") as file:
            scale_factor = float(file.readline())
        snapshot_ids.add(snap_idx)
        redshift = (1.0 / scale_factor) - 1.0
        scale_factors.add(scale_factor)
        redshifts.add(redshift)


    results = {
        "snapshot_ids" : np.array(sorted(snapshot_ids)), 
        "scale_factors" : np.array(sorted(scale_factors)), 
        "redshifts" : np.array(sorted(redshifts, reverse=True))
    }

    if 'inputspec_snapshot.txt' in [filepath.name for filepath in power_spec_dir.glob("*")]:
        with open(Path(power_spec_dir, 'inputspec_snapshot.txt'), "r") as file:
            first_line = file.readline().split()
            # First line: snapshot_id, redshift (e.g., "99  78.2563")
            ic_redshift = float(first_line[0])
        results['initial'] = np.array([ic_redshift])

    return results



def get_spins_and_shapes(
        spin_and_shape_path: Path,
        return_concatenated: bool = True
    ) -> dict:
    shape_data = {}

    with h5py.File(spin_and_shape_path, 'r') as h5_file:
        # Handle snapshots (e.g., 'snapshot_009')
        for snapshot_str in h5_file.keys():
            snapshot_id = int(snapshot_str.split('_')[-1])
            shape_data[snapshot_id] = {}

            # Handle mass definitions (e.g., 'M200m')
            for mass_def_str in h5_file[snapshot_str].keys():
                if mass_def_str == "MM200m": 
                    shape_data[snapshot_id]["M200m"] = {}
                else:
                    shape_data[snapshot_id][mass_def_str] = {}

                mass_def_group = h5_file[snapshot_str][mass_def_str]

                # Handle mass bins (e.g., 'M12')
                for mass_bin_str in mass_def_group.keys():
                    mass_bin = int(mass_bin_str.split('M')[-1])

                    if mass_def_str == "MM200m":
                        shape_data[snapshot_id]["M200m"][mass_bin] = {}
                    else:
                        shape_data[snapshot_id][mass_def_str][mass_bin] = {}

                    mass_bin_group = mass_def_group[mass_bin_str]

                    # Handle all measurements for this mass bin
                    for measure_type in mass_bin_group.keys():
                        measure_data = mass_bin_group[measure_type]

                        # Handle both array datasets and scalar datasets
                        if measure_data.shape == ():  
                            if mass_def_str == "MM200m": # scalar value
                                shape_data[snapshot_id]["M200m"][mass_bin][measure_type] = \
                                    measure_data[()]
                            else:
                                shape_data[snapshot_id][mass_def_str][mass_bin][measure_type] = \
                                    measure_data[()]
                        elif mass_def_str == "MM200m": # array value
                            shape_data[snapshot_id]["M200m"][mass_bin][measure_type] = \
                                measure_data[:]
                        else:
                            shape_data[snapshot_id][mass_def_str][mass_bin][measure_type] = \
                                measure_data[:]
                            
            if return_concatenated:
                shape_data[snapshot_id] = concatenate_physical_catalog(shape_data[snapshot_id])
                            
    return shape_data



def load_spin_and_shapes_data(
        spin_and_shape_dir: Path, 
        from_primary_catalog: bool = False,
        return_concatenated: bool = True
    ) -> OrderedDict:

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    pattern = (
        "L*_N*_primary_spin_and_shapes.hdf5"
        if from_primary_catalog else 
        "L*_N*_box*_spin_and_shapes.hdf5"
    )

    for filepath in spin_and_shape_dir.glob(pattern):

        box_size, num_particles, box_seed = filepath.name.split("_")[:3]
        L, N, seed_num = int(box_size[1:]), int(num_particles[1:]), int(box_seed[3:])
        file_data = get_spins_and_shapes(
            spin_and_shape_path=filepath,
            return_concatenated=return_concatenated
        )
        data[N][L][seed_num] = file_data

    return OrderedDict({
        N : {
            L : {
                seed_num : data[N][L][seed_num]
                for seed_num in sorted(data[N][L])
            }
            for L in sorted(data[N])
        }
        for N in sorted(data)
    })


def concatenate_physical_catalog(catalog: PhysicalCatalog) -> ConcatenatedPhysicalCatalog:
    collected_catalog = defaultdict(lambda: defaultdict(list))

    for mdef_key, mdef_data in catalog.items():

        for mass_bin_data in mdef_data.values():

            for profile_type, data in mass_bin_data.items():
                collected_catalog[mdef_key][profile_type].append(data)

    # pdb.set_trace()

    concatenated_catalog = {
        mdef_key: {
            profile_type: np.concatenate(data, axis=0)
            for profile_type, data in mdef_data.items()
        }
        for mdef_key, mdef_data in collected_catalog.items()
    }

    for mdef_key in concatenated_catalog:
        order = np.argsort(concatenated_catalog[mdef_key]["group_ids"])
        for profile_type in concatenated_catalog[mdef_key]:
            concatenated_catalog[mdef_key][profile_type] = (
                concatenated_catalog[mdef_key][profile_type][order]
            )

    return concatenated_catalog

def load_physical_catalog_data(
        filepath: Path, 
        return_concatenated: bool = False
    ) -> PhysicalCatalog | ConcatenatedPhysicalCatalog:
    
    catalog_data = {}

    with h5py.File(filepath, "r") as f:
        # top-level groups: "M{mdef}"
        for mdef_group_name, mdef_group in f.items():
            # strip leading "M" to get original mdef key
            mdef_key = mdef_group_name[1:]
            catalog_data[mdef_key] = {}

            # second-level groups: "M{mass_bin}"
            for mass_bin_group_name, mass_bin_group in mdef_group.items():
                # strip "M" and convert to int
                mass_bin = int(mass_bin_group_name[1:])
                catalog_data[mdef_key][mass_bin] = {}

                # datasets inside: one per profile_type
                for profile_type, ds in mass_bin_group.items():
                    # load the entire array into memory
                    catalog_data[mdef_key][mass_bin][profile_type] = ds[()]

    return (
        concatenate_physical_catalog(catalog_data) 
        if return_concatenated else 
        catalog_data
    )

# Finish this out before going to sleep...
def load_all_physical_catalogs(
        directory: Path, 
        comoving_box_size: int,
        num_particles: int,
        seed_num: int = -1,
        sampling_mdef: str = "200m",
        is_from_primary: bool = False,
        return_concatenated: bool = False
    ) -> dict[str, dict[int, dict[str, dict[int, dict[str, np.ndarray]]]]]:

    snapshot_ids = get_phys_catalog_snap_ids(directory, seed_idx=seed_num)

    catalogs = {}

    for snapshot_id in tqdm(snapshot_ids, desc="Loading Physical Catalogs"):

        phys_catalog_path = get_physical_catalog_path(
            sim_data_dir=directory,
            box_size=comoving_box_size,
            num_particles=num_particles,
            snap_idx=snapshot_id,
            seed_num=seed_num,
            is_from_primary=is_from_primary,
        )

        if not phys_catalog_path.exists():
            print(f"File not found for snapshot {snapshot_id}")
            continue

        catalog = load_physical_catalog_data(
            filepath=phys_catalog_path, 
            return_concatenated=return_concatenated
        )
        catalogs[snapshot_id] = catalog[sampling_mdef]

    return catalogs


def load_saved_halo_particles(filepath: Path) -> dict[str, int | float | np.ndarray]:
    with h5py.File(filepath, "r") as f:

        return {
            "halo_id": int(f["halo_id"][()]),
            "snapshot_id": int(f["snapshot_id"][()]),
            "scale_factor": float(f["scale_factor"][()]),
            "redsift": float(f["redshift"][()]),
            "age": float(f["age"][()]),
            "lookback_time": float(f["lookback_time"][()]),
            "query_radius": float(f["query_radius"][()]),
            "particle_mass": float(f["particle_mass"][()]),
            "halo_position": np.array(f["halo_position"][()]),
            "coordinates": np.array(f["coordinates"][()]),
        }