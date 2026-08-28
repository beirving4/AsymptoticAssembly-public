from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")


import numpy as np
from tqdm.auto import tqdm
from argparse import Namespace
from itertools import combinations
from collections import defaultdict
from collections.abc import Collection

from ..distribution.slope import (
    get_mass_distribution_milestones, 
    get_so_values
)

from ..particles.orbital import (
    get_radial_distances
)
from ..cosmo.model import Cosmology
from ..particles.collection import (
    BoxParticleCollection, BoundedParticleCollection
)

from ..bounded.fof import FOFGroup, FOFGroups, FOFEvoData, FOFGroupEvoData


def update_mpb_boundary_values(
        snap_idx: int,
        milestones: dict[str, float | np.ndarray],
        so_values: dict[str, dict[str, float]],
        so_values_evo: dict[int, dict[str, dict[str, list]]],
        milestones_evo: dict[int, dict[str, list]],
        query_radii_evo: dict[int, dict[str, list]],  
    ) -> None:
    
    for key, value in so_values.items():    
        query_radii_evo[snap_idx]["200m"].append(so_values["R"]["200m"])
        for mdef, mdef_value in value.items():
            so_values_evo[snap_idx][key][mdef].append(mdef_value)

    for key, value in milestones.items():
            milestones_evo[snap_idx][key].append(value)
            query_radii_evo[snap_idx]["enclosed_steepest"].append(milestones["last_bound"])
            query_radii_evo[snap_idx]["splashback"].append(milestones["splashback"])

def extract_descendant_radial_distances(
        box_particles: BoxParticleCollection,
        query_radii: dict[str, float | np.ndarray],
        group: FOFGroup,
        contained_halo_ids: dict[str, dict[int, np.ndarray]],
        prog_radial_distances: dict[int, dict[str, list]],
        bounded_counts: dict[int, dict[str, list]],
        cosmo: Cosmology,
    ) -> None:

    for radii_name, query_radius in query_radii.items():
        # radial_subset = region_particles.get_radial_subset(query_radius)

        group.set_particles(
            box_particles=box_particles,
            r_query=query_radius,
            cosmo=cosmo,
            use_comoving=False
        )

        contained_halo_ids[radii_name][group.id_number] = group.particles.ids
        prog_radial_distances[group.moment.snapshot_id][radii_name].append(
            group.particles.radial_distances 
        )

        bounded_counts[group.moment.snapshot_id][radii_name].append(
            group.particles.radial_distances.size
        )


def extract_prog_radial_distances(
        box_particles: BoxParticleCollection,
        query_radii: dict[str, float | np.ndarray],
        desc_group_id: int,
        group: FOFGroup,
        contained_halo_ids: dict[str, dict[int, np.ndarray]],
        prog_radial_distances: dict[int, dict[str, list]],
        bounded_counts: dict[int, dict[str, list]],
    ) -> None:

    for radii_name, query_radius in query_radii.items():

        particle_subset = box_particles.get_particles(
            ids=contained_halo_ids[radii_name][desc_group_id]
        )
        radial_distances = get_radial_distances(
            positions=particle_subset.coordinates,
            box_size=box_particles.box_size,
            center_of_mass=np.asarray(group.coordinate),
        )
        prog_radial_distances[group.moment.snapshot_id][radii_name].append(
            radial_distances
        )

        bounded_counts[group.moment.snapshot_id][radii_name].append(
            radial_distances[radial_distances <= query_radius].size
        )

def extract_radial_distances(
        box_particles: BoxParticleCollection,
        cosmo: Cosmology,
        group: FOFGroup,
        snap_idx: int,
        desc_snap_id: int,
        desc_group_id: int,
        query_radii: dict[str, float | np.ndarray],
        contained_halo_ids: dict[str, dict[int, np.ndarray]],
        prog_radial_distances: dict[int, dict[str, list]],
        bounded_counts: dict[int, dict[str, list]],
    ) -> None:

    if (snap_idx == desc_snap_id):

        extract_descendant_radial_distances(
            box_particles=box_particles,
            cosmo=cosmo,
            group=group,
            contained_halo_ids=contained_halo_ids,
            prog_radial_distances=prog_radial_distances,
            query_radii = query_radii,
            bounded_counts=bounded_counts,
        )

    else:

        extract_prog_radial_distances(
            box_particles=box_particles,
            desc_group_id=desc_group_id,
            group = group,
            contained_halo_ids=contained_halo_ids,
            prog_radial_distances=prog_radial_distances,
            query_radii = query_radii,
            bounded_counts=bounded_counts
        )


def update_halo_particle_evolution(
        box_particles: BoxParticleCollection, 
        halo: FOFGroup,
        cosmo: Cosmology,
        snap_idx: int,
        desc_snap_id: int,
        desc_group_id: int,
        contained_halo_ids: dict[str, dict[int, np.ndarray]],
        milestones_evo: dict[int, dict[str, list]],
        so_values_evo: dict[int, dict[str, dict[str, list]]],
        prog_radial_distances: dict[int, dict[str, list]], 
        query_radii_evo: dict[int, dict[str, list]], 
        bounded_counts: dict[int, dict[str, list]],
        num_radial_points: int = 50, 
        use_comoving: bool = True, 
        return_in_physical_units: bool = True,
        min_radius: float = 0.01, 
        max_radius: float = 1.0
    ) -> None:

    region_particles = BoundedParticleCollection.from_box_particles(
        box_particles=box_particles,
        group_coordinates=np.asarray(halo.coordinate), 
        group_velocities=np.asarray(halo.velocity),
        query_radii=halo.region_radius,
        cosmo=cosmo,
        use_comoving=use_comoving,
    )

    mass_distribution = region_particles.get_mass_distribution(
        num_radial_points=num_radial_points, 
        return_in_physical_units=return_in_physical_units
    )

    so_values = get_so_values(
        profiles=mass_distribution,
        z=halo.moment.redshift,
        cosmo=cosmo,
    )

    milestones = get_mass_distribution_milestones(
        profiles=mass_distribution, 
        min_radius=min_radius * so_values["R"]["vir"],
        max_radius=max_radius * so_values["R"]["vir"]
    )

    update_mpb_boundary_values(
        snap_idx=snap_idx,
        milestones=milestones,
        so_values=so_values,
        so_values_evo=so_values_evo,
        milestones_evo=milestones_evo,
        query_radii_evo=query_radii_evo
    )

    extract_radial_distances(
        box_particles=box_particles,
        cosmo=cosmo,
        snap_idx=snap_idx,
        desc_snap_id=desc_snap_id,
        desc_group_id=desc_group_id,
        group=halo,
        contained_halo_ids=contained_halo_ids,
        prog_radial_distances=prog_radial_distances,
        query_radii = {
            "enclosed_steepest" : milestones["last_bound"],
            "splashback" : milestones["splashback"],
            "200m" : so_values["R"]["200m"]
        },
        bounded_counts=bounded_counts,
    )


def halo_population_particle_evolution(
        box_particles: BoxParticleCollection,
        fof_groups: Collection[FOFGroup],
        cosmo: Cosmology,
        snap_idx: int,
        desc_snap_id: int,
        desc_group_ids: Collection[int],
        contained_halo_ids: dict[str, dict[int, np.ndarray]],
        milestones_evo: dict[int, dict[str, list]],
        so_values_evo: dict[int, dict[str, dict[str, list]]],
        prog_radial_distances: dict[int, dict[str, list]], 
        query_radii_evo: dict[int, dict[str, list]],
        bounded_counts: dict[int, dict[str, list]],
        num_radial_points: int = 50,
        use_comoving: bool = True, 
        return_in_physical_units: bool = True,
        min_radius: float = 0.01, 
        max_radius: float = 1.0
    ):

    if not isinstance(desc_group_ids, np.ndarray):
        desc_group_ids = np.array(desc_group_ids)
    

    for i, group in tqdm(
        enumerate(fof_groups), 
        desc=f"Extracting info from snap {snap_idx}",
        total=len(fof_groups)
    ): 
        
        update_halo_particle_evolution(
            box_particles=box_particles,
            halo=group,
            cosmo=cosmo,
            snap_idx=snap_idx,
            desc_snap_id=desc_snap_id,
            desc_group_id=desc_group_ids[i],
            contained_halo_ids=contained_halo_ids,
            milestones_evo=milestones_evo,
            so_values_evo=so_values_evo,
            prog_radial_distances=prog_radial_distances,
            query_radii_evo=query_radii_evo,
            bounded_counts=bounded_counts,
            num_radial_points=num_radial_points,
            use_comoving=use_comoving,
            return_in_physical_units=return_in_physical_units,
            min_radius=min_radius,
            max_radius=max_radius
        )


def concatenate_prog_radial_distances(
        prog_radial_distances: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, np.ndarray]]:

    return {
        snap_idx: {
            radii_name: np.concatenate(radial_values)
            for radii_name, radial_values in radii_data.items()
        }
        for snap_idx, radii_data in prog_radial_distances.items()
    }

def concatenate_milestone_evo(
        milestones_evo: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, np.ndarray]]:

    def is_valid_value(value: float | int | np.ndarray) -> bool:
        if isinstance(value, np.ndarray):
            return (value.ndim < 2) and (value.size == 1)
        return True
        

    return {
        snap_idx: {
            milestones_name: np.stack(milestones_values)
            for milestones_name, milestones_values in milestones_data.items()
            if is_valid_value(milestones_values[0])
        }
        for snap_idx, milestones_data in milestones_evo.items()
    }

def concatenate_so_values_evo(
        so_values_evo: dict[int, dict[str, dict[str, list]]]
    ) -> dict[int, dict[str, dict[str, np.ndarray]]]:

    return {
        snap_idx: {
            so_name: {
                mdef: np.stack(mdef_values)
                for mdef, mdef_values in so_data.items()
            }
            for so_name, so_data in so_values.items()
        }
        for snap_idx, so_values in so_values_evo.items()
    }



def get_radial_boundary_evo(
        query_radii_evo: dict[int, dict[str, list]]
    ) -> dict[str, dict[int, float]]: 

    boundary_evo = defaultdict(dict)

    for snap_idx, radii_data in query_radii_evo.items():
        for radii_name, radial_values in radii_data.items():
            boundary_evo[radii_name][snap_idx] = np.nanmedian(radial_values)

    return boundary_evo

def get_snapshot_pairings(snap_ids: Collection[int]) -> Collection[tuple[int, int]]:
    return sorted([tuple(sorted(pair)) for pair in combinations(snap_ids, 2)])

def get_radial_distributions2D(
        prog_radial_distances: dict[int, dict[str, np.ndarray]],
        radial_boundary_evo: dict[str, dict[int, float]],
        num_mesh_bins: int = 50
    ) -> dict[str, dict[tuple[int, int], dict]]: 

    distributions = defaultdict(lambda: defaultdict(dict))

    for scaling_key, boundary_evo in radial_boundary_evo.items():
        for before_snap_idx, after_snap_idx in get_snapshot_pairings(boundary_evo.keys()):

            before = prog_radial_distances[before_snap_idx][scaling_key]
            after = prog_radial_distances[after_snap_idx][scaling_key]

            # Mask out invalid values
            mask = (before > 0) & (after > 0) & np.isfinite(before) & np.isfinite(after)
            before, after = before[mask], after[mask]

            # Bin edges
            before_bins = np.logspace(
                np.log10(before.min()), np.log10(before.max()), num_mesh_bins
            )
            after_bins = np.logspace(
                np.log10(after.min()), np.log10(after.max()), num_mesh_bins
            )

            # Create 2D histogram
            counts, xedges, yedges = np.histogram2d(
                before,
                after,
                bins=(before_bins, after_bins),
                density=True
            )

            distributions[scaling_key][(before_snap_idx, after_snap_idx)] = {
                "counts" : counts,
                "xedges" : xedges,
                "yedges" : yedges
            }

    return distributions



def get_bounded_stats(
        bounded_counts: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, dict[str, float | np.ndarray]]]:

    bounded_stats = defaultdict(lambda: defaultdict(dict))

    for snap_idx, count_data in bounded_counts.items():
        if snap_idx == max(bounded_counts): continue
        for radii_name, particle_counts in count_data.items():
            prog_counts = np.asarray(particle_counts)
            desc_counts = np.asarray(bounded_counts[max(bounded_counts)][radii_name])

            bounded_stats[snap_idx][radii_name] = {
                "halo_pct" : np.divide(prog_counts, desc_counts),
                "total_bound_pcts" : prog_counts.sum() / desc_counts.sum()
            }

    return bounded_stats


def vectorize_bounded_counts(
        bounded_counts: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, np.ndarray]]:

    return {
        snap_idx: {
            radii_name: np.asarray(counts)
            for radii_name, counts in count_data.items()
        }
        for snap_idx, count_data in bounded_counts.items()
    }

def vectorize_query_radii_evo(
        query_radii_evo: dict[int, dict[str, list]]
    ) -> dict[int, dict[str, np.ndarray]]:

    return {
        snap_idx: {
            radii_name: np.asarray(counts)
            for radii_name, counts in count_data.items()
        }
        for snap_idx, count_data in query_radii_evo.items()
    }