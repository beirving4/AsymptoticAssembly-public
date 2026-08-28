"""Physical-catalogue file naming.

Deterministic, filesystem-free construction of a physical-catalogue path
under a caller-supplied root. The naming convention is a property of the
data products, so it is public and portable; choosing the root (laptop,
external volume, or a cluster allocation) is research infrastructure and
stays private.
"""
from __future__ import annotations

from pathlib import Path


def get_physical_catalog_path(
    sim_data_dir: Path,
    box_size: int,
    num_particles: int,
    snap_idx: int,
    seed_num: int = 1,
    is_from_primary: bool = False,
) -> Path:
    """Path of one physical catalogue below ``sim_data_dir``.

    Catalogues live in a ``physical_catalogs`` directory beside the rest of
    the simulation products; the primary run is labelled ``primary`` and every
    other run by its seed. Pure path construction: nothing is validated,
    checked, created, or read, and the caller's root spelling (relative or
    absolute) is preserved.
    """
    run = "primary" if is_from_primary else f"box{seed_num}"
    fname = f"L{box_size}_N{num_particles}_{run}_phys_catalog_{snap_idx:03d}.hdf5"

    return Path(sim_data_dir, "physical_catalogs", fname)
