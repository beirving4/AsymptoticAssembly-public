"""Simulation save-name construction.

Deterministic, filesystem-free naming for one simulation's analysis products.
The convention is a property of the products, so it is public and portable;
choosing the directory those products live in is research infrastructure and
stays private. Same seam as `asymptotic.utils.physical_catalog` (B1c).
"""
from __future__ import annotations


def seed_path_extension(seed_num: int) -> str:
    """Path/name fragment identifying one simulation of a suite.

    Two seeds name distinguished runs; every other value, including negative
    ones, keeps its sign in a ``box`` fragment.
    """
    match seed_num:
        case -2:
            return "Planck"
        case -1:
            return "primary"
        case _:
            return f"box{seed_num}"


def get_sim_save_name(box_size: int, num_particles: int, seed_num: int) -> str:
    """Stem shared by every analysis product of one simulation."""
    return f"L{box_size}_N{num_particles}_{seed_path_extension(seed_num)}"
