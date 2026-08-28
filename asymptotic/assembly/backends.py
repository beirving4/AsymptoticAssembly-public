"""Compatibility re-export; canonical home is :mod:`asymptotic.merger_trees.backends`.

This established import path is kept temporarily for existing callers.
New code should import from ``asymptotic.merger_trees.backends``.
"""
from ..merger_trees.backends import (
    BASE_QUANTITIES,
    CANONICAL_APERTURES,
    DEFAULT_QUANTITIES,
    MergerTreeBackend,
    SHAPE_ENERGETICS_QUANTITIES,
    SUBHALO_FIELDS,
    canonical_name,
    open_merger_tree,
)

__all__ = [
    "BASE_QUANTITIES",
    "CANONICAL_APERTURES",
    "DEFAULT_QUANTITIES",
    "MergerTreeBackend",
    "SHAPE_ENERGETICS_QUANTITIES",
    "SUBHALO_FIELDS",
    "canonical_name",
    "open_merger_tree",
]
