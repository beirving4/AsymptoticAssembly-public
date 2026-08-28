"""Compatibility re-export; canonical home is :mod:`asymptotic.merger_trees.topology`.

This established import path is kept temporarily for existing callers.
New code should import from ``asymptotic.merger_trees.topology``.
"""
from ..merger_trees.topology import (
    BranchTopology,
    build_main_branches,
)

__all__ = [
    "BranchTopology",
    "build_main_branches",
]
