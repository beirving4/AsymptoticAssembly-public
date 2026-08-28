"""Compatibility re-export; canonical home is :mod:`asymptotic.merger_trees.full_tree`.

This established import path is kept temporarily for existing callers.
New code should import from ``asymptotic.merger_trees.full_tree``.
"""
from ..merger_trees.full_tree import (
    DEFAULT_DESC_SNAP,
    FullTreeStore,
    TesseraTreeBackend,
)

__all__ = [
    "DEFAULT_DESC_SNAP",
    "FullTreeStore",
    "TesseraTreeBackend",
]
