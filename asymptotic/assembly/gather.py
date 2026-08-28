"""Aperture/quantity configuration + gather orchestration.

Thin layer over ``backend.gather``: given an aperture and a list of canonical
quantities, pull each aligned ``[n_halos, n_snaps]`` array, skipping (with a
warning) any quantity the chosen backend cannot supply.
"""
from __future__ import annotations

import warnings

import numpy as np
from attrs import define

from ..merger_trees.backends import (
    CANONICAL_APERTURES,
    DEFAULT_QUANTITIES,
    MergerTreeBackend,
)
from ..merger_trees.topology import BranchTopology


@define(slots=True)
class GatherConfig:
    apertures: tuple[str, ...] = CANONICAL_APERTURES
    quantities: tuple[str, ...] = DEFAULT_QUANTITIES

    @classmethod
    def default_primary(cls) -> "GatherConfig":
        return cls()


def gather_aperture(
    backend: MergerTreeBackend,
    topo: BranchTopology,
    aperture: str,
    quantities: tuple[str, ...],
    warn_missing: bool = True,
) -> dict[str, np.ndarray]:
    """Return ``{quantity: [n_halos, n_snaps]}`` for those available at ``aperture``."""
    out: dict[str, np.ndarray] = {}
    for q in quantities:
        if not backend.available(q, aperture):
            if warn_missing:
                warnings.warn(
                    f"{q}@{aperture} unavailable in {backend.name} backend; skipping.",
                    stacklevel=2,
                )
            continue
        out[q] = backend.gather(q, aperture, topo)
    return out
