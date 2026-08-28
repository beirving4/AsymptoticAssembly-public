"""Branch-catalog workflow — Tessera backend loading.

Lightweight by contract: importing this module must not pull in NumPy, h5py,
Tessera, or any private infrastructure. The driver imports Tessera *before* the
numerical stack on purpose (an HDF5/OpenMP conflict), so `_load_tessera` is the
single place that ordering is expressed, and every caller goes through it.

H1b2b2a scope: the loader only. The workflow spec, registry entry, public
action, and the scientific implementation module arrive in later packages.
"""
from __future__ import annotations

import importlib
from types import ModuleType

#: backends in preference order; the second is the in-place build
BACKENDS: tuple[str, ...] = ("tessera", "_tessera")

#: the accepted missing-backend message, unchanged from the legacy driver
MISSING_BACKEND = (
    "tessera package not found. Please ensure it is installed or built.\n"
    "Check your Python path or build the package from source."
)


def _load_tessera() -> ModuleType:
    """Import the Tessera backend, or raise the deterministic error.

    Idempotent: repeated calls return the identical module object through
    Python's import cache, so no local caching is needed. A backend that is
    installed but fails on one of *its own* imports propagates unchanged —
    only a genuinely absent backend is reported as missing.
    """
    absent: ModuleNotFoundError | None = None

    for name in BACKENDS:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                # the backend exists; something it imports does not
                raise
            absent = error

    raise ImportError(MISSING_BACKEND) from absent
