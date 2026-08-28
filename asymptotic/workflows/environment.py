"""Thread-environment configuration, applied before any numerical import.

Standard library only, and the policy is not reinvented here: it delegates to
``asymptotic.utils.threads``, which already implements the project's rules —
an existing environment value wins, OpenMP gets the requested count with safe
binding defaults, and the BLAS/NumExpr libraries default to **1** so a
threaded OpenMP region does not nest a threaded BLAS underneath it.
"""
from __future__ import annotations

import os

from ..utils.threads import _assert_threads_match_env, _ensure_omp_binding
from .specs import ComputeSpec

#: OpenMP thread count and its binding defaults
OMP_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
#: nested-parallelism guards: these default to 1, never to the OpenMP count
BLAS_VARS: tuple[str, ...] = (
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
THREAD_ENV_VARS: tuple[str, ...] = OMP_VARS + BLAS_VARS

#: callsite label for the existing thread assertion/debug path
_CALLSITE = "workflows.environment"


def configure_thread_environment(compute: ComputeSpec) -> dict[str, str]:
    """Apply ``threads_per_process`` through the project's thread policy.

    Returns only the variables this call actually set, so a runner can report
    what it changed. An all-``None`` ``ComputeSpec`` — the default — changes
    nothing and returns an empty mapping.
    """
    if compute.threads_per_process is None:
        return {}

    unset_before = [name for name in THREAD_ENV_VARS if not os.environ.get(name)]
    # setdefault semantics throughout: a scheduler or user value wins
    _ensure_omp_binding(compute.threads_per_process)
    # a requested count that disagrees with the winning environment follows
    # the existing assertion (AA_ASSERT_THREADS) / debug-logging path
    _assert_threads_match_env(compute.threads_per_process, _CALLSITE)

    return {name: os.environ[name]
            for name in unset_before if os.environ.get(name)}
