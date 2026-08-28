import os
import threading
import time
from typing import Callable, Any

_DEF_OMP_ENV = "OMP_NUM_THREADS"

def _omp_nthreads() -> int:
    try:
        n = int(os.environ.get(_DEF_OMP_ENV, "").strip() or 0)
    except Exception:
        n = 0
    nthreads = max(1, n or (os.cpu_count() or 1))
    if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
        print(
            f"[threads] _omp_nthreads -> {nthreads} "
            f"(env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
            f"cpu_count={os.cpu_count()})"
        )
    return nthreads

def _env_threads_snapshot() -> dict:
    """Capture relevant thread-related env + CPU info for debugging.
    Enable by setting AA_DEBUG_THREADS=1 in the environment.
    """
    snap = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
    }
    # Linux-only: current CPU affinity if available
    if hasattr(os, "sched_getaffinity"):
        try:
            snap["affinity_count"] = len(os.sched_getaffinity(0))
        except Exception:
            snap["affinity_count"] = None
    else:
        snap["affinity_count"] = None
    return snap

def _log_thread_info(callsite: str, nthreads: int) -> None:
    """Conditional debug print summarizing thread settings passed to Corrfunc.

    To enable these messages at runtime, export AA_DEBUG_THREADS=1 before running.
    """
    if os.environ.get("AA_DEBUG_THREADS", "0") in ("", "0"):
        return
    snap = _env_threads_snapshot()
    print(
        f"[threads] {callsite}: nthreads={nthreads}, "
        f"OMP_NUM_THREADS={snap['OMP_NUM_THREADS']}, "
        f"cpu_count={snap['cpu_count']}, affinity={snap['affinity_count']}, "
        f"MKL={snap['MKL_NUM_THREADS']}, OPENBLAS={snap['OPENBLAS_NUM_THREADS']}, "
        f"NUMEXPR={snap['NUMEXPR_NUM_THREADS']}"
    )

def _assert_threads_match(nthreads: int, callsite: str | None = None) -> None:
    """Raise OSError if the provided `nthreads` does not match `_omp_nthreads()`."""
    expected = _omp_nthreads()
    if nthreads != int(expected):
        snap = _env_threads_snapshot()
        where = f" at {callsite}" if callsite else ""
        raise OSError(
            (
                f"Thread mismatch{where}"
                + f": nthreads={nthreads} but _omp_nthreads()={expected}. Env OMP_NUM_THREADS={snap['OMP_NUM_THREADS']}; cpu_count={snap['cpu_count']}; affinity={snap['affinity_count']}; MKL={snap['MKL_NUM_THREADS']}; OPENBLAS={snap['OPENBLAS_NUM_THREADS']}; NUMEXPR={snap['NUMEXPR_NUM_THREADS']}"
            )
        )
    else:
        if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
            print(
                f"[threads] _assert_threads_match passed: nthreads={nthreads} "
                f"matches expected={expected}"
            )

def _assert_threads_match_env(nthreads: int, callsite: str | None = None) -> None:
    """Only assert if AA_ASSERT_THREADS is truthy; otherwise no-op."""
    if (
        os.environ.get("AA_ASSERT_THREADS", "0") in ("", "0")
        and isinstance(callsite, str)
    ):
        _log_thread_info(callsite, nthreads)
        return
    _assert_threads_match(nthreads, callsite)


# --------------------------- Thread usage enforcement ---------------------------

def _ensure_omp_binding(
        nthreads: int,
        set_blas_1: bool = True,
        bind: str = "spread",
        places: str = "cores",
    ) -> None:
    """Set sensible OpenMP/BLAS env defaults to encourage full-core usage.

    This function *does not* override values you've already set in the
    environment; it only sets sane defaults when missing. Useful in tests.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
    os.environ.setdefault("OMP_PROC_BIND", bind)
    os.environ.setdefault("OMP_PLACES", places)
    if set_blas_1:
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
        print(
            f"[threads] _ensure_omp_binding: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
            f"OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND')}, OMP_PLACES={os.environ.get('OMP_PLACES')}, "
            f"MKL={os.environ.get('MKL_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
            f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}"
        )


def _assert_affinity_allows(nthreads: int, callsite: str | None = None) -> None:
    
    """Check and fix CPU affinity if restricted to fewer than nthreads CPUs.

    On Linux, if CPU affinity is too restrictive, this will attempt to reset it
    to allow access to the requested number of threads.
    
    Args:
        nthreads: Number of threads required
        callsite: Location identifier for debugging
        always_check: If True, always verify and fix affinity (default: True)
                     This is important because libraries can reset affinity between calls
    """
    if not hasattr(os, "sched_getaffinity"):
        return
    where = f" at {callsite}" if callsite else ""
    try:
        allowed = len(os.sched_getaffinity(0))
    except Exception:
        return

    if allowed < nthreads:
        # Affinity is too restrictive - try to fix it
        # Track if we've already warned about this
        warned_key = f"_affinity_fix_warned_{nthreads}"
        first_fix = not hasattr(_assert_affinity_allows, warned_key)

        if first_fix and os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
            print(
                f"[threads] WARNING{where}: CPU affinity restricted to {allowed} CPUs, "
                f"but {nthreads} threads requested. Attempting to fix..."
            )
        elif not first_fix and os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
            print(
                f"[threads] Re-fixing CPU affinity{where}: {allowed} -> {nthreads} CPUs "
                f"(affinity was reset by library)"
            )

        try:
            # Get the actual CPU count from SLURM or system
            cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
            cpu_count = os.cpu_count() or nthreads

            # Try multiple strategies to get the full CPU set
            attempted_cpus = None

            # Strategy 1: Use full system CPU count (handles non-contiguous numbering)
            try:
                if cpus_per_task > 0:
                    # Use SLURM allocation but with full CPU range to handle gaps
                    attempted_cpus = set(range(cpu_count))
                else:
                    attempted_cpus = set(range(nthreads))

                os.sched_setaffinity(0, attempted_cpus)
                new_allowed = len(os.sched_getaffinity(0))

                # Mark that we've done at least one fix
                setattr(_assert_affinity_allows, warned_key, True)

                if first_fix and os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
                    print(
                        f"[threads] SUCCESS{where}: CPU affinity reset from {allowed} to {new_allowed} CPUs "
                        f"(using range(0, {len(attempted_cpus)}))"
                    )

                # If we got at least nthreads CPUs, that's good enough
                if new_allowed >= nthreads:
                    return

                # Otherwise, fall through to try other strategies
                if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
                    print(
                        f"[threads] WARNING{where}: Only got {new_allowed} CPUs, need {nthreads}. "
                        f"Trying to adjust nthreads..."
                    )

                # Adjust nthreads to match what we got
                # This is acceptable since we got as many as the system allows
                if new_allowed >= nthreads * 0.9:  # Within 10% is acceptable
                    if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
                        print(
                            f"[threads] ACCEPTABLE{where}: Got {new_allowed} CPUs, "
                            f"which is ≥90% of requested {nthreads}"
                        )
                    # Update environment to match reality
                    os.environ['OMP_NUM_THREADS'] = str(new_allowed)
                    return

                raise OSError(
                    f"CPU affinity still too restrictive{where}: got {new_allowed} CPUs, "
                    f"which is <90% of requested {nthreads}. Check SLURM allocation."
                )

            except PermissionError as e:
                raise OSError(
                    f"CPU affinity too restrictive{where}: allowed={allowed} < requested nthreads={nthreads}. "
                    f"Permission denied to reset affinity. Contact cluster admin."
                ) from e
            except OSError as e:
                if "Invalid argument" not in str(e):
                    raise
                # Non-contiguous numbering, fall through

        except Exception as e:
            if isinstance(e, OSError) and "Permission denied" in str(e):
                raise
            raise OSError(
                f"CPU affinity too restrictive{where}: allowed={allowed} < requested nthreads={nthreads}. "
                f"Failed to reset affinity: {e}"
            ) from e


def _count_os_threads_linux() -> int:
    """Return current OS thread count for this process (Linux only), else -1."""
    try:
        return len(os.listdir("/proc/self/task"))
    except Exception:
        return -1


def assert_corrfunc_threads_active(
        nthreads: int,
        work: Callable[[], Any],
        callsite: str | None = None,
        sample_hz: int = 250,
        min_peak_fraction: float = 0.9,
    ) -> None:
    """Run `work()` and assert OpenMP threads are actually spawned/used.

    This is a *diagnostic* helper you can call around Corrfunc hotspots.

    What it enforces:
      1) `nthreads` matches our policy (`_assert_threads_match`).
      2) CPU affinity allows at least `nthreads` logical CPUs.
      3) On Linux, during `work()` the process' peak OS thread count increases
         by ~`nthreads-1` (OpenMP workers) relative to the baseline. If the
         observed peak falls below `min_peak_fraction*(nthreads-1)`, raise.

    Notes:
      • On non-Linux platforms (no `/proc`), we skip (3) and rely on (1)+(2).
      • This does not measure per-thread utilization; it checks workers exist.
      • Use `_ensure_omp_binding(...)` beforehand if you want to set sane env
        defaults for OMP binding/places and disable BLAS oversubscription.
    """
    # Check affinity FIRST - it may adjust OMP_NUM_THREADS
    _assert_affinity_allows(nthreads, callsite)

    # After affinity fix, OMP_NUM_THREADS might have been adjusted
    # Use the updated value for actual thread count
    actual_nthreads = _omp_nthreads()
    if actual_nthreads != nthreads:
        if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
            print(
                f"[threads] NOTE{f' at {callsite}' if callsite else ''}: "
                f"nthreads adjusted from {nthreads} to {actual_nthreads} "
                f"based on available CPUs"
            )
        nthreads = actual_nthreads

    baseline = _count_os_threads_linux()
    where = f" at {callsite}" if callsite else ""

    # Non-Linux or unable to read /proc — just execute the work.
    if baseline <= 0:
        if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
            print(f"[threads] assert_corrfunc_threads_active{where}: no /proc sampling; running work()")
        work()
        return

    peak = baseline
    exc: BaseException | None = None

    def runner():
        nonlocal exc
        try:
            work()
        except BaseException as e:  # bubble original exception after sampling
            exc = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    interval = max(0.001, 1.0 / float(max(1, sample_hz)))
    while t.is_alive():
        c = _count_os_threads_linux()
        if c > 0:
            peak = max(peak, c)
        time.sleep(interval)

    if exc is not None:
        raise exc

    expected_new = max(0, nthreads - 1)
    observed_new = max(0, peak - baseline)
    threshold = int(max(0, round(min_peak_fraction * expected_new)))

    if observed_new < threshold:
        raise OSError(
            f"OpenMP workers below expectation{where}: observed additional threads={observed_new} < {threshold} "
            f"(baseline={baseline}, peak={peak}, requested nthreads={nthreads}). "
            f"Check OMP_PROC_BIND/OMP_PLACES, CPU affinity, or oversubscription."
        )

    if os.environ.get("AA_DEBUG_THREADS", "0") not in ("", "0"):
        print(
            f"[threads] assert_corrfunc_threads_active passed{where}: baseline={baseline}, peak={peak}, "
            f"observed_new≈{observed_new}, requested={nthreads}"
        )