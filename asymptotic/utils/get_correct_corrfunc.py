from __future__ import annotations

import os
import sys
import subprocess
import importlib
import shutil
import platform
from argparse import ArgumentParser
from typing import Tuple

__all__ = [
    "add_corrfunc_cli_args",
    "init_corrfunc_from_args",
    "ensure_corrfunc",
]

# ---------- CLI helper ----------

def add_corrfunc_cli_args(parser: ArgumentParser) -> None:
    """
    Add Corrfunc-related build/runtime options to an argparse parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to augment.
    """
    parser.add_argument(
        "--rebuild_corrfunc",
        action="store_true",
        help="Force rebuild Corrfunc from source with native CPU flags on this node.",
    )
    parser.add_argument(
        "--corrfunc_source",
        type=str,
        default="pypi",
        help=(
            "Source to install Corrfunc from if rebuilding. Use 'pypi' or a VCS URL "
            "like 'git+https://github.com/manodeep/Corrfunc.git#egg=Corrfunc'."
        ),
    )
    parser.add_argument(
        "--omp_threads",
        type=int,
        default=None,
        help="If set, export OMP_NUM_THREADS to this value for Corrfunc/OpenMP.",
    )
    parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="C compiler to use when rebuilding Corrfunc (e.g., gcc, clang).",
    )
    parser.add_argument(
        "--pip_no_build_isolation",
        action="store_true",
        help="Pass --no-build-isolation to pip when rebuilding Corrfunc.",
    )
    parser.add_argument(
        "--pip_user",
        action="store_true",
        help="Pass --user to pip when rebuilding Corrfunc.",
    )
    parser.add_argument(
        "--pip_verbose",
        type=int,
        default=0,
        help="Verbosity level for pip install when rebuilding Corrfunc (0=quiet).",
    )

# ---------- Build helpers ----------

def _pick_compiler_and_flags(cc_arg: str | None) -> Tuple[str | None, str, str]:
    """
    Choose a C compiler and sensible default CFLAGS/LDFLAGS based on the platform.
    Returns (cc, cflags, ldflags). If cc_arg is provided, it takes precedence.
    """
    sysname = platform.system()
    machine = platform.machine()

    # Prefer explicit argument, then $CC, then a detected compiler
    cc = cc_arg or os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")

    # Start from environment (so user-config is preserved) and only fill defaults if empty
    env_cflags = os.environ.get("CFLAGS", "").strip()
    env_ldflags = os.environ.get("LDFLAGS", "").strip()

    default_cflags = "-O3 -fPIC -march=native -mtune=native"
    default_ldflags = ""

    # OpenMP flags are compiler/OS specific
    if sysname == "Linux":
        # GCC & modern Clang on Linux: -fopenmp both compile & link
        default_cflags += " -fopenmp"
        default_ldflags += " -fopenmp"
    elif sysname == "Darwin":
        # Apple clang: users typically `brew install libomp`
        # Try to be helpful if Homebrew exists
        if shutil.which("brew"):
            default_cflags += " -Xpreprocessor -fopenmp"
            default_ldflags += " -lomp"
        # Else leave defaults; build may be single-threaded if libomp missing
    # Use environment if already set; else our defaults
    cflags = env_cflags or default_cflags
    ldflags = env_ldflags or default_ldflags

    print(f"[Corrfunc] Platform: {sysname} ({machine}), CC={cc or '(auto)'}")
    print(f"[Corrfunc] Defaulted CFLAGS='{cflags}' LDFLAGS='{ldflags}'")

    return cc, cflags, ldflags


def ensure_corrfunc(
    rebuild: bool = False,
    source: str = "pypi",
    cflags: str | None = None,
    ldflags: str | None = None,
    cc: str | None = None,
    no_build_isolation: bool = True,
    user: bool | None = None,
    pip_verbose: int = 0,
):
    """
    Ensure Corrfunc is importable. If `rebuild` is True or import fails, rebuild from source
    with CPU-local flags so OpenMP/C SIMD match the executing node.

    Parameters
    ----------
    rebuild : bool
        If True, force a rebuild via pip from source.
    source : str
        'pypi' (default) or a VCS URL like
        'git+https://github.com/manodeep/Corrfunc.git#egg=Corrfunc'.
    cflags, ldflags : str | None
        Extra flags to append to CFLAGS / LDFLAGS for the build. If None, sensible defaults
        are used (-O3 -fPIC -fopenmp -march=native -mtune=native on Linux).
    cc : str | None
        C compiler to set via $CC (e.g., 'gcc', 'clang').
    no_build_isolation : bool
        If True, pass --no-build-isolation to pip install.
    user : bool | None
        If True, pass --user to pip install; if None, do not pass anything.
    pip_verbose : int
        Verbosity level for pip install; number of -v flags to append.

    Returns
    -------
    Corrfunc module object
    """
    try:
        import Corrfunc as _cf  # type: ignore
        if not rebuild:
            return _cf
    except Exception:
        _cf = None

    env = os.environ.copy()

    # Auto-select flags/compiler if not provided
    sel_cc, sel_cflags, sel_ldflags = _pick_compiler_and_flags(cc)

    final_cflags = cflags if cflags is not None else sel_cflags
    final_ldflags = ldflags if ldflags is not None else sel_ldflags
    final_cc = cc if cc is not None else sel_cc

    env["CFLAGS"] = (env.get("CFLAGS", "") + " " + final_cflags).strip()
    env["LDFLAGS"] = (env.get("LDFLAGS", "") + " " + final_ldflags).strip()
    if final_cc:
        env["CC"] = final_cc

    # Prefer rebuilding the in-environment package so the compiled extension matches this node
    pkg = source if (source and source != "pypi") else "Corrfunc"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--force-reinstall",
        "--no-binary",
        ":all:",
        pkg,
    ]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    if user is True:
        cmd.append("--user")
    if pip_verbose > 0:
        cmd.extend(["-v"] * pip_verbose)

    print(f"[Corrfunc] Rebuilding from source via: {' '.join(cmd)}")
    print(
        f"[Corrfunc] Using CC='{env.get('CC','(default)')}', "
        f"CFLAGS='{env['CFLAGS']}', LDFLAGS='{env['LDFLAGS']}'"
    )
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Print last ~50 lines of stderr/stdout for diagnostics
        stderr_lines = e.stderr.splitlines() if e.stderr else []
        stdout_lines = e.stdout.splitlines() if e.stdout else []
        print("[Corrfunc] ERROR: pip install failed with the following output (last ~50 lines):")
        for line in stderr_lines[-50:]:
            print(line)
        for line in stdout_lines[-50:]:
            print(line)
        raise RuntimeError(
            "[Corrfunc] Failed to build Corrfunc from source. "
            "Consider trying one or more of the following:\n"
            "  - Use --pip_no_build_isolation to disable build isolation.\n"
            "  - Use --pip_user to install to user site packages.\n"
            "  - Specify a different compiler with --cc.\n"
            "  - Provide a local or alternative source with --corrfunc_source.\n"
            "See the output above for more details."
        ) from e

    if "Corrfunc" in sys.modules:
        del sys.modules["Corrfunc"]
    return importlib.import_module("Corrfunc")

# ---------- One-shot initializer from parsed args ----------

def init_corrfunc_from_args(args):
    """
    Given parsed CLI args (with add_corrfunc_cli_args applied), set OMP threads
    and ensure Corrfunc is importable/built for the current node. Returns the module.
    """
    # Optionally pin OpenMP threads from CLI or SLURM env
    omp_threads = getattr(args, "omp_threads", None)
    if omp_threads is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                omp_threads = int(slurm_cpus)
                os.environ["OMP_NUM_THREADS"] = str(omp_threads)
                print(f"OMP_NUM_THREADS set from SLURM_CPUS_PER_TASK to {omp_threads}")
            except ValueError:
                pass
    else:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        print(f"OMP_NUM_THREADS set to {os.environ['OMP_NUM_THREADS']}")

    Corrfunc = ensure_corrfunc(
        rebuild=getattr(args, "rebuild_corrfunc", False),
        source=getattr(args, "corrfunc_source", "pypi"),
        cc=getattr(args, "cc", None),
        no_build_isolation=getattr(args, "pip_no_build_isolation", True),
        user=getattr(args, "pip_user", False),
        pip_verbose=getattr(args, "pip_verbose", 0),
    )
    try:
        cf_ver = getattr(Corrfunc, "__version__", "<unknown>")
    except Exception:
        cf_ver = "<unknown>"
    print(f"Corrfunc ready (version: {cf_ver}).")
    return Corrfunc