"""
cic_pdf_loader.py
=================
Self-contained module for loading comoving and physical CIC density PDF
results from HDF5 jackknife files.

Includes:
    - All PDF normalization and moment helpers
    - The HDF5 group and file loaders
    - Filename parsers for comoving and physical CIC files
    - A consolidated loading loop that produces identically-structured
      nested dicts for either case
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

import h5py
import numpy as np
from scipy.integrate import trapezoid
from tqdm.auto import tqdm


# ============================================================
# Basic helpers
# ============================================================

RTOL = 1e-6
ATOL = 1e-12


def show(name: str, obj: Any) -> None:
    """Print HDF5 object info for debugging."""
    kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
    shape = getattr(obj, "shape", "")
    dtype = getattr(obj, "dtype", "")
    extra = f" shape={shape} dtype={dtype}" if kind == "Dataset" else ""
    print(f"{kind}: {name}{extra}")


def _normalize_pdf_over_x_trapz(
    pdf: np.ndarray,
    x_centers: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Normalize P(x) so integral P(x) dx = 1 using trapezoid rule over x_centers.
    """
    pdf = np.asarray(pdf, dtype=np.float64)
    x_centers = np.asarray(x_centers, dtype=np.float64)

    area = float(trapezoid(pdf, x=x_centers))
    if area > 0 and np.isfinite(area) and not np.isclose(area, 1.0, rtol=RTOL, atol=ATOL):
        pdf = pdf / area
        area = float(trapezoid(pdf, x=x_centers))
    return pdf, area


def _normalize_midpoint(
    P: np.ndarray,
    dx: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Normalize a PDF tabulated at bin centers under midpoint rule:
        integral P(x) dx ~ sum_i P_i Delta x_i
    """
    P = np.asarray(P, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    area = float(np.sum(P * dx))
    if area > 0 and np.isfinite(area):
        P = P / area
    return P, area


def _moment_midpoint(
    P: np.ndarray,
    x: np.ndarray,
    dx: np.ndarray,
    power: int,
) -> float:
    """
    Compute integral x^power P(x) dx under midpoint rule.
    """
    P = np.asarray(P, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    return float(np.sum((x**power) * P * dx))


def _variance_about_one(
    P: np.ndarray,
    x: np.ndarray,
    dx: np.ndarray,
) -> float:
    """
    Compute integral (x - 1)^2 P(x) dx under midpoint rule.
    """
    P = np.asarray(P, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    return float(np.sum(((x - 1.0) ** 2) * P * dx))


def _enforce_mean_one_and_unit_area(
    x_edges: np.ndarray,
    x_centers: np.ndarray,
    x_widths: np.ndarray,
    pdf: np.ndarray,
    vol_pdf: np.ndarray,
    n_passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Enforce simultaneously (under midpoint rule):
      (i)  sum P_V Delta x = 1
      (ii) sum x P_V Delta x = 1
      (iii) sum P_M Delta x = 1

    Returns
    -------
    x_edges_out, x_centers_out, x_widths_out, pdf_out, vol_pdf_out, xbar_final
    """
    x_edges = np.asarray(x_edges, dtype=np.float64)
    x_centers = np.asarray(x_centers, dtype=np.float64)
    x_widths = np.asarray(x_widths, dtype=np.float64)
    pdf = np.asarray(pdf, dtype=np.float64)
    vol_pdf = np.asarray(vol_pdf, dtype=np.float64)

    xbar_final = float("nan")

    for _ in range(n_passes):
        pdf, _ = _normalize_midpoint(pdf, x_widths)
        vol_pdf, _ = _normalize_midpoint(vol_pdf, x_widths)

        xbar = float(np.sum(x_centers * vol_pdf * x_widths))
        xbar_final = xbar
        if not (np.isfinite(xbar) and xbar > 0):
            break

        # x' = x / xbar,  P'(x') = xbar * P(x)
        x_edges = x_edges / xbar
        x_centers = x_centers / xbar
        x_widths = x_widths / xbar
        pdf = pdf * xbar
        vol_pdf = vol_pdf * xbar

    pdf, _ = _normalize_midpoint(pdf, x_widths)
    vol_pdf, _ = _normalize_midpoint(vol_pdf, x_widths)
    xbar_final = float(np.sum(x_centers * vol_pdf * x_widths))

    return x_edges, x_centers, x_widths, pdf, vol_pdf, xbar_final


# ============================================================
# Group loader
# ============================================================

def load_pdf_group(
    grp: h5py.Group,
    x_edges_base: np.ndarray,
    x_centers_base: np.ndarray,
    x_widths_base: np.ndarray,
) -> dict[str, Any]:
    """
    Load one histogram/PDF group and compute derived quantities.

    Parameters
    ----------
    grp
        HDF5 group containing histogram/pdf arrays.
    x_edges_base, x_centers_base, x_widths_base
        Raw x = 1+delta binning.

    Returns
    -------
    data : dict
    """
    histogram = np.asarray(grp["histogram"][:], dtype=np.float64)
    pdf = np.asarray(grp["pdf"][:], dtype=np.float64)  # stored mass-weighted by default

    # First-pass normalization in x
    pdf, pdf_integral_trapz_pre = _normalize_pdf_over_x_trapz(pdf, x_centers_base)

    # Derive volume-weighted PDF from mass histogram
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_hist = np.where(x_centers_base > 0, histogram / x_centers_base, 0.0)

    W_total = float(np.sum(vol_hist))
    if W_total > 0 and np.isfinite(W_total):
        vol_pdf = vol_hist / (W_total * x_widths_base)
        vol_pdf, vol_integral_trapz_pre = _normalize_pdf_over_x_trapz(vol_pdf, x_centers_base)
    else:
        vol_pdf = np.zeros_like(pdf)
        vol_integral_trapz_pre = 0.0

    # Enforce consistent x-axis + normalization
    x_edges_out, x_centers_out, x_widths_out, pdf, vol_pdf, xbar_final = _enforce_mean_one_and_unit_area(
        x_edges_base,
        x_centers_base,
        x_widths_base,
        pdf,
        vol_pdf,
        n_passes=2,
    )

    # Derived x-space curves
    scaled_pdf = pdf * x_centers_out
    var_pdf = pdf * x_centers_out**2
    vol_scaled_pdf = vol_pdf * x_centers_out

    # Variance / sigma
    variance_from_PV = _variance_about_one(vol_pdf, x_centers_out, x_widths_out)
    sigma_from_PV = float(np.sqrt(max(variance_from_PV, 0.0)))

    variance_from_PM = _variance_about_one(pdf, x_centers_out, x_widths_out)
    sigma_from_PM = float(np.sqrt(max(variance_from_PM, 0.0)))

    # Higher moments / diagnostics
    mean_x_from_PM = _moment_midpoint(pdf, x_centers_out, x_widths_out, power=1)
    second_moment_from_PV = _moment_midpoint(vol_pdf, x_centers_out, x_widths_out, power=2)
    second_moment_from_PM = _moment_midpoint(pdf, x_centers_out, x_widths_out, power=2)

    dlnx_out = np.diff(np.log(x_edges_out))

    data: dict[str, Any] = {
        "n_particles": grp.attrs["n_particles"],
        "mean_overdensity": grp.attrs["mean_overdensity"],
        "median_overdensity": grp.attrs["median_overdensity"],
        "histogram": histogram,
        "pdf": pdf,
        "scaled_pdf": scaled_pdf,
        "var_pdf": var_pdf,
        "vol_hist": vol_hist,
        "vol_pdf": vol_pdf,
        "vol_scaled_pdf": vol_scaled_pdf,
        # Variance / sigma
        "variance_from_PV": variance_from_PV,
        "sigma_from_PV": sigma_from_PV,
        "variance_from_PM": variance_from_PM,
        "sigma_from_PM": sigma_from_PM,
        "second_moment_from_PV": second_moment_from_PV,
        "second_moment_from_PM": second_moment_from_PM,
        "mean_x_from_PM": mean_x_from_PM,
        # Post-fix midpoint diagnostics
        "pdf_integral_dx": float(np.sum(pdf * x_widths_out)),
        "vol_pdf_integral_dx": float(np.sum(vol_pdf * x_widths_out)),
        "mean_x_from_PV": float(np.sum(x_centers_out * vol_pdf * x_widths_out)),
        "scaled_integral_dlnx": float(np.sum(scaled_pdf * dlnx_out)),
        "vol_scaled_integral_dlnx": float(np.sum(vol_scaled_pdf * dlnx_out)),
        "xbar_final": xbar_final,
        # Pre-fix trapz diagnostics
        "pdf_integral_trapz_pre": pdf_integral_trapz_pre,
        "vol_pdf_integral_trapz_pre": vol_integral_trapz_pre,
        # Common rescaled x-axis
        "_x_edges": x_edges_out,
        "_x_centers": x_centers_out,
        "_x_widths": x_widths_out,
    }

    if "histogram_error" in grp:
        data["histogram_error"] = grp["histogram_error"][:]
    if "pdf_error" in grp:
        data["pdf_error"] = grp["pdf_error"][:]

    return data


# ============================================================
# Main HDF5 loader
# ============================================================

def load_results_from_hdf5(
    load_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load histogram and PDF results from HDF5.

    In these files, the binned variable is x = 1 + delta (dimensionless),
    and the bins are log-spaced. We treat all PDFs as PDFs over linear x:
        integral P(x) dx = 1

    We enforce the definition of 1+delta by rescaling the x-axis so that the
    volume-weighted mean satisfies:
        <x>_V = integral x P_V(x) dx = 1

    We also force exact normalization under the midpoint rule used elsewhere.

    Parameters
    ----------
    load_path
        Path to the HDF5 file.

    Returns
    -------
    hist_results, snapshot_info
    """
    snap_idx = int(load_path.stem.split("_")[-2])

    try:
        h5py_file = h5py.File(load_path, "r")
        h5py_file.close()
    except OSError as e:
        raise TimeoutError(
            f"Failed to open the {snap_idx} snapshot results from {load_path}. "
            "The file may be corrupted or incomplete."
        ) from e

    with h5py.File(load_path, "r") as f:
        # Header
        hdr = f["Header"]
        snapshot_info = {
            "snapshot_idx": hdr.attrs["snapshot_idx"],
            "scale_factor": hdr.attrs["scale_factor"],
            "redshift": hdr.attrs["redshift"],
            "box_size": hdr.attrs["box_size"],
            "grid_size": hdr.attrs["grid_size"],
        }

        hist_results: dict[str, Any] = {
            "mean_density": hdr.attrs["mean_density"],
            "jackknife": hdr.attrs["jackknife"],
        }

        if hist_results["jackknife"]:
            hist_results["n_subboxes"] = hdr.attrs["n_subboxes"]
            hist_results["n_subboxes_per_dim"] = hdr.attrs["n_subboxes_per_dim"]

        # Raw x = 1+delta bins
        bins = f["Bins"]
        x_edges_raw = np.asarray(bins["edges"][:], dtype=np.float64)
        x_centers_raw = np.asarray(bins["centers"][:], dtype=np.float64)
        x_widths_raw = np.diff(x_edges_raw)

        hist_results["bin_edges"] = x_edges_raw
        hist_results["bin_centers"] = x_centers_raw
        hist_results["bin_widths"] = x_widths_raw

        # Backwards-compatible names
        hist_results["rho_edges"] = x_edges_raw
        hist_results["rho_centers"] = x_centers_raw
        hist_results["rho_widths"] = x_widths_raw

        # Load "all" first
        hist_results["all"] = load_pdf_group(
            grp=f["All"],
            x_edges_base=x_edges_raw,
            x_centers_base=x_centers_raw,
            x_widths_base=x_widths_raw,
        )

        # Use enforced axis from "all"
        x_edges_enf = hist_results["all"]["_x_edges"]
        x_centers_enf = hist_results["all"]["_x_centers"]
        x_widths_enf = hist_results["all"]["_x_widths"]

        hist_results["rho_edges"] = x_edges_enf
        hist_results["rho_centers"] = x_centers_enf
        hist_results["rho_widths"] = x_widths_enf

        hist_results["rho_edges_raw"] = x_edges_raw
        hist_results["rho_centers_raw"] = x_centers_raw
        hist_results["rho_widths_raw"] = x_widths_raw

    return hist_results, snapshot_info


# ============================================================
# Filename parsing
# ============================================================

def _parse_comoving_filename(
    filename: Path,
    L: int,
) -> tuple[int, int, float] | None:
    """
    Parse a comoving CIC filename.

    Expected stems:
        ..._cic_{cic_N}_{snap_idx}_jackknife
        ..._cic_base_{snap_idx}_jackknife   (cic_N = 1024)

    Returns (snap_idx, cic_N, Rs) or None if not a comoving CIC file.
    """
    stem = filename.stem

    if "cic" not in stem:
        return None
    if "cic_phys" in stem:
        return None

    if "optimal" in stem: 
        return None

    snap_idx = int(stem.split("_")[-2])


    cic_N = 1024 if "base" in stem else int(stem.split("_")[-3])
    Rs = L / cic_N
    return snap_idx, cic_N, Rs


def _parse_physical_filename(
    filename: Path,
    L: int,
) -> tuple[int, int, float] | None:
    """
    Parse a physical CIC filename.

    Expected stems:
        ..._cic_phys_R{Rs}_{snap_idx}_jackknife

    Returns (snap_idx, cic_N, Rs) or None if not a physical CIC file.
    """
    stem = filename.stem

    if "cic_phys" not in stem:
        return None

    snap_idx = int(stem.split("_")[-2])
    Rs = float(stem.split("_")[-3][1:])
    cic_N = round(L / Rs)

    return snap_idx, cic_N, Rs


# ============================================================
# Consolidated loading loop
# ============================================================

def load_cic_pdf_results(
    boxes: tuple[int, ...] = (32, 128, 512, 2048),
    mode: Literal["comoving", "physical"] = "comoving",
    get_pdf_dir: Callable[[int], Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load CIC density PDF results for all boxes and smoothing scales.

    Parameters
    ----------
    boxes : tuple of int
        Box sizes to load (cMpc/h).
    mode : "comoving" or "physical"
        Which set of CIC files to load.
    get_pdf_dir : callable
        Function (L) -> Path to the directory containing
        the HDF5 jackknife files.
    verbose : bool
        If True, show progress bars and print warnings.

    Returns
    -------
    dict with keys:
        'pdf_results'        : nested dict [L][cic_N][snap_idx] -> hist_results
        'snapshot_info'      : nested dict [L][cic_N][snap_idx] -> snapshot_info
        'smoothing_lengths'  : nested dict [L][cic_N] -> Rs
        'sigma_results'      : nested dict [L][cic_N][snap_idx] -> sigma_from_PV
        'mode'               : "comoving" or "physical"
    """
    if get_pdf_dir is None:
        raise ValueError("get_pdf_dir must be provided")

    parser = _parse_comoving_filename if mode == "comoving" else _parse_physical_filename

    pdf_results = defaultdict(lambda: defaultdict(dict))
    snapshot_info = defaultdict(lambda: defaultdict(dict))
    smoothing_lengths = defaultdict(dict)
    sigma_results = defaultdict(lambda: defaultdict(dict))

    for L in boxes:
        pdf_dir = get_pdf_dir(L)
        if not pdf_dir.exists():
            if verbose:
                print(f"  Directory not found for L={L}: {pdf_dir}, skipping.")
            continue

        files = sorted(pdf_dir.glob("*jackknife.hdf5"))

        iterator = (
            tqdm(files, desc=f"L={L} ({mode})")
            if verbose
            else files
        )

        for filename in iterator:
            parsed = parser(filename, L)
            if parsed is None:
                continue

            snap_idx, cic_N, Rs = parsed

            try:
                hist_results, snap_info = load_results_from_hdf5(
                    load_path=filename,
                )
            except TimeoutError:
                if verbose:
                    print(f"  TimeoutError loading {filename}, skipping.")
                continue

            pdf_results[L][cic_N][snap_idx] = hist_results
            snapshot_info[L][cic_N][snap_idx] = snap_info
            smoothing_lengths[L][cic_N] = Rs
            sigma_results[L][cic_N][snap_idx] = hist_results["all"]["sigma_from_PV"]

    return {
        "pdf_results": dict(pdf_results),
        "snapshot_info": dict(snapshot_info),
        "smoothing_lengths": dict(smoothing_lengths),
        "sigma_results": dict(sigma_results),
        "mode": mode,
    }

def _parse_variance_comoving_filename(
    filename: Path,
    n_particles_per_dim: int = 512,
    box_size: int = 1024,
) -> tuple[int, int, float] | None:
    """
    Parse a comoving CIC filename for a variance simulation (fixed L=1024, N=512^3).

    Expected stems:
        ..._cic_{cic_N}_{snap_idx}_jackknife
        ..._cic_base_{snap_idx}_jackknife   (cic_N = n_particles_per_dim)
    """
    stem = filename.stem

    if "cic" not in stem:
        return None
    if "cic_phys" in stem:
        return None
    if "optimal" in stem:
        return None

    snap_idx = int(stem.split("_")[-2])
    cic_N = n_particles_per_dim if "base" in stem else int(stem.split("_")[-3])
    Rs = box_size / cic_N
    return snap_idx, cic_N, Rs


def _parse_variance_physical_filename(
    filename: Path,
    box_size: int = 1024,
) -> tuple[int, int, float] | None:
    """
    Parse a physical CIC filename for a variance simulation (fixed L=1024).

    Expected stems:
        ..._cic_phys_R{Rs}_{snap_idx}_jackknife
    """
    stem = filename.stem

    if "cic_phys" not in stem:
        return None

    snap_idx = int(stem.split("_")[-2])
    Rs = float(stem.split("_")[-3][1:])
    cic_N = round(box_size / Rs)
    return snap_idx, cic_N, Rs


def load_variance_cic_pdf_results(
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18),
    mode: Literal["comoving", "physical"] = "comoving",
    get_pdf_dir: Callable[[int], Path] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load CIC density PDF results across variance seeds (fixed L=1024, N=512^3).

    Parameters
    ----------
    seeds : tuple of int
        Seed numbers identifying each simulation realization.
    mode : "comoving" or "physical"
        Which set of CIC files to load.
    get_pdf_dir : callable
        Function (seed_num) -> Path to the directory containing
        the HDF5 jackknife files for that seed.
    box_size : int
        Box size in cMpc/h (default 1024).
    verbose : bool
        If True, show progress bars and print warnings.

    Returns
    -------
    dict with keys:
        'pdf_results'        : nested dict [seed_num][cic_N][snap_idx] -> hist_results
        'snapshot_info'      : nested dict [seed_num][cic_N][snap_idx] -> snapshot_info
        'smoothing_lengths'  : nested dict [seed_num][cic_N] -> Rs
        'sigma_results'      : nested dict [seed_num][cic_N][snap_idx] -> sigma_from_PV
        'mode'               : "comoving" or "physical"
    """
    if get_pdf_dir is None:
        raise ValueError("get_pdf_dir must be provided")

    parser = _parse_variance_comoving_filename if mode == "comoving" else _parse_variance_physical_filename

    pdf_results = defaultdict(lambda: defaultdict(dict))
    snapshot_info = defaultdict(lambda: defaultdict(dict))
    smoothing_lengths = defaultdict(dict)
    sigma_results = defaultdict(lambda: defaultdict(dict))

    for seed_num in seeds:
        pdf_dir = get_pdf_dir(seed_num)
        if not pdf_dir.exists():
            if verbose:
                print(f"  Directory not found for seed={seed_num}: {pdf_dir}, skipping.")
            continue

        files = sorted(pdf_dir.glob("*jackknife.hdf5"))

        iterator = (
            tqdm(files, desc=f"seed={seed_num} ({mode})")
            if verbose
            else files
        )

        for filename in iterator:
            parsed = parser(filename)
            if parsed is None:
                continue

            snap_idx, cic_N, Rs = parsed

            try:
                hist_results, snap_info = load_results_from_hdf5(
                    load_path=filename,
                )
            except TimeoutError:
                if verbose:
                    print(f"  TimeoutError loading {filename}, skipping.")
                continue

            pdf_results[seed_num][cic_N][snap_idx] = hist_results
            snapshot_info[seed_num][cic_N][snap_idx] = snap_info
            smoothing_lengths[seed_num][cic_N] = Rs
            sigma_results[seed_num][cic_N][snap_idx] = hist_results["all"]["sigma_from_PV"]

    return {
        "pdf_results": dict(pdf_results),
        "snapshot_info": dict(snapshot_info),
        "smoothing_lengths": dict(smoothing_lengths),
        "sigma_results": dict(sigma_results),
        "mode": mode,
    }


def load_cosmo_cic_pdf_results(
    cosmo_names: tuple[str, ...] = ("primary", "toy_model_a", "toy_model_b", "planck"),
    mode: Literal["comoving", "physical"] = "comoving",
    get_pdf_dirs: tuple[Callable[[int], Path], ...] | None = None,
    box_sizes: int | tuple[int, ...] = 512,
    n_particles_per_dim: int | tuple[int, ...] = 512,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load CIC density PDF results across different cosmologies.

    Each cosmology can have its own box size and particle count, controlled
    by *box_sizes* and *n_particles_per_dim*.  Pass a single int when all
    cosmologies share the same value, or a tuple (one per cosmology) when
    they differ.

    Parameters
    ----------
    cosmo_names : tuple of str
        Labels for each cosmological model (e.g. "primary", "toy_model_a").
    mode : "comoving" or "physical"
        Which set of CIC files to load.
    get_pdf_dirs : tuple of callable
        One callable per cosmology, each taking a single int argument (L)
        and returning the Path to the directory containing the HDF5
        jackknife files. Must be the same length as *cosmo_names*.

    box_sizes : int or tuple of int
        Box side-length(s) in cMpc/h, passed to *get_pdf_dirs* and used
        for filename parsing.  A single int is broadcast to all
        cosmologies; a tuple must match the length of *cosmo_names*.
    n_particles_per_dim : int or tuple of int
        Number of particles per dimension (used to resolve "base" CIC files).
        A single int is broadcast; a tuple must match *cosmo_names*.
    verbose : bool
        If True, show progress bars and print warnings.

    Returns
    -------
    dict with keys:
        'pdf_results'        : nested dict [cosmo_name][cic_N][snap_idx] -> hist_results
        'snapshot_info'      : nested dict [cosmo_name][cic_N][snap_idx] -> snapshot_info
        'smoothing_lengths'  : nested dict [cosmo_name][cic_N] -> Rs
        'sigma_results'      : nested dict [cosmo_name][cic_N][snap_idx] -> sigma_from_PV
        'mode'               : "comoving" or "physical"
    """
    n_cosmo = len(cosmo_names)

    if get_pdf_dirs is None:
        raise ValueError("get_pdf_dirs must be provided (one callable per cosmo_name)")
    if len(get_pdf_dirs) != n_cosmo:
        raise ValueError(
            f"cosmo_names has {n_cosmo} entries but get_pdf_dirs has {len(get_pdf_dirs)}"
        )

    # Broadcast scalar -> per-cosmology tuple
    if isinstance(box_sizes, int):
        box_sizes = tuple(box_sizes for _ in cosmo_names)
    if isinstance(n_particles_per_dim, int):
        n_particles_per_dim = tuple(n_particles_per_dim for _ in cosmo_names)

    if len(box_sizes) != n_cosmo:
        raise ValueError(
            f"cosmo_names has {n_cosmo} entries but box_sizes has {len(box_sizes)}"
        )
    if len(n_particles_per_dim) != n_cosmo:
        raise ValueError(
            f"cosmo_names has {n_cosmo} entries but n_particles_per_dim has {len(n_particles_per_dim)}"
        )

    pdf_results = defaultdict(lambda: defaultdict(dict))
    snapshot_info = defaultdict(lambda: defaultdict(dict))
    smoothing_lengths = defaultdict(dict)
    sigma_results = defaultdict(lambda: defaultdict(dict))

    for cosmo_name, get_dir, L, N in zip(cosmo_names, get_pdf_dirs, box_sizes, n_particles_per_dim):
        pdf_dir = get_dir(L)
        if not pdf_dir.exists():
            if verbose:
                print(f"  Directory not found for {cosmo_name}: {pdf_dir}, skipping.")
            continue

        # Build per-cosmology parser with the correct L and N
        if mode == "comoving":
            parser = lambda fn, _L=L, _N=N: _parse_variance_comoving_filename(
                fn, n_particles_per_dim=_N, box_size=_L
            )
        else:
            parser = lambda fn, _L=L: _parse_variance_physical_filename(fn, box_size=_L)

        files = sorted(pdf_dir.glob("*jackknife.hdf5"))

        iterator = (
            tqdm(files, desc=f"{cosmo_name} ({mode})")
            if verbose
            else files
        )

        for filename in iterator:
            parsed = parser(filename)
            if parsed is None:
                continue

            snap_idx, cic_N, Rs = parsed

            try:
                hist_results, snap_info = load_results_from_hdf5(
                    load_path=filename,
                )
            except TimeoutError:
                if verbose:
                    print(f"  TimeoutError loading {filename}, skipping.")
                continue

            pdf_results[cosmo_name][cic_N][snap_idx] = hist_results
            snapshot_info[cosmo_name][cic_N][snap_idx] = snap_info
            smoothing_lengths[cosmo_name][cic_N] = Rs
            sigma_results[cosmo_name][cic_N][snap_idx] = hist_results["all"]["sigma_from_PV"]

    return {
        "pdf_results": dict(pdf_results),
        "snapshot_info": dict(snapshot_info),
        "smoothing_lengths": dict(smoothing_lengths),
        "sigma_results": dict(sigma_results),
        "mode": mode,
    }