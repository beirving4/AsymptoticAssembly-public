import numpy as np, pdb

from typing import Callable, Sequence, Optional

from .type_defs import RelationType
from ..utils.jackknife import (
    jackknife_resampling,
    jackknife_estimate,
    jackknife_error,
    get_jackknife_volume
)
def _compute_abundance_dict(
        jk_samples: list[np.ndarray],
        samples_bins: dict[str, np.ndarray],
        jackknife_volume: float,
        are_bins_logged: bool,
        reduce_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
        hist_samples: list[np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
    """Build abundance dictionaries from jackknife samples using a reducer.

    Parameters
    ----------
    jk_samples
        Jackknife-resampled raw samples (one 1D array per fold).
    samples_bins
        Dict with "edges" and "centers".
    jackknife_volume
        Effective volume per jackknife sample.
    are_bins_logged
        If True, bins are in log10-space for the underlying variable.
    reduce_fn
        A function mapping a list of per-fold arrays -> a single array. Use
        `jackknife_estimate` for central estimates and `jackknife_error` for errors.
    hist_samples
        Optionally pass precomputed per-fold histograms to avoid recomputation.

    Returns
    -------
    dict
        Keys: "histogram", "histogram_density", "number_density",
        "differential", "cumulative".
    """
    # Per-fold histograms (counts per bin)
    if hist_samples is None:
        hist_samples = [
            np.histogram(sample, bins=samples_bins["edges"])[0]
            for sample in jk_samples
        ]

    # Per-fold cumulative counts from high->low bin
    cumulative_samples = [np.cumsum(h[::-1])[::-1] for h in hist_samples]

    # Reduce across folds
    hist = reduce_fn(hist_samples)
    cum = reduce_fn(cumulative_samples)

    # Convert to densities
    dn = hist / jackknife_volume
    cum_density = cum / jackknife_volume

    # Bin widths in ln x
    if are_bins_logged:
        dlnx = np.diff(np.log(10.0 ** samples_bins["edges"]))
    else:
        dlnx = np.diff(np.log(samples_bins["edges"]))

    dndlnx = dn / dlnx

    return {
        "histogram": hist,
        "histogram_density": dn,
        "number_density": dndlnx / samples_bins["centers"],
        "differential": dndlnx,
        "cumulative": cum_density,
    }


def get_logged_sample_bins(samples: np.ndarray, bin_count: int) -> dict[str, np.ndarray]:
    bin_edges = np.histogram_bin_edges(samples, bins=bin_count)
    bin_centers = 10.0**(bin_edges[:-1] + 0.5 * np.diff(bin_edges))
    return {"edges" : bin_edges, "centers" : bin_centers}

def get_linear_sample_bins(samples: np.ndarray, bin_count: int) -> dict[str, np.ndarray]:
    bin_edges = np.histogram_bin_edges(samples, bins=bin_count)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {"edges" : bin_edges, "centers" : bin_centers}
def get_sample_bins(
        samples: np.ndarray, 
        bin_count: int,
        are_bins_logged: bool = False
    ) -> dict[str, np.ndarray]:

    if are_bins_logged:
        return get_logged_sample_bins(samples, bin_count)
    else:
        return get_linear_sample_bins(samples, bin_count)

def get_mass_bins(masses: np.ndarray, bin_count: int) -> dict[str, np.ndarray]:
    return get_sample_bins(masses, bin_count, are_bins_logged=True)

def compute_abundance_function_data(
        samples: np.ndarray,
        comoving_volume: float,
        bin_count: int | None = None,
        samples_bins: dict[str, np.ndarray] | None = None,
        are_bins_logged: bool = False
    ) -> dict[str, dict[str, np.ndarray]]:

    samples = samples[np.isfinite(samples)]

    if samples_bins is None:
        if bin_count is None:
            raise ValueError("Either bin_count or samples_bins must be provided")
        else:
            samples_bins = get_sample_bins(
                samples=samples, 
                bin_count=bin_count, 
                are_bins_logged=are_bins_logged
            )

    hist = np.histogram(samples, bins=samples_bins["edges"])[0]
    dlnx = (
        np.diff(np.log(10.0**samples_bins["edges"])) 
        if are_bins_logged else 
        np.diff(np.log(samples_bins["edges"]))
    )
    dn = hist / comoving_volume 
    dndlnx = dn / dlnx

    estimate = {
        "histogram" : hist,
        "histogram_density" : hist / comoving_volume,
        "number_density" : dndlnx / samples_bins["centers"],
        "differential" : dndlnx,
        "cumulative" : np.cumsum(dn[::-1])[::-1],
    }

    return {"samples_bins" : samples_bins, "estimate" : estimate}

def get_mass_function_data(
        masses: np.ndarray, 
        comoving_volume: float,
        present_day_density: float | None = None,
        bin_count: int | None = None,
        mass_bins: dict[str, np.ndarray] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]: 

    abundance_data = compute_abundance_function_data(
        samples=masses,
        comoving_volume=comoving_volume,
        bin_count=bin_count,
        samples_bins=mass_bins,
        are_bins_logged=True
    )

    if present_day_density is not None:
        bin_centers = abundance_data["samples_bins"]["centers"]
        n = abundance_data["estimate"]["number_density"]
        Msq_over_rho = bin_centers**2 / present_day_density
        abundance_data["estimate"]["normalized"] = n * Msq_over_rho

    return {
        "mass_bins" : abundance_data["samples_bins"],
        "estimate" : abundance_data["estimate"]
    }
        

def get_abundance_function_data(
        samples: np.ndarray, 
        comoving_volume: float,
        relation_type: str, 
        present_day_density: float | None = None,
        bin_count: int | None = None,
        samples_bins: dict[str, np.ndarray] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:

    match relation_type:
        case RelationType.HALO_MASS_FUNCTION.value:
            if present_day_density is None:
                raise ValueError(
                    "present_day_density must be provided for mass function data."
                )

            return get_mass_function_data(
                masses=samples,
                comoving_volume=comoving_volume,
                present_day_density=present_day_density,
                bin_count=bin_count,
                mass_bins=samples_bins
            )
        case _:
            raise ValueError(
                f"Unknown relation type: {relation_type}. "
                "Must be 'mass'."
            )
 

def compute_jackknifed_abundance_function_data(
        sample_sets: list[np.ndarray],
        comoving_volume: float,
        bin_count: int | None = None,
        samples_bins: dict[str, np.ndarray] | None = None,
        are_bins_logged: bool = False,
        return_fold_samples: bool = False
    ) -> dict[str, dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]]:

    n_samples, box_size = len(sample_sets), np.cbrt(comoving_volume)
    samples = np.hstack(sample_sets)[np.isfinite(np.hstack(sample_sets))]

    if samples_bins is None:
        if bin_count is None:
            raise ValueError("Either bin_count or samples_bins must be provided")
        else:
            samples_bins = get_sample_bins(
                samples=samples,
                bin_count=bin_count,
                are_bins_logged=are_bins_logged
            )

    jackknife_volume = get_jackknife_volume(n_samples, box_size)
    jackknife_samples = jackknife_resampling(sample_sets)

    # Optionally precompute histograms once if caller will want them later
    pre_hist = None
    if return_fold_samples:
        pre_hist = [
            np.histogram(s, bins=samples_bins["edges"])[0]
            for s in jackknife_samples
        ]

    estimate = _compute_abundance_dict(
        jk_samples=jackknife_samples,
        samples_bins=samples_bins,
        jackknife_volume=jackknife_volume,
        are_bins_logged=are_bins_logged,
        reduce_fn=jackknife_estimate,
        hist_samples=pre_hist,
    )

    errors = _compute_abundance_dict(
        jk_samples=jackknife_samples,
        samples_bins=samples_bins,
        jackknife_volume=jackknife_volume,
        are_bins_logged=are_bins_logged,
        reduce_fn=jackknife_error,
        hist_samples=pre_hist,
    )

    result = {
        "estimate": estimate,
        "errors": errors,
        "samples_bins": samples_bins,
    }

    if return_fold_samples:
        # Build per-subbox abundance dicts using the original sub-box samples.
        # Use estimate reducer (n_samples = 1 semantics) and per-subbox volume.
        subbox_volume = comoving_volume / n_samples
        fold_samples: dict[int, dict[str, np.ndarray]] = {}
        for i, sub_samples in enumerate(sample_sets, start=1):
            fold_samples[i] = _compute_abundance_dict(
                jk_samples=[sub_samples],
                samples_bins=samples_bins,
                jackknife_volume=subbox_volume,
                are_bins_logged=are_bins_logged,
                reduce_fn=jackknife_estimate,
                hist_samples=None,
            )
        result["fold_samples"] = fold_samples

    return result


def get_jackknifed_mass_function_data(
        mass_samples: list[np.ndarray],
        comoving_volume: float,
        present_day_density: float | None = None,
        bin_count: int | None = None,
        mass_bins: dict[str, np.ndarray] | None = None,
        return_fold_samples: bool = False
    ) -> dict[str, dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]]:

    jk_mf_data = compute_jackknifed_abundance_function_data(
        sample_sets=mass_samples,
        comoving_volume=comoving_volume,
        bin_count=bin_count,
        samples_bins=mass_bins,
        are_bins_logged=True,
        return_fold_samples=return_fold_samples
    )

    if present_day_density is not None:
        bin_centers = jk_mf_data["samples_bins"]["centers"]
        n = jk_mf_data["estimate"]["number_density"]
        delta_n = jk_mf_data["errors"]["number_density"]
        Msq_over_rho = bin_centers**2 / present_day_density
        jk_mf_data["estimate"]["normalized"] = n * Msq_over_rho
        jk_mf_data["errors"]["normalized"] = delta_n * Msq_over_rho

        # Per-fold normalized (no per-fold errors by design)
        if "fold_samples" in jk_mf_data:
            for fold_id, fold_dict in jk_mf_data["fold_samples"].items():
                jk_mf_data["fold_samples"][fold_id]["normalized"] = fold_dict["number_density"] * Msq_over_rho

    result = {
        "mass_bins": jk_mf_data["samples_bins"],
        "estimate": jk_mf_data["estimate"],
        "errors": jk_mf_data["errors"],
    }

    if "fold_samples" in jk_mf_data:
        result["fold_samples"] = jk_mf_data["fold_samples"]

    return result


def get_jackknifed_abundance_data(
        samples: list[np.ndarray],
        comoving_volume: float,
        relation_type: str,
        present_day_density: float | None = None,
        bin_count: int | None = None,
        samples_bins: dict[str, np.ndarray] | None = None,
        return_fold_samples: bool = False
    ) -> dict[str, dict[str, np.ndarray]]:

    common_cfg = {
        "comoving_volume": comoving_volume,
        "bin_count": bin_count,
        "return_fold_samples": return_fold_samples
    }

    match relation_type:
        case RelationType.HALO_MASS_FUNCTION.value:
            if present_day_density is None:
                raise ValueError(
                    "present_day_density must be provided for mass function"
                )
            return get_jackknifed_mass_function_data(
                mass_samples=samples,
                present_day_density=present_day_density,
                mass_bins=samples_bins,
                **common_cfg
            )
        case _:
            raise ValueError(
                f"Unknown relation type: {relation_type}. "
                "Must be 'mass'."
            )
