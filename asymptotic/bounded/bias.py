from __future__ import annotations

import numpy as np

from attrs import define, field, fields as attrs_fields
from scipy.interpolate import interp1d
from collections import OrderedDict
from pathlib import Path
import h5py

from ..fields.correlation import TwoPointCorrelationData

from ..fields.two_point import compute_tpcf, get_tpcf_particle_subsample_size
from ..simulation.evo import EvolutionData
from ..utils.jackknife import (
    get_jackknife_subbox_samples,
    jackknife_resampling,
    jackknife_estimate,
    jackknife_error,
)

@define(slots=True)
class HaloBiasData:
    property_values: np.ndarray
    estimates: np.ndarray
    errors: np.ndarray

    @classmethod
    def from_data(
            cls,
            comoving_halo_positions: np.ndarray,
            comoving_particle_coordinates: np.ndarray,
            property_values: np.ndarray,
            property_bins: dict[int | str, tuple[float, float]] | None,
            sub_box_info: dict[int, list[tuple[float, float]]],
            boxsize: float,
            rmin: float,
            rmax: float,
            nbins: int,
            target_radius: float | None = None,
            radial_index: int | None = None,
            use_natural: bool = False,
            eps: float = 1e-12,
            positive_only: bool = True,
            return_folds: bool = False,
            interpolate_to_radius: float | None = None,
            allow_loglog: bool = False,
        ) -> HaloBiasData:

        """Construct a **single-property** halo-bias dataset via jackknife.

        This computes the scalar bias at a chosen r-bin for each property bin
        using :func:`get_bias_by_property`, and converts the per-bin dict to arrays.

        Parameters
        ----------
        property_values : (Nh,) array
            Per-halo property (e.g., mass, spin) used for binning.
        property_bins : dict[key -> (min, max)]
            Binning for the property. If ``None``, 10 quantile bins are used.
        """

        # If no user-provided bins, build 10 quantile bins over finite values
        if property_bins is None:
            pv = np.asarray(property_values, dtype=float)
            mask = np.isfinite(pv)
            if mask.sum() < 2:
                raise ValueError("Not enough finite property values to build bins.")
            qs = np.linspace(0.0, 100.0, 11)
            edges = np.unique(np.nanpercentile(pv[mask], qs))
            if edges.size < 2:
                raise ValueError("Could not derive valid bin edges from property values.")
            property_bins = {i: (edges[i], edges[i+1]) for i in range(edges.size - 1)}

        # Run the property-binned bias calculation
        results = get_bias_by_property(
            halo_coordinates=comoving_halo_positions,
            halo_property=property_values,
            property_bins=property_bins,
            particle_coordinates=comoving_particle_coordinates,
            sub_box_info=sub_box_info,
            boxsize=boxsize,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            target_radius=target_radius,
            radial_index=radial_index,
            use_natural=use_natural,
            eps=eps,
            positive_only=positive_only,
            return_folds=return_folds,
            interpolate_to_radius=interpolate_to_radius,
            allow_loglog=allow_loglog,
        )

        # Convert per-bin dict to arrays (sorted by bin key order)
        ordered_keys = list(property_bins.keys())
        # Representative property value per bin: midpoint of (min, max)
        prop_centers = np.array([
            0.5 * (property_bins[k][0] + property_bins[k][1]) for k in ordered_keys
        ], dtype=float)
        est = np.array([results[k]["bias"] for k in ordered_keys], dtype=float)
        err = np.array([results[k]["bias_err"] for k in ordered_keys], dtype=float)

        return cls(property_values=prop_centers, estimates=est, errors=err)


    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "property_values": self.property_values,
            "estimates": self.estimates,
            "errors": self.errors,
        }
    
    @property
    def as_array(self) -> np.ndarray:
        return np.column_stack([self.property_values, self.estimates, self.errors])


@define(slots=True)
class HaloBiasEvoData(EvolutionData):
    data: OrderedDict[int, HaloBiasData]

@define(slots=True)
class HaloBiasDataset:
    mass: HaloBiasData | None = field(default=None)
    peak_height: HaloBiasData | None = field(default=None)
    spin: HaloBiasData | None = field(default=None)
    concentration: HaloBiasData | None = field(default=None)
    age: HaloBiasData | None = field(default=None)
    velocity_dispersion: HaloBiasData | None = field(default=None)
    radial_velocity: HaloBiasData | None = field(default=None)
    sphericity: HaloBiasData | None = field(default=None)
    ellipticity: HaloBiasData | None = field(default=None)
    triaxiality: HaloBiasData | None = field(default=None)

    def __getitem__(self, property_name: str) -> HaloBiasData | None:
        return getattr(self, property_name)

    @classmethod
    def from_data(
            cls,
            comoving_halo_positions: np.ndarray,
            comoving_particle_coordinates: np.ndarray,
            halo_properties: dict[str, np.ndarray],
            bins_dict: dict[str, dict[int | str, tuple[float, float]]] | None,
            sub_box_info: dict[int, list[tuple[float, float]]],
            boxsize: float,
            rmin: float,
            rmax: float,
            nbins: int,
            target_radius: float | None = None,
            radial_index: int | None = None,
            use_natural: bool = False,
            eps: float = 1e-12,
            positive_only: bool = True,
            return_folds: bool = False,
            interpolate_to_radius: float | None = None,
            allow_loglog: bool = False,
        ) -> HaloBiasDataset:
        """Construct a **multi-property** halo-bias dataset via jackknife.

        This version supports *partial* property sets. Only properties present in
        both the input `halo_properties` and as fields on this class are computed;
        all others remain `None`.
        """
        # Recognized property names from the attrs class definition
        allowed_fields = {f.name for f in attrs_fields(cls)}
        requested = {k: v for k, v in halo_properties.items() if k in allowed_fields}
        if not requested:
            raise ValueError(
                "No recognized properties found. Allowed keys: " + ", ".join(sorted(allowed_fields))
            )

        # Build bins for any requested property missing from bins_dict
        def _autobins(values: np.ndarray, n: int = 10) -> dict[int, tuple[float, float]]:
            vals = np.asarray(values, dtype=float)
            mask = np.isfinite(vals)
            if mask.sum() < 2:
                raise ValueError("Not enough finite values to bin property.")
            qs = np.linspace(0.0, 100.0, n + 1)
            edges = np.unique(np.nanpercentile(vals[mask], qs))
            if edges.size < 2:
                raise ValueError("Failed to derive bin edges for property.")
            return {i: (edges[i], edges[i+1]) for i in range(edges.size - 1)}

        bins_dict = {} if bins_dict is None else dict(bins_dict)
        effective_bins: dict[str, dict[int | str, tuple[float, float]]] = {}
        for name, vals in requested.items():
            if name in bins_dict and bins_dict[name]:
                effective_bins[name] = bins_dict[name]
            else:
                effective_bins[name] = _autobins(vals, n=10)

        # Compute biases for all requested properties (precomputes ξ_mm folds once)
        results_by_property = _bias_for_many_properties(
            halo_coords=comoving_halo_positions,
            halo_props_dict=requested,
            bins_dict=effective_bins,
            particle_coords=comoving_particle_coordinates,
            sub_box_info=sub_box_info,
            boxsize=boxsize,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            target_radius=target_radius,
            radial_index=radial_index,
            use_natural=use_natural,
            eps=eps,
            positive_only=positive_only,
            return_folds=return_folds,
            interpolate_to_radius=interpolate_to_radius,
            allow_loglog=allow_loglog,
        )

        # Assemble payload only for computed properties; others remain None
        dataset_payload: dict[str, HaloBiasData] = {}
        for prop_name, per_bin in results_by_property.items():
            ordered_keys = list(effective_bins[prop_name].keys())
            centers = np.array([
                0.5 * (effective_bins[prop_name][k][0] + effective_bins[prop_name][k][1])
                for k in ordered_keys
            ], dtype=float)
            est = np.array([per_bin[k]["bias"] for k in ordered_keys], dtype=float)
            err = np.array([per_bin[k]["bias_err"] for k in ordered_keys], dtype=float)
            dataset_payload[prop_name] = HaloBiasData(
                property_values=centers,
                estimates=est,
                errors=err,
            )

        # Return an instance with provided properties set; others default to None
        return cls(**dataset_payload)
    
    @property
    def as_nested_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """Return a dictionary representation of this dataset."""
        return {
            f.name: getattr(self, f.name).as_dict for f in attrs_fields(self)
            if isinstance(getattr(self, f.name), HaloBiasData)
        }
    
    @property
    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a dictionary representation of this dataset."""
        return {
            f.name: getattr(self, f.name).as_array for f in attrs_fields(self)
            if isinstance(getattr(self, f.name), HaloBiasData)
        }

    def save(self, filepath: Path) -> None: 
        save_halo_bias_dataset(filepath, self.as_dict)

    @classmethod
    def load(cls, filepath: Path) -> HaloBiasDataset:
        data = load_halo_bias_dataset(filepath)
        return cls(
            **{
                k: HaloBiasData(
                    property_values=v[:, 0],
                    estimates=v[:, 1],
                    errors=v[:, 2],
                )
                for k, v in data.items()
            }
        )




@define(slots=True)
class HaloBiasEvoDataset(EvolutionData):
    data: OrderedDict[int, HaloBiasDataset]




def _check_rmax_vs_box(rmax: float, boxsize: float) -> None:
    if rmax > boxsize / 2.0:
        raise ValueError(
            f"rmax={rmax} exceeds boxsize/2={boxsize/2.0}. Choose rmax <= boxsize/2 to avoid empty RR bins."
        )


def _compute_fold_tpcf(
        coords: np.ndarray,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        use_natural: bool,
        eps: float,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single (non-JK) TPCF and return (r, xi)."""
    res = compute_tpcf(
        comoving_coordinates=coords,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        boxsize=boxsize,
        weights=weights,
        eps=eps,
        use_natural=use_natural,
    )
    return res[:, 0], res[:, 1]

def _get_resamples(
        comoving_coordinates: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
    ) -> list[np.ndarray]:
    """Get jackknife resamples for the given coordinates and sub-box information."""
    parts = get_jackknife_subbox_samples(
        sub_box_info=sub_box_info,
        coordinates=comoving_coordinates,
        return_as_mask=False
    )
    return jackknife_resampling(parts)

def _maybe_subsample_coords(
        coords: np.ndarray,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        num_target_pairs: int,
        rng: np.random.Generator | None,
        enable: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Optionally subsample a coordinate catalog *before* jackknife masking.

    Returns (coords_sub, idx) where idx are the kept indices into the original
    array (or None if no subsampling occurred).
    """
    if (not enable) or (rng is None):
        return coords, None
    n = coords.shape[0]
    if n <= 2:
        return coords, None

    keep = get_tpcf_particle_subsample_size(
        n, boxsize, rmin, rmax, nbins, num_target_pairs
    )
    if keep >= n:
        return coords, None

    idx = rng.choice(n, size=keep, replace=False)
    return coords[idx], idx


def _maybe_subsample_coords_and_property(
        coords: np.ndarray,
        prop: np.ndarray,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        num_target_pairs: int,
        rng: np.random.Generator | None,
        enable: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample coords and apply the same selection to the property array.
    Returns (coords_sub, prop_sub).
    """
    coords_sub, idx = _maybe_subsample_coords(
        coords,
        boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
        num_target_pairs=num_target_pairs, rng=rng, enable=enable
    )

    return (coords, prop) if idx is None else (coords_sub, prop[idx])


# Helper to precompute matter-matter xi per JK fold for bias-by-property
def _precompute_mm_folds(
        particle_coordinates: np.ndarray,
        part_resamples: list[np.ndarray],
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        use_natural: bool,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Precompute matter–matter xi per JK fold and define a shared r-grid.

    Returns
    -------
    mm_fold_xi : (m, nbins) array
        Per-fold xi_mm(r) values (NaNs where insufficient pairs).
    r_grid : (nbins,) array
        The common r-grid used across folds.
    """
    mm_fold_xi: list[np.ndarray] = []
    r_grid: np.ndarray | None = None
    for pc in part_resamples:
        if pc.shape[0] < 2:
            if r_grid is None:
                r_grid, _ = _compute_fold_tpcf(
                    coords=particle_coordinates[:2],
                    boxsize=boxsize,
                    rmin=rmin,
                    rmax=rmax,
                    nbins=nbins,
                    use_natural=use_natural,
                    eps=eps,
                )
            mm_fold_xi.append(np.full_like(r_grid, np.nan, dtype=float))
            continue
        r_mm, xi_mm = _compute_fold_tpcf(
            coords=pc,
            boxsize=boxsize,
            rmin=rmin,
            rmax=rmax,
            nbins=nbins,
            use_natural=use_natural,
            eps=eps,
        )
        if r_grid is None:
            r_grid = r_mm
        mm_fold_xi.append(xi_mm)

    mm_fold_xi_arr = (
        np.vstack(mm_fold_xi) if len(mm_fold_xi) else np.empty((0, nbins))
    )
    return mm_fold_xi_arr, r_grid

def get_bias_radial_profile(
        halo_coordinates: np.ndarray,
        particle_coordinates: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        use_natural: bool = False,
        eps: float = 1e-12,
        positive_only: bool = True,
        halo_weights: np.ndarray | None = None,
        particle_weights: np.ndarray | None = None,
        return_folds: bool = False,
        # Subsampling controls (performed *before* JK masking)
        use_recommended_subsample: bool = False,
        num_target_pairs: int = 100_000,
        subsample_rng_seed: int | None = None,
        subsample_halos: bool = False,
        subsample_particles: bool = False,
    ) -> dict[str, np.ndarray]:
    """Jackknife estimate of the **radial bias profile** b(r) from raw coordinates.

    Per fold (leave-one-subbox-out):
      1) xi_hh(r) on included halos;
      2) xi_mm(r) on included particles;
      3) b_i(r) = sqrt(xi_hh/xi_mm) on valid/positive bins.

    Optional pre-JK subsampling reduces catalogs before masking, to target a
    recommended pair-count budget (see two_point.get_tpcf_particle_subsample_size).

    Returns
    -------
    {
      'r': (nbins,),
      'bias': (nbins,),
      'bias_err': (nbins,),
      ['fold_bias']: (m, nbins)  # only when return_folds=True
    }
    """
    _check_rmax_vs_box(rmax, boxsize)

    rng = np.random.default_rng(subsample_rng_seed) if use_recommended_subsample else None

    # Optional pre-JK subsampling
    halos_in, _ = _maybe_subsample_coords(
        halo_coordinates, boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
        num_target_pairs=num_target_pairs, rng=rng, enable=subsample_halos
    )
    parts_in, _ = _maybe_subsample_coords(
        particle_coordinates, boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
        num_target_pairs=num_target_pairs, rng=rng, enable=subsample_particles
    )

    # Partition catalogs into sub-box groups (using possibly-subsampled arrays)
    halo_resamples = _get_resamples(
        comoving_coordinates=halos_in,
        sub_box_info=sub_box_info
    )
    particle_resamples = _get_resamples(
        comoving_coordinates=parts_in,
        sub_box_info=sub_box_info
    )
    m = len(halo_resamples)

    # Per-fold bias curves
    fold_bias_list: list[np.ndarray] = []
    r_shared: np.ndarray | None = None

    for i in range(m):
        hc = halo_resamples[i]
        pc = particle_resamples[i]
        if hc.shape[0] < 2 or pc.shape[0] < 2:
            if r_shared is None:
                r_shared, _ = _compute_fold_tpcf(
                    coords=parts_in[:2],
                    boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
                    use_natural=use_natural, eps=eps
                )
            fold_bias_list.append(np.full_like(r_shared, np.nan, dtype=float))
            continue

        r_hh, xi_hh = _compute_fold_tpcf(
            coords=hc, boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
            use_natural=use_natural, eps=eps, weights=halo_weights
        )
        _, xi_mm = _compute_fold_tpcf(
            coords=pc, boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
            use_natural=use_natural, eps=eps, weights=particle_weights
        )

        if r_shared is None:
            r_shared = r_hh

        valid = np.isfinite(xi_hh) & np.isfinite(xi_mm)
        if positive_only:
            valid &= (xi_hh > 0) & (xi_mm > 0)

        b_i = np.full_like(r_shared, np.nan, dtype=float)
        b_i[valid] = np.sqrt(xi_hh[valid] / xi_mm[valid])
        fold_bias_list.append(b_i)

    fold_bias = (
        np.vstack(fold_bias_list) 
        if len(fold_bias_list) else
        np.empty((0, nbins))
    )

    bias = np.empty_like(r_shared, dtype=float)
    bias_err = np.empty_like(r_shared, dtype=float)
    for j in range(r_shared.size):
        vals = [
            np.array([fb[j]], dtype=float)
            for fb in fold_bias_list
            if np.isfinite(fb[j])
        ]
        if vals:
            bias[j] = float(jackknife_estimate(vals).ravel()[0])
            bias_err[j] = float(jackknife_error(vals).ravel()[0])
        else:
            bias[j] = np.nan
            bias_err[j] = np.nan

    out = {"r": r_shared, "bias": bias, "bias_err": bias_err}
    if return_folds:
        out["fold_bias"] = fold_bias
    return out

def _select_halos_in_property_bin(
        halo_coordinates: np.ndarray,
        halo_property: np.ndarray,
        pmin: float,
        pmax: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (coords_in_bin, selection_mask, count) for a property bin."""
    sel = (halo_property >= pmin) & (halo_property < pmax)
    halos_sel = halo_coordinates[sel]
    return halos_sel, sel, int(halos_sel.shape[0])


def _choose_rbin_index(
        r_grid: np.ndarray,
        target_radius: float | None,
        radial_index: int | None,
    ) -> int:
    """Pick an r-bin index based on target_radius or radial_index with safe fallback."""
    if target_radius is not None and r_grid is not None:
        return int(np.argmin(np.abs(r_grid - target_radius)))
    return int(radial_index) if radial_index is not None else r_grid.size // 2

from scipy.interpolate import interp1d  # already imported above; safe to re-import

def _interp_value_at_radius(
        r: np.ndarray,
        y: np.ndarray,
        r_target: float,
        allow_loglog: bool = False,
    ) -> float:
    """
    Interpolate y(r) at r_target.

    If `allow_loglog` and all (r>0, y>0), interpolate in log–log space;
    else do linear-in-r interpolation. Returns NaN if out of bounds.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    if r_target <= 0 or r.size == 0 or y.size == 0:
        return float("nan")

    try:
        if allow_loglog and np.all(r > 0) and np.all(y > 0):
            f = interp1d(
                np.log(r), np.log(y),
                kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=False
            )
            val = f(np.log(r_target))
            return float(np.exp(val)) if np.isfinite(val) else float("nan")
        else:
            f = interp1d(
                r, y,
                kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=False
            )
            val = f(r_target)
            return float(val) if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def _compute_per_fold_bias_at_radius(
        halo_resamples: list[np.ndarray],
        mm_fold_r: np.ndarray,
        mm_fold_xi: np.ndarray,
        r_target: float,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        use_natural: bool,
        eps: float,
        positive_only: bool,
        allow_loglog: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Interpolate xi_hh and xi_mm per fold to r_target and compute b_i.
    Returns (per_fold_bias, fold_vals) where fold_vals is a list of (1,) arrays
    suitable for jackknife_estimate/error.
    """
    nfolds = mm_fold_xi.shape[0]
    per_fold_bias = np.full((nfolds,), np.nan, dtype=float)
    fold_vals: list[np.ndarray] = []

    for i, hc in enumerate(halo_resamples):
        if hc.shape[0] < 2:
            continue

        r_hh, xi_hh = _compute_fold_tpcf(
            coords=hc, boxsize=boxsize,
            rmin=rmin, rmax=rmax, nbins=nbins,
            use_natural=use_natural, eps=eps
        )
        xi_mm = mm_fold_xi[i]

        xh = _interp_value_at_radius(r_hh, xi_hh, r_target, allow_loglog=allow_loglog)
        xm = _interp_value_at_radius(mm_fold_r, xi_mm, r_target, allow_loglog=allow_loglog)

        if not (np.isfinite(xh) and np.isfinite(xm)):
            continue
        if positive_only and (xh <= 0 or xm <= 0):
            continue

        b_i = np.sqrt(xh / xm)
        per_fold_bias[i] = b_i
        fold_vals.append(np.array([b_i], dtype=float))

    return per_fold_bias, fold_vals

def _compute_per_fold_bias_at_index(
        halo_resamples: list[np.ndarray],
        mm_fold_xi: np.ndarray,
        j_idx: int,
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        use_natural: bool,
        eps: float,
        positive_only: bool,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute b_i per fold at a chosen r-bin. Returns (per_fold_bias, fold_vals).

    `fold_vals` is a list of (1,) arrays, suitable for jackknife_estimate/error.
    """
    nfolds = mm_fold_xi.shape[0]
    per_fold_bias = np.full((nfolds,), np.nan, dtype=float)
    fold_vals: list[np.ndarray] = []

    for i, hc in enumerate(halo_resamples):
        if hc.shape[0] < 2:
            continue
        _, xi_hh = _compute_fold_tpcf(
            coords=hc, boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
            use_natural=use_natural, eps=eps
        )
        xi_mm = mm_fold_xi[i]
        if j_idx >= xi_hh.size or j_idx >= xi_mm.size:
            continue
        if not (np.isfinite(xi_hh[j_idx]) and np.isfinite(xi_mm[j_idx])):
            continue
        if positive_only and (xi_hh[j_idx] <= 0 or xi_mm[j_idx] <= 0):
            continue
        b_i = np.sqrt(xi_hh[j_idx] / xi_mm[j_idx])
        per_fold_bias[i] = b_i
        fold_vals.append(np.array([b_i], dtype=float))

    return per_fold_bias, fold_vals


def _jk_scalar_from_folds(fold_vals: list[np.ndarray]) -> tuple[float, float]:
    """Jackknife estimate and 1σ error for scalar targets.

    Expects a list of (1,) arrays; returns (estimate, error) as floats or (nan, nan)
    if no valid folds are provided.
    """
    if not fold_vals:
        return np.nan, np.nan
    return (
        float(jackknife_estimate(fold_vals).ravel()[0]),
        float(jackknife_error(fold_vals).ravel()[0]),
    )


def _empty_property_result(
        n_sel: int,
        r_grid: np.ndarray | None,
        mm_fold_xi: np.ndarray,
        return_folds: bool,
    ) -> dict[str, float | int | np.ndarray]:
    """Package a result dict for bins with insufficient halos."""
    out: dict[str, float | int | np.ndarray] = {
        "bias": np.nan,
        "bias_err": np.nan,
        "n_halos": n_sel,
        "radius": np.nan if r_grid is None else float(r_grid[0]),
    }
    if return_folds:
        out["fold_bias"] = np.full((mm_fold_xi.shape[0],), np.nan, dtype=float)
    return out


def _package_property_result(
        b_hat: float,
        b_err: float,
        n_sel: int,
        r_value: float,
        per_fold_bias: np.ndarray,
        return_folds: bool,
    ) -> dict[str, float | int | np.ndarray]:
    """Assemble the per-bin output dictionary in a single place."""
    out: dict[str, float | int | np.ndarray] = {
        "bias": b_hat,
        "bias_err": b_err,
        "n_halos": n_sel,
        "radius": r_value,
    }
    if return_folds:
        out["fold_bias"] = per_fold_bias
    return out


def get_bias_by_property(
        halo_coordinates: np.ndarray,
        halo_property: np.ndarray,
        property_bins: dict[int | str, tuple[float, float]],
        particle_coordinates: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        target_radius: float | None = None,
        radial_index: int | None = None,
        use_natural: bool = False,
        eps: float = 1e-12,
        positive_only: bool = True,
        return_folds: bool = False,
        interpolate_to_radius: float | None = None,
        allow_loglog: bool = False,
        *,
        # Subsampling controls (performed *before* JK masking)
        use_recommended_subsample: bool = False,
        num_target_pairs: int = 100_000,
        subsample_rng_seed: int | None = None,
        subsample_halos: bool = False,
        subsample_particles: bool = False,
    ) -> dict[int | str, dict[str, float | int | np.ndarray]]:
    """Compute scalar halo bias per property bin via jackknife.

    If `interpolate_to_radius` is provided, interpolate xi(r) to that radius
    per fold; otherwise evaluate at the nearest (or specified) bin index.

    Optional pre-JK subsampling reduces catalogs before masking, to target a
    recommended pair-count budget (see two_point.get_tpcf_particle_subsample_size).
    """
    _check_rmax_vs_box(rmax, boxsize)

    rng = np.random.default_rng(subsample_rng_seed) if use_recommended_subsample else None

    # Optional pre-JK subsampling of the *full* catalogs
    halo_coordinates, halo_property = _maybe_subsample_coords_and_property(
        halo_coordinates, np.asarray(halo_property, dtype=float),
        boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
        num_target_pairs=num_target_pairs, rng=rng, enable=subsample_halos
    )
    particle_coordinates, _ = _maybe_subsample_coords(
        particle_coordinates,
        boxsize=boxsize, rmin=rmin, rmax=rmax, nbins=nbins,
        num_target_pairs=num_target_pairs, rng=rng, enable=subsample_particles
    )

    # Particle resamples and precomputed xi_mm per fold (defines shared r-grid)
    part_groups = get_jackknife_subbox_samples(
        sub_box_info=sub_box_info,
        coordinates=particle_coordinates,
        return_as_mask=False,
    )
    part_resamples = jackknife_resampling(part_groups)
    mm_fold_xi, r_grid = _precompute_mm_folds(
        particle_coordinates=particle_coordinates,
        part_resamples=part_resamples,
        boxsize=boxsize,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        use_natural=use_natural,
        eps=eps,
    )

    results: dict[int | str, dict[str, float | int | np.ndarray]] = {}

    for key, (pmin, pmax) in property_bins.items():
        halos_sel, _, n_sel = _select_halos_in_property_bin(
            halo_coordinates=halo_coordinates,
            halo_property=halo_property,
            pmin=pmin, pmax=pmax,
        )

        if n_sel < 2:
            results[key] = _empty_property_result(
                n_sel=int(n_sel), r_grid=r_grid, mm_fold_xi=mm_fold_xi, return_folds=return_folds
            )
            continue

        halo_groups = get_jackknife_subbox_samples(
            sub_box_info=sub_box_info,
            coordinates=halos_sel,
            return_as_mask=False,
        )
        halo_resamples = jackknife_resampling(halo_groups)

        # Evaluate per fold either by interpolation or at a chosen bin index
        if (interpolate_to_radius is not None) and (r_grid is not None):
            r_value = float(interpolate_to_radius)
            per_fold_bias, fold_vals = _compute_per_fold_bias_at_radius(
                halo_resamples=halo_resamples,
                mm_fold_r=r_grid,
                mm_fold_xi=mm_fold_xi,
                r_target=r_value,
                boxsize=boxsize,
                rmin=rmin,
                rmax=rmax,
                nbins=nbins,
                use_natural=use_natural,
                eps=eps,
                positive_only=positive_only,
                allow_loglog=allow_loglog,
            )
        else:
            # Choose r-bin index
            j_idx = _choose_rbin_index(r_grid, target_radius, radial_index)
            r_value = float(r_grid[j_idx]) if r_grid is not None else np.nan
            per_fold_bias, fold_vals = _compute_per_fold_bias_at_index(
                halo_resamples=halo_resamples,
                mm_fold_xi=mm_fold_xi,
                j_idx=j_idx,
                boxsize=boxsize,
                rmin=rmin,
                rmax=rmax,
                nbins=nbins,
                use_natural=use_natural,
                eps=eps,
                positive_only=positive_only,
            )

        b_hat, b_err = _jk_scalar_from_folds(fold_vals)

        results[key] = _package_property_result(
            b_hat=b_hat,
            b_err=b_err,
            n_sel=int(n_sel),
            r_value=r_value,
            per_fold_bias=per_fold_bias,
            return_folds=return_folds,
        )

    return results


def _bias_for_many_properties(
        halo_coords: np.ndarray,
        halo_props_dict: dict[str, np.ndarray],
        bins_dict: dict[str, dict[int | str, tuple[float, float]]],
        particle_coords: np.ndarray,
        sub_box_info: dict[int, list[tuple[float, float]]],
        boxsize: float,
        rmin: float,
        rmax: float,
        nbins: int,
        target_radius: float | None = None,
        radial_index: int | None = None,
        use_natural: bool = False,
        eps: float = 1e-12,
        positive_only: bool = True,
        return_folds: bool = False,
        interpolate_to_radius: float | None = None,
        allow_loglog: bool = False,
    ) -> dict[str, dict[int | str, dict[str, float | int | np.ndarray]]]:
    """Compute bias-vs-property for many properties, reusing ξ_mm folds once."""
    _check_rmax_vs_box(rmax, boxsize)

    # Precompute particle resamples & xi_mm folds once
    part_groups = get_jackknife_subbox_samples(
        sub_box_info=sub_box_info,
        coordinates=particle_coords,
        return_as_mask=False,
    )
    part_resamples = jackknife_resampling(part_groups)
    mm_fold_xi, r_grid = _precompute_mm_folds(
        particle_coordinates=particle_coords,
        part_resamples=part_resamples,
        boxsize=boxsize,
        rmin=rmin,
        rmax=rmax,
        nbins=nbins,
        use_natural=use_natural,
        eps=eps,
    )

    out: dict[str, dict[int | str, dict[str, float | int | np.ndarray]]] = {}
    for prop_name, prop_vals in halo_props_dict.items():
        bins = bins_dict[prop_name]
        prop_results: dict[int | str, dict[str, float | int | np.ndarray]] = {}

        for key, (pmin, pmax) in bins.items():
            sel = (prop_vals >= pmin) & (prop_vals < pmax)
            halos_sel = halo_coords[sel]
            n_sel = halos_sel.shape[0]
            if n_sel < 2:
                prop_results[key] = _empty_property_result(n_sel=int(n_sel), r_grid=r_grid, mm_fold_xi=mm_fold_xi, return_folds=return_folds)
                continue

            halo_groups = get_jackknife_subbox_samples(
                sub_box_info=sub_box_info,
                coordinates=halos_sel,
                return_as_mask=False,
            )
            halo_resamples = jackknife_resampling(halo_groups)

            if interpolate_to_radius is not None:
                per_fold_bias, fold_vals = _compute_per_fold_bias_at_radius(
                    halo_resamples=halo_resamples,
                    mm_fold_r=r_grid,
                    mm_fold_xi=mm_fold_xi,
                    r_target=float(interpolate_to_radius),
                    boxsize=boxsize,
                    rmin=rmin,
                    rmax=rmax,
                    nbins=nbins,
                    use_natural=use_natural,
                    eps=eps,
                    positive_only=positive_only,
                    allow_loglog=allow_loglog,
                )
                r_value = float(interpolate_to_radius)
            else:
                j_idx = _choose_rbin_index(
                    r_grid=r_grid, target_radius=target_radius, radial_index=radial_index
                )
                per_fold_bias, fold_vals = _compute_per_fold_bias_at_index(
                    halo_resamples=halo_resamples,
                    mm_fold_xi=mm_fold_xi,
                    j_idx=j_idx,
                    boxsize=boxsize,
                    rmin=rmin,
                    rmax=rmax,
                    nbins=nbins,
                    use_natural=use_natural,
                    eps=eps,
                    positive_only=positive_only,
                )
                r_value = float(r_grid[j_idx]) if r_grid is not None else np.nan

            b_hat, b_err = _jk_scalar_from_folds(fold_vals)
            prop_results[key] = _package_property_result(
                b_hat=b_hat,
                b_err=b_err,
                n_sel=int(n_sel),
                r_value=r_value,
                per_fold_bias=per_fold_bias,
                return_folds=return_folds,
            )

        out[prop_name] = prop_results

    return out


def save_halo_bias_dataset(filepath: Path, data: dict[str, np.ndarray]) -> None:
    """
    Save a HaloBiasDataset to an HDF5 file.

    Parameters
    ----------
    filepath : Path
        Target .h5/.hdf5 file path.
    data : dict[str, np.ndarray]
        The output of HaloBiasDataset.as_dict(). Each value is an (N, 3) array
        with columns [property_values, estimates, errors].
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as h5:
        # Optional: a tiny version tag
        h5.attrs["format"] = "HaloBiasDataset.v1"
        for name, arr in data.items():
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"Dataset for '{name}' must be a 2D array with 3 columns; got shape {arr.shape}."
                )
            if name in h5:
                del h5[name]
            h5.create_dataset(name, data=arr)


def load_halo_bias_dataset(filepath: Path) -> dict[str, np.ndarray]:
    """
    Load a HaloBiasDataset from an HDF5 file written by `save_halo_bias_dataset`.

    Returns a dict compatible with `HaloBiasDataset.as_dict()` where each value
    is an (N, 3) array [property_values, estimates, errors].
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: {filepath}")

    out: dict[str, np.ndarray] = {}
    with h5py.File(filepath, "r") as h5:
        # Accept files even if the attribute is missing
        for name, node in h5.items():
            if isinstance(node, h5py.Dataset):
                arr = np.array(node[()])
                # Normalize empty datasets to shape (0,3)
                if arr.ndim == 1 and arr.size == 0:
                    arr = arr.reshape((0, 3))
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        f"Dataset '{name}' must be a 2D array with 3 columns; got shape {arr.shape}."
                    )
                out[name] = arr
            else:
                # If a group is encountered (future-proofing), attempt to reconstruct
                # from child datasets named 'property_values', 'estimates', 'errors'.
                grp = node
                keys = [k for k in ("property_values", "estimates", "errors") if k in grp]
                if len(keys) == 3:
                    pv = np.array(grp["property_values"][()]).reshape(-1)
                    est = np.array(grp["estimates"][()]).reshape(-1)
                    err = np.array(grp["errors"][()]).reshape(-1)
                    if not (pv.shape == est.shape == err.shape):
                        raise ValueError(
                            f"Group '{name}' children must share the same length; got lengths "
                            f"{pv.shape}, {est.shape}, {err.shape}."
                        )
                    out[name] = np.column_stack([pv, est, err])
                # else: ignore unknown groups

    return out