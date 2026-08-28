"""
Freeze-out metrics for TPCF correlation matrices.

This module provides tools to quantify the "freeze-out" of the two-point
correlation function's covariance/correlation structure by comparing
matrices at different epochs on a fixed physical separation grid.

The key insight is that comparing correlation matrices at fixed comoving
bins conflates dynamical evolution with the fact that at later scale factors,
the same comoving bin corresponds to larger physical scales. By mapping to
a fixed physical grid via s(a) = r_phys / a, we isolate the true freeze-out
behavior.
"""
from __future__ import annotations

from typing import Literal
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
from attrs import define, field


@define(slots=True)
class FreezeOutMetrics:
    """Container for freeze-out distance metrics.
    
    Attributes
    ----------
    scale_factors : np.ndarray
        Scale factors at each epoch, shape (N_snap,).
    snapshot_ids : np.ndarray
        Snapshot IDs corresponding to each epoch, shape (N_snap,).
    r_phys_grid : np.ndarray
        The fixed physical separation grid used, shape (N_bins,).
    frobenius_global : np.ndarray
        Normalized Frobenius distance to final correlation matrix at each epoch,
        shape (N_snap,). Values are NaN for epochs that could not be computed.
    band_boundaries : dict[str, tuple[float, float]] | None
        Physical scale band boundaries, e.g. {"small": (0.2, 2.0), ...}.
    frobenius_bands : dict[str, np.ndarray] | None
        Per-band Frobenius distances, {band_name: (N_snap,)}.
    valid_bins_per_epoch : np.ndarray
        Number of valid (non-NaN) bins at each epoch after interpolation,
        shape (N_snap,). Useful for diagnosing edge effects.
    fixed_mask_global : np.ndarray | None
        Boolean mask of bins used when use_fixed_mask=True (global).
        Shape (N_bins,). None if use_fixed_mask=False.
    fixed_masks_bands : dict[str, np.ndarray] | None
        Per-band boolean masks used when use_fixed_mask=True.
        {band_name: (N_band_bins,)}. None if use_fixed_mask=False.
    used_fixed_mask : bool
        Whether fixed-mask mode was used for this computation.
    """
    scale_factors: np.ndarray
    snapshot_ids: np.ndarray
    r_phys_grid: np.ndarray
    frobenius_global: np.ndarray
    band_boundaries: dict[str, tuple[float, float]] | None = field(default=None)
    frobenius_bands: dict[str, np.ndarray] | None = field(default=None)
    valid_bins_per_epoch: np.ndarray = field(factory=lambda: np.array([]))
    fixed_mask_global: np.ndarray | None = field(default=None)
    fixed_masks_bands: dict[str, np.ndarray] | None = field(default=None)
    used_fixed_mask: bool = field(default=False)

    def __repr__(self) -> str:
        n_snap = len(self.scale_factors)
        n_bins = len(self.r_phys_grid)
        a_range = f"[{self.scale_factors.min():.3f}, {self.scale_factors.max():.3f}]"
        r_range = f"[{self.r_phys_grid.min():.2f}, {self.r_phys_grid.max():.2f}]"
        bands_str = f", {len(self.band_boundaries)} bands" if self.band_boundaries else ""
        mask_str = ", fixed-mask" if self.used_fixed_mask else ""
        return (
            f"FreezeOutMetrics({n_snap} epochs, {n_bins} physical bins, "
            f"a∈{a_range}, r∈{r_range} Mpc/h{bands_str}{mask_str})"
        )

    def coverage_report(self, print_report: bool = True) -> dict:
        """Generate a summary of mask coverage for documentation.
        
        This method reports on the bins used in the freeze-out computation,
        particularly useful when using fixed-mask mode to document which
        physical scales are consistently sampled across all epochs.
        
        Parameters
        ----------
        print_report : bool, default True
            If True, print a formatted report to stdout.
            
        Returns
        -------
        dict
            Dictionary containing coverage statistics:
            - 'mode': 'fixed-mask' or 'variable-mask'
            - 'n_bins_total': Total bins in physical grid
            - 'n_bins_global': Number of bins in global mask (or min valid if variable)
            - 'r_phys_range_global': (r_min, r_max) of global mask
            - 'fraction_global': Fraction of total grid retained
            - 'bands': dict of per-band statistics (if bands defined)
                - {band_name: {'n_bins': int, 'r_phys_range': (min, max),
                               'fraction': float, 'nominal_range': (min, max)}}
        """
        report = {
            'mode': 'fixed-mask' if self.used_fixed_mask else 'variable-mask',
            'n_bins_total': len(self.r_phys_grid),
        }
        
        # Global coverage
        if self.used_fixed_mask and self.fixed_mask_global is not None:
            mask = self.fixed_mask_global
            n_global = int(np.sum(mask))
            if n_global > 0:
                r_masked = self.r_phys_grid[mask]
                r_range = (float(r_masked.min()), float(r_masked.max()))
            else:
                r_range = (np.nan, np.nan)
            frac = n_global / len(self.r_phys_grid)
        else:
            # Variable mask: report based on valid_bins_per_epoch
            n_global = int(np.min(self.valid_bins_per_epoch)) if len(self.valid_bins_per_epoch) > 0 else 0
            # For variable mask, we can't give exact range since it varies
            r_range = (float(self.r_phys_grid.min()), float(self.r_phys_grid.max()))
            frac = n_global / len(self.r_phys_grid) if len(self.r_phys_grid) > 0 else 0.0
        
        report['n_bins_global'] = n_global
        report['r_phys_range_global'] = r_range
        report['fraction_global'] = frac
        
        # Per-band coverage
        if self.band_boundaries is not None:
            bands_report = {}
            for band_name, (r_lo, r_hi) in self.band_boundaries.items():
                # Bins in the nominal band definition
                band_mask_nominal = (self.r_phys_grid >= r_lo) & (self.r_phys_grid <= r_hi)
                n_nominal = int(np.sum(band_mask_nominal))
                
                if self.used_fixed_mask and self.fixed_masks_bands is not None:
                    band_mask = self.fixed_masks_bands.get(band_name)
                    if band_mask is not None:
                        # band_mask is over the full r_phys_grid
                        n_band = int(np.sum(band_mask))
                        if n_band > 0:
                            # Get actual range covered (band_mask is full-grid sized)
                            r_band = self.r_phys_grid[band_mask]
                            r_band_range = (float(r_band.min()), float(r_band.max()))
                        else:
                            r_band_range = (np.nan, np.nan)
                    else:
                        n_band = 0
                        r_band_range = (np.nan, np.nan)
                else:
                    # Variable mask: approximate with nominal
                    n_band = n_nominal
                    r_band_range = (r_lo, r_hi)
                
                frac_band = n_band / n_nominal if n_nominal > 0 else 0.0
                bands_report[band_name] = {
                    'n_bins': n_band,
                    'r_phys_range': r_band_range,
                    'fraction': frac_band,
                    'nominal_range': (r_lo, r_hi),
                    'n_bins_nominal': n_nominal,
                }
            report['bands'] = bands_report
        
        if print_report:
            self._print_coverage_report(report)
        
        return report

    def _print_coverage_report(self, report: dict) -> None:
        """Print a formatted coverage report."""
        print("=" * 60)
        print("FREEZE-OUT MASK COVERAGE REPORT")
        print("=" * 60)
        print(f"Mode: {report['mode']}")
        print(f"Scale factor range: [{self.scale_factors.min():.3f}, {self.scale_factors.max():.3f}]")
        print(f"Number of epochs: {len(self.scale_factors)}")
        print("-" * 60)
        print("GLOBAL COVERAGE:")
        print(f"  Total bins in physical grid: {report['n_bins_total']}")
        print(f"  Bins in global mask: {report['n_bins_global']}")
        r_lo, r_hi = report['r_phys_range_global']
        print(f"  Physical range covered: [{r_lo:.3f}, {r_hi:.3f}] Mpc/h")
        print(f"  Retention fraction: {report['fraction_global']:.1%}")
        
        if 'bands' in report:
            print("-" * 60)
            print("PER-BAND COVERAGE:")
            for band_name, band_stats in report['bands'].items():
                print(f"\n  {band_name.upper()}:")
                nom_lo, nom_hi = band_stats['nominal_range']
                print(f"    Nominal range: [{nom_lo:.2f}, {nom_hi:.2f}] Mpc/h")
                print(f"    Bins (nominal): {band_stats['n_bins_nominal']}")
                print(f"    Bins (retained): {band_stats['n_bins']}")
                r_lo, r_hi = band_stats['r_phys_range']
                if not np.isnan(r_lo):
                    print(f"    Actual range: [{r_lo:.3f}, {r_hi:.3f}] Mpc/h")
                else:
                    print(f"    Actual range: (no valid bins)")
                print(f"    Retention: {band_stats['fraction']:.1%}")
        print("=" * 60)


def interpolate_jk_curves_to_physical_grid(
    s_bins_comoving: np.ndarray,
    xi_jk: np.ndarray,
    scale_factor: float,
    r_phys_grid: np.ndarray,
    interp_kind: str = "linear",
) -> np.ndarray:
    """Interpolate jackknife correlation curves from comoving to physical grid.
    
    Parameters
    ----------
    s_bins_comoving : np.ndarray
        Comoving separation bin centers, shape (M,).
    xi_jk : np.ndarray
        Jackknife correlation curves, shape (J, M) where J is number of
        jackknife regions and M is number of comoving bins.
    scale_factor : float
        Scale factor for this epoch.
    r_phys_grid : np.ndarray
        Target physical separation grid, shape (K,).
    interp_kind : str
        Interpolation method for scipy.interpolate.interp1d. Default "linear".
        
    Returns
    -------
    np.ndarray
        Interpolated jackknife curves on physical grid, shape (J, K).
        Values outside the comoving range are set to NaN.
    """
    # Map physical grid to comoving at this epoch
    s_target = r_phys_grid / scale_factor
    
    # Interpolate in log-space for better behavior with power-law correlation
    log_s_bins = np.log10(s_bins_comoving)
    log_s_target = np.log10(s_target)
    
    n_jk, _ = xi_jk.shape
    n_phys = len(r_phys_grid)
    xi_phys = np.full((n_jk, n_phys), np.nan)
    
    for j in range(n_jk):
        # Create interpolator for this jackknife curve
        # Use bounds_error=False to get NaN for out-of-range
        interp_func = interp1d(
            log_s_bins,
            xi_jk[j],
            kind=interp_kind,
            bounds_error=False,
            fill_value=np.nan,
        )
        xi_phys[j] = interp_func(log_s_target)
    
    return xi_phys


def cov_matrix_from_jk(
    xi_jk: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute covariance matrix from jackknife samples.
    
    Uses the standard jackknife covariance estimator:
        C_ij = (J-1)/J * sum_j (xi_j - xi_mean)(xi_j - xi_mean)^T
    
    Parameters
    ----------
    xi_jk : np.ndarray
        Jackknife correlation curves, shape (J, K).
    mask : np.ndarray | None
        Boolean mask of valid bins, shape (K,). If None, all bins are used.
        
    Returns
    -------
    cov : np.ndarray
        Covariance matrix, shape (K, K) or (K_valid, K_valid) if masked.
    xi_mean : np.ndarray
        Mean correlation function.
    valid_mask : np.ndarray
        Boolean mask of bins used (non-NaN across all jackknife samples).
    """
    n_jk, n_bins = xi_jk.shape
    
    # Build validity mask: bins where ALL jackknife curves are finite
    if mask is None:
        valid_mask = np.all(np.isfinite(xi_jk), axis=0)
    else:
        valid_mask = mask & np.all(np.isfinite(xi_jk), axis=0)
    
    if not np.any(valid_mask):
        # No valid bins
        return (
            np.full((n_bins, n_bins), np.nan),
            np.full(n_bins, np.nan),
            valid_mask,
        )
    
    # Extract valid portion
    xi_valid = xi_jk[:, valid_mask]
    xi_mean_valid = np.mean(xi_valid, axis=0)
    
    # Jackknife covariance: (J-1)/J * sum of outer products
    delta = xi_valid - xi_mean_valid  # shape (J, K_valid)
    cov_valid = ((n_jk - 1) / n_jk) * (delta.T @ delta)
    
    # Embed back into full-size matrix if needed
    n_valid = np.sum(valid_mask)
    if n_valid == n_bins:
        return cov_valid, xi_mean_valid, valid_mask
    
    # Create full-size arrays with NaN for invalid bins
    cov_full = np.full((n_bins, n_bins), np.nan)
    xi_mean_full = np.full(n_bins, np.nan)
    
    valid_idx = np.where(valid_mask)[0]
    cov_full[np.ix_(valid_idx, valid_idx)] = cov_valid
    xi_mean_full[valid_mask] = xi_mean_valid
    
    return cov_full, xi_mean_full, valid_mask


def corr_matrix_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix.
    
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix, shape (K, K).
        
    Returns
    -------
    np.ndarray
        Correlation matrix, shape (K, K). Diagonal elements with zero or
        negative variance are set to NaN in the corresponding row/column.
    """
    diag = np.diag(cov)
    
    # Handle non-positive diagonal elements
    valid_diag = diag > 0
    if not np.all(valid_diag):
        # Create a copy to avoid modifying input
        cov = cov.copy()
        invalid_idx = ~valid_diag
        cov[invalid_idx, :] = np.nan
        cov[:, invalid_idx] = np.nan
        diag = np.where(valid_diag, diag, 1.0)  # Avoid division warnings
    
    stdev = np.sqrt(diag)
    corr = cov / np.outer(stdev, stdev)
    
    return corr


def corr_matrix_on_physical_grid_from_jk(
    s_bins_comoving: np.ndarray,
    xi_jk: np.ndarray,
    scale_factor: float,
    r_phys_grid: np.ndarray,
    return_cov: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build correlation matrix on fixed physical grid from jackknife curves.
    
    This is the main workhorse function that:
    1. Maps comoving jackknife curves to a fixed physical grid
    2. Computes the jackknife covariance on that grid
    3. Converts to correlation matrix
    
    Parameters
    ----------
    s_bins_comoving : np.ndarray
        Comoving separation bin centers, shape (M,).
    xi_jk : np.ndarray
        Jackknife correlation curves, shape (J, M).
    scale_factor : float
        Scale factor for this epoch.
    r_phys_grid : np.ndarray
        Target physical separation grid, shape (K,).
    return_cov : bool
        If True, return covariance matrix instead of correlation matrix.
        
    Returns
    -------
    matrix : np.ndarray
        Correlation (or covariance) matrix on physical grid, shape (K, K).
    xi_mean : np.ndarray
        Mean correlation function on physical grid, shape (K,).
    n_valid : int
        Number of valid bins (non-NaN in the diagonal).
    """
    # Step 1: Interpolate to physical grid
    xi_phys = interpolate_jk_curves_to_physical_grid(
        s_bins_comoving=s_bins_comoving,
        xi_jk=xi_jk,
        scale_factor=scale_factor,
        r_phys_grid=r_phys_grid,
    )
    
    # Step 2: Compute covariance
    cov, xi_mean, valid_mask = cov_matrix_from_jk(xi_phys)
    n_valid = int(np.sum(valid_mask))
    
    if return_cov:
        return cov, xi_mean, n_valid
    
    # Step 3: Convert to correlation
    corr = corr_matrix_from_cov(cov)
    
    return corr, xi_mean, n_valid


def frobenius_distance(
    A: np.ndarray,
    B: np.ndarray,
    normalize: bool = True,
    use_valid_only: bool = True,
) -> float:
    """Compute (normalized) Frobenius distance between two matrices.
    
    Parameters
    ----------
    A : np.ndarray
        First matrix.
    B : np.ndarray
        Second matrix (reference for normalization).
    normalize : bool
        If True, normalize by ||B||_F.
    use_valid_only : bool
        If True, only use elements that are finite in both matrices.
        
    Returns
    -------
    float
        Frobenius distance (or NaN if no valid elements).
    """
    if use_valid_only:
        valid = np.isfinite(A) & np.isfinite(B)
        if not np.any(valid):
            return np.nan
        diff = A[valid] - B[valid]
        norm_B = np.linalg.norm(B[valid])
    else:
        diff = A - B
        norm_B = np.linalg.norm(B)
    
    dist = np.linalg.norm(diff)
    
    if normalize and norm_B > 0:
        return dist / norm_B
    return dist


def create_log_spaced_bands(
    r_min: float,
    r_max: float,
    n_bands: int,
) -> dict[str, tuple[float, float]]:
    """Create logarithmically spaced scale bands.
    
    Parameters
    ----------
    r_min : float
        Minimum physical scale.
    r_max : float
        Maximum physical scale.
    n_bands : int
        Number of bands to create.
        
    Returns
    -------
    dict[str, tuple[float, float]]
        Band boundaries, e.g. {"band_0": (0.2, 1.0), "band_1": (1.0, 5.0), ...}.
    """
    edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bands + 1)
    bands = {}
    for i in range(n_bands):
        bands[f"band_{i}"] = (edges[i], edges[i + 1])
    return bands


def get_band_indices(
    r_phys_grid: np.ndarray,
    band_boundaries: tuple[float, float],
) -> np.ndarray:
    """Get indices of physical grid bins that fall within a band.
    
    Parameters
    ----------
    r_phys_grid : np.ndarray
        Physical separation grid.
    band_boundaries : tuple[float, float]
        (r_min, r_max) for the band.
        
    Returns
    -------
    np.ndarray
        Boolean mask of bins within the band.
    """
    r_min, r_max = band_boundaries
    return (r_phys_grid >= r_min) & (r_phys_grid <= r_max)


def extract_jk_curves_from_folds(
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract jackknife curves from folds array.
    
    Parameters
    ----------
    folds : np.ndarray
        Folds array with shape (n_bins, 1 + n_jk), where column 0 is radii
        and remaining columns are per-fold xi values.
        
    Returns
    -------
    s_bins : np.ndarray
        Separation bin centers, shape (n_bins,).
    xi_jk : np.ndarray
        Jackknife curves, shape (n_jk, n_bins).
    """
    s_bins = folds[:, 0]
    xi_jk = folds[:, 1:].T  # Transpose to (n_jk, n_bins)
    return s_bins, xi_jk


def compute_freezeout_metrics(
    folds_evo: OrderedDict[int, np.ndarray | None],
    scale_factors: OrderedDict[int, float],
    r_phys_grid: np.ndarray | None = None,
    r_phys_min: float = 0.2,
    r_phys_max: float = 50.0,
    n_phys_bins: int = 30,
    final_snap_id: int | None = None,
    scale_bands: dict[str, tuple[float, float]] | None = None,
    n_bands: int | None = None,
    min_scale_factor: float = 0.0,
    max_scale_factor: float = 100.0,
) -> FreezeOutMetrics:
    """Compute freeze-out distance metrics for correlation matrices.
    
    This function computes the evolution of the correlation matrix relative
    to its final state, on a fixed physical separation grid. The key metric
    is the normalized Frobenius distance, which should decrease as the
    correlation structure "freezes out".
    
    Parameters
    ----------
    folds_evo : OrderedDict[int, np.ndarray | None]
        Mapping from snapshot ID to folds array. Folds array has shape
        (n_bins, 1 + n_jk) where column 0 is radii.
    scale_factors : OrderedDict[int, float]
        Mapping from snapshot ID to scale factor.
    r_phys_grid : np.ndarray | None
        Fixed physical separation grid. If None, creates log-spaced grid
        from r_phys_min to r_phys_max with n_phys_bins bins.
    r_phys_min : float
        Minimum physical scale if creating grid. Default 0.2 Mpc/h.
    r_phys_max : float
        Maximum physical scale if creating grid. Default 50.0 Mpc/h.
    n_phys_bins : int
        Number of bins if creating grid. Default 30.
    final_snap_id : int | None
        Snapshot ID to use as the final reference. If None, uses the
        latest snapshot with valid folds data.
    scale_bands : dict[str, tuple[float, float]] | None
        Explicit band boundaries for hierarchical analysis.
        E.g. {"small": (0.2, 2.0), "mid": (2.0, 10.0), "large": (10.0, 50.0)}.
    n_bands : int | None
        If provided and scale_bands is None, create this many log-spaced bands.
    min_scale_factor : float
        Minimum scale factor to include. Default 0.0.
    max_scale_factor : float
        Maximum scale factor to include. Default 100.0.
        
    Returns
    -------
    FreezeOutMetrics
        Container with freeze-out distance metrics.
        
    Raises
    ------
    ValueError
        If no valid folds data is available or final snapshot has no folds.
    """
    # Build physical grid if not provided
    if r_phys_grid is None:
        r_phys_grid = np.logspace(
            np.log10(r_phys_min),
            np.log10(r_phys_max),
            n_phys_bins,
        )
    
    # Filter snapshots by scale factor range and valid folds
    valid_snaps = []
    for snap_id, folds in folds_evo.items():
        if folds is None:
            continue
        a = scale_factors.get(snap_id)
        if a is None:
            continue
        if min_scale_factor <= a <= max_scale_factor:
            valid_snaps.append((snap_id, a, folds))
    
    if not valid_snaps:
        raise ValueError("No valid snapshots with folds data in scale factor range")
    
    # Sort by scale factor
    valid_snaps.sort(key=lambda x: x[1])
    
    # Determine final snapshot
    if final_snap_id is not None:
        final_data = next(
            ((sid, a, f) for sid, a, f in valid_snaps if sid == final_snap_id),
            None,
        )
        if final_data is None:
            raise ValueError(
                f"Final snapshot {final_snap_id} not found or has no valid folds"
            )
    else:
        final_data = valid_snaps[-1]
    
    final_snap_id, final_a, final_folds = final_data
    
    # Compute final correlation matrix
    s_bins_final, xi_jk_final = extract_jk_curves_from_folds(final_folds)
    corr_final, _, n_valid_final = corr_matrix_on_physical_grid_from_jk(
        s_bins_comoving=s_bins_final,
        xi_jk=xi_jk_final,
        scale_factor=final_a,
        r_phys_grid=r_phys_grid,
    )
    
    # Set up bands
    band_boundaries = scale_bands
    if band_boundaries is None and n_bands is not None:
        band_boundaries = create_log_spaced_bands(
            r_min=r_phys_grid.min(),
            r_max=r_phys_grid.max(),
            n_bands=n_bands,
        )
    
    # Initialize output arrays
    n_snap = len(valid_snaps)
    snap_ids_out = np.array([s[0] for s in valid_snaps])
    scale_factors_out = np.array([s[1] for s in valid_snaps])
    frobenius_global = np.full(n_snap, np.nan)
    valid_bins_per_epoch = np.zeros(n_snap, dtype=int)
    
    frobenius_bands: dict[str, np.ndarray] | None = None
    if band_boundaries is not None:
        frobenius_bands = {
            name: np.full(n_snap, np.nan) for name in band_boundaries
        }
    
    # Compute metrics for each epoch
    for i, (snap_id, a, folds) in enumerate(valid_snaps):
        s_bins, xi_jk = extract_jk_curves_from_folds(folds)
        
        corr_a, _, n_valid = corr_matrix_on_physical_grid_from_jk(
            s_bins_comoving=s_bins,
            xi_jk=xi_jk,
            scale_factor=a,
            r_phys_grid=r_phys_grid,
        )
        valid_bins_per_epoch[i] = n_valid
        
        if n_valid == 0:
            continue
        
        # Global Frobenius distance
        frobenius_global[i] = frobenius_distance(
            corr_a, corr_final, normalize=True, use_valid_only=True
        )
        
        # Per-band distances
        if band_boundaries is not None and frobenius_bands is not None:
            for band_name, (r_lo, r_hi) in band_boundaries.items():
                band_mask = get_band_indices(r_phys_grid, (r_lo, r_hi))
                if not np.any(band_mask):
                    continue
                
                # Extract sub-matrices for this band
                band_idx = np.where(band_mask)[0]
                corr_a_band = corr_a[np.ix_(band_idx, band_idx)]
                corr_final_band = corr_final[np.ix_(band_idx, band_idx)]
                
                frobenius_bands[band_name][i] = frobenius_distance(
                    corr_a_band, corr_final_band, normalize=True, use_valid_only=True
                )
    
    return FreezeOutMetrics(
        scale_factors=scale_factors_out,
        snapshot_ids=snap_ids_out,
        r_phys_grid=r_phys_grid,
        frobenius_global=frobenius_global,
        band_boundaries=band_boundaries,
        frobenius_bands=frobenius_bands,
        valid_bins_per_epoch=valid_bins_per_epoch,
    )


# =============================================================================
# TreeCorr-based freeze-out metrics
# =============================================================================

def extract_treecorr_design_matrix(
    treecorr_nn,  # treecorr.NNCorrelation
    method: str = "jackknife",
    cross_patch_weight: str = "match",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the covariance design matrix from a TreeCorr NNCorrelation object.
    
    The design matrix rows are the xi vectors for each jackknife/bootstrap sample.
    TreeCorr's cross-patch weighting is preserved.
    
    Parameters
    ----------
    treecorr_nn : treecorr.NNCorrelation
        The TreeCorr correlation object (must have been processed with patches).
    method : str
        Variance method: 'jackknife', 'sample', 'bootstrap', 'marked_bootstrap'.
        Default 'jackknife'.
    cross_patch_weight : str
        Cross-patch weighting scheme: 'simple', 'mean', 'match', 'geom'.
        Default 'match' (recommended by Mohammad & Percival 2022).
        
    Returns
    -------
    design_matrix : np.ndarray
        Design matrix, shape (n_samples, n_bins). Each row is a xi vector.
    weights : np.ndarray
        Weight for each row (used by 'sample' method).
    rnom : np.ndarray
        Nominal bin centers (comoving), shape (n_bins,).
    """
    A, w = treecorr_nn.build_cov_design_matrix(
        method=method,
        cross_patch_weight=cross_patch_weight,
    )
    return A, w, treecorr_nn.rnom.copy()


def interpolate_design_matrix_to_physical_grid(
    design_matrix: np.ndarray,
    s_bins_comoving: np.ndarray,
    scale_factor: float,
    r_phys_grid: np.ndarray,
    interp_kind: str = "linear",
) -> np.ndarray:
    """Interpolate a design matrix from comoving bins to a physical grid.
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix, shape (n_samples, n_comoving_bins).
    s_bins_comoving : np.ndarray
        Comoving separation bin centers, shape (n_comoving_bins,).
    scale_factor : float
        Scale factor for this epoch.
    r_phys_grid : np.ndarray
        Target physical separation grid, shape (n_phys_bins,).
    interp_kind : str
        Interpolation method. Default "linear".
        
    Returns
    -------
    np.ndarray
        Interpolated design matrix, shape (n_samples, n_phys_bins).
    """
    # This is the same as interpolate_jk_curves_to_physical_grid
    # but with a more general name
    return interpolate_jk_curves_to_physical_grid(
        s_bins_comoving=s_bins_comoving,
        xi_jk=design_matrix,
        scale_factor=scale_factor,
        r_phys_grid=r_phys_grid,
        interp_kind=interp_kind,
    )


def cov_matrix_from_treecorr_design_matrix(
    design_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "jackknife",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute covariance matrix from TreeCorr design matrix.
    
    Applies the correct formula depending on the method used.
    
    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix, shape (n_samples, n_bins).
    weights : np.ndarray | None
        Per-row weights (only used for 'sample' method).
    method : str
        The method that was used to build the design matrix.
        
    Returns
    -------
    cov : np.ndarray
        Covariance matrix, shape (n_bins, n_bins).
    xi_mean : np.ndarray
        Mean xi vector, shape (n_bins,).
    valid_mask : np.ndarray
        Boolean mask of valid bins.
    """
    n_samples, n_bins = design_matrix.shape
    
    # Build validity mask: bins where ALL samples are finite
    valid_mask = np.all(np.isfinite(design_matrix), axis=0)
    
    if not np.any(valid_mask):
        return (
            np.full((n_bins, n_bins), np.nan),
            np.full(n_bins, np.nan),
            valid_mask,
        )
    
    # Extract valid portion
    xi_valid = design_matrix[:, valid_mask]
    xi_mean_valid = np.mean(xi_valid, axis=0)
    delta = xi_valid - xi_mean_valid
    
    # Apply correct formula based on method
    if method == "jackknife":
        # C = (N-1)/N * sum_i (v_i - v_mean)(v_i - v_mean)^T
        cov_valid = ((n_samples - 1) / n_samples) * (delta.T @ delta)
    elif method == "sample":
        # C = 1/(N-1) * sum_i w_i (v_i - v_mean)(v_i - v_mean)^T
        if weights is None:
            weights = np.ones(n_samples) / n_samples
        else:
            weights = weights / np.sum(weights)
        cov_valid = (1.0 / (n_samples - 1)) * ((weights[:, None] * delta).T @ delta)
    else:
        # bootstrap, marked_bootstrap: C = 1/(N-1) * sum
        cov_valid = (1.0 / (n_samples - 1)) * (delta.T @ delta)
    
    # Embed back into full-size matrix
    n_valid = np.sum(valid_mask)
    if n_valid == n_bins:
        return cov_valid, xi_mean_valid, valid_mask
    
    cov_full = np.full((n_bins, n_bins), np.nan)
    xi_mean_full = np.full(n_bins, np.nan)
    valid_idx = np.where(valid_mask)[0]
    cov_full[np.ix_(valid_idx, valid_idx)] = cov_valid
    xi_mean_full[valid_mask] = xi_mean_valid
    
    return cov_full, xi_mean_full, valid_mask


def corr_matrix_on_physical_grid_from_treecorr(
    treecorr_nn,  # treecorr.NNCorrelation
    scale_factor: float,
    r_phys_grid: np.ndarray,
    method: str = "jackknife",
    cross_patch_weight: str = "match",
    return_cov: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build correlation matrix on physical grid using TreeCorr's design matrix.
    
    This function:
    1. Extracts TreeCorr's covariance design matrix (preserving cross-patch weights)
    2. Interpolates each row to the fixed physical grid
    3. Recomputes the covariance/correlation from the interpolated matrix
    
    Parameters
    ----------
    treecorr_nn : treecorr.NNCorrelation
        TreeCorr correlation object (must have patch information).
    scale_factor : float
        Scale factor for this epoch.
    r_phys_grid : np.ndarray
        Target physical separation grid, shape (K,).
    method : str
        Variance method for TreeCorr. Default 'jackknife'.
    cross_patch_weight : str
        Cross-patch weighting. Default 'match'.
    return_cov : bool
        If True, return covariance instead of correlation matrix.
        
    Returns
    -------
    matrix : np.ndarray
        Correlation (or covariance) matrix on physical grid.
    xi_mean : np.ndarray
        Mean xi on physical grid.
    n_valid : int
        Number of valid bins.
    """
    # Step 1: Extract design matrix from TreeCorr
    design_matrix, weights, s_bins = extract_treecorr_design_matrix(
        treecorr_nn=treecorr_nn,
        method=method,
        cross_patch_weight=cross_patch_weight,
    )
    
    # Step 2: Interpolate to physical grid
    design_phys = interpolate_design_matrix_to_physical_grid(
        design_matrix=design_matrix,
        s_bins_comoving=s_bins,
        scale_factor=scale_factor,
        r_phys_grid=r_phys_grid,
    )
    
    # Step 3: Compute covariance from interpolated design matrix
    cov, xi_mean, valid_mask = cov_matrix_from_treecorr_design_matrix(
        design_matrix=design_phys,
        weights=weights,
        method=method,
    )
    n_valid = int(np.sum(valid_mask))
    
    if return_cov:
        return cov, xi_mean, n_valid
    
    # Step 4: Convert to correlation
    corr = corr_matrix_from_cov(cov)
    return corr, xi_mean, n_valid


def compute_freezeout_metrics_treecorr(
    treecorr_nn_evo: OrderedDict[int, object],  # OrderedDict[int, treecorr.NNCorrelation]
    scale_factors: OrderedDict[int, float],
    r_phys_grid: np.ndarray | None = None,
    r_phys_min: float = 0.2,
    r_phys_max: float = 50.0,
    n_phys_bins: int = 30,
    final_snap_id: int | None = None,
    scale_bands: dict[str, tuple[float, float]] | None = None,
    n_bands: int | None = None,
    min_scale_factor: float = 0.0,
    max_scale_factor: float = 100.0,
    method: str = "jackknife",
    cross_patch_weight: str = "match",
    use_fixed_mask: bool = False,
    fixed_mask_min_scale_factor: float | None = None,
) -> FreezeOutMetrics:
    """Compute freeze-out metrics using TreeCorr's covariance estimation.
    
    This version uses TreeCorr's `build_cov_design_matrix` to extract the
    jackknife/bootstrap samples with proper cross-patch weighting, then
    interpolates to the physical grid and recomputes the covariance.
    
    Parameters
    ----------
    treecorr_nn_evo : OrderedDict[int, treecorr.NNCorrelation]
        Mapping from snapshot ID to TreeCorr NNCorrelation objects.
    scale_factors : OrderedDict[int, float]
        Mapping from snapshot ID to scale factor.
    r_phys_grid : np.ndarray | None
        Fixed physical separation grid. If None, creates log-spaced grid.
    r_phys_min : float
        Minimum physical scale if creating grid. Default 0.2 Mpc/h.
    r_phys_max : float
        Maximum physical scale if creating grid. Default 50.0 Mpc/h.
    n_phys_bins : int
        Number of bins if creating grid. Default 30.
    final_snap_id : int | None
        Snapshot ID to use as final reference. If None, uses latest.
    scale_bands : dict[str, tuple[float, float]] | None
        Explicit band boundaries for hierarchical analysis.
    n_bands : int | None
        If provided, create this many log-spaced bands.
    min_scale_factor : float
        Minimum scale factor to include. Default 0.0.
    max_scale_factor : float
        Maximum scale factor to include. Default 100.0.
    method : str
        TreeCorr variance method. Default 'jackknife'.
    cross_patch_weight : str
        Cross-patch weighting scheme. Default 'match'.
    use_fixed_mask : bool
        If True, compute a fixed bin mask as the intersection of valid bins
        across all epochs (or epochs above fixed_mask_min_scale_factor).
        This ensures that Frobenius distances reflect only physical evolution,
        not changing bin validity. Default False.
    fixed_mask_min_scale_factor : float | None
        Minimum scale factor for epochs included in the fixed mask computation.
        If None, uses all epochs in the range. Useful for excluding early
        epochs where coverage is very sparse.
        
    Returns
    -------
    FreezeOutMetrics
        Container with freeze-out distance metrics. If use_fixed_mask=True,
        includes fixed_mask_global, fixed_masks_bands, and used_fixed_mask=True.
    """
    # Build physical grid if not provided
    if r_phys_grid is None:
        r_phys_grid = np.logspace(
            np.log10(r_phys_min),
            np.log10(r_phys_max),
            n_phys_bins,
        )
    
    # Filter snapshots by scale factor range and valid treecorr objects
    valid_snaps = []
    for snap_id, nn in treecorr_nn_evo.items():
        if nn is None:
            continue
        a = scale_factors.get(snap_id)
        if a is None:
            continue
        if min_scale_factor <= a <= max_scale_factor:
            valid_snaps.append((snap_id, a, nn))
    
    if not valid_snaps:
        raise ValueError("No valid snapshots with TreeCorr data in scale factor range")
    
    # Sort by scale factor
    valid_snaps.sort(key=lambda x: x[1])
    
    # Determine final snapshot
    if final_snap_id is not None:
        final_data = next(
            ((sid, a, nn) for sid, a, nn in valid_snaps if sid == final_snap_id),
            None,
        )
        if final_data is None:
            raise ValueError(
                f"Final snapshot {final_snap_id} not found or has no valid TreeCorr data"
            )
    else:
        final_data = valid_snaps[-1]
    
    final_snap_id, final_a, final_nn = final_data
    
    # Set up bands
    band_boundaries = scale_bands
    if band_boundaries is None and n_bands is not None:
        band_boundaries = create_log_spaced_bands(
            r_min=r_phys_grid.min(),
            r_max=r_phys_grid.max(),
            n_bands=n_bands,
        )
    
    # Initialize output arrays
    n_snap = len(valid_snaps)
    n_bins = len(r_phys_grid)
    snap_ids_out = np.array([s[0] for s in valid_snaps])
    scale_factors_out = np.array([s[1] for s in valid_snaps])
    frobenius_global = np.full(n_snap, np.nan)
    valid_bins_per_epoch = np.zeros(n_snap, dtype=int)
    
    frobenius_bands: dict[str, np.ndarray] | None = None
    if band_boundaries is not None:
        frobenius_bands = {
            name: np.full(n_snap, np.nan) for name in band_boundaries
        }
    
    # First pass: compute all correlation matrices and track valid bins
    corr_matrices = []
    valid_masks_per_epoch = []
    
    for i, (snap_id, a, nn) in enumerate(valid_snaps):
        corr_a, _, n_valid = corr_matrix_on_physical_grid_from_treecorr(
            treecorr_nn=nn,
            scale_factor=a,
            r_phys_grid=r_phys_grid,
            method=method,
            cross_patch_weight=cross_patch_weight,
        )
        corr_matrices.append(corr_a)
        valid_bins_per_epoch[i] = n_valid
        
        # Track which bins are valid (diagonal finite means row/col is valid)
        diag_valid = np.isfinite(np.diag(corr_a))
        valid_masks_per_epoch.append(diag_valid)
    
    # Compute fixed masks if requested
    fixed_mask_global: np.ndarray | None = None
    fixed_masks_bands: dict[str, np.ndarray] | None = None
    
    if use_fixed_mask:
        # Determine which epochs to include in mask computation
        mask_min_a = fixed_mask_min_scale_factor if fixed_mask_min_scale_factor is not None else min_scale_factor
        
        # Compute intersection of valid bins across relevant epochs
        fixed_mask_global = np.ones(n_bins, dtype=bool)
        for i, (snap_id, a, nn) in enumerate(valid_snaps):
            if a >= mask_min_a:
                fixed_mask_global &= valid_masks_per_epoch[i]
        
        # Also ensure final epoch is valid
        final_idx = next(i for i, (sid, _, _) in enumerate(valid_snaps) if sid == final_snap_id)
        fixed_mask_global &= valid_masks_per_epoch[final_idx]
        
        # Compute per-band fixed masks
        if band_boundaries is not None:
            fixed_masks_bands = {}
            for band_name, (r_lo, r_hi) in band_boundaries.items():
                band_mask = get_band_indices(r_phys_grid, (r_lo, r_hi))
                # Intersection of band bins with globally valid bins
                fixed_masks_bands[band_name] = band_mask & fixed_mask_global
    
    # Get final correlation matrix
    final_idx = next(i for i, (sid, _, _) in enumerate(valid_snaps) if sid == final_snap_id)
    corr_final = corr_matrices[final_idx]
    
    # Second pass: compute Frobenius distances
    for i, (snap_id, a, nn) in enumerate(valid_snaps):
        corr_a = corr_matrices[i]
        
        if valid_bins_per_epoch[i] == 0:
            continue
        
        # Global Frobenius distance
        if use_fixed_mask and fixed_mask_global is not None:
            # Use only bins that are valid across all epochs
            if np.sum(fixed_mask_global) == 0:
                frobenius_global[i] = np.nan
            else:
                mask_idx = np.where(fixed_mask_global)[0]
                corr_a_masked = corr_a[np.ix_(mask_idx, mask_idx)]
                corr_final_masked = corr_final[np.ix_(mask_idx, mask_idx)]
                frobenius_global[i] = frobenius_distance(
                    corr_a_masked, corr_final_masked, normalize=True, use_valid_only=False
                )
        else:
            frobenius_global[i] = frobenius_distance(
                corr_a, corr_final, normalize=True, use_valid_only=True
            )
        
        # Per-band distances
        if band_boundaries is not None and frobenius_bands is not None:
            for band_name, (r_lo, r_hi) in band_boundaries.items():
                if use_fixed_mask and fixed_masks_bands is not None:
                    # Use fixed mask for this band
                    band_fixed_mask = fixed_masks_bands[band_name]
                    if np.sum(band_fixed_mask) == 0:
                        frobenius_bands[band_name][i] = np.nan
                        continue
                    
                    mask_idx = np.where(band_fixed_mask)[0]
                    corr_a_band = corr_a[np.ix_(mask_idx, mask_idx)]
                    corr_final_band = corr_final[np.ix_(mask_idx, mask_idx)]
                    frobenius_bands[band_name][i] = frobenius_distance(
                        corr_a_band, corr_final_band, normalize=True, use_valid_only=False
                    )
                else:
                    band_mask = get_band_indices(r_phys_grid, (r_lo, r_hi))
                    if not np.any(band_mask):
                        continue
                    
                    band_idx = np.where(band_mask)[0]
                    corr_a_band = corr_a[np.ix_(band_idx, band_idx)]
                    corr_final_band = corr_final[np.ix_(band_idx, band_idx)]
                    
                    frobenius_bands[band_name][i] = frobenius_distance(
                        corr_a_band, corr_final_band, normalize=True, use_valid_only=True
                    )
    
    return FreezeOutMetrics(
        scale_factors=scale_factors_out,
        snapshot_ids=snap_ids_out,
        r_phys_grid=r_phys_grid,
        frobenius_global=frobenius_global,
        band_boundaries=band_boundaries,
        frobenius_bands=frobenius_bands,
        valid_bins_per_epoch=valid_bins_per_epoch,
        fixed_mask_global=fixed_mask_global,
        fixed_masks_bands=fixed_masks_bands,
        used_fixed_mask=use_fixed_mask,
    )
