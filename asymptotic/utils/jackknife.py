import numpy as np

def jackknife_resampling(samples: list[np.ndarray]) -> list[np.ndarray]:
    if len(samples) <= 1:
        return samples
    else:
        return [
            np.hstack(samples[:i] + samples[i + 1 :]) 
            for i in range(len(samples))
        ]

def jackknife_estimate(samples: list[np.ndarray]) -> np.ndarray:
    return np.vstack(samples).mean(axis=0)

def jackknife_error(samples: list[np.ndarray]) -> np.ndarray:
    return np.vstack(samples).std(axis=0) * np.sqrt(len(samples) - 1)


def get_jackknife_volume(n_samples: int, box_size: float) -> float:
    """Compute the volume of the jackknifed sub-sample."""
    return (box_size ** 3) * (max(1, n_samples - 1) / n_samples)


def get_jackknife_results(samples: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute the jackknife estimate and error from a list of samples."""
    jk_samples = jackknife_resampling(samples)
    return {
        "estimate": jackknife_estimate(jk_samples),
        "error": jackknife_error(jk_samples)
    }

def get_jackknife_subbox_samples(
        sub_box_info: dict[int, list[tuple[float, float]]], # {1: [(x_i, x_f), (y_i, y_f), (z_i, z_f)], ...}
        coordinates: np.ndarray,
        return_as_mask: bool = True
    ) -> list[np.ndarray]:

    mask = lambda coords: np.logical_and.reduce([
        (coordinates[:, 0] >= coords[0][0]) & (coordinates[:, 0] <= coords[0][1]),
        (coordinates[:, 1] >= coords[1][0]) & (coordinates[:, 1] <= coords[1][1]),
        (coordinates[:, 2] >= coords[2][0]) & (coordinates[:, 2] <= coords[2][1])
    ])

    return [
        mask(coords) if return_as_mask else coordinates[mask(coords)] 
        for coords in sub_box_info.values()
    ]


def jackknife_correlation_matrix(jk_folds: list[np.ndarray], std_err: np.ndarray) -> np.ndarray:
    """
    Build a correlation matrix from delete-1 jackknife folds for any binned statistic.

    C_{ij} = (n-1)/n * sum_k (x_k,i - xbar_i)(x_k,j - xbar_j) / (sigma_i sigma_j)

    Parameters
    ----------
    xi_folds : list of (nbins,) arrays
        Per-fold vectors (leave-one-out samples) of the statistic of interest.
    xi_err : (nbins,) array
        Jackknife standard error per bin (already computed) for that statistic.

    Returns
    -------
    corr : (nbins, nbins) array
        Correlation coefficients between bins.
    """


    nfolds = len(jk_folds)
    if nfolds < 2:
        # Not enough folds to estimate covariance
        nb = int(np.asarray(std_err).shape[0])
        return np.eye(nb, dtype=float)

    X = np.vstack([np.asarray(fold, dtype=float) for fold in jk_folds])  # (nfolds, nbins)
    Xbar = np.mean(X, axis=0)
    diffs = X - Xbar
    # Standard delete-1 jackknife covariance: (n-1)/n * sum_i (x_i - xbar)(x_i - xbar)^T
    cov = (nfolds - 1.0) / nfolds * (diffs.T @ diffs)

    # Correlation matrix using the jackknife errors on the diagonal
    denom = np.outer(std_err, std_err)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, cov / denom, 0.0)

    # Ensure diagonal is exactly 1 for bins with nonzero error
    for i in range(corr.shape[0]):
        corr[i, i] = 1.0 if std_err[i] > 0.0 else 0.0

    return corr