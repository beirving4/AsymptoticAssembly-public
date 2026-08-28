from __future__ import annotations

import numpy as np, pdb

from enum import StrEnum
from attrs import define, field
from scipy.interpolate import interp1d
from scipy.integrate import quad

from .filter import get_filter_function
        

def compute_mass_variance(
        wavenumbers: np.ndarray,
        power_spectrum: np.ndarray,
        filter_radius: float,
        filter_type: str = "tophat",
    ) -> float:

    """Compute the mass variance σ²(R) for a given filter radius R.

    Args:
        wavenumbers (np.ndarray): Array of wavenumbers k.
        power_spectrum (np.ndarray): Corresponding power spectrum P(k).
        filter_radius (float): Filter radius R.
        filter_type (str): Type of filter function to use ('tophat', 'gaussian', 'sharp_k').

    Returns:
        float: The mass variance σ²(R).
    """
    # Get the filter function
    filter_function = get_filter_function(filter_type)

    # Compute kR
    kR = wavenumbers * filter_radius

    # Compute the integrand
    W_kR = filter_function(kR)
    integrand = kR**2 * power_spectrum * W_kR**2

    return quad(integrand, 0, np.inf)[0] / (2 * np.pi**2)


def get_mass_variance_from_masses(
        masses: np.ndarray,
        power_spectrum: np.ndarray,
        filter_radius: float,
        present_day_mean_density: float,
        filter_type: str = "tophat"
    ) -> np.ndarray:

    """Compute the mass variance σ²(M) for an array of masses M.

    Args:
        masses (np.ndarray): Array of masses M.
        power_spectrum (np.ndarray): Corresponding power spectrum P(k).
        filter_radius (float): Filter radius R.
        present_day_mean_density (float): Present day mean density ρ0.
        filter_type (str): Type of filter function to use ('tophat', 'gaussian', 'sharp_k').

    Returns:
        np.ndarray: Array of mass variances σ²(M).
    """
    lagrange_radii = (3 * masses / (4 * np.pi * present_day_mean_density))**(1/3)
    wavenumbers = (2 * np.pi) / lagrange_radii

    return np.array([
        compute_mass_variance(
            wavenumbers=wavenumbers[i],
            power_spectrum=power_spectrum,
            filter_radius=filter_radius,
            filter_type=filter_type
        ) for i in range(len(masses))
    ])


def get_mass_variance_from_radii(
        radii: np.ndarray,
        power_spectrum: np.ndarray,
        filter_type: str = "tophat"
    ) -> np.ndarray:

    """Compute the mass variance σ²(R) for an array of filter radii R.

    Args:
        radii (np.ndarray): Array of filter radii R.
        power_spectrum (np.ndarray): Corresponding power spectrum P(k).
        filter_type (str): Type of filter function to use ('tophat', 'gaussian', 'sharp_k').

    Returns:
        np.ndarray: Array of mass variances σ²(R).
    """
    wavenumbers = (2 * np.pi) / radii

    return np.array([
        compute_mass_variance(
            wavenumbers=wavenumbers[i],
            power_spectrum=power_spectrum,
            filter_radius=radii[i],
            filter_type=filter_type
        ) for i in range(len(radii))
    ])


def get_mass_variance(
        values: np.ndarray,
        power_spectrum: np.ndarray,
        filter_radius: float,
        present_day_mean_density: float,
        filter_type: str = "tophat",
        from_masses: bool = True
    ) -> np.ndarray:

    return (
        get_mass_variance_from_masses(
            masses=values,
            power_spectrum=power_spectrum,
            filter_radius=filter_radius,
            present_day_mean_density=present_day_mean_density,
            filter_type=filter_type
        ) if from_masses else 
        get_mass_variance_from_radii(
            radii=values,
            power_spectrum=power_spectrum,
            filter_type=filter_type
        )
    )