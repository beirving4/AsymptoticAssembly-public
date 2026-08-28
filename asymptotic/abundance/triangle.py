from __future__ import annotations

import numpy as np, pdb

from attrs import define, field 
from scipy.interpolate import interp1d

MaskType = bool | np.ndarray


@define(slots=True)
class CoverageTriangle:
    min_count: int 
    min_density: float

    min_sample_value: float
    max_sample_value: float

    abundance_bounds: np.ndarray

    min_peak_height: float = field(repr=False)
    max_peak_height: float = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"CoverageTriangle(min={self.min_sample_value:.2f}, "
            f"max={self.max_sample_value:.2f})"
        )

    @classmethod
    def from_abundance(
            cls, 
            log10_differential: np.ndarray,
            bin_centers: np.ndarray,
            comoving_volume: float,
            min_count: int, 
            min_sample_value: float, 
            min_peak_height: float,
            max_peak_height: float
        ) -> CoverageTriangle:

        min_number_density = min_count / comoving_volume
        dlnx = np.diff(np.log(bin_centers))[0]
        min_dndlnx = np.log10(min_number_density / dlnx)

        above_min_value = mass_bin_centers >= min_sample_value
        above_density = 10.0**log10_differential >= 10.0**min_dndlnx
        bounds = above_min_value * above_density
        
        try: 
            cropped_phi = log10_differential[bounds]
            sample_interp = interp1d(cropped_phi, bin_centers[bounds])
            value_at_min_dndlnx = mass_interp(min_dndlnx)
        except ValueError: 
            value_at_min_dndlnx = bin_centers.max()

        return cls(
            min_count=min_count,
            min_density=min_dndlnx,
            min_sample_value=np.log10(min_mass), 
            max_sample_value=np.log10(value_at_min_dndlnx), 
            abundance_bounds=bounds,
            min_peak_height=min_peak_height,
            max_peak_height=max_peak_height
        )
    
    @classmethod
    def from_histogram(
            cls, 
            histogram: np.ndarray,
            bin_centers: np.ndarray,
            comoving_volume: float,
            min_count: int, 
            min_sample_value: float, 
            min_peak_height: float,
            max_peak_height: float
        ) -> CoverageTriangle:

        cumulative_counts = np.cumsum(histogram[::-1])[::-1]
        dlnx = np.diff(np.log(bin_centers))[0]
        phi = (histogram / comoving_volume) / dlnx
        min_dndlnx = np.log10(min_count / comoving_volume / dlnx)

        above_min_value = bin_centers >= min_sample_value
        above_count = cumulative_counts >= min_count
        bounds = above_min_value * above_count
        
        try: 
            cropped_log10_phi = np.log10(phi[bounds])
            sample_interp = interp1d(cropped_log10_phi, bin_centers[bounds])
            value_at_min_dndlnx = sample_interp(min_dndlnx)
        except ValueError: 
            value_at_min_dndlnx = bin_centers.max()

        return cls(
            min_count=min_count,
            min_density=min_dndlnx,
            min_sample_value=np.log10(min_sample_value), 
            max_sample_value=np.log10(value_at_min_dndlnx), 
            abundance_bounds=bounds,
            min_peak_height=min_peak_height,
            max_peak_height=max_peak_height
        )
    
    def is_above_min_sample_value(self, value: float | np.ndarray) -> MaskType:
        return value >= self.min_sample_value
    
    def values_above_min(self, values: np.ndarray) -> np.ndarry:
        return values[self.is_above_min_sample_value(values)]
    
    def is_below_max_sample_value(self, value: float | np.ndarray) -> MaskType:
        return value <= self.max_sample_value

    def mass_below_max(self, values: np.ndarray) -> np.ndarray:
        return values[self.is_below_max_sample_value(values)]
    
    def is_value_in_bounds(self, value: float | np.ndarray) -> MaskType:
        return np.logical_and(
            self.is_above_min_sample_value(value), 
            self.is_below_max_sample_value(value)
        )

    def values_in_bounds(self, values: np.ndarray) -> np.ndarray:
        return values[self.is_values_in_bounds(values)]
    

'''

@define 
class CoverageTriangle:
    min_count: int 
    min_density: float

    min_mass: float
    max_mass: float

    mass_function_bounds: np.ndarray

    min_peak_height: float = field(repr=False)
    max_peak_height: float = field(repr=False)

    def __repr__(self) -> str: 
        return f"CoverageTriangle(min={self.min_mass:.2f}, max={self.max_mass:.2f})"

    @classmethod
    def from_mass_function(
            cls, 
            log10_differential: np.ndarray,
            mass_bin_centers: np.ndarray,
            comoving_volume: float,
            min_count: int, 
            min_mass: float, 
            min_peak_height: float,
            max_peak_height: float
        ) -> CoverageTriangle:

        min_number_density = min_count / comoving_volume
        dlnM = np.diff(np.log(mass_bin_centers))[0]
        min_dndlnM = np.log10(min_number_density / dlnM)

        above_mass = mass_bin_centers >= min_mass
        above_density = 10.0**log10_differential >= 10.0**min_dndlnM
        bounds = above_mass * above_density
        
        try: 
            cropped_phi = log10_differential[bounds]
            mass_interp = interp1d(cropped_phi, mass_bin_centers[bounds])
            mass_at_min_dndlnM = mass_interp(min_dndlnM)
        except ValueError: 
            mass_at_min_dndlnM = mass_bin_centers.max()

        # pdb.set_trace()

        return cls(
            min_count=min_count,
            min_density=min_dndlnM,
            min_mass=np.log10(min_mass), 
            max_mass=np.log10(mass_at_min_dndlnM), 
            mass_function_bounds=bounds,
            min_peak_height=min_peak_height,
            max_peak_height=max_peak_height
        )
    
    @classmethod
    def from_histogram(
            cls, 
            histogram: np.ndarray,
            mass_bin_centers: np.ndarray,
            comoving_volume: float,
            min_count: int, 
            min_mass: float, 
            min_peak_height: float,
            max_peak_height: float
        ) -> CoverageTriangle:

        cumulative_counts = np.cumsum(histogram[::-1])[::-1]
        dlnM = np.diff(np.log(mass_bin_centers))[0]
        phi = (histogram / comoving_volume) / dlnM
        min_dndlnM = np.log10(min_count / comoving_volume / dlnM)

        above_mass = mass_bin_centers >= min_mass
        above_count = cumulative_counts >= min_count
        bounds = above_mass * above_count
        
        try: 
            cropped_log10_phi = np.log10(phi[bounds])
            mass_interp = interp1d(cropped_log10_phi, mass_bin_centers[bounds])
            mass_at_min_dndlnM = mass_interp(min_dndlnM)
        except ValueError: 
            mass_at_min_dndlnM = mass_bin_centers.max()

        return cls(
            min_count=min_count,
            min_density=min_dndlnM,
            min_mass=np.log10(min_mass), 
            max_mass=np.log10(mass_at_min_dndlnM), 
            mass_function_bounds=bounds,
            min_peak_height=min_peak_height,
            max_peak_height=max_peak_height
        )
    
    def is_above_min_mass(self, mass: float | np.ndarray) -> MaskType:
        return mass >= self.min_mass
    
    def mass_above_min(self, masses: np.ndarray) -> np.ndarry:
        return masses[self.is_above_min_mass(masses)]
    
    def is_below_max_mass(self, mass: float | np.ndarray) -> MaskType:
        return mass <= self.max_mass 

    def mass_below_max(self, masses: np.ndarray) -> np.ndarray:
        return masses[self.is_below_max_mass(masses)]
    
    def is_mass_in_bounds(self, mass: float | np.ndarray) -> MaskType:
        if isinstance(mass, float):
            return (mass >= self.min_mass) and (mass <= self.max_mass)
        else:
            return (mass >= self.min_mass) * (mass <= self.max_mass)

    def mass_in_bounds(self, masses: np.ndarray) -> np.ndarray:
        return masses[self.is_mass_in_bounds(masses)]

'''